#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 2 training (ICI prior) using image features = concat(attr_hat, dis_hat)
predicted by PARA Phase1 ResNet50.

Modes:
  - use_external=True  (Full / in-domain):
      uses user attributes + user_id embedding + image(attr_hat, dis_hat)
      and enables external interaction term.
  - use_external=False (Reduced / cross-domain):
      uses ONLY user_id embedding (no user attrs) + image(attr_hat, dis_hat)
      and disables external interaction term.

Training data:
  - ALWAYS PARA train_interactions.csv from ici_splits/para (Phase1/2 trained on PARA only)

Output:
  - checkpoint with:
      phase1 meta, user_id_to_idx, (optional) user_feat_cols, etc.
"""

import os
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Phase1 model definition (must match phase1_train_resnet.py)
# -----------------------------

class ResNet50Phase1(nn.Module):
    def __init__(self, n_attr: int, n_bins: int):
        super().__init__()
        base = models.resnet50(weights=None)
        feat_dim = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.attr_head = nn.Linear(feat_dim, n_attr)
        self.dis_head = nn.Linear(feat_dim, n_bins)

    def forward(self, x):
        feat = self.backbone(x)
        attr = self.attr_head(feat)
        dis_logits = self.dis_head(feat)
        return feat, attr, dis_logits


def load_phase1_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    attr_cols = ckpt["attr_cols"]
    dis_cols = ckpt["dis_cols"]
    model = ResNet50Phase1(n_attr=len(attr_cols), n_bins=len(dis_cols))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, attr_cols, dis_cols


def default_transform():
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


@torch.inference_mode()
def extract_attr_dis_features(model, image_paths: List[str], device: str, batch_size: int = 64) -> Dict[str, np.ndarray]:
    tfm = default_transform()
    feats: Dict[str, np.ndarray] = {}

    batch_imgs, batch_paths = [], []

    def flush():
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs, dim=0).to(device)
        _, attr, dis_logits = model(x)
        dis = torch.softmax(dis_logits, dim=-1)
        f = torch.cat([attr, dis], dim=-1)  # [B, n_attr+n_bins]
        f = f.detach().cpu().numpy().astype(np.float32)
        for p, vec in zip(batch_paths, f):
            feats[p] = vec
        batch_imgs.clear()
        batch_paths.clear()

    for p in tqdm(image_paths, desc="Phase1Feat[attr+dis]"):
        img = Image.open(p).convert("RGB")
        x = tfm(img)
        batch_imgs.append(x)
        batch_paths.append(p)
        if len(batch_imgs) >= batch_size:
            flush()
    flush()
    return feats


# -----------------------------
# Phase2 Dataset
# -----------------------------

def _onehot_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    return pd.get_dummies(df, columns=cols, dummy_na=True)


class Phase2Dataset(Dataset):
    """
    Returns:
      user_idx: Long
      img_feat: Float [Din]
      user_feat: Float [Udim]  (may be None if use_external=False)
      y: Float scalar
    """
    def __init__(self, df: pd.DataFrame, user_id_to_idx: Dict[str, int], img_feat_map: Dict[str, np.ndarray],
                 user_feat_table: Optional[pd.DataFrame], user_feat_cols: Optional[List[str]]):
        self.df = df.reset_index(drop=True)
        self.user_id_to_idx = user_id_to_idx
        self.img_feat_map = img_feat_map
        self.user_feat_table = user_feat_table  # indexed by user_id
        self.user_feat_cols = user_feat_cols

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        uid = str(row["user_id"])
        ip = str(row["image_path"])
        y = float(row["score"])

        uidx = self.user_id_to_idx[uid]
        img_feat = self.img_feat_map[ip]

        if self.user_feat_table is None:
            return (
                torch.tensor(uidx, dtype=torch.long),
                torch.tensor(img_feat, dtype=torch.float32),
                None,
                torch.tensor(y, dtype=torch.float32),
            )

        ufeat = self.user_feat_table.loc[uid, self.user_feat_cols].to_numpy(dtype=np.float32)
        return (
            torch.tensor(uidx, dtype=torch.long),
            torch.tensor(img_feat, dtype=torch.float32),
            torch.tensor(ufeat, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


def collate_fn(batch):
    user_idx = torch.stack([b[0] for b in batch], dim=0)
    img_feat = torch.stack([b[1] for b in batch], dim=0)
    ys = torch.stack([b[3] for b in batch], dim=0)
    # user_feat may be None
    if batch[0][2] is None:
        return user_idx, img_feat, None, ys
    user_feat = torch.stack([b[2] for b in batch], dim=0)
    return user_idx, img_feat, user_feat, ys


# -----------------------------
# Phase2 Model
# -----------------------------

class Phase2ICI(nn.Module):
    """
    - d=64
    - MLP hidden layers=1, hidden=256
    - use_external:
        True  -> uses user attributes + external term
        False -> NO user attributes, NO external term
    """
    def __init__(self, num_users: int, img_in_dim: int, user_feat_dim: int,
                 d: int = 64, hidden: int = 256, use_external: bool = False):
        super().__init__()
        self.use_external = use_external

        self.user_emb = nn.Embedding(num_users, d)
        self.img_proj = nn.Linear(img_in_dim, d)

        if use_external:
            # user attributes are only used in external/full mode
            self.user_feat_proj = nn.Linear(user_feat_dim, d)
        else:
            self.user_feat_proj = None

        self.mlp = nn.Sequential(
            nn.Linear(d * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        self.ext_mlp = nn.Sequential(
            nn.Linear(d * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, user_idx, img_feat, user_feat=None):
        u = self.user_emb(user_idx)     # [B,d]
        if self.use_external:
            # full mode: add demographic embedding
            u = u + self.user_feat_proj(user_feat)

        v = self.img_proj(img_feat)     # [B,d]
        internal = self.mlp(torch.cat([u, v], dim=-1)).squeeze(-1)

        if not self.use_external:
            return internal

        # external (batch proxy)
        u_ctx = u.mean(dim=0, keepdim=True).expand_as(u)
        v_ctx = v.mean(dim=0, keepdim=True).expand_as(v)
        external = self.ext_mlp(torch.cat([u * u_ctx, v * v_ctx], dim=-1)).squeeze(-1)
        return internal + external


# -----------------------------
# Train
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", required=True, help="ici_splits/para")
    ap.add_argument("--phase1_ckpt", required=True)
    ap.add_argument("--out_ckpt", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--use_external", action="store_true",
                    help="If set: use user attributes + external interaction (full mode). "
                         "If not set: image-only + internal-only (reduced mode).")
    ap.add_argument("--max_rows", type=int, default=None)
    ap.add_argument("--feat_batch", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- train interactions ---
    train_csv = os.path.join(args.split_dir, "train_interactions.csv")
    df = pd.read_csv(train_csv).dropna(subset=["user_id", "image_path", "score"])
    if args.max_rows is not None and args.max_rows < len(df):
        df = df.sample(n=args.max_rows, random_state=42).reset_index(drop=True)

    user_ids = sorted(df["user_id"].astype(str).unique().tolist())
    user_id_to_idx = {u: i for i, u in enumerate(user_ids)}
    img_paths = sorted(df["image_path"].astype(str).unique().tolist())

    # --- phase1 feature extraction (attr_hat + dis_hat) ---
    phase1, attr_cols, dis_cols = load_phase1_model(args.phase1_ckpt, device=device)
    img_feat_map = extract_attr_dis_features(phase1, img_paths, device=device, batch_size=args.feat_batch)
    img_in_dim = len(next(iter(img_feat_map.values())))
    print(f"[info] img_in_dim={img_in_dim} (= {len(attr_cols)} attr_hat + {len(dis_cols)} dis_hat)")

    # --- user attributes only when use_external=True ---
    user_feat_table = None
    user_feat_cols = None
    user_feat_dim = 0

    if args.use_external:
        df_u = pd.read_csv(os.path.join(args.split_dir, "user_attrs.csv"))
        df_u = df_u.rename(columns={"userId": "user_id"}) if "userId" in df_u.columns else df_u
        df_u["user_id"] = df_u["user_id"].astype(str)

        cat_cols = [c for c in ["age", "gender", "EducationalLevel", "artExperience", "photographyExperience"] if c in df_u.columns]
        df_u = _onehot_columns(df_u, cat_cols)

        user_feat_cols = [c for c in df_u.columns if c != "user_id"]
        for c in user_feat_cols:
            df_u[c] = pd.to_numeric(df_u[c], errors="coerce").fillna(0.0)

        user_feat_table = df_u.set_index("user_id")

        # filter df to users having attrs (should be most)
        ok = df["user_id"].astype(str).isin(user_feat_table.index)
        df = df[ok].reset_index(drop=True)

        user_feat_dim = len(user_feat_cols)
        print(f"[info] user_feat_dim={user_feat_dim} (use_external=True)")

    # dataset/loader
    ds = Phase2Dataset(df, user_id_to_idx, img_feat_map, user_feat_table, user_feat_cols)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True,
                        collate_fn=collate_fn)

    # model
    model = Phase2ICI(
        num_users=len(user_ids),
        img_in_dim=img_in_dim,
        user_feat_dim=max(1, user_feat_dim),  # dummy for reduced mode
        d=args.d,
        hidden=args.hidden,
        use_external=args.use_external,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for user_idx, img_feat, user_feat, y in loader:
            user_idx = user_idx.to(device)
            img_feat = img_feat.to(device)
            y = y.to(device)
            if args.use_external:
                user_feat = user_feat.to(device)

            pred = model(user_idx, img_feat, user_feat=user_feat)
            loss = F.mse_loss(pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())

        print(f"[ep {ep:02d}] mse={sum(losses)/max(1,len(losses)):.4f}")

    os.makedirs(os.path.dirname(args.out_ckpt) or ".", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "meta": {
            "phase1_ckpt": args.phase1_ckpt,
            "attr_cols": attr_cols,
            "dis_cols": dis_cols,
            "img_in_dim": img_in_dim,
            "user_id_to_idx": user_id_to_idx,
            "d": args.d,
            "hidden": args.hidden,
            "use_external": args.use_external,
            "user_feat_cols": user_feat_cols if args.use_external else None,
        }
    }, args.out_ckpt)

    print(f"[save] {args.out_ckpt}")


if __name__ == "__main__":
    main()