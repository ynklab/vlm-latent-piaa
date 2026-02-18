#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 2 (Aesthetic prior model building) - faithful-ish implementation of PIAA-ICI.

Key design (per your request):
- Image graph: ALWAYS K nodes corresponding to Phase1 predicted attr_hat (K-dim).
  (cross-domain also uses attr_hat nodes)
- User graph:
    * use_external=True  -> user attributes nodes + user_id node, external enabled
    * use_external=False -> single node = user_id only, external disabled
- dis_hat is NOT used inside interactions; it is fused at the end via FC_s(dis_hat) (Eq. 13).

Input:
  --split_dir    ici_splits/para
  --phase1_ckpt  runs/ici_phase1_para_resnet50.pth
  --out_ckpt     runs/ici_phase2_*.pth
  --use_external (flag)

Data:
  - train_interactions.csv: user_id, image_path, score (personalized)
  - user_attrs.csv (PARA-UserInfo): only needed when use_external=True

Training:
  - Adam, lr=1e-4, epochs=50 (paper)
  - d=64, hidden=256, MLP depth=1 (paper)

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
        attr = self.attr_head(feat)          # [B,K]
        dis_logits = self.dis_head(feat)     # [B,Bin]
        return feat, attr, dis_logits


def default_transform():
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def load_phase1_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    attr_cols = ckpt["attr_cols"]
    dis_cols = ckpt["dis_cols"]
    model = ResNet50Phase1(n_attr=len(attr_cols), n_bins=len(dis_cols))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, attr_cols, dis_cols


@torch.inference_mode()
def extract_attr_dis_for_paths(
    model: nn.Module,
    image_paths: List[str],
    device: str,
    batch_size: int = 64,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Returns:
      attr_map: image_path -> attr_hat [K]
      dis_map : image_path -> dis_hat  [Bin] (softmax)
    """
    tfm = default_transform()
    attr_map: Dict[str, np.ndarray] = {}
    dis_map: Dict[str, np.ndarray] = {}

    imgs, paths = [], []

    def flush():
        if not imgs:
            return
        x = torch.stack(imgs, dim=0).to(device)
        _, attr, dis_logits = model(x)
        dis = torch.softmax(dis_logits, dim=-1)
        attr_np = attr.detach().cpu().numpy().astype(np.float32)
        dis_np = dis.detach().cpu().numpy().astype(np.float32)
        for p, a, d in zip(paths, attr_np, dis_np):
            attr_map[p] = a
            dis_map[p] = d
        imgs.clear()
        paths.clear()

    for p in tqdm(image_paths, desc="Phase1 forward (attr_hat, dis_hat)"):
        img = Image.open(p).convert("RGB")
        imgs.append(tfm(img))
        paths.append(p)
        if len(imgs) >= batch_size:
            flush()
    flush()
    return attr_map, dis_map


# -----------------------------
# User attributes encoding (PARA)
# -----------------------------

PARA_CAT_FIELDS = ["age", "gender", "EducationalLevel", "artExperience", "photographyExperience"]
PARA_NUM_FIELDS = ["personality-E", "personality-A", "personality-N", "personality-O", "personality-C"]

def build_para_user_attr_tables(user_attrs_csv: str, user_ids: List[str]):
    """
    Returns:
      user_feat_cat: dict[field] -> dict[user_id] -> int category index
      cat_vocab: dict[field] -> list of categories
      user_feat_num: dict[user_id] -> np.ndarray [len(PARA_NUM_FIELDS)] (float32)
    """
    df = pd.read_csv(user_attrs_csv)
    df = df.rename(columns={"userId": "user_id"}) if "userId" in df.columns else df
    df["user_id"] = df["user_id"].astype(str)
    df = df[df["user_id"].isin(set(user_ids))].copy().set_index("user_id")

    cat_vocab: Dict[str, List[str]] = {}
    user_feat_cat: Dict[str, Dict[str, int]] = {}
    for f in PARA_CAT_FIELDS:
        if f not in df.columns:
            raise RuntimeError(f"Missing categorical field in PARA user_attrs.csv: {f}")
        cats = sorted(df[f].astype(str).fillna("NA").unique().tolist())
        cat_vocab[f] = cats
        cat_to_idx = {c: i for i, c in enumerate(cats)}
        user_feat_cat[f] = {uid: cat_to_idx[str(df.loc[uid, f])] for uid in df.index}

    # numeric
    for f in PARA_NUM_FIELDS:
        if f not in df.columns:
            raise RuntimeError(f"Missing numeric field in PARA user_attrs.csv: {f}")
    user_feat_num: Dict[str, np.ndarray] = {}
    for uid in df.index:
        vals = []
        for f in PARA_NUM_FIELDS:
            v = pd.to_numeric(df.loc[uid, f], errors="coerce")
            if np.isnan(v):
                v = 0.0
            vals.append(float(v))
        user_feat_num[uid] = np.array(vals, dtype=np.float32)

    return user_feat_cat, cat_vocab, user_feat_num


# -----------------------------
# Dataset
# -----------------------------

class Phase2Dataset(Dataset):
    """
    Each item:
      user_idx: Long
      attr_hat: Float [K]
      dis_hat : Float [Bin]
      user_cat: dict[field]->Long (optional, only when use_external=True)
      user_num: Float [P] (optional)
      y: Float
    """
    def __init__(
        self,
        df: pd.DataFrame,
        user_id_to_idx: Dict[str, int],
        attr_map: Dict[str, np.ndarray],
        dis_map: Dict[str, np.ndarray],
        use_external: bool,
        user_feat_cat: Optional[Dict[str, Dict[str, int]]] = None,
        user_feat_num: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.user_id_to_idx = user_id_to_idx
        self.attr_map = attr_map
        self.dis_map = dis_map
        self.use_external = use_external
        self.user_feat_cat = user_feat_cat
        self.user_feat_num = user_feat_num

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        uid = str(r["user_id"])
        ip = str(r["image_path"])
        y = float(r["score"])

        uidx = self.user_id_to_idx[uid]
        attr_hat = self.attr_map[ip]
        dis_hat = self.dis_map[ip]

        if not self.use_external:
            return (
                torch.tensor(uidx, dtype=torch.long),
                torch.tensor(attr_hat, dtype=torch.float32),
                torch.tensor(dis_hat, dtype=torch.float32),
                None, None,
                torch.tensor(y, dtype=torch.float32),
            )

        # categorical fields -> ints
        cat = {f: self.user_feat_cat[f][uid] for f in PARA_CAT_FIELDS}
        num = self.user_feat_num[uid]

        return (
            torch.tensor(uidx, dtype=torch.long),
            torch.tensor(attr_hat, dtype=torch.float32),
            torch.tensor(dis_hat, dtype=torch.float32),
            {f: torch.tensor(cat[f], dtype=torch.long) for f in PARA_CAT_FIELDS},
            torch.tensor(num, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


def collate_fn(batch):
    user_idx = torch.stack([b[0] for b in batch], dim=0)
    attr_hat = torch.stack([b[1] for b in batch], dim=0)
    dis_hat  = torch.stack([b[2] for b in batch], dim=0)
    y = torch.stack([b[5] for b in batch], dim=0)

    if batch[0][3] is None:
        return user_idx, attr_hat, dis_hat, None, None, y

    cat_dict = {f: torch.stack([b[3][f] for b in batch], dim=0) for f in PARA_CAT_FIELDS}
    num = torch.stack([b[4] for b in batch], dim=0)
    return user_idx, attr_hat, dis_hat, cat_dict, num, y


# -----------------------------
# Phase2 ICI model (faithful-ish)
# -----------------------------

class PairMLP(nn.Module):
    """MLP for pairwise interaction: concat(a,b) -> d"""
    def __init__(self, d: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2*d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d),
        )
    def forward(self, a, b):
        return self.net(torch.cat([a, b], dim=-1))


class Phase2ICI(nn.Module):
    """
    Graph-style Phase2 with:
      - internal interactions on user graph and image graph (complete graph)
      - external interactions (Bi-Interaction) ONLY when use_external=True
      - GRU fusion (information fusion)
      - post-interaction aggregation
      - final score: y_hat = b_hat + FC_s(dis_hat)

    Cross-domain mode:
      - use_external=False => user graph is single node (user_id), external off
      - image graph still uses K attr nodes
    """

    def __init__(
        self,
        num_users: int,
        K_img: int,
        Bin: int,
        d: int = 64,
        hidden: int = 256,
        use_external: bool = False,
        # user nodes when full mode
        cat_vocab_sizes: Optional[Dict[str, int]] = None,
        num_user_num: int = 5,
    ):
        super().__init__()
        self.use_external = use_external
        self.d = d
        self.K_img = K_img

        # ---- Image attribute nodes: K nodes, each scalar -> d
        self.img_node_proj = nn.Linear(1, d)

        # ---- User graph nodes:
        # reduced: single node = user_id embedding
        self.user_id_emb = nn.Embedding(num_users, d)

        if use_external:
            assert cat_vocab_sizes is not None
            # categorical fields each -> embedding(d)
            self.user_cat_emb = nn.ModuleDict({
                f: nn.Embedding(cat_vocab_sizes[f], d) for f in PARA_CAT_FIELDS
            })
            # numeric personality fields -> per-field linear(1->d)
            self.user_num_proj = nn.ModuleList([nn.Linear(1, d) for _ in range(num_user_num)])
            self.K_user = len(PARA_CAT_FIELDS) + num_user_num  # user graph nodes
        else:
            self.user_cat_emb = None
            self.user_num_proj = None
            self.K_user = 1  # single node

        # ---- Internal interactions: pairwise MLP + sum aggregation
        self.int_mlp_user = PairMLP(d, hidden)
        self.int_mlp_img  = PairMLP(d, hidden)

        # ---- External interactions (Bi-Interaction) only in full mode
        # c_i = sum_j (u_i ⊙ v_j)  then linear to d
        if use_external:
            self.ext_proj_user = nn.Linear(d, d)  # after sum of elementwise product
            self.ext_proj_img  = nn.Linear(d, d)
        else:
            self.ext_proj_user = None
            self.ext_proj_img = None

        # ---- GRU fusion: hidden=u_i, input=[e_i, c_i] (2d -> d)
        self.gru_in = nn.Linear(2*d, d)
        self.gru_user = nn.GRUCell(input_size=d, hidden_size=d)
        self.gru_img  = nn.GRUCell(input_size=d, hidden_size=d)

        # ---- post-interaction to bias term b_hat
        self.bias_mlp = nn.Sequential(
            nn.Linear(2*d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        # ---- FC_s(dis_hat)  (Eq.13)
        self.dis_fc = nn.Linear(Bin, 1)

    def _internal_complete_graph(self, nodes: torch.Tensor, pair_mlp: PairMLP) -> torch.Tensor:
        """
        nodes: [B, K, d]
        returns e: [B, K, d], where e_i = sum_j MLP(u_i, u_j)
        """
        B, K, d = nodes.shape
        # compute all pairs by broadcasting
        a = nodes.unsqueeze(2).expand(B, K, K, d)  # [B,K,K,d]
        b = nodes.unsqueeze(1).expand(B, K, K, d)  # [B,K,K,d]
        e_ij = pair_mlp(a.reshape(B*K*K, d), b.reshape(B*K*K, d)).reshape(B, K, K, d)
        e_i = e_ij.sum(dim=2)  # sum over j
        return e_i

    def _external_bi_interaction(self, user_nodes: torch.Tensor, img_nodes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        user_nodes: [B, Ku, d]
        img_nodes:  [B, Ki, d]
        returns:
          c_user: [B, Ku, d], c_img: [B, Ki, d]
        """
        B, Ku, d = user_nodes.shape
        Ki = img_nodes.shape[1]

        # user side: c_i = sum_j (u_i ⊙ v_j)
        u = user_nodes.unsqueeze(2).expand(B, Ku, Ki, d)
        v = img_nodes.unsqueeze(1).expand(B, Ku, Ki, d)
        prod = u * v
        c_user = prod.sum(dim=2)  # [B,Ku,d]
        c_user = self.ext_proj_user(c_user)

        # image side similarly: sum over user nodes
        # c_j = sum_i (v_j ⊙ u_i)
        v2 = img_nodes.unsqueeze(2).expand(B, Ki, Ku, d)
        u2 = user_nodes.unsqueeze(1).expand(B, Ki, Ku, d)
        prod2 = v2 * u2
        c_img = prod2.sum(dim=2)  # [B,Ki,d]
        c_img = self.ext_proj_img(c_img)

        return c_user, c_img

    def forward(self, user_idx, attr_hat, dis_hat, user_cat=None, user_num=None):
        """
        user_idx: [B]
        attr_hat: [B, K_img]
        dis_hat : [B, Bin]
        user_cat: dict[field]->[B] (only full)
        user_num: [B, P] (only full) P=5 personality dims
        """
        B = user_idx.size(0)
        d = self.d

        # ---- Image nodes from attr_hat (K nodes, scalar -> d)
        # nodes: [B, K_img, 1] -> [B, K_img, d]
        img_nodes = self.img_node_proj(attr_hat.unsqueeze(-1))

        # ---- User nodes
        if not self.use_external:
            # single node: user_id embedding
            u0 = self.user_id_emb(user_idx).unsqueeze(1)  # [B,1,d]
            user_nodes = u0
        else:
            # categorical nodes
            cat_nodes = []
            for f in PARA_CAT_FIELDS:
                cat_nodes.append(self.user_cat_emb[f](user_cat[f]))  # [B,d]
            # numeric nodes (each scalar -> d)
            num_nodes = []
            # user_num: [B,5]
            for k in range(user_num.size(1)):
                num_nodes.append(self.user_num_proj[k](user_num[:, k:k+1]))  # [B,d]

            user_nodes = torch.stack(cat_nodes + num_nodes, dim=1)  # [B,Ku,d]

        # ---- Internal interactions
        e_user = self._internal_complete_graph(user_nodes, self.int_mlp_user)  # [B,Ku,d]
        e_img  = self._internal_complete_graph(img_nodes,  self.int_mlp_img)   # [B,Ki,d]

        # ---- External interactions
        if self.use_external:
            c_user, c_img = self._external_bi_interaction(user_nodes, img_nodes)  # [B,Ku,d], [B,Ki,d]
        else:
            c_user = torch.zeros_like(user_nodes)
            c_img  = torch.zeros_like(img_nodes)

        # ---- Information fusion (GRU): u'_i = GRU(u_i, e_i, c_i)
        # Implement as GRUCell(hidden=u_i, input=Linear([e_i,c_i]))
        in_user = self.gru_in(torch.cat([e_user, c_user], dim=-1))  # [B,Ku,d]
        in_img  = self.gru_in(torch.cat([e_img,  c_img],  dim=-1))  # [B,Ki,d]

        # apply GRUCell per node (loop over K, small)
        user_nodes_f = []
        for k in range(user_nodes.size(1)):
            user_nodes_f.append(self.gru_user(in_user[:, k, :], user_nodes[:, k, :]))
        user_nodes_f = torch.stack(user_nodes_f, dim=1)  # [B,Ku,d]

        img_nodes_f = []
        for k in range(img_nodes.size(1)):
            img_nodes_f.append(self.gru_img(in_img[:, k, :], img_nodes[:, k, :]))
        img_nodes_f = torch.stack(img_nodes_f, dim=1)  # [B,Ki,d]

        # ---- Post-interaction feature aggregation (sum pooling)
        b_user = user_nodes_f.sum(dim=1)  # [B,d]
        b_img  = img_nodes_f.sum(dim=1)   # [B,d]
        b_hat = self.bias_mlp(torch.cat([b_user, b_img], dim=-1)).squeeze(-1)  # [B]

        # ---- Add FC_s(dis_hat)  (Eq.13)
        dis_term = self.dis_fc(dis_hat).squeeze(-1)  # [B]
        y_hat = b_hat + dis_term
        return y_hat


# -----------------------------
# Train loop
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
                    help="Full mode: use user attributes + external interactions. "
                         "If not set: reduced mode (user_id single node, no external).")
    ap.add_argument("--max_rows", type=int, default=None)
    ap.add_argument("--feat_batch", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- load training interactions (PARA only) ----
    df = pd.read_csv(os.path.join(args.split_dir, "train_interactions.csv"))
    df = df.dropna(subset=["user_id", "image_path", "score"])
    if args.max_rows is not None and args.max_rows < len(df):
        df = df.sample(n=args.max_rows, random_state=42).reset_index(drop=True)

    user_ids = sorted(df["user_id"].astype(str).unique().tolist())
    user_id_to_idx = {u: i for i, u in enumerate(user_ids)}
    img_paths = sorted(df["image_path"].astype(str).unique().tolist())

    # ---- Phase1 forward for attr_hat/dis_hat ----
    phase1, attr_cols, dis_cols = load_phase1_model(args.phase1_ckpt, device=device)
    attr_map, dis_map = extract_attr_dis_for_paths(phase1, img_paths, device=device, batch_size=args.feat_batch)

    K_img = len(attr_cols)
    Bin = len(dis_cols)
    print(f"[info] K_img={K_img} (attr_hat), Bin={Bin} (dis_hat)")
    print(f"[info] users={len(user_ids)} images={len(img_paths)} interactions={len(df)}")

    # ---- user attrs only in full mode ----
    if args.use_external:
        user_feat_cat, cat_vocab, user_feat_num = build_para_user_attr_tables(
            os.path.join(args.split_dir, "user_attrs.csv"),
            user_ids=user_ids
        )
        cat_vocab_sizes = {f: len(cat_vocab[f]) for f in PARA_CAT_FIELDS}
    else:
        user_feat_cat = None
        user_feat_num = None
        cat_vocab_sizes = None

    # ---- dataset/loader ----
    ds = Phase2Dataset(
        df=df,
        user_id_to_idx=user_id_to_idx,
        attr_map=attr_map,
        dis_map=dis_map,
        use_external=args.use_external,
        user_feat_cat=user_feat_cat,
        user_feat_num=user_feat_num,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # ---- model ----
    model = Phase2ICI(
        num_users=len(user_ids),
        K_img=K_img,
        Bin=Bin,
        d=args.d,
        hidden=args.hidden,
        use_external=args.use_external,
        cat_vocab_sizes=cat_vocab_sizes,
        num_user_num=len(PARA_NUM_FIELDS),
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for user_idx, attr_hat, dis_hat, user_cat, user_num, y in loader:
            user_idx = user_idx.to(device)
            attr_hat = attr_hat.to(device)
            dis_hat = dis_hat.to(device)
            y = y.to(device)

            if args.use_external:
                user_cat = {k: v.to(device) for k, v in user_cat.items()}
                user_num = user_num.to(device)
                pred = model(user_idx, attr_hat, dis_hat, user_cat=user_cat, user_num=user_num)
            else:
                pred = model(user_idx, attr_hat, dis_hat)

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
            "K_img": K_img,
            "Bin": Bin,
            "d": args.d,
            "hidden": args.hidden,
            "use_external": args.use_external,
            "user_id_to_idx": user_id_to_idx,
            "cat_vocab_sizes": cat_vocab_sizes if args.use_external else None,
            "cat_vocab": cat_vocab if args.use_external else None,
        }
    }, args.out_ckpt)

    print(f"[save] {args.out_ckpt}")


if __name__ == "__main__":
    main()