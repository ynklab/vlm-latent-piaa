#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Phase 3 (Personalized aesthetic model fine-tuning) - faithful to Sec. 3.4.

- Fine-tunes the aesthetic prior model (Phase2) for each user using that user's support set.
- Phase1 is fixed and used to compute (attr_hat, dis_hat).
- Output predictions on test set and save CSV.

Paper settings:
- lr = 1e-5, epochs = 50 for Phase3.
"""

import os
import csv
import math
import copy
import argparse
import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models

from utils.para import get_personalized_para_dataset
from utils.lapis import get_personalized_lapis_dataset


# -----------------------------
# Phase1 (must match your phase1_train_resnet.py)
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
def extract_attr_dis_batch(model, paths: List[str], device: str, batch_size: int = 64):
    tfm = default_transform()
    xs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        xs.append(tfm(img))
    attrs, diss = [], []
    for i in range(0, len(xs), batch_size):
        x = torch.stack(xs[i:i+batch_size], dim=0).to(device)
        _, attr, dis_logits = model(x)
        dis = torch.softmax(dis_logits, dim=-1)
        attrs.append(attr.detach().cpu().numpy().astype(np.float32))
        diss.append(dis.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(attrs, axis=0), np.concatenate(diss, axis=0)


# -----------------------------
# Phase2 model (same as your rewritten phase2_train_graph.py)
# -----------------------------

PARA_CAT_FIELDS = ["age", "gender", "EducationalLevel", "artExperience", "photographyExperience"]
PARA_NUM_FIELDS = ["personality-E", "personality-A", "personality-N", "personality-O", "personality-C"]


class PairMLP(nn.Module):
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
    def __init__(
        self,
        num_users: int,
        K_img: int,
        Bin: int,
        d: int = 64,
        hidden: int = 256,
        use_external: bool = False,
        cat_vocab_sizes: Optional[Dict[str, int]] = None,
        num_user_num: int = 5,
    ):
        super().__init__()
        self.use_external = use_external
        self.d = d

        self.img_node_proj = nn.Linear(1, d)
        self.user_id_emb = nn.Embedding(num_users, d)

        if use_external:
            assert cat_vocab_sizes is not None
            self.user_cat_emb = nn.ModuleDict({
                f: nn.Embedding(cat_vocab_sizes[f], d) for f in PARA_CAT_FIELDS
            })
            self.user_num_proj = nn.ModuleList([nn.Linear(1, d) for _ in range(num_user_num)])
            self.K_user = len(PARA_CAT_FIELDS) + num_user_num
        else:
            self.user_cat_emb = None
            self.user_num_proj = None
            self.K_user = 1

        self.int_mlp_user = PairMLP(d, hidden)
        self.int_mlp_img  = PairMLP(d, hidden)

        if use_external:
            self.ext_proj_user = nn.Linear(d, d)
            self.ext_proj_img  = nn.Linear(d, d)
        else:
            self.ext_proj_user = None
            self.ext_proj_img  = None

        self.gru_in = nn.Linear(2*d, d)
        self.gru_user = nn.GRUCell(input_size=d, hidden_size=d)
        self.gru_img  = nn.GRUCell(input_size=d, hidden_size=d)

        self.bias_mlp = nn.Sequential(
            nn.Linear(2*d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.dis_fc = nn.Linear(Bin, 1)

    def _internal_complete_graph(self, nodes: torch.Tensor, pair_mlp: PairMLP) -> torch.Tensor:
        B, K, d = nodes.shape
        a = nodes.unsqueeze(2).expand(B, K, K, d)
        b = nodes.unsqueeze(1).expand(B, K, K, d)
        e_ij = pair_mlp(a.reshape(B*K*K, d), b.reshape(B*K*K, d)).reshape(B, K, K, d)
        return e_ij.sum(dim=2)

    def _external_bi(self, user_nodes: torch.Tensor, img_nodes: torch.Tensor):
        B, Ku, d = user_nodes.shape
        Ki = img_nodes.shape[1]
        u = user_nodes.unsqueeze(2).expand(B, Ku, Ki, d)
        v = img_nodes.unsqueeze(1).expand(B, Ku, Ki, d)
        c_user = self.ext_proj_user((u * v).sum(dim=2))

        v2 = img_nodes.unsqueeze(2).expand(B, Ki, Ku, d)
        u2 = user_nodes.unsqueeze(1).expand(B, Ki, Ku, d)
        c_img = self.ext_proj_img((v2 * u2).sum(dim=2))
        return c_user, c_img

    def forward(self, user_idx, attr_hat, dis_hat, user_cat=None, user_num=None):
        img_nodes = self.img_node_proj(attr_hat.unsqueeze(-1))  # [B,K,d]
        if not self.use_external:
            user_nodes = self.user_id_emb(user_idx).unsqueeze(1)  # [B,1,d]
        else:
            cat_nodes = [self.user_cat_emb[f](user_cat[f]) for f in PARA_CAT_FIELDS]
            num_nodes = [self.user_num_proj[k](user_num[:, k:k+1]) for k in range(user_num.size(1))]
            user_nodes = torch.stack(cat_nodes + num_nodes, dim=1)  # [B,Ku,d]

        e_user = self._internal_complete_graph(user_nodes, self.int_mlp_user)
        e_img  = self._internal_complete_graph(img_nodes,  self.int_mlp_img)

        if self.use_external:
            c_user, c_img = self._external_bi(user_nodes, img_nodes)
        else:
            c_user = torch.zeros_like(user_nodes)
            c_img  = torch.zeros_like(img_nodes)

        in_user = self.gru_in(torch.cat([e_user, c_user], dim=-1))
        in_img  = self.gru_in(torch.cat([e_img,  c_img],  dim=-1))

        user_nodes_f = torch.stack([self.gru_user(in_user[:, k, :], user_nodes[:, k, :])
                                    for k in range(user_nodes.size(1))], dim=1)
        img_nodes_f = torch.stack([self.gru_img(in_img[:, k, :], img_nodes[:, k, :])
                                   for k in range(img_nodes.size(1))], dim=1)

        b_user = user_nodes_f.sum(dim=1)
        b_img  = img_nodes_f.sum(dim=1)
        b_hat = self.bias_mlp(torch.cat([b_user, b_img], dim=-1)).squeeze(-1)
        dis_term = self.dis_fc(dis_hat).squeeze(-1)
        return b_hat + dis_term


# -----------------------------
# PARA user attr encoding (needed only for full mode)
# -----------------------------

def build_para_user_attr_maps_from_fixed_vocab(user_attrs_csv: str, cat_vocab: Dict[str, List[str]]):
    """
    Build user category index maps using cat_vocab saved in Phase2 ckpt.

    Any unseen category -> 0 (assumed NA/UNK bucket).
    Any unseen user -> 0 for all categorical + zeros for numeric.
    """
    df = pd.read_csv(user_attrs_csv)
    df = df.rename(columns={"userId": "user_id"}) if "userId" in df.columns else df
    df["user_id"] = df["user_id"].astype(str)
    df = df.set_index("user_id")

    # category maps: field -> {user_id: idx}
    cat_map: Dict[str, Dict[str, int]] = {}
    for f in PARA_CAT_FIELDS:
        vocab = cat_vocab[f]
        c2i = {c: i for i, c in enumerate(vocab)}
        m = {}
        for uid in df.index:
            val = str(df.loc[uid, f]) if pd.notna(df.loc[uid, f]) else "NA"
            m[uid] = int(c2i.get(val, 0))  # unseen -> 0
        cat_map[f] = m

    # numeric personality fields
    num_map: Dict[str, np.ndarray] = {}
    for uid in df.index:
        vals = []
        for f in PARA_NUM_FIELDS:
            v = pd.to_numeric(df.loc[uid, f], errors="coerce")
            if np.isnan(v):
                v = 0.0
            vals.append(float(v))
        num_map[uid] = np.array(vals, dtype=np.float32)

    return cat_map, num_map

def encode_user_full(uid: str, cat_vocab_sizes: Dict[str, int], cat_map, num_map):
    uid = str(uid)
    out_cat = {}
    for f in PARA_CAT_FIELDS:
        # unseen user/category -> 0
        idx = int(cat_map[f].get(uid, 0))
        idx = max(0, min(idx, cat_vocab_sizes[f] - 1))
        out_cat[f] = idx
    out_num = num_map.get(uid, np.zeros((len(PARA_NUM_FIELDS),), dtype=np.float32))
    return out_cat, out_num


# -----------------------------
# Fine-tune PRIOR model per user (faithful to Sec. 3.4)
# -----------------------------

def set_trainable_scope(model: nn.Module, scope: str):
    """
    scope:
      - "all": tune all Phase2 params (faithful)
      - "no_phase1": same (Phase1 not in this script)
      - "user_only": only user_id_emb
      - "user_plus_head": user_id_emb + bias_mlp + dis_fc
    """
    for p in model.parameters():
        p.requires_grad = False

    if scope == "user_only":
        model.user_id_emb.weight.requires_grad = True
        return

    if scope == "user_plus_head":
        model.user_id_emb.weight.requires_grad = True
        for p in model.bias_mlp.parameters():
            p.requires_grad = True
        for p in model.dis_fc.parameters():
            p.requires_grad = True
        return

    if scope == "all":
        for p in model.parameters():
            p.requires_grad = True
        return

    raise ValueError(f"Unknown tune scope: {scope}")


def finetune_one_user(
    base_state: Dict[str, torch.Tensor],
    model_template: Phase2ICI,
    device: str,
    user_local_idx: int,
    sup_attr: np.ndarray,
    sup_dis: np.ndarray,
    sup_y: np.ndarray,
    user_cat: Optional[Dict[str, int]],
    user_num: Optional[np.ndarray],
    epochs: int,
    lr: float,
    tune_scope: str,
) -> Phase2ICI:
    """
    Create a fresh copy of Phase2 model, load base weights, then fine-tune on support set.
    Returns the fine-tuned model for this user.
    """
    model = copy.deepcopy(model_template).to(device)
    model.load_state_dict(base_state, strict=True)

    set_trainable_scope(model, tune_scope)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

    x_attr = torch.tensor(sup_attr, dtype=torch.float32, device=device)
    x_dis  = torch.tensor(sup_dis, dtype=torch.float32, device=device)
    y = torch.tensor(sup_y, dtype=torch.float32, device=device)
    uidx = torch.full((x_attr.size(0),), user_local_idx, dtype=torch.long, device=device)

    if model.use_external:
        assert user_cat is not None and user_num is not None
        cat_t = {f: torch.full((x_attr.size(0),), int(user_cat[f]), dtype=torch.long, device=device) for f in PARA_CAT_FIELDS}
        num_t = torch.tensor(np.repeat(user_num[None, :], x_attr.size(0), axis=0), dtype=torch.float32, device=device)
    else:
        cat_t, num_t = None, None

    model.train()
    for _ in range(epochs):
        pred = model(uidx, x_attr, x_dis, user_cat=cat_t, user_num=num_t)
        loss = F.mse_loss(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return model


@torch.inference_mode()
def predict_user(model: Phase2ICI, device: str, user_local_idx: int,
                 te_attr: np.ndarray, te_dis: np.ndarray,
                 user_cat: Optional[Dict[str, int]], user_num: Optional[np.ndarray]) -> np.ndarray:
    model.eval()
    x_attr = torch.tensor(te_attr, dtype=torch.float32, device=device)
    x_dis  = torch.tensor(te_dis, dtype=torch.float32, device=device)
    uidx = torch.full((x_attr.size(0),), user_local_idx, dtype=torch.long, device=device)

    if model.use_external:
        cat_t = {f: torch.full((x_attr.size(0),), int(user_cat[f]), dtype=torch.long, device=device) for f in PARA_CAT_FIELDS}
        num_t = torch.tensor(np.repeat(user_num[None, :], x_attr.size(0), axis=0), dtype=torch.float32, device=device)
    else:
        cat_t, num_t = None, None

    pred = model(uidx, x_attr, x_dis, user_cat=cat_t, user_num=num_t).detach().cpu().numpy().astype(np.float32)
    return pred


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["para", "lapis"], required=True)
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--split_root", default="ici_splits")
    ap.add_argument("--phase1_ckpt", required=True)
    ap.add_argument("--phase2_ckpt", required=True)
    ap.add_argument("--support_set", choices=["small", "large"], default="small")
    ap.add_argument("--epochs", type=int, default=50)     # paper
    ap.add_argument("--lr", type=float, default=1e-5)     # paper
    ap.add_argument("--tune_scope", choices=["all", "user_only", "user_plus_head"], default="all",
                    help="Faithful is 'all' (fine-tune the prior model).")
    ap.add_argument("--quick_users", type=int, default=None)
    ap.add_argument("--feat_batch", type=int, default=64)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load phase2 ckpt/meta
    ckpt2 = torch.load(args.phase2_ckpt, map_location="cpu")
    meta2 = ckpt2["meta"]
    use_external = bool(meta2["use_external"])
    d = int(meta2["d"])
    hidden = int(meta2["hidden"])
    K_img = int(meta2["K_img"])
    Bin = int(meta2["Bin"])
    cat_vocab_sizes = meta2.get("cat_vocab_sizes", None)
    cat_vocab = meta2.get("cat_vocab", None)
    if use_external and cat_vocab is None:
        raise RuntimeError("Phase2 ckpt meta has no 'cat_vocab'. Re-run Phase2 saving cat_vocab.")

    # load phase1
    phase1, attr_cols, dis_cols = load_phase1_model(args.phase1_ckpt, device=device)
    assert len(attr_cols) == K_img
    assert len(dis_cols) == Bin

    # load eval split
    if args.dataset == "para":
        personalized = get_personalized_para_dataset(seed=42, dataset_dir=args.dataset_dir)
        eval_users = sorted(list(personalized.keys()))
    else:
        personalized = get_personalized_lapis_dataset(seed=42, dataset_dir=args.dataset_dir)
        eval_users = sorted([str(x) for x in personalized.keys()])

    if args.quick_users is not None:
        eval_users = eval_users[:args.quick_users]

    # build a template model with eval-user embedding size (no leakage)
    model_template = Phase2ICI(
        num_users=len(eval_users),
        K_img=K_img,
        Bin=Bin,
        d=d,
        hidden=hidden,
        use_external=use_external,
        cat_vocab_sizes=cat_vocab_sizes,
        num_user_num=len(PARA_NUM_FIELDS),
    )

    # load base weights into template EXCEPT user_id_emb (shape mismatch)
    sd = ckpt2["model_state_dict"]
    base_state = model_template.state_dict()
    for k, v in sd.items():
        if k.startswith("user_id_emb."):
            continue
        base_state[k] = v
    # keep template's user_id_emb random init
    model_template.load_state_dict(base_state, strict=True)
    base_state = model_template.state_dict()  # frozen base weights (with random eval user emb init)

    # full mode only valid for PARA
    if use_external and args.dataset != "para":
        raise RuntimeError("Phase2 ckpt is full (use_external=True). For cross-domain LAPIS, use reduced ckpt.")

    if use_external:
        # build vocab/maps from PARA train users (to match training)
        # Use the exact vocab saved in Phase2 ckpt (no rebuilding)
        cat_map, num_map = build_para_user_attr_maps_from_fixed_vocab(
            os.path.join(args.split_root, "para", "user_attrs.csv"),
            cat_vocab
        )
    else:
        cat_vocab, cat_map, num_map = None, None, None

    method_name = f"ici_phase3_{'full' if use_external else 'reduced'}_{args.support_set}_tune-{args.tune_scope}"

    rows = []
    for local_uid_idx, uid in enumerate(tqdm(eval_users, desc=f"Phase3[{args.dataset}]")):
        pdata = personalized[int(uid)] if args.dataset == "lapis" else personalized[uid]
        support_items = pdata.support_small if args.support_set == "small" else pdata.support_large
        test_items = pdata.test

        sup_paths = [it.image_path for it in support_items]
        sup_y = np.array([float(it.score) for it in support_items], dtype=np.float32)
        sup_attr, sup_dis = extract_attr_dis_batch(phase1, sup_paths, device=device, batch_size=args.feat_batch)

        te_paths = [it.image_path for it in test_items]
        te_y = np.array([float(it.score) for it in test_items], dtype=np.float32)
        te_attr, te_dis = extract_attr_dis_batch(phase1, te_paths, device=device, batch_size=args.feat_batch)

        if use_external:
            user_cat, user_num = encode_user_full(uid, cat_vocab_sizes, cat_map, num_map)
        else:
            user_cat, user_num = None, None

        # fine-tune prior model for this user
        tuned = finetune_one_user(
            base_state=base_state,
            model_template=model_template,
            device=device,
            user_local_idx=local_uid_idx,
            sup_attr=sup_attr,
            sup_dis=sup_dis,
            sup_y=sup_y,
            user_cat=user_cat,
            user_num=user_num,
            epochs=args.epochs,
            lr=args.lr,
            tune_scope=args.tune_scope,
        )

        preds = predict_user(
            tuned, device=device, user_local_idx=local_uid_idx,
            te_attr=te_attr, te_dis=te_dis,
            user_cat=user_cat, user_num=user_num
        )

        for p, gt, pr in zip(te_paths, te_y, preds):
            rows.append({
                "user_id": uid,
                "image_path": p,
                "model_id": f"ICI(PARA Phase1/2)::phase2={os.path.basename(args.phase2_ckpt)}",
                "support_set": args.support_set,
                "method": method_name,
                "giaa": math.nan,
                "piaa_pred": float(pr),
                "user_score": float(gt),
            })

        # free per-user model
        del tuned
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    fieldnames = ["user_id","image_path","model_id","support_set","method","giaa","piaa_pred","user_score"]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[done] wrote {len(rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
