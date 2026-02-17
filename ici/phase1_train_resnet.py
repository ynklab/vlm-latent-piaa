#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader

from datasets_phase1 import PARAPhase1Dataset
import torchvision.models as models


def emd_loss_from_logits(pred_logits: torch.Tensor, target_dis: torch.Tensor) -> torch.Tensor:
    """
    pred_logits: [B, K]
    target_dis : [B, K] (sum=1)
    """
    pred_prob = F.softmax(pred_logits, dim=-1)
    cdf_pred = torch.cumsum(pred_prob, dim=-1)
    cdf_true = torch.cumsum(target_dis, dim=-1)
    return torch.mean(torch.abs(cdf_pred - cdf_true))


class ResNet50Phase1(nn.Module):
    def __init__(self, n_attr: int, n_bins: int):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        feat_dim = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.attr_head = nn.Linear(feat_dim, n_attr)
        self.dis_head = nn.Linear(feat_dim, n_bins)

    def forward(self, x):
        feat = self.backbone(x)
        attr = self.attr_head(feat)  # [B, n_attr]
        dis_logits = self.dis_head(feat)  # [B, n_bins]
        return feat, attr, dis_logits


def train_one_epoch(model, loader, opt, device, lambda_emd: float) -> Tuple[float, float, float]:
    model.train()
    loss_attr_sum, loss_emd_sum, loss_sum = 0.0, 0.0, 0.0
    n = 0
    for x, y_attr, y_dis in loader:
        x = x.to(device)
        y_attr = y_attr.to(device)
        y_dis = y_dis.to(device)

        _, pred_attr, pred_dis_logits = model(x)
        loss_attr = F.mse_loss(pred_attr, y_attr)
        loss_emd = emd_loss_from_logits(pred_dis_logits, y_dis)
        loss = loss_attr + lambda_emd * loss_emd

        opt.zero_grad()
        loss.backward()
        opt.step()

        bs = x.size(0)
        n += bs
        loss_attr_sum += float(loss_attr.item()) * bs
        loss_emd_sum += float(loss_emd.item()) * bs
        loss_sum += float(loss.item()) * bs

    return loss_sum / n, loss_attr_sum / n, loss_emd_sum / n


@torch.inference_mode()
def eval_one_epoch(model, loader, device, lambda_emd: float) -> Tuple[float, float, float]:
    model.eval()
    loss_attr_sum, loss_emd_sum, loss_sum = 0.0, 0.0, 0.0
    n = 0
    for x, y_attr, y_dis in loader:
        x = x.to(device)
        y_attr = y_attr.to(device)
        y_dis = y_dis.to(device)

        _, pred_attr, pred_dis_logits = model(x)
        loss_attr = F.mse_loss(pred_attr, y_attr)
        loss_emd = emd_loss_from_logits(pred_dis_logits, y_dis)
        loss = loss_attr + lambda_emd * loss_emd

        bs = x.size(0)
        n += bs
        loss_attr_sum += float(loss_attr.item()) * bs
        loss_emd_sum += float(loss_emd.item()) * bs
        loss_sum += float(loss.item()) * bs

    return loss_sum / n, loss_attr_sum / n, loss_emd_sum / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_dir", required=True, help="e.g. ici_splits/para")
    ap.add_argument("--out_ckpt", required=True, help="output .pth")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda_emd", type=float, default=0.1)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # transforms (ResNet ImageNet style)
    tfm = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    ds_tr = PARAPhase1Dataset(args.split_dir, split="train", val_ratio=args.val_ratio, seed=args.seed, transform=tfm)
    ds_va = PARAPhase1Dataset(args.split_dir, split="val",   val_ratio=args.val_ratio, seed=args.seed, transform=tfm,
                              attr_cols=ds_tr.meta.attr_cols, dis_cols=ds_tr.meta.dis_cols)

    n_attr = len(ds_tr.meta.attr_cols)
    n_bins = len(ds_tr.meta.dis_cols)
    print(f"[info] n_attr={n_attr} attr_cols={ds_tr.meta.attr_cols}")
    print(f"[info] n_bins={n_bins} dis_cols={ds_tr.meta.dis_cols}")
    print(f"[info] train={len(ds_tr)} val={len(ds_va)}")

    model = ResNet50Phase1(n_attr=n_attr, n_bins=n_bins).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    tr_loader = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    va_loader = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    best_val = float("inf")
    os.makedirs(os.path.dirname(args.out_ckpt) or ".", exist_ok=True)

    for ep in range(1, args.epochs + 1):
        tr_loss, tr_attr, tr_emd = train_one_epoch(model, tr_loader, opt, device, args.lambda_emd)
        va_loss, va_attr, va_emd = eval_one_epoch(model, va_loader, device, args.lambda_emd)
        print(f"[ep {ep:02d}] train: loss={tr_loss:.4f} (attr={tr_attr:.4f}, emd={tr_emd:.4f}) | "
              f"val: loss={va_loss:.4f} (attr={va_attr:.4f}, emd={va_emd:.4f})")

        if va_loss < best_val:
            best_val = va_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "attr_cols": ds_tr.meta.attr_cols,
                "dis_cols": ds_tr.meta.dis_cols,
            }, args.out_ckpt)
            print(f"[save] best ckpt -> {args.out_ckpt}")

    print("[done]")


if __name__ == "__main__":
    main()