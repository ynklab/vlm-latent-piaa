#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train per-user residual MLP models for PIAA on PARA using mm_embed features.

For each user u:
  - We have support set S_u (support_small or support_large) and test set T_u
    from get_personalized_para_dataset.
  - For each image i:
      GIAA_pred(i)   : precomputed GIAA prediction (from vlm_giaa_para.py).
      user_score_i   : user-specific aesthetic score.
      feature z_i    : mm_embed feature from a chosen source/layer.
  - Target residual:
      r_i = user_score_i - GIAA_pred(i)
  - We fit a small MLP r ~ z on the support set.
  - On the test set, PIAA_pred(i) = GIAA_pred(i) + MLP(z_i).

Outputs a CSV with one row per user × test image:
  user_id, image_path, model_id, support_set, method, giaa, piaa_pred, user_score
"""

import os
import csv
import math
import argparse
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from utils.para import get_personalized_para_dataset
from utils.mm_embed import load_mm_model, build_inputs, extract_all_pools


# ---------- Prompt (for mm_embed features) ----------

AESTHETIC_ATTRS_FOR_PROMPT = [
    "BalancingElements", "ColorHarmony", "Content", "DoF",
    "Light", "MotionBlur", "Object", "Repetition",
    "RuleOfThirds", "Symmetry", "VividColor",
]

def make_prompt(mode: str) -> str:
    if mode == "base":
        return "Assess the aesthetics of this image."
    elif mode == "format":
        return (
            "Assess the overall aesthetic quality of this image. "
            "Please rate it on a scale from 1 to 5. "
            "Output only the numeric score, and do not output any other text."
        )
    elif mode == "attributes":
        attrs = ", ".join(AESTHETIC_ATTRS_FOR_PROMPT)
        return (
            "Assess the aesthetics of this image with respect to the following attributes: "
            f"{attrs}. "
            "You do not need to output the attributes explicitly; just use them as internal criteria."
        )
    elif mode == "unrelated":
        return "Describe the weather today in one sentence."
    else:
        raise ValueError(f"Unknown prompt_mode: {mode}")


# ---------- GIAA loader ----------

def load_giaa_map(
    giaa_csv: str,
    model_id_filter: str | None = None,
) -> Tuple[Dict[str, float], str]:
    """
    Read GIAA CSV and build a dict: image_path -> giaa (float).

    Expected columns in CSV (from vlm_giaa_para.py):
      model_id, split, image_path, giaa, raw_output

    If model_id_filter is given, only that model_id is used.
    If not, CSV must contain exactly one model_id.
    """
    image_to_giaa: Dict[str, float] = {}
    model_ids: set[str] = set()

    with open(giaa_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not {"model_id", "image_path", "giaa"}.issubset(reader.fieldnames or []):
            raise ValueError("GIAA CSV must contain columns: model_id, image_path, giaa")

        for row in reader:
            mid = row["model_id"]
            path = row["image_path"]
            try:
                score = float(row["giaa"])
            except Exception:
                score = math.nan
            model_ids.add(mid)

            if model_id_filter is not None and mid != model_id_filter:
                continue

            image_to_giaa[path] = score

    if model_id_filter is not None:
        model_id_used = model_id_filter
    else:
        if len(model_ids) != 1:
            raise ValueError(
                f"GIAA CSV contains multiple model_ids: {model_ids}. "
                f"Please specify --model_id_filter."
            )
        model_id_used = next(iter(model_ids))

    return image_to_giaa, model_id_used


# ---------- Feature helper ----------

def extract_feature_vector(
    pools,
    source: str,
    layer_idx: int,
) -> np.ndarray:
    """
    Given AllPools from mm_embed.extract_all_pools and a source/layer,
    return a 1D numpy feature vector.
    """
    if source == "llm_text":
        vec = pools.llm_text[layer_idx]
    elif source == "llm_visual":
        vec = pools.llm_visual[layer_idx]
    elif source == "llm_text_tail":
        vec = pools.llm_text_tail[layer_idx]
    elif source == "vision":
        if pools.vision_layers is None:
            raise RuntimeError("vision_layers is None; vision source not available for this model.")
        vec = pools.vision_layers[layer_idx]
    elif source == "bridge_text":
        vec = pools.bridge_text[0]
    elif source == "bridge_visual":
        vec = pools.bridge_visual[0]
    else:
        raise ValueError(f"Unknown feature_source: {source}")

    return vec.astype(np.float32)


# ---------- Residual MLP ----------

class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, hidden1: int = 128, hidden2: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D]
        return self.net(x).squeeze(-1)  # [B]


def train_mlp_for_user(
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    max_epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    patience: int = 10,
) -> Tuple[ResidualMLP, StandardScaler]:
    """
    Train a small MLP on (X, y) for a single user.
    X: [N, D], y: [N]
    Returns: (trained_model, fitted_scaler)
    """
    N, D = X.shape
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    # Standardize features (per-user scaler)
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)

    Xs_t = torch.from_numpy(Xs)
    y_t = torch.from_numpy(y)

    # train/val split (if enough samples)
    if N >= 15:
        idx = np.arange(N)
        np.random.shuffle(idx)
        split = max(1, int(0.8 * N))
        idx_tr = idx[:split]
        idx_va = idx[split:]
    else:
        # small support: no val split, use all as train
        idx_tr = np.arange(N)
        idx_va = None

    train_ds = TensorDataset(Xs_t[idx_tr], y_t[idx_tr])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = None
    if idx_va is not None and len(idx_va) > 0:
        val_ds = TensorDataset(Xs_t[idx_va], y_t[idx_va])
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = ResidualMLP(input_dim=D).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    wait = 0

    for epoch in range(max_epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # validation
        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    val_losses.append(loss.item())
            val_loss = float(np.mean(val_losses))
        else:
            # no val: use train loss as proxy
            val_loss = float(np.mean(train_losses))

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, scaler


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--giaa_csv",
        required=True,
        help="Path to GIAA prediction CSV (from vlm_giaa_para.py).",
    )
    ap.add_argument(
        "--dataset_dir",
        default="datasets/PARA",
        help="Path to PARA dataset root.",
    )
    ap.add_argument(
        "--model_id",
        required=True,
        help="Multimodal model id for mm_embed (e.g. Qwen/Qwen3-VL-2B-Instruct, google/gemma-3-4b-it).",
    )
    ap.add_argument(
        "--model_id_filter",
        default=None,
        help="If giaa_csv contains multiple model_ids, specify which one to use.",
    )
    ap.add_argument(
        "--feature_source",
        required=True,
        choices=[
            "llm_text",
            "llm_visual",
            "llm_text_tail",
            "vision",
            "bridge_text",
            "bridge_visual",
        ],
        help="Which feature source from mm_embed.AllPools to use.",
    )
    ap.add_argument(
        "--feature_layer",
        type=int,
        required=True,
        help="Layer index (0-based) for the chosen feature_source.",
    )
    ap.add_argument(
        "--support_set",
        default="large",
        choices=["small", "large"],
        help="Which support set from get_personalized_para_dataset to use.",
    )
    ap.add_argument(
        "--prompt_mode",
        default="base",
        choices=["base", "format", "attributes", "unrelated"],
        help="Prompt preset used when extracting features with mm_embed.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for get_personalized_para_dataset (must match when generating splits).",
    )
    ap.add_argument(
        "--quick",
        type=int,
        default=None,
        help="If set, limit to at most N users (for debugging). Use 1 for a single-user check.",
    )
    ap.add_argument(
        "--out_csv",
        required=True,
        help="Path to output CSV for per-user test predictions.",
    )
    args = ap.parse_args()

    # 1) Load GIAA map
    image_to_giaa, giaa_model_id = load_giaa_map(args.giaa_csv, args.model_id_filter)
    print(f"[info] using GIAA model_id={giaa_model_id}, entries={len(image_to_giaa)}")

    # 2) Load personalized PARA dataset
    print("[info] loading personalized PARA dataset...")
    personalized = get_personalized_para_dataset(seed=args.seed, dataset_dir=args.dataset_dir)
    all_user_ids = sorted(personalized.keys())
    print(f"[info] num users in personalized dataset: {len(all_user_ids)}")

    # quick: limit number of users
    if args.quick is not None and args.quick < len(all_user_ids):
        user_ids = all_user_ids[:args.quick]
        print(f"[info] quick mode: using first {len(user_ids)} users out of {len(all_user_ids)}")
    else:
        user_ids = all_user_ids

    # 3) Collect all image_paths for selected users
    all_paths = set()
    for user_id in user_ids:
        pdata = personalized[user_id]
        for item in pdata.support_small + pdata.support_large + pdata.test:
            all_paths.add(item.image_path)
    print(f"[info] total unique images in selected users' splits: {len(all_paths)}")

    # 4) Load VLM for mm_embed features
    print(f"[info] loading VLM for features: {args.model_id}")
    model, processor = load_mm_model(args.model_id, dtype="auto", device_map="auto", attn_impl=None)
    model.eval()
    device = model.device
    prompt = make_prompt(args.prompt_mode)
    print(f"[info] prompt_mode={args.prompt_mode}, prompt={prompt!r}")

    # 5) Precompute features for all relevant images
    feat_cache: Dict[str, np.ndarray] = {}
    print("[info] extracting features for all selected images...")
    for path in tqdm(sorted(all_paths), desc="Embed"):
        if path not in image_to_giaa:
            continue
        img = Image.open(path).convert("RGB")
        inputs = build_inputs(processor, img, prompt)
        pools = extract_all_pools(model, inputs, processor=processor)
        try:
            vec = extract_feature_vector(pools, args.feature_source, args.feature_layer)
        except IndexError:
            raise IndexError(
                f"feature_layer={args.feature_layer} is out of range for source={args.feature_source} "
                f"(check number of layers for this model/source)."
            )
        feat_cache[path] = vec
    print(f"[info] feature cache size: {len(feat_cache)} (images with both GIAA and features)")

    # 6) Per-user training (MLP) and prediction
    rows: List[dict] = []
    method_name = f"residual_mlp_{args.feature_source}_L{args.feature_layer}"

    for user_id in tqdm(user_ids, desc="Users"):
        pdata = personalized[user_id]
        if args.support_set == "small":
            support_items = pdata.support_small
        else:
            support_items = pdata.support_large
        test_items = pdata.test

        # Build support set
        X_support = []
        y_support = []
        for item in support_items:
            path = item.image_path
            if path not in image_to_giaa or path not in feat_cache:
                continue
            giaa = image_to_giaa[path]
            user_score = float(item.score)
            residual = user_score - giaa
            X_support.append(feat_cache[path])
            y_support.append(residual)

        X_support = np.array(X_support, dtype=np.float32)
        y_support = np.array(y_support, dtype=np.float32)

        if len(y_support) < 2:
            # not enough support points to train; skip this user
            continue

        # Train MLP for this user
        mlp, scaler = train_mlp_for_user(
            X_support, y_support, device=device,
            max_epochs=100,
            batch_size=16,
            lr=1e-3,
            weight_decay=1e-2,
            patience=10,
        )

        # Predict on test set
        mlp.eval()
        for item in test_items:
            path = item.image_path
            if path not in image_to_giaa or path not in feat_cache:
                continue
            giaa = image_to_giaa[path]
            z = feat_cache[path][None, :]        # [1,D]
            # Apply same scaler
            z_scaled = scaler.transform(z).astype(np.float32)
            z_t = torch.from_numpy(z_scaled).to(device)
            with torch.no_grad():
                residual_pred = mlp(z_t)[0].item()
            piaa_pred = giaa + residual_pred
            rows.append(
                {
                    "user_id": user_id,
                    "image_path": path,
                    "model_id": args.model_id,
                    "support_set": args.support_set,
                    "method": method_name,
                    "giaa": giaa,
                    "piaa_pred": piaa_pred,
                    "user_score": float(item.score),
                }
            )

    # 7) Save CSV
    out_path = args.out_csv
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fieldnames = [
        "user_id",
        "image_path",
        "model_id",
        "support_set",
        "method",
        "giaa",
        "piaa_pred",
        "user_score",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[done] wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()