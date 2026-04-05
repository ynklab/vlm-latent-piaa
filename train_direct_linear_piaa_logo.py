#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LOGO Ridge regression on LAPIS using 2_styles.

For each user:
  - Leave-one-2_style-out
  - Train Ridge on remaining style
  - Evaluate on held-out style

Input:
  - LOGO split CSV from lapis_logo_2styles_split.py
  - mm_embed features (computed on the fly)

Output:
  user_id, holdout_tag, source, layer,
  n_train, n_test,
  rho, r2
"""

import os
import math
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import torch

from utils.mm_embed import load_mm_model, build_inputs, extract_all_pools


# ---------- Metrics ----------

def compute_metrics(y_true, y_pred):
    rho = spearmanr(y_true, y_pred).correlation
    if np.isnan(rho):
        rho = 0.0
    mse = np.mean((y_true - y_pred) ** 2)
    var = np.var(y_true) + 1e-12
    r2 = 1.0 - mse / var
    return rho, r2


# ---------- Feature Extraction ----------

def extract_feature(pools, source, layer):
    if source == "llm_text":
        return pools.llm_text[layer]
    elif source == "llm_visual":
        return pools.llm_visual[layer]
    elif source == "vision":
        return pools.vision_layers[layer]
    else:
        raise ValueError("Unsupported source")


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logo_csv", required=True)
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--source", default="llm_text")
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--min_train", type=int, default=10)
    ap.add_argument("--min_test", type=int, default=5)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.logo_csv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = load_mm_model(args.model_id)
    model.eval()

    # Precompute all features once
    unique_paths = df["image_path"].unique()
    pools_cache = {}

    print("[info] extracting mm_embed features...")
    for p in tqdm(unique_paths):
        img = Image.open(p).convert("RGB")
        inputs = build_inputs(processor, img, "Assess aesthetics.")
        pools = extract_all_pools(model, inputs, processor=processor)
        pools_cache[p] = pools

    results = []
    skipped = []

    for (user_id, holdout_tag), g in df.groupby(["user_id", "holdout_tag"]):

        train_df = g[g["logo_split"] == "train"]
        test_df  = g[g["logo_split"] == "test"]

        if len(train_df) < args.min_train or len(test_df) < args.min_test:
            skipped.append((user_id, holdout_tag))
            continue

        X_train = []
        y_train = []
        X_test  = []
        y_test  = []

        for _, row in train_df.iterrows():
            pools = pools_cache[row["image_path"]]
            feat = extract_feature(pools, args.source, args.layer)
            X_train.append(feat)
            y_train.append(row.get("user_score", np.nan))

        for _, row in test_df.iterrows():
            pools = pools_cache[row["image_path"]]
            feat = extract_feature(pools, args.source, args.layer)
            X_test.append(feat)
            y_test.append(row.get("user_score", np.nan))

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test  = np.array(X_test)
        y_test  = np.array(y_test)

        pipe = make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=np.logspace(-3, 3, 13))
        )
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        rho, r2 = compute_metrics(y_test, y_pred)

        results.append({
            "user_id": user_id,
            "holdout_tag": holdout_tag,
            "source": args.source,
            "layer": args.layer,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "rho": rho,
            "r2": r2,
        })

    pd.DataFrame(results).to_csv(args.out_csv, index=False)

    print("\n[done]")
    print("Results:", len(results))
    print("Skipped:", len(skipped))


if __name__ == "__main__":
    main()