#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scatter plot: per-user GT std vs per-user Pred std.

Input:
  - input_dir: directory containing PIAA result CSVs.
    Each CSV must have columns:
      user_id,image_path,model_id,support_set,method,giaa,piaa_pred,user_score

Output:
  - out_dir/std_scatter__<model_id>__<method>__<support_set>.png
    (one plot per (model_id, method, support_set))

Plot:
  x-axis: std of user_score per user (GT std)
  y-axis: std of piaa_pred per user (Pred std)

Notes:
  - Only users with >= min_items items are included (default: 5).
  - NaNs are dropped.
"""

import os
import re
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


REQUIRED_COLS = {
    "user_id",
    "image_path",
    "model_id",
    "support_set",
    "method",
    "piaa_pred",
    "user_score",
}

def sanitize(s: str) -> str:
    s = str(s)
    return re.sub(r"[^0-9A-Za-z._\\-]+", "_", s)

def load_all_csvs(input_dir: str) -> pd.DataFrame:
    if not os.path.isdir(input_dir):
        raise RuntimeError(f"input_dir is not a directory: {input_dir}")

    dfs = []
    for name in sorted(os.listdir(input_dir)):
        if not name.lower().endswith(".csv"):
            continue
        path = os.path.join(input_dir, name)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[warn] failed to read {path}: {e}")
            continue
        if not REQUIRED_COLS.issubset(df.columns):
            print(f"[info] skip {path} (missing cols: {REQUIRED_COLS - set(df.columns)})")
            continue
        dfs.append(df)

    if not dfs:
        raise RuntimeError(f"No valid CSV found in {input_dir}")
    return pd.concat(dfs, ignore_index=True)

def compute_user_std(df: pd.DataFrame, min_items: int) -> pd.DataFrame:
    """
    Compute per-user std for GT and Pred within a fixed (model_id, method, support_set).
    Returns a DF with:
      user_id, gt_std, pred_std, n_items
    """
    # coerce numeric
    df = df.copy()
    df["user_score"] = pd.to_numeric(df["user_score"], errors="coerce")
    df["piaa_pred"] = pd.to_numeric(df["piaa_pred"], errors="coerce")

    rows = []
    for uid, g in df.groupby("user_id"):
        g = g.dropna(subset=["user_score", "piaa_pred"])
        if len(g) < min_items:
            continue
        gt_std = float(np.std(g["user_score"].to_numpy(dtype=float)))
        pr_std = float(np.std(g["piaa_pred"].to_numpy(dtype=float)))
        rows.append({"user_id": uid, "gt_std": gt_std, "pred_std": pr_std, "n_items": len(g)})
    return pd.DataFrame(rows)

def plot_scatter(df_u: pd.DataFrame, title: str, out_png: str):
    if df_u.empty:
        return False

    x = df_u["gt_std"].to_numpy(dtype=float)
    y = df_u["pred_std"].to_numpy(dtype=float)

    # correlations
    pearson = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else float("nan")
    rho = spearmanr(x, y).correlation if len(x) > 1 else float("nan")

    plt.close("all")
    fig, ax = plt.subplots(figsize=(5.6, 4.4))

    ax.scatter(x, y, alpha=0.7, s=18)

    # y=x reference line
    lim = max(float(np.nanmax(x)), float(np.nanmax(y)), 1e-6)
    ax.plot([0, lim], [0, lim], linestyle="--", linewidth=1.0)

    ax.set_xlabel("GT std (per user)")
    ax.set_ylabel("Pred std (per user)")
    ax.set_title(title)

    ax.grid(True, linestyle="--", alpha=0.3)

    txt = f"n_users={len(df_u)}\nPearson={pearson:.3f}\nSpearman={rho:.3f}"
    ax.text(0.98, 0.02, txt, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none"))

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Directory containing PIAA result CSVs.")
    ap.add_argument("--out_dir", required=True, help="Output directory for scatter plots.")
    ap.add_argument("--min_items", type=int, default=5, help="Min items per user to compute std.")
    ap.add_argument("--only_model_id", default=None, help="If set, filter to this model_id only.")
    ap.add_argument("--only_method", default=None, help="If set, filter to this method only.")
    ap.add_argument("--only_support_set", default=None, help="If set, filter to this support_set only.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_all_csvs(args.input_dir)

    if args.only_model_id is not None:
        df = df[df["model_id"].astype(str) == args.only_model_id]
    if args.only_method is not None:
        df = df[df["method"].astype(str) == args.only_method]
    if args.only_support_set is not None:
        df = df[df["support_set"].astype(str) == args.only_support_set]

    if df.empty:
        print("[warn] No rows after filtering.")
        return

    groups = df.groupby(["model_id", "method", "support_set"])
    for (model_id, method, sup), g in groups:
        df_u = compute_user_std(g, min_items=args.min_items)
        out_png = os.path.join(
            args.out_dir,
            f"std_scatter__{sanitize(model_id)}__{sanitize(method)}__{sanitize(sup)}.png"
        )
        title = f"{model_id}\n{method} | support={sup}"
        ok = plot_scatter(df_u, title, out_png)
        if ok:
            print(f"[plot] {out_png}")

    print("[done]")

if __name__ == "__main__":
    main()