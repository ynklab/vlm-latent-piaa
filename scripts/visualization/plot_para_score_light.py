#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot PARA: 'score' vs 'light' using utils.para.get_para_dataset.

Usage:
  python -m scripts.visualization.plot_para_score_light --dataset_dir datasets/PARA --out_png out/para_score_light_scatter.png

Options:
  --dataset_dir   path to PARA dataset root (default: datasets/PARA)
  --out_png       output PNG path (default: para_score_light_scatter.png)
  --max_points    max number of points to plot (random subsample if dataset is large)
"""

import os
import argparse
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import the PARA loader from utils. Assumes utils/para.py is on PYTHONPATH or in project.
try:
    from utils.para import get_para_dataset
except Exception as e:
    raise RuntimeError(f"Failed to import utils.para.get_para_dataset: {e}")


def items_to_df(items):
    rows = []
    for it in items:
        row = {}
        # attributes is expected to be a dict (quality, composition, color, dof, light, ...)
        if hasattr(it, "attributes") and isinstance(it.attributes, dict):
            row.update(it.attributes)
        # overall score might be stored in .score
        if hasattr(it, "score"):
            # ensure numeric
            try:
                row["score"] = float(it.score)
            except Exception:
                row["score"] = None
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default="datasets/PARA", help="Root directory of PARA dataset")
    ap.add_argument("--out_png", default="para_score_light_scatter.png", help="Output PNG path")
    ap.add_argument("--max_points", type=int, default=5000, help="If dataset is larger, randomly subsample to this many points")
    ap.add_argument("--aadb_like", action="store_true", help="(ignored) kept for API compatibility")
    args = ap.parse_args()

    if not os.path.isdir(args.dataset_dir):
        raise SystemExit(f"dataset_dir not found: {args.dataset_dir}")

    # load both splits (get_para_dataset(None) returns train+test)
    items = get_para_dataset(None, dataset_dir=args.dataset_dir)
    if not items:
        raise SystemExit("No items loaded from PARA. Check dataset_dir and files.")

    df = items_to_df(items)
    if df.empty:
        raise SystemExit("Constructed dataframe is empty (no numeric attributes found).")

    # ensure 'light' and 'score' exist
    if "light" not in df.columns:
        raise SystemExit(f"'light' attribute not found in PARA items. Available columns: {list(df.columns)}")
    if "score" not in df.columns:
        raise SystemExit(f"'score' not found in PARA items. Available columns: {list(df.columns)}")

    plot_df = df[["score", "light"]].dropna().astype(float)
    if plot_df.empty:
        raise SystemExit("No (score, light) pairs after dropping NaNs.")

    # subsample if too large
    n_total = len(plot_df)
    if args.max_points is not None and n_total > args.max_points:
        sampled_idx = random.sample(list(plot_df.index), args.max_points)
        plot_df = plot_df.loc[sampled_idx].reset_index(drop=True)

    # plot
    plt.close("all")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(plot_df["light"], plot_df["score"], alpha=0.6, s=18)
    ax.set_xlabel("light")
    ax.set_ylabel("score")
    ax.set_title("PARA: score vs light")

    # add linear trend line (robust to constant arrays)
    try:
        x = plot_df["light"].to_numpy()
        y = plot_df["score"].to_numpy()
        if x.size >= 2 and np.std(x) > 0:
            coef = np.polyfit(x, y, 1)
            poly1d_fn = np.poly1d(coef)
            xs = np.linspace(np.min(x), np.max(x), 200)
            ax.plot(xs, poly1d_fn(xs), linestyle="--", linewidth=1.5)
            ax.text(0.02, 0.96, f"trend: y={coef[0]:.3f}x+{coef[1]:.3f}", transform=ax.transAxes, fontsize=9, va="top")
    except Exception:
        pass

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
    fig.savefig(args.out_png, dpi=200)
    plt.close(fig)
    print(f"Saved scatter plot to: {args.out_png}")
    # also print some summary
    print(f"Plotted {len(plot_df)} points (original dataset had {n_total}).")
    print(f"score: mean={plot_df['score'].mean():.3f}, std={plot_df['score'].std():.3f}")
    print(f"light: mean={plot_df['light'].mean():.3f}, std={plot_df['light'].std():.3f}")


if __name__ == "__main__":
    main()
