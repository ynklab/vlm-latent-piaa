#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualize y_true vs y_pred scatter plots for each PIAA method.

Input:
  - A directory containing PIAA baseline CSVs (from piaa_from_giaa.py,
    residual/direct/hidden_attr/LoRA/CoT, etc.) with columns:

      user_id, image_path, model_id, support_set, method, giaa, piaa_pred, user_score, ...

Behavior:
  - Load all such CSVs from --input_dir.
  - Optionally filter by model_id and/or support_set.
  - For each distinct `method`, plot a scatter:

      x-axis: user_score (ground truth PIAA)
      y-axis: piaa_pred (predicted PIAA)

    and draw the y=x line for reference.

Output:
  - One PNG per (model_id, support_set, method) combination:

      <out_dir>/<sanitize(model_id)>__<support_set>__<method>__scatter.png
"""

import os
import re
import math
import argparse
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import spearmanr


def sanitize(s: str) -> str:
    if s is None or not isinstance(s, str) or s.strip() == "":
        return "unknown"
    return re.sub(r"[^0-9A-Za-z._\\-]+", "_", s)


def load_baseline_from_dir(input_dir: str) -> pd.DataFrame:
    """
    Read CSVs in the given directory,
    and return the concatenation of only files with the required PIAA baseline columns.
    """
    required = {
        "user_id",
        "image_path",
        "model_id",
        "support_set",
        "method",
        "giaa",
        "piaa_pred",
        "user_score",
    }
    dfs: List[pd.DataFrame] = []

    if not os.path.isdir(input_dir):
        raise RuntimeError(f"input_dir is not a directory: {input_dir}")

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(".csv")]
    print(f"[info] found {len(files)} CSV files in {input_dir}")

    for name in files:
        path = os.path.join(input_dir, name)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[warn] failed to read {path}: {e}, skip")
            continue

        if not required.issubset(df.columns):
            print(f"[info] skip {path} (missing baseline columns)")
            continue

        print(f"[info] loaded baseline CSV: {path} (rows={len(df)})")
        dfs.append(df)

    if not dfs:
        raise RuntimeError(f"No valid baseline CSVs found in directory: {input_dir}")

    df_all = pd.concat(dfs, ignore_index=True)
    return df_all


def plot_scatter_for_method(
    df: pd.DataFrame,
    model_id: str,
    support_set: str,
    method: str,
    out_path: str,
    max_points: int | None = None,
    figsize=(6, 6),
    dpi: int = 160,
) -> bool:
    """
    df: rows for a given (model_id, support_set, method) combination.
    """
    if df.empty:
        return False

    # Extract values and drop NaNs
    x = df["user_score"].to_numpy(dtype=float)
    y = df["piaa_pred"].to_numpy(dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    if x.size == 0:
        return False

    # Optional subsampling
    if max_points is not None and x.size > max_points:
        idx = np.random.RandomState(0).choice(x.size, size=max_points, replace=False)
        x = x[idx]
        y = y[idx]

    # Compute Spearman's rho for the title
    rho = spearmanr(x, y).correlation
    if np.isnan(rho):
        rho = 0.0

    plt.close("all")
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(x, y, s=8, alpha=0.4)

    # y = x reference line
    vmin = min(x.min(), y.min())
    vmax = max(x.max(), y.max())
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 1.0, 5.0
    pad = 0.1 * (vmax - vmin + 1e-6)
    vmin -= pad
    vmax += pad
    ax.plot([vmin, vmax], [vmin, vmax], linestyle="--", color="gray", linewidth=1.0)

    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)

    ax.set_xlabel("Ground truth score (user_score)")
    ax.set_ylabel("Predicted score (piaa_pred)")
    ax.set_title(
        f"{model_id}\n"
        f"support_set={support_set}, method={method}\n"
        f"n={len(x)}, Spearman ρ={rho:.3f}"
    )
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] saved {out_path}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing PIAA baseline CSVs.",
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Directory to save scatter plots.",
    )
    ap.add_argument(
        "--model_id_filter",
        default=None,
        help="If set, only use rows with this model_id.",
    )
    ap.add_argument(
        "--support_set_filter",
        default=None,
        help="If set, only use rows with this support_set (e.g. small, large, none).",
    )
    ap.add_argument(
        "--min_points",
        type=int,
        default=30,
        help="Minimum number of points required to plot a scatter for a method.",
    )
    ap.add_argument(
        "--max_points",
        type=int,
        default=5000,
        help="Maximum number of points to plot per method (subsample if larger).",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[info] loading baseline data...")
    df_all = load_baseline_from_dir(args.input_dir)
    print(f"[info] total rows = {len(df_all)}")

    # Filtering
    df = df_all.copy()
    if args.model_id_filter is not None:
        df = df[df["model_id"] == args.model_id_filter].copy()
        print(f"[info] filtered by model_id={args.model_id_filter}, rows={len(df)}")
    if args.support_set_filter is not None:
        df = df[df["support_set"] == args.support_set_filter].copy()
        print(f"[info] filtered by support_set={args.support_set_filter}, rows={len(df)}")

    if df.empty:
        print("[warn] no rows left after filtering, nothing to plot.")
        return

    # Draw a scatter plot for each (model_id, support_set, method)
    grouped = df.groupby(["model_id", "support_set", "method"])

    for (model_id, support_set, method), g in tqdm(grouped, desc="Methods"):
        if len(g) < args.min_points:
            print(f"[info] skip method={method} (model_id={model_id}, support_set={support_set}) "
                  f"due to too few points (n={len(g)})")
            continue

        fname = f"{sanitize(model_id)}__{sanitize(str(support_set))}__{sanitize(method)}__scatter.png"
        out_path = os.path.join(args.out_dir, fname)
        plot_scatter_for_method(
            g,
            model_id=model_id,
            support_set=str(support_set),
            method=method,
            out_path=out_path,
            max_points=args.max_points,
        )

    print("[done] all scatter plots generated.")


if __name__ == "__main__":
    main()