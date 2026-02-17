#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot score distributions for PIAA experiments.

For a given PIAA result directory (CSV files):
  - Plot GT (user_score) histogram (one figure).
  - Plot predicted score (piaa_pred) histogram for each (method, support_set).

Plots:
  - Simple histogram
  - y-axis fixed to (0, 50)
  - x-axis: score (1..5)
"""

import os
import re
import argparse
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


REQUIRED_COLS = {
    "user_id",
    "image_path",
    "model_id",
    "support_set",
    "method",
    "piaa_pred",
    "user_score",
}


# ---------- Utils ----------

def sanitize(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z._\\-]+", "_", str(s))


def load_piaa_from_dir(input_dir: str) -> pd.DataFrame:
    if not os.path.isdir(input_dir):
        raise RuntimeError(f"input_dir is not a directory: {input_dir}")

    dfs: List[pd.DataFrame] = []
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
            print(f"[info] skip {path} (missing columns)")
            continue

        dfs.append(df)

    if not dfs:
        raise RuntimeError(f"No valid PIAA CSVs found in {input_dir}")

    return pd.concat(dfs, ignore_index=True)


def plot_hist(
    values: np.ndarray,
    title: str,
    out_path: str,
    bins: int = 20,
):
    if len(values) == 0:
        return False

    plt.close("all")
    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    ax.hist(values, bins=bins)

    ax.set_xlim(1.0, 5.0)
    # ax.set_ylim(0, 50)
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.set_title(title)

    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Directory containing PIAA result CSVs.")
    ap.add_argument("--out_dir", required=True, help="Output directory for histograms.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_piaa_from_dir(args.input_dir)

    # ---- GT histogram (one figure) ----
    gt_vals = pd.to_numeric(df["user_score"], errors="coerce").dropna().to_numpy()
    gt_out = os.path.join(args.out_dir, "gt", "gt_score_hist.png")
    plot_hist(
        gt_vals,
        title="GT score distribution",
        out_path=gt_out,
    )
    print(f"[plot] {gt_out}")

    # ---- Method-wise predicted score histograms ----
    method_dir = os.path.join(args.out_dir, "methods")
    os.makedirs(method_dir, exist_ok=True)

    groups = df.groupby(["method", "support_set"])
    for (method, sup), g in groups:
        vals = pd.to_numeric(g["piaa_pred"], errors="coerce").dropna().to_numpy()
        if len(vals) == 0:
            continue

        fname = f"{sanitize(method)}__{sanitize(sup)}.png"
        out_path = os.path.join(method_dir, fname)

        title = f"{method} (support={sup})"
        plot_hist(
            vals,
            title=title,
            out_path=out_path,
        )
        print(f"[plot] {out_path}")

    print("[done]")


if __name__ == "__main__":
    main()