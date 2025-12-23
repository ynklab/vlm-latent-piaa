#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
For PARA or LAPIS, compute per-annotator correlation between their ratings and
"ground truth PIAA" (per-image consensus rating), and plot a histogram.

Ground truth options:
  - default: simple per-image mean over all annotators
  - --leave_one_out: per-image mean over all annotators except the annotator
    whose agreement is being measured (more strict, but slightly more complex)

Datasets:
  - PARA  : uses utils.para._load_personalized_data
            columns: userId, image_path, aestheticScore (etc.)
            we use "aestheticScore" as the rating.
  - LAPIS : uses utils.lapis._load_personalized_data
            columns: participant_id, image_id, rating (0..100)
            we convert rating to 1..5 scale as in other scripts:
              score = rating/100 * 4 + 1

Outputs:
  - Histogram PNG of per-annotator correlation coefficients
  - Prints summary stats (mean, std, min, max, count)

Usage examples:

  # PARA, Spearman, simple mean as GT
  python plot_annotator_piaa_agreement_hist.py \
    --dataset para \
    --dataset_dir datasets/PARA \
    --out_png runs/para_annotator_agreement_hist.png

  # LAPIS, Spearman, leave-one-out GT, min 30 ratings per annotator
  python plot_annotator_piaa_agreement_hist.py \
    --dataset lapis \
    --dataset_dir datasets/LAPIS \
    --out_png runs/lapis_annotator_agreement_hist_loo.png \
    --leave_one_out \
    --min_items 30
"""

import os
import argparse
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

# ---- dataset utils ----

try:
    from utils.para import _load_personalized_data as para_load_personal
except Exception:
    para_load_personal = None

try:
    from utils.lapis import _load_personalized_data as lapis_load_personal
except Exception:
    lapis_load_personal = None


# ---- helpers ----

def build_personal_df(dataset: str, dataset_dir: str) -> pd.DataFrame:
    """
    Build a unified DataFrame with columns:

      annotator_id : str/int
      image_key    : hashable image id (path or id)
      score        : float (1..5 scale)

    for the specified dataset ("para" or "lapis").
    """
    if dataset == "para":
        if para_load_personal is None:
            raise RuntimeError("utils.para._load_personal_data not importable.")
        df = para_load_personal(dataset_dir)
        # expect columns: userId, image_path, aestheticScore, ...
        if "userId" not in df.columns or "image_path" not in df.columns or "aestheticScore" not in df.columns:
            raise RuntimeError(f"PARA personalized data missing required columns. Found: {df.columns}")
        out = pd.DataFrame(
            {
                "annotator_id": df["userId"].astype(str),
                "image_key": df["image_path"].astype(str),
                "score": df["aestheticScore"].astype(float),
            }
        )
    else:  # lapis
        if lapis_load_personal is None:
            raise RuntimeError("utils.lapis._load_personal_data not importable.")
        df = lapis_load_personal(dataset_dir)
        # expect columns: participant_id, image_id, rating, ...
        if "participant_id" not in df.columns or "image_id" not in df.columns or "rating" not in df.columns:
            raise RuntimeError(f"LAPIS personalized data missing required columns. Found: {df.columns}")
        # convert rating 0..100 -> 1..5
        score = (df["rating"].astype(float) / 100.0) * 4.0 + 1.0
        out = pd.DataFrame(
            {
                "annotator_id": df["participant_id"].astype(str),
                "image_key": df["image_id"].astype(str),
                "score": score,
            }
        )

    # drop NaNs
    out = out.dropna(subset=["score", "annotator_id", "image_key"])
    return out


def compute_annotator_correlations(
    df: pd.DataFrame,
    leave_one_out: bool = False,
    metric: str = "spearman",
    min_items: int = 10,
) -> List[float]:
    """
    df: columns = annotator_id, image_key, score (float)

    Returns list of per-annotator correlation coefficients between:
      x = their scores
      y = ground truth (per-image consensus mean)
    according to the chosen options.
    """
    # per-image sum & count
    img_stats = df.groupby("image_key")["score"].agg(["sum", "count"]).reset_index()
    img_stats = img_stats.rename(columns={"sum": "img_sum", "count": "img_count"})
    df = df.merge(img_stats, on="image_key", how="left")

    if leave_one_out:
        # GT = mean over other annotators (exclude self)
        # if img_count == 1, gt_excl is NaN
        df["gt"] = (df["img_sum"] - df["score"]) / (df["img_count"] - 1)
        df.loc[df["img_count"] <= 1, "gt"] = np.nan
    else:
        # GT = mean over all annotators
        df["gt"] = df["img_sum"] / df["img_count"]

    rhos: List[float] = []

    grouped = df.groupby("annotator_id")
    for annotator, g in grouped:
        g_valid = g.dropna(subset=["gt", "score"])
        if len(g_valid) < min_items:
            continue
        x = g_valid["score"].to_numpy(dtype=float)
        y = g_valid["gt"].to_numpy(dtype=float)
        if np.all(x == x[0]) or np.all(y == y[0]):
            # constant array → correlation undefined, skip
            continue
        if metric == "spearman":
            r = spearmanr(x, y).correlation
        else:
            r = pearsonr(x, y)[0]
        if not np.isnan(r):
            rhos.append(float(r))

    return rhos


def plot_histogram(rhos: List[float], out_png: str, title: str):
    if not rhos:
        raise RuntimeError("No valid per-annotator correlations to plot.")

    rhos_arr = np.array(rhos, dtype=float)

    plt.close("all")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.hist(rhos_arr, bins=30, range=(-1.0, 1.0), alpha=0.75, edgecolor="black")
    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel("Correlation with GIAA scores", fontsize=18)
    ax.set_ylabel("Number of annotators", fontsize=18)
    ax.set_title(title, fontsize=18)

    # Summary stats as text
    txt = (
        f"n_annotators = {len(rhos_arr)}\n"
        f"mean = {rhos_arr.mean():.3f}\n"
        f"std  = {rhos_arr.std():.3f}\n"
        f"min  = {rhos_arr.min():.3f}\n"
        f"max  = {rhos_arr.max():.3f}"
    )
    # ax.text(
    #     0.98,
    #     0.95,
    #     txt,
    #     transform=ax.transAxes,
    #     ha="left",
    #     va="top",
    #     fontsize=8,
    #     bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    # )

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=400)
    plt.close(fig)
    print(f"[save] histogram -> {out_png}")
    print("[stats]")
    print(txt.replace("\n", "  "))


# ---- main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        required=True,
        choices=["para", "lapis"],
        help="Dataset name (para or lapis).",
    )
    ap.add_argument(
        "--dataset_dir",
        required=True,
        help="Dataset root directory (datasets/PARA or datasets/LAPIS).",
    )
    ap.add_argument(
        "--out_png",
        required=True,
        help="Path to output PNG file for the histogram.",
    )
    ap.add_argument(
        "--metric",
        choices=["spearman", "pearson"],
        default="spearman",
        help="Correlation metric to use (default: spearman).",
    )
    ap.add_argument(
        "--leave_one_out",
        action="store_true",
        help="Use leave-one-out mean as ground truth (exclude annotator's own rating).",
    )
    ap.add_argument(
        "--min_items",
        type=int,
        default=10,
        help="Minimum number of rated images per annotator to be included.",
    )

    args = ap.parse_args()

    df = build_personal_df(args.dataset, args.dataset_dir)
    print(f"[info] loaded personalized data: {len(df)} rows, "
          f"{df['annotator_id'].nunique()} annotators, {df['image_key'].nunique()} images.")

    rhos = compute_annotator_correlations(
        df,
        leave_one_out=args.leave_one_out,
        metric=args.metric,
        min_items=args.min_items,
    )

    if args.dataset == "para":
        dataset_label = "PARA"
    else:
        dataset_label = "LAPIS"

    loo_label = " (leave-one-out)" if args.leave_one_out else ""
    # title = f"{dataset_label}: annotator vs GT PIAA {args.metric.title()} correlation{loo_label}"
    title = dataset_label

    plot_histogram(rhos, args.out_png, title)


if __name__ == "__main__":
    main()