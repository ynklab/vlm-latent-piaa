#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
For PARA or LAPIS, plot per-image relationship between:

  x = General aesthetic score (GIAA)
  y = Variance of personalized ratings across users

Uses:
  - utils.para.get_para_dataset, utils.para._load_personalized_data
  - utils.lapis.get_lapis_dataset, utils.lapis._load_personalized_data

Usage examples:

  # PARA
  python -m scripts.visualization.plot_general_vs_personal_variance \
    --dataset para \
    --dataset_dir datasets/PARA \
    --out_png runs/para_general_vs_personal_var.png

  # LAPIS
  python -m scripts.visualization.plot_general_vs_personal_variance \
    --dataset lapis \
    --dataset_dir datasets/LAPIS \
    --out_png runs/lapis_general_vs_personal_var.png
"""

import os
import argparse
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------- dataset utils ----------

try:
    from utils.para import get_para_dataset, _load_personalized_data as para_load_personal
except Exception:
    get_para_dataset = None
    para_load_personal = None

try:
    from utils.lapis import get_lapis_dataset, _load_personalized_data as lapis_load_personal
except Exception:
    get_lapis_dataset = None
    lapis_load_personal = None


# --------- core plotting helpers ----------

def _scatter_plot(df: pd.DataFrame,
                  x_col: str,
                  y_col: str,
                  out_png: str,
                  title: str,
                  max_points: int | None = 5000) -> None:
    """Generic scatter plot helper."""
    plot_df = df[[x_col, y_col]].dropna().copy()
    if plot_df.empty:
        raise RuntimeError("No data to plot after dropping NaNs.")

    n_total = len(plot_df)
    if max_points is not None and n_total > max_points:
        # random subsample for readability
        plot_df = plot_df.sample(n=max_points, random_state=42)
        print(f"[info] subsampled {n_total} -> {len(plot_df)} points for plotting.")

    x = plot_df[x_col].to_numpy(dtype=float)
    y = plot_df[y_col].to_numpy(dtype=float)

    # compute simple correlations
    pearson = np.corrcoef(x, y)[0, 1] if len(x) > 1 else np.nan
    # Spearman
    from scipy.stats import spearmanr
    rho = spearmanr(x, y).correlation if len(x) > 1 else np.nan

    plt.close("all")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, alpha=0.5, s=15)
    ax.set_xlabel("General score (per image)")
    ax.set_ylabel("Variance of personalized scores (per image)")
    ax.set_title(title)

    # add a small text with basic stats
    txt = f"n_images={len(plot_df)}\nPearson={pearson:.3f}\nSpearman={rho:.3f}"
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=8,
            va="top", ha="left", bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[save] scatter -> {out_png}")


# --------- dataset-specific builders ----------

def build_para_df(dataset_dir: str) -> pd.DataFrame:
    """
    Build a DataFrame with columns:
      image_path, general_score, personal_var, n_raters
    for PARA.
    """
    if get_para_dataset is None or para_load_personal is None:
        raise RuntimeError("utils.para not available; make sure it is importable.")

    # General (GIAA): mean aesthetic score per image
    items = get_para_dataset(None, dataset_dir=dataset_dir)
    if not items:
        raise RuntimeError("get_para_dataset returned no items; check dataset_dir.")

    rows_g = []
    for it in items:
        rows_g.append(
            {
                "image_path": it.image_path,
                "general_score": float(it.score),
            }
        )
    df_g = pd.DataFrame(rows_g)

    # Personalized: all individual ratings from PARA-Images.csv
    df_p = para_load_personal(dataset_dir)
    # df_p has columns incl. 'image_path' and 'aestheticScore'
    if "image_path" not in df_p.columns or "aestheticScore" not in df_p.columns:
        raise RuntimeError(f"PARA personalized data missing required columns. Found: {df_p.columns}")

    # per-image variance of personalized scores
    grp = df_p.groupby("image_path")["aestheticScore"]
    personal_var = grp.var(ddof=0)  # population variance
    n_raters = grp.size()
    df_var = pd.DataFrame(
        {
            "image_path": personal_var.index,
            "personal_var": personal_var.values,
            "n_raters": n_raters.values,
        }
    )

    # join
    df = pd.merge(df_g, df_var, on="image_path", how="inner")
    print(f"[info][PARA] joined images: {len(df)}")
    return df


def build_lapis_df(dataset_dir: str) -> pd.DataFrame:
    """
    Build a DataFrame with columns:
      image_id, general_score, personal_var, n_raters
    for LAPIS.
    """
    if get_lapis_dataset is None or lapis_load_personal is None:
        raise RuntimeError("utils.lapis not available; make sure it is importable.")

    # General: LAPIS_GIAA_* splits
    items = get_lapis_dataset(None, dataset_dir=dataset_dir)
    if not items:
        raise RuntimeError("get_lapis_dataset returned no items; check dataset_dir.")

    rows_g = []
    for it in items:
        rows_g.append(
            {
                "image_id": it.image_id,
                "general_score": float(it.score),
            }
        )
    df_g = pd.DataFrame(rows_g)

    # Personalized: all individual ratings from LAPIS_PIAA.csv
    df_p = lapis_load_personal(dataset_dir)
    # columns: 'image_id', 'rating', ...
    if "image_id" not in df_p.columns or "rating" not in df_p.columns:
        raise RuntimeError(f"LAPIS personalized data missing required columns. Found: {df_p.columns}")

    # Convert rating (0-100) to 1..5 scale, same as general_score
    df_p = df_p.copy()
    df_p["score"] = (df_p["rating"] / 100.0) * 4.0 + 1.0

    grp = df_p.groupby("image_id")["score"]
    personal_var = grp.var(ddof=0)
    n_raters = grp.size()
    df_var = pd.DataFrame(
        {
            "image_id": personal_var.index,
            "personal_var": personal_var.values,
            "n_raters": n_raters.values,
        }
    )

    # join
    df = pd.merge(df_g, df_var, on="image_id", how="inner")
    print(f"[info][LAPIS] joined images: {len(df)}")
    return df


# --------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        required=True,
        choices=["para", "lapis"],
        help="Dataset to analyze (para or lapis).",
    )
    ap.add_argument(
        "--dataset_dir",
        required=True,
        help="Path to dataset root directory (datasets/PARA or datasets/LAPIS).",
    )
    ap.add_argument(
        "--out_png",
        required=True,
        help="Output PNG path for the scatter plot.",
    )
    ap.add_argument(
        "--max_points",
        type=int,
        default=5000,
        help="Max points to plot (random subsampling for readability).",
    )
    args = ap.parse_args()

    if args.dataset == "para":
        df = build_para_df(args.dataset_dir)
        title = "PARA: General score vs variance of personalized scores"
    else:  # lapis
        df = build_lapis_df(args.dataset_dir)
        title = "LAPIS: General score vs variance of personalized scores"

    # もし「評価者数が少ない画像は避けたい」ならここでフィルタも可能:
    # df = df[df["n_raters"] >= 5]

    _scatter_plot(
        df=df,
        x_col="general_score",
        y_col="personal_var",
        out_png=args.out_png,
        title=title,
        max_points=args.max_points,
    )

    # 参考までに分散やスコアの簡単な統計も表示
    print("[stats] general_score:", df["general_score"].describe())
    print("[stats] personal_var :", df["personal_var"].describe())
    print("[stats] n_raters     :", df["n_raters"].describe())


if __name__ == "__main__":
    main()
