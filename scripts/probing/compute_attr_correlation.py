#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute correlation matrices (Spearman and/or Pearson) between aesthetic attributes
for PARA / AADB and save CSV + heatmap PNG.

Usage examples:
  python -m scripts.probing.compute_attr_correlation --dataset para --dataset_dir datasets/PARA --out_dir runs/stats --method both
  python -m scripts.probing.compute_attr_correlation --dataset aadb --dataset_dir datasets/aadb --out_dir runs/stats --method spearman
"""

import os
import argparse
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

# dataset utils (assumes these modules exist in utils/)
from utils.para import get_para_dataset
from utils.aadb import get_aadb_dataset


def items_to_dataframe_para(items) -> pd.DataFrame:
    rows = []
    for it in items:
        row = {}
        if hasattr(it, "attributes") and isinstance(it.attributes, dict):
            row.update(it.attributes)
        if hasattr(it, "score"):
            row.setdefault("score", float(it.score))
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def items_to_dataframe_aadb(items) -> pd.DataFrame:
    rows = []
    for it in items:
        row = {}
        if hasattr(it, "attributes") and isinstance(it.attributes, dict):
            row.update(it.attributes)
        if hasattr(it, "score"):
            row["score"] = float(it.score)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def compute_spearman_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (corr_df, pval_df) for Spearman."""
    df_num = df.select_dtypes(include=[np.number]).copy()
    cols = df_num.columns.tolist()
    n = len(cols)
    corr = np.zeros((n, n), dtype=float)
    pval = np.ones((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            x = df_num.iloc[:, i].to_numpy(dtype=float)
            y = df_num.iloc[:, j].to_numpy(dtype=float)
            mask = (~np.isnan(x)) & (~np.isnan(y))
            if mask.sum() < 2:
                corr[i, j] = np.nan
                pval[i, j] = np.nan
            else:
                r, p = spearmanr(x[mask], y[mask])
                corr[i, j] = float(r)
                pval[i, j] = float(p)
    corr_df = pd.DataFrame(corr, index=cols, columns=cols)
    pval_df = pd.DataFrame(pval, index=cols, columns=cols)
    return corr_df, pval_df


def compute_pearson_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (corr_df, pval_df) for Pearson."""
    df_num = df.select_dtypes(include=[np.number]).copy()
    cols = df_num.columns.tolist()
    n = len(cols)
    corr = np.zeros((n, n), dtype=float)
    pval = np.ones((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            x = df_num.iloc[:, i].to_numpy(dtype=float)
            y = df_num.iloc[:, j].to_numpy(dtype=float)
            mask = (~np.isnan(x)) & (~np.isnan(y))
            if mask.sum() < 2:
                corr[i, j] = np.nan
                pval[i, j] = np.nan
            else:
                r, p = pearsonr(x[mask], y[mask])
                corr[i, j] = float(r)
                pval[i, j] = float(p)
    corr_df = pd.DataFrame(corr, index=cols, columns=cols)
    pval_df = pd.DataFrame(pval, index=cols, columns=cols)
    return corr_df, pval_df


def save_heatmap(corr_df: pd.DataFrame, out_png: str, title: str = "Correlation", vmin: float = -1.0, vmax: float = 1.0):
    plt.close("all")
    fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(corr_df.columns)), max(5, 0.45 * len(corr_df.columns))))
    im = ax.imshow(corr_df.values, vmin=vmin, vmax=vmax, cmap="coolwarm", aspect="equal")
    ax.set_xticks(np.arange(len(corr_df.columns)))
    ax.set_yticks(np.arange(len(corr_df.index)))
    columns = [
        "Overall Score" if col == 'score' else col
        for col in corr_df.columns
    ]
    indices = [
        "Overall Score" if idx == 'score' else idx  
        for idx in corr_df.index
    ]

    ax.set_xticklabels(columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(indices, fontsize=8)
    # show values
    for i in range(len(corr_df.index)):
        for j in range(len(corr_df.columns)):
            val = corr_df.iat[i, j]
            if np.isnan(val):
                txt = ""
            else:
                txt = f"{val:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=400)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["para", "aadb"], help="Dataset to analyze.")
    ap.add_argument("--dataset_dir", required=True, help="Path to dataset root directory.")
    ap.add_argument("--out_dir", required=True, help="Output directory to save matrices and images.")
    ap.add_argument("--method", choices=["spearman", "pearson", "both"], default="both",
                    help="Which correlation method(s) to compute.")
    ap.add_argument("--aadb_splits", nargs="+", default=["train", "validation", "test"], help="splits of AADB to load")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.dataset == "para":
        items = get_para_dataset(None, dataset_dir=args.dataset_dir)
        df = items_to_dataframe_para(items)
    else:  # aadb
        all_items = []
        for s in args.aadb_splits:
            items = get_aadb_dataset(s, dataset_dir=args.dataset_dir)
            all_items.extend(items)
        df = items_to_dataframe_aadb(all_items)

    if df.empty:
        raise RuntimeError("Constructed dataframe is empty; check dataset_dir and dataset availability.")

    # drop columns with all NaN
    df = df.dropna(axis=1, how="all")

    methods = [args.method] if args.method in ("spearman", "pearson") else ["spearman", "pearson"]
    for m in methods:
        if m == "spearman":
            corr_df, pval_df = compute_spearman_matrix(df)
        else:
            corr_df, pval_df = compute_pearson_matrix(df)

        base = os.path.join(args.out_dir, f"{args.dataset}_{m}")
        corr_csv = base + "_corr.csv"
        pval_csv = base + "_pval.csv"
        corr_df.to_csv(corr_csv)
        pval_df.to_csv(pval_csv)
        print(f"[save] {m} correlation CSV -> {corr_csv}")
        print(f"[save] {m} p-value CSV      -> {pval_csv}")

        pdf_path = base + "_corr.pdf"
        save_heatmap(corr_df, pdf_path, title=f"{m.title()} correlation")
        print(f"[save] {m} heatmap PDF -> {pdf_path}")

    print("[done]")


if __name__ == "__main__":
    main()
