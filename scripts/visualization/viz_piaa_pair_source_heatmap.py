#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize pair-grid PIAA summary CSVs as source-pair heatmaps.

Example:
  python -m scripts.visualization.viz_piaa_pair_source_heatmap \
    --input_root outputs/piaa_pair/lapis \
    --row_source vision \
    --col_source llm_text \
    --out_dir outputs/viz/piaa_pair_heatmap/lapis
"""

import os
import csv
import argparse
from typing import Dict, List, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRIC = "mean_rho"
MODEL_TITLE_MAP = {
    "gemma3-4b": "Gemma 3 4B",
    "qwen3vl-2b": "Qwen3-VL 2B",
}


def load_summary(path: str) -> List[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["l1"] = int(row["l1"])
            row["l2"] = int(row["l2"])
            row["n_users"] = int(row["n_users"])
            row[METRIC] = float(row[METRIC])
            rows.append(row)
    return rows


def collect_pair_rows(rows: List[dict], row_source: str, col_source: str) -> Tuple[List[int], List[int], Dict[Tuple[int, int], dict]]:
    row_layers = set()
    col_layers = set()
    pair_to_row: Dict[Tuple[int, int], dict] = {}

    for row in rows:
        s1, l1 = row["s1"], row["l1"]
        s2, l2 = row["s2"], row["l2"]

        if {s1, s2} != {row_source, col_source}:
            continue
        if row_source == col_source and s1 != s2:
            continue

        if s1 == row_source and s2 == col_source:
            row_layer, col_layer = l1, l2
        elif s1 == col_source and s2 == row_source:
            row_layer, col_layer = l2, l1
        else:
            continue

        key = (row_layer, col_layer)
        if key in pair_to_row:
            prev = pair_to_row[key]
            if abs(prev[METRIC] - row[METRIC]) > 1e-12:
                raise ValueError(
                    f"Inconsistent duplicate for {key} in {row_source} x {col_source}: "
                    f"{prev[METRIC]} vs {row[METRIC]}"
                )
        else:
            pair_to_row[key] = row
            row_layers.add(row_layer)
            col_layers.add(col_layer)

    return sorted(row_layers), sorted(col_layers), pair_to_row


def build_matrix(
    row_layers: List[int],
    col_layers: List[int],
    pair_to_row: Dict[Tuple[int, int], dict],
) -> np.ndarray:
    mat = np.full((len(row_layers), len(col_layers)), np.nan, dtype=float)
    for i, row_layer in enumerate(row_layers):
        for j, col_layer in enumerate(col_layers):
            row = pair_to_row.get((row_layer, col_layer))
            if row is not None:
                mat[i, j] = row[METRIC]
    return mat


def plot_heatmaps(
    model_label: str,
    row_source: str,
    col_source: str,
    row_layers: List[int],
    col_layers: List[int],
    mat: np.ndarray,
    out_path: str,
) -> None:
    plt.close("all")
    fig, ax = plt.subplots(figsize=(6.8, 5.6), constrained_layout=True)

    finite = mat[np.isfinite(mat)]
    if finite.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = float(np.min(finite)), float(np.max(finite))
        if abs(vmin - vmax) <= 1e-12:
            pad = 1e-3 if vmin == 0.0 else max(1e-3, abs(vmin) * 0.01)
            vmin -= pad
            vmax += pad

    im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title("Mean Spearman Correlation", fontsize=13)
    ax.set_xlabel("LT")
    ax.set_ylabel("V")
    ax.set_xticks(np.arange(len(col_layers)))
    ax.set_yticks(np.arange(len(row_layers)))
    ax.set_xticklabels(col_layers)
    ax.set_yticklabels(row_layers)

    midpoint = (vmin + vmax) / 2.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isnan(val):
                continue
            color = "white" if val < midpoint else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8, color=color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(model_label, fontsize=12)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def format_model_title(model_name: str) -> str:
    return MODEL_TITLE_MAP.get(model_name, model_name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", required=True, help="Root directory containing per-model summary.csv files.")
    ap.add_argument("--row_source", default="vision")
    ap.add_argument("--col_source", default="llm_text")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    model_dirs = []
    for name in sorted(os.listdir(args.input_root)):
        summary_path = os.path.join(args.input_root, name, "summary.csv")
        if os.path.isfile(summary_path):
            model_dirs.append((name, summary_path))

    if not model_dirs:
        raise FileNotFoundError(f"No summary.csv found under: {args.input_root}")

    for model_name, summary_path in model_dirs:
        rows = load_summary(summary_path)
        row_layers, col_layers, pair_to_row = collect_pair_rows(rows, args.row_source, args.col_source)
        if not pair_to_row:
            print(f"[skip] no rows for {model_name}: {args.row_source} x {args.col_source}")
            continue

        row_layers = sorted(row_layers, reverse=True)
        mat = build_matrix(row_layers, col_layers, pair_to_row)

        model_out_dir = os.path.join(args.out_dir, model_name)
        pdf_path = os.path.join(model_out_dir, f"{args.row_source}_x_{args.col_source}.pdf")

        plot_heatmaps(
            model_label=format_model_title(model_name),
            row_source=args.row_source,
            col_source=args.col_source,
            row_layers=row_layers,
            col_layers=col_layers,
            mat=mat,
            out_path=pdf_path,
        )

        print(f"[save] heatmap -> {pdf_path}")

    print("[done]")


if __name__ == "__main__":
    main()
