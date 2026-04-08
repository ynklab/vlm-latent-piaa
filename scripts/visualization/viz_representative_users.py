#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualize representative users for ALL PIAA methods found in a directory
as 10x10 confusion-matrix-style heatmaps with bin width 0.5.

Input:
  - A directory containing PIAA baseline CSVs (from piaa_from_giaa.py,
    residual/direct/hidden_attr/LoRA/CoT, etc.) with columns:

      user_id, image_path, model_id, support_set, method, giaa, piaa_pred, user_score, ...

Behavior:
  - Load all such CSVs from --input_dir.
  - For each (model_id, support_set, method) combination:
      * Compute per-user Spearman rho (always rho).
      * Select representative users (per method combo):
          - top_k     users with highest rho
          - mid_k     users around the median rho
          - bottom_k  users with lowest rho
      * For each representative user:
          - Build a 10x10 confusion-matrix-like count matrix with bin width 0.5:

                bin edges: 0.75, 1.25, 1.75, ..., 5.75  (11 edges)
                bins     : [0.75,1.25), [1.25,1.75), ..., [5.25,5.75)

                rows   = GT bins (user_score)
                cols   = Pred bins (piaa_pred)

          - Plot this as a heatmap with counts.

          - Save to:

              <out_dir>/<sanitize(method)>/
                <sanitize(model_id)>__<support_set>__user_<user_id>__<group>_rho_confmat.png

Where group ∈ {"top", "mid", "bottom"}.
"""

import os
import re
import argparse
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.metrics import r2_score


def sanitize(s: str) -> str:
    if s is None or not isinstance(s, str) or s.strip() == "":
        return "unknown"
    return re.sub(r"[^0-9A-Za-z._\\-]+", "_", s)


def load_baseline_from_dir(input_dir: str) -> pd.DataFrame:
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

    return pd.concat(dfs, ignore_index=True)


def compute_user_metrics(df: pd.DataFrame, min_items: int = 2) -> pd.DataFrame:
    """
    df: filtered to one (model_id, support_set, method).
    Return: DataFrame with columns [user_id, n_items, rho, r2].
    """
    groups = df.groupby("user_id")
    rows = []
    for user_id, g in groups:
        y_true = g["user_score"].to_numpy(dtype=float)
        y_pred = g["piaa_pred"].to_numpy(dtype=float)
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        if len(y_true) < min_items:
            continue

        rho = spearmanr(y_true, y_pred).correlation
        if np.isnan(rho):
            rho = 0.0
        try:
            r2 = float(r2_score(y_true, y_pred))
        except Exception:
            r2 = float("nan")

        rows.append(
            {
                "user_id": user_id,
                "n_items": len(y_true),
                "rho": rho,
                "r2": r2,
            }
        )
    return pd.DataFrame(rows)


def pick_representative_users(df_user: pd.DataFrame, top_k: int, mid_k: int, bottom_k: int):
    """
    df_user: per-user metrics DataFrame with columns ['user_id', 'rho', 'r2', 'n_items'].
    Return dict with keys: 'top', 'mid', 'bottom', each a list of user_id.
    Metric is always rho (descending).
    """
    if df_user.empty:
        return {"top": [], "mid": [], "bottom": []}

    df_sorted = df_user.sort_values(by="rho", ascending=False).reset_index(drop=True)
    n = len(df_sorted)

    # top
    top_users = df_sorted.head(top_k)["user_id"].tolist()

    # bottom
    bottom_users = df_sorted.tail(bottom_k)["user_id"].tolist()

    # middle: around median
    mid_users = []
    if mid_k > 0:
        mid_center = n // 2
        half = mid_k // 2
        start = max(0, mid_center - half)
        end = min(n, start + mid_k)
        mid_users = df_sorted.iloc[start:end]["user_id"].tolist()

    return {"top": top_users, "mid": mid_users, "bottom": bottom_users}


# --- 10-bin setup with width 0.5: edges 0.75, 1.25, ..., 5.75 / centers 1.0, 1.5, ..., 5.5 ---

BIN_EDGES = np.arange(0.75, 5.75 + 1e-6, 0.5)  # 0.75..5.75 (step=0.5) → 11 edges
BIN_N = len(BIN_EDGES) - 1                      # 10 bins
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2.0  # 1.0,1.5,...,5.5


def build_confusion_matrix_for_user(g: pd.DataFrame) -> np.ndarray:
    """
    g: rows for a single user_id (single method).
    Build 10x10 confusion-matrix-like count matrix with bin width 0.5:

      rows = GT bins (user_score in [0.75,1.25), [1.25,1.75), ..., [5.25,5.75))
      cols = Pred bins (piaa_pred on the same scale)
    """
    y_true = g["user_score"].to_numpy(dtype=float)
    y_pred = g["piaa_pred"].to_numpy(dtype=float)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    mat = np.zeros((BIN_N, BIN_N), dtype=int)
    if y_true.size == 0:
        return mat

    # bin index: 0..BIN_N-1
    gt_bins = np.digitize(y_true, BIN_EDGES, right=False) - 1
    pr_bins = np.digitize(y_pred, BIN_EDGES, right=False) - 1
    gt_bins = np.clip(gt_bins, 0, BIN_N - 1)
    pr_bins = np.clip(pr_bins, 0, BIN_N - 1)

    for g_bin, p_bin in zip(gt_bins, pr_bins):
        mat[g_bin, p_bin] += 1

    return mat


def plot_user_confmat(
    g: pd.DataFrame,
    model_id: str,
    support_set: str,
    method: str,
    group_name: str,
    rho_value: float,
    out_path: str,
    figsize=(5, 4),
    dpi: int = 160,
) -> bool:
    """
    g: rows for a single user_id (single method).
    Plot 10x10 confusion-matrix-like heatmap.
    rows: GT bins, cols: Pred bins (both in bins of width 0.5).
    """
    if g.empty:
        return False

    mat = build_confusion_matrix_for_user(g)
    if mat.sum() == 0:
        return False

    plt.close("all")
    fig, ax = plt.subplots(figsize=figsize)

    # Set extent to [min_edge, max_edge] so each cell width is about 0.5
    extent = [BIN_EDGES[0], BIN_EDGES[-1], BIN_EDGES[0], BIN_EDGES[-1]]
    im = ax.imshow(
        mat,
        interpolation="nearest",
        cmap="Blues",
        origin="lower",   # lower side corresponds to bin index 0 (lowest score range)
        extent=extent,
        aspect="auto",
    )

    # Axis labels use bin centers (1.0, 1.5, ..., 5.5)
    ax.set_xticks(BIN_CENTERS)
    ax.set_xticklabels([f"{c:.1f}" for c in BIN_CENTERS], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(BIN_CENTERS)
    ax.set_yticklabels([f"{c:.1f}" for c in BIN_CENTERS], fontsize=8)

    ax.set_xlabel("Predicted score (bin centers, width=0.5)")
    ax.set_ylabel("Ground truth score (bin centers, width=0.5)")
    ax.set_title(
        f"{model_id}\n"
        f"user_id={g['user_id'].iloc[0]}, support_set={support_set}, method={method} ({group_name})\n"
        f"rho={rho_value:.3f}"
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Count")

    # Draw counts at cell centers: coordinates are (pred_center, gt_center)
    thresh = mat.max() / 2.0 if mat.max() > 0 else 0
    for i in range(BIN_N):   # GT bins
        for j in range(BIN_N):  # Pred bins
            value = mat[i, j]
            if value == 0:
                continue
            color = "white" if value > thresh else "black"
            ax.text(
                BIN_CENTERS[j],
                BIN_CENTERS[i],
                str(value),
                ha="center",
                va="center",
                color=color,
                fontsize=7,
            )

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
        help="Directory to save representative user confusion-matrix heatmaps.",
    )
    ap.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of top users to visualize per (model_id, support_set, method).",
    )
    ap.add_argument(
        "--mid_k",
        type=int,
        default=3,
        help="Number of middle users to visualize per (model_id, support_set, method).",
    )
    ap.add_argument(
        "--bottom_k",
        type=int,
        default=3,
        help="Number of bottom users to visualize per (model_id, support_set, method).",
    )
    ap.add_argument(
        "--min_items_per_user",
        type=int,
        default=5,
        help="Minimum number of items per user to be considered.",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[info] loading baseline data...")
    df_all = load_baseline_from_dir(args.input_dir)
    print(f"[info] total rows = {len(df_all)}")

    # Visualize representative users for each (model_id, support_set, method)
    grouped = df_all.groupby(["model_id", "support_set", "method"])

    for (model_id, support_set, method), g_method in tqdm(grouped, desc="Method combos"):
        print(
            f"\n[info] combo: model_id={model_id}, support_set={support_set}, method={method}, "
            f"rows={len(g_method)}"
        )

        # per-user metrics (rho)
        df_user = compute_user_metrics(g_method, min_items=args.min_items_per_user)
        print(f"[info]   per-user rows (min_items={args.min_items_per_user}) = {len(df_user)}")
        if df_user.empty:
            print("[info]   -> no users with enough items, skip.")
            continue

        reps = pick_representative_users(
            df_user,
            top_k=args.top_k,
            mid_k=args.mid_k,
            bottom_k=args.bottom_k,
        )

        print(f"[info]   representative users (rho-based):")
        for k in ["top", "mid", "bottom"]:
            print(f"    {k}: {reps[k]}")

        # Subdirectories for each method
        method_dir = os.path.join(args.out_dir, sanitize(str(method)))
        os.makedirs(method_dir, exist_ok=True)

        # Draw a confusion-matrix-style heatmap for each representative user
        for group_name, user_ids in reps.items():
            for user_id in user_ids:
                g_user = g_method[g_method["user_id"] == user_id].copy()
                if g_user.empty:
                    continue
                rho_val = df_user[df_user["user_id"] == user_id]["rho"].iloc[0]

                fname = (
                    f"{sanitize(str(model_id))}__{sanitize(str(support_set))}__"
                    f"user_{sanitize(str(user_id))}__{group_name}_rho_confmat.png"
                )
                out_path = os.path.join(method_dir, fname)
                plot_user_confmat(
                    g_user,
                    model_id=str(model_id),
                    support_set=str(support_set),
                    method=str(method),
                    group_name=group_name,
                    rho_value=rho_val,
                    out_path=out_path,
                )

    print("[done] representative user confusion-matrix heatmaps generated for all method combos.")


if __name__ == "__main__":
    main()