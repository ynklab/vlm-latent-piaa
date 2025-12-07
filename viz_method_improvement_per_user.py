#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualize per-user improvement across PIAA methods
using 'raw' as a fixed baseline.

Input:
  - --input_dir: directory containing PIAA baseline CSVs.
    Each CSV must have columns:

      user_id, image_path, model_id, support_set, method, giaa, piaa_pred, user_score, ...

Behavior:
  - Load all such CSVs from --input_dir.
  - Compute per-user Spearman rho for each (model_id, support_set, method, user_id).
  - For each (model_id, support_set) combination:
      * Use method 'raw' as baseline.
      * For every other method M (compare_methods):
          - Consider users that appear in both raw and M.
          - Build two scatter plots:

            (a) raw_rho vs method_rho:
                  x: rho_raw(u),  y: rho_M(u)

            (b) raw_rho vs delta_rho:
                  x: rho_raw(u),  y: rho_M(u) - rho_raw(u)

          - Save plots under:

                <out_dir>/<sanitize(method)>/
                  <model>__<support>__raw__vs__<method>__rho_scatter.png
                  <model>__<support>__raw__vs__<method>__delta_rho_scatter.png
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


# ---------- Utils ----------

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


def compute_per_user_rho(df: pd.DataFrame, min_items: int = 2) -> pd.DataFrame:
    """
    df: rows with columns (model_id, support_set, method, user_id, piaa_pred, user_score).
    Return: DataFrame with columns [model_id, support_set, method, user_id, n_items, rho].
    """
    group_cols = ["model_id", "support_set", "method", "user_id"]
    groups = df.groupby(group_cols)
    rows = []
    for key, g in groups:
        model_id, support_set, method, user_id = key
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
        rows.append(
            {
                "model_id": model_id,
                "support_set": support_set,
                "method": method,
                "user_id": user_id,
                "n_items": len(y_true),
                "rho": rho,
            }
        )
    return pd.DataFrame(rows)


def plot_baseline_vs_method(
    x: np.ndarray,
    y: np.ndarray,
    baseline_name: str,
    method_name: str,
    model_id: str,
    support_set: str,
    out_path: str,
    figsize=(6, 6),
    dpi: int = 160,
):
    """
    x: per-user rho for baseline
    y: per-user rho for method
    """
    if x.size == 0:
        return

    mean_base = float(np.mean(x))
    mean_meth = float(np.mean(y))
    frac_improved = float(np.mean((y - x) > 0))

    plt.close("all")
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(x, y, s=10, alpha=0.6)

    # y=x line
    vmin = min(x.min(), y.min()) - 0.05
    vmax = max(x.max(), y.max()) + 0.05
    ax.plot([vmin, vmax], [vmin, vmax], linestyle="--", color="gray", linewidth=1.0)

    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_xlabel(f"{baseline_name} per-user rho")
    ax.set_ylabel(f"{method_name} per-user rho")

    ax.set_title(
        f"{model_id} | support_set={support_set}\n"
        f"{baseline_name} vs {method_name} (per-user rho)\n"
        f"mean_rho(base)={mean_base:.3f}, mean_rho(method)={mean_meth:.3f}, "
        f"frac_improved={frac_improved:.2f}"
    )
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] saved {out_path}")


def plot_baseline_vs_delta(
    x: np.ndarray,
    delta: np.ndarray,
    baseline_name: str,
    method_name: str,
    model_id: str,
    support_set: str,
    out_path: str,
    figsize=(6, 5),
    dpi: int = 160,
):
    """
    x: per-user rho for baseline
    delta: per-user (rho_method - rho_baseline)
    """
    if x.size == 0:
        return

    mean_delta = float(np.mean(delta))
    frac_improved = float(np.mean(delta > 0))

    plt.close("all")
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(x, delta, s=10, alpha=0.6)

    # y=0 line
    xmin, xmax = x.min() - 0.05, x.max() + 0.05
    ax.axhline(0.0, linestyle="--", color="gray", linewidth=1.0)

    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(f"{baseline_name} per-user rho")
    ax.set_ylabel(f"{method_name} delta rho (method - base)")

    ax.set_title(
        f"{model_id} | support_set={support_set}\n"
        f"{baseline_name} vs {method_name} (per-user delta rho)\n"
        f"mean_delta={mean_delta:.3f}, frac_improved={frac_improved:.2f}"
    )
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] saved {out_path}")


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
        help="Directory to save per-user improvement scatter plots.",
    )
    ap.add_argument(
        "--min_items_per_user",
        type=int,
        default=5,
        help="Minimum number of items per user to compute rho.",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[info] loading baseline data...")
    df_all = load_baseline_from_dir(args.input_dir)
    print(f"[info] total rows = {len(df_all)}")

    # per-user rho for all (model_id, support_set, method)
    print("[info] computing per-user rho...")
    df_rho = compute_per_user_rho(df_all, min_items=args.min_items_per_user)
    print(f"[info] per-user metric rows = {len(df_rho)}")

    if df_rho.empty:
        print("[warn] no users with enough items; nothing to plot.")
        return

    # (model_id, support_set) ごとに pivotして 'raw' を baseline に比較
    grouped = df_rho.groupby(["model_id", "support_set"])

    for (model_id, support_set), g_ms in tqdm(grouped, desc="Model/Support combos"):
        print(
            f"\n[info] combo: model_id={model_id}, support_set={support_set}, "
            f"methods={g_ms['method'].unique().tolist()}"
        )

        pivot = g_ms.pivot_table(
            index="user_id",
            columns="method",
            values="rho",
            aggfunc="mean",
        )

        if "raw" not in pivot.columns:
            print("[info]   -> no 'raw' method found for this combo, skip.")
            continue

        base_name = "raw"
        base_rho = pivot[base_name].to_numpy(dtype=float)

        # compare_methods = all methods except 'raw'
        compare_methods = [m for m in pivot.columns if m != base_name]
        if not compare_methods:
            print("[info]   -> only 'raw' present, nothing to compare; skip.")
            continue

        for method in compare_methods:
            meth_rho = pivot[method].to_numpy(dtype=float)

            # 両方NaNでないユーザだけを対象にする
            mask = ~np.isnan(base_rho) & ~np.isnan(meth_rho)
            x = base_rho[mask]
            y = meth_rho[mask]
            if x.size == 0:
                print(f"[info]   method={method}: no overlapping users, skip.")
                continue

            delta = y - x

            method_dir = os.path.join(args.out_dir, sanitize(str(method)))
            os.makedirs(method_dir, exist_ok=True)

            # (a) raw rho vs method rho
            fname_rho = (
                f"{sanitize(str(model_id))}__{sanitize(str(support_set))}__"
                f"{sanitize(base_name)}__vs__{sanitize(str(method))}__rho_scatter.png"
            )
            out_rho = os.path.join(method_dir, fname_rho)
            plot_baseline_vs_method(
                x,
                y,
                baseline_name=base_name,
                method_name=str(method),
                model_id=str(model_id),
                support_set=str(support_set),
                out_path=out_rho,
            )

            # (b) raw rho vs delta rho
            fname_delta = (
                f"{sanitize(str(model_id))}__{sanitize(str(support_set))}__"
                f"{sanitize(base_name)}__vs__{sanitize(str(method))}__delta_rho_scatter.png"
            )
            out_delta = os.path.join(method_dir, fname_delta)
            plot_baseline_vs_delta(
                x,
                delta,
                baseline_name=base_name,
                method_name=str(method),
                model_id=str(model_id),
                support_set=str(support_set),
                out_path=out_delta,
            )

    print("[done] per-user improvement plots generated for all (model_id, support_set) combos.")
    

if __name__ == "__main__":
    main()