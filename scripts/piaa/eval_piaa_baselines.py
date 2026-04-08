#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate PIAA baselines from CSV outputs of `piaa_from_giaa.py`.

Expected input CSV columns:
  user_id, image_path, model_id, support_set, method, giaa, piaa_pred, user_score

This script:

1. Loads all CSVs in `--input_dir` that match the schema above.
2. Computes per-user metrics for each `(model_id, support_set, method, user_id)`:
   - Spearman's rho
   - R^2 (`sklearn.metrics.r2_score`)
3. Aggregates per-user metrics by `(model_id, support_set, method)` and
   computes `mean_rho`, `std_rho`, `mean_r2`, `std_r2`, and `num_users`.
4. Saves:
   - per-user metrics -> `<out_dir>/per_user_metrics.csv`
   - summary metrics  -> `<out_dir>/summary_metrics.csv`
   - users with low R^2 -> `<out_dir>/bad_users_r2.csv`
5. Produces visualizations:
   - bar charts of mean_rho and mean_r2 by `(support_set, method)`
   - box plots of per-user rho / r2 (`showfliers=False`)
"""

import os
import re
import math
import argparse
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import r2_score


def sanitize(s: str) -> str:
    if s is None or not isinstance(s, str) or s.strip() == "":
        return "unknown"
    return re.sub(r"[^0-9A-Za-z._\\-]+", "_", s)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute Spearman's rho and R^2.
    Pairs containing NaN values are dropped.
    """
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return {"rho": 0.0, "r2": float("nan")}

    # Spearman
    rho = spearmanr(y_true, y_pred).correlation
    if np.isnan(rho):
        rho = 0.0

    # R²
    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = float("nan")

    return {"rho": float(rho), "r2": r2}


def load_from_dir(input_dir: str) -> pd.DataFrame:
    """
    Read all CSVs in the given directory and return the concatenation
    of only those files that contain the required columns.
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

    for name in os.listdir(input_dir):
        if not name.lower().endswith(".csv"):
            continue
        path = os.path.join(input_dir, name)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[warn] failed to read {path}: {e}, skip")
            continue

        if not required.issubset(df.columns):
            # Files such as per_user_metrics.csv or summary_metrics.csv are filtered out here.
            print(f"[info] skip {path} (missing required columns)")
            continue

        print(f"[info] loaded baseline CSV: {path} (rows={len(df)})")
        dfs.append(df)

    if not dfs:
        raise RuntimeError(f"No valid baseline CSVs found in directory: {input_dir}")

    df_all = pd.concat(dfs, ignore_index=True)
    return df_all


def compute_per_user_metrics(df_all: pd.DataFrame, min_items: int = 2) -> pd.DataFrame:
    """
    Compute per-user metrics for each `(model_id, support_set, method, user_id)`.
    Skip users with fewer than `min_items` examples.
    """
    groups = df_all.groupby(["model_id", "support_set", "method", "user_id"])

    rows = []
    for (model_id, support_set, method, user_id), g in groups:
        y_true = g["user_score"].to_numpy(dtype=np.float32)
        y_pred = g["piaa_pred"].to_numpy(dtype=np.float32)

        if len(y_true) < min_items:
            continue

        m = _metrics(y_true, y_pred)
        rows.append(
            {
                "model_id": model_id,
                "support_set": support_set,
                "method": method,
                "user_id": user_id,
                "n_items": len(y_true),
                "rho": m["rho"],
                "r2": m["r2"],
            }
        )

    df_user = pd.DataFrame(rows)
    return df_user


def compute_summary(df_user: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-user metrics by `(model_id, support_set, method)`.
    """
    if df_user.empty:
        return pd.DataFrame()

    grouped = df_user.groupby(["model_id", "support_set", "method"])
    rows = []
    for (model_id, support_set, method), g in grouped:
        num_users = len(g)
        mean_rho = g["rho"].mean()
        std_rho = g["rho"].std()
        mean_r2 = g["r2"].mean()
        std_r2 = g["r2"].std()

        rows.append(
            {
                "model_id": model_id,
                "support_set": support_set,
                "method": method,
                "num_users": num_users,
                "mean_rho": mean_rho,
                "std_rho": std_rho,
                "mean_r2": mean_r2,
                "std_r2": std_r2,
            }
        )

    df_summary = pd.DataFrame(rows)
    return df_summary


def plot_bar_for_model(
    df_summary: pd.DataFrame,
    out_dir: str,
    metric: str,
    err: str,
    figsize=(10, 5),
    dpi: int = 160,
):
    """
    Draw a bar chart of metric vs `(support_set, method)` for each model_id.
    One bar corresponds to the `(mean_metric +/- std_metric)` of one pair.

    - For raw, ignore support_set and use the label "raw".
    - For small/large, use the label "<method>_<small/large>".
    - Ordering: raw -> small -> large -> others.
    """
    if df_summary.empty:
        print("[viz] summary dataframe is empty, nothing to plot.")
        return

    os.makedirs(out_dir, exist_ok=True)

    for model_id in df_summary["model_id"].unique():
        df_m = df_summary[df_summary["model_id"] == model_id].copy()
        if df_m.empty:
            continue

        def canon(row):
            return "raw" if row["method"] == "raw" else row["support_set"]

        df_m["support_canon"] = df_m.apply(canon, axis=1)

        keys = ["model_id", "method", "support_canon"]
        agg = (
            df_m.groupby(keys)
            .agg(
                num_users=("num_users", "sum"),
                mean_metric=(metric, "mean"),
                std_metric=(err, "mean"),
            )
            .reset_index()
        )

        support_order = {"raw": 0, "small": 1, "large": 2}
        agg["support_rank"] = agg["support_canon"].map(
            lambda s: support_order.get(s, 99)
        )
        agg = agg.sort_values(by=["support_rank", "method"])

        labels = []
        for _, r in agg.iterrows():
            if r["support_canon"] == "raw":
                labels.append(str(r["method"]))
            else:
                labels.append(f"{r['method']}_{r['support_canon']}")

        vals = agg["mean_metric"].to_numpy()
        errs = agg["std_metric"].to_numpy()

        x = np.arange(len(labels))
        width = 0.7

        plt.close("all")
        fig, ax = plt.subplots(figsize=figsize)
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i / max(1, len(labels))) for i in range(len(labels))]
        ax.bar(x, vals, width, yerr=errs, capsize=4, color=colors)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(f"{model_id} | {metric}")
        ax.grid(True, linestyle="--", alpha=0.3)

        plt.tight_layout()
        fname = f"{sanitize(model_id)}__{metric}.png"
        out_path = os.path.join(out_dir, fname)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[viz] saved: {out_path}")


def plot_box_for_model(
    df_user: pd.DataFrame,
    out_dir: str,
    metric: str = "rho",
    figsize=(10, 5),
    dpi: int = 160,
):
    """
    Use per-user metrics (`df_user`) to visualize the distribution of each
    `(support_set, method)` pair as a box plot for every model_id.

    Ordering:
      raw -> small -> large, and within each group sort by method name.
      Labels follow the same convention as the bar chart:
        - raw: method only
        - otherwise: "<method>_<support_canon>"

    Since `showfliers=False`, extreme outliers are hidden to make the main
    distribution easier to read.
    """
    if df_user.empty:
        print("[viz-box] per-user dataframe is empty, nothing to plot.")
        return

    os.makedirs(out_dir, exist_ok=True)

    for model_id in df_user["model_id"].unique():
        df_m = df_user[df_user["model_id"] == model_id].copy()
        if df_m.empty:
            continue

        def canon(row):
            return "raw" if row["method"] == "raw" else row["support_set"]

        df_m["support_canon"] = df_m.apply(canon, axis=1)

        groups = df_m.groupby(["support_canon", "method"])

        support_order = {"raw": 0, "small": 1, "large": 2}
        sorted_keys = sorted(
            groups.groups.keys(),
            key=lambda k: (support_order.get(k[0], 99), k[1]),
        )

        data_list = []
        labels = []
        for (support_canon, method) in sorted_keys:
            g = groups.get_group((support_canon, method))
            vals = g[metric].to_numpy(dtype=float)
            if vals.size == 0:
                continue
            data_list.append(vals)
            if support_canon == "raw":
                labels.append(str(method))
            else:
                labels.append(f"{method}_{support_canon}")

        if not data_list:
            continue

        plt.close("all")
        fig, ax = plt.subplots(figsize=figsize)

        bp = ax.boxplot(
            data_list,
            labels=labels,
            patch_artist=True,
            notch=False,
            showfliers=False,  # Hide outliers.
        )

        cmap = plt.get_cmap("tab20")
        for i, box in enumerate(bp["boxes"]):
            box.set_facecolor(cmap(i / max(1, len(data_list))))

        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(f"{model_id} | per-user {metric} distribution")
        ax.grid(True, linestyle="--", alpha=0.3)

        plt.tight_layout()
        fname = f"{sanitize(model_id)}__{metric}_boxplot.png"
        out_path = os.path.join(out_dir, fname)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[viz-box] saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_dir",
        required=True,
        help=(
            "Directory containing PIAA baseline CSVs (from piaa_from_giaa_para.py). "
            "All CSVs in this directory that contain the required columns will be used."
        ),
    )
    ap.add_argument(
        "--out_dir",
        default="viz_piaa_baselines",
        help="Output directory for metrics and plots.",
    )
    ap.add_argument(
        "--min_items_per_user",
        type=int,
        default=2,
        help="Minimum number of items per user to compute metrics.",
    )
    ap.add_argument(
        "--bad_r2_threshold",
        type=float,
        default=-10.0,
        help="Threshold for flagging 'bad' users by R^2 (e.g. -10.0).",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Load all baseline CSVs from directory
    print(f"[info] loading CSVs from dir={args.input_dir} ...")
    df_all = load_from_dir(args.input_dir)
    print(f"[info] total rows = {len(df_all)}")

    # 2) Per-user metrics
    print("[info] computing per-user metrics...")
    df_user = compute_per_user_metrics(df_all, min_items=args.min_items_per_user)
    per_user_path = os.path.join(args.out_dir, "per_user_metrics.csv")
    df_user.to_csv(per_user_path, index=False)
    print(f"[save] per-user metrics -> {per_user_path} (rows={len(df_user)})")

    # 2.5) Save bad users (R^2 < threshold).
    bad_mask = df_user["r2"] < args.bad_r2_threshold
    df_bad = df_user[bad_mask].copy()
    bad_path = os.path.join(args.out_dir, "bad_users_r2.csv")
    df_bad.to_csv(bad_path, index=False)
    print(
        f"[save] bad users (r2 < {args.bad_r2_threshold}) -> {bad_path} "
        f"(rows={len(df_bad)})"
    )

    # 3) Summary metrics
    print("[info] computing summary metrics...")
    df_summary = compute_summary(df_user)
    summary_path = os.path.join(args.out_dir, "summary_metrics.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"[save] summary metrics -> {summary_path} (rows={len(df_summary)})")

    # 4) Visualization: mean_rho
    print("[viz] plotting mean_rho...")
    plot_bar_for_model(
        df_summary=df_summary,
        out_dir=args.out_dir,
        metric="mean_rho",
        err="std_rho",
    )

    # 5) Visualization: mean_r2
    print("[viz] plotting mean_r2...")
    plot_bar_for_model(
        df_summary=df_summary,
        out_dir=args.out_dir,
        metric="mean_r2",
        err="std_r2",
    )

    # 6) per-user boxplots
    print("[viz-box] plotting per-user rho boxplots...")
    plot_box_for_model(
        df_user=df_user,
        out_dir=args.out_dir,
        metric="rho",
    )

    print("[viz-box] plotting per-user r2 boxplots...")
    plot_box_for_model(
        df_user=df_user,
        out_dir=args.out_dir,
        metric="r2",
    )


if __name__ == "__main__":
    main()
