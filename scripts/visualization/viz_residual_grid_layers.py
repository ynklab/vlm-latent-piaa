#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualize layerwise performance for linear PIAA models (residual / direct) on PARA/LAPIS.

Input:
  - A directory containing CSVs produced by:
      - train_residual_linear_grid_piaa.py
      - train_direct_linear_grid_piaa.py
    Each CSV must have columns:

      user_id, image_path, model_id, support_set, method, giaa, piaa_pred, user_score

    method には例えば以下のような形式を想定:
      - residual_linear_<feature_source>_L<layer>
      - direct_linear_<feature_source>_L<layer>
      - residual_linear_giaa_gt_<feature_source>_L<layer>
      - direct_linear_giaa_gt_<feature_source>_L<layer>

Output:
  - For each (model_id, family, feature_source), this script creates line charts:

      <out_dir>/<sanitize(model_id)>__<family>__<feature_source>__rho_layers.png
      <out_dir>/<sanitize(model_id)>__<family>__<feature_source>__r2_layers.png

    where:
      - family: e.g. residual_linear, direct_linear, residual_linear_giaa_gt, ...
      - feature_source: e.g. llm_text, vision, bridge_visual
      - x-axis: layer index
      - y-axis: mean per-user metric (rho or r2)
      - one line per support_set (small / large / etc), if available.
"""

import os
import re
import math
import argparse
from typing import List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import r2_score


# ---------- Utils ----------

def sanitize(s: str) -> str:
    if s is None or not isinstance(s, str) or s.strip() == "":
        return "unknown"
    return re.sub(r"[^0-9A-Za-z._\\-]+", "_", s)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Spearman ρ と R² を計算。NaN を含むペアはドロップ。
    """
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return {"rho": 0.0, "r2": float("nan")}

    rho = spearmanr(y_true, y_pred).correlation
    if np.isnan(rho):
        rho = 0.0

    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = float("nan")

    return {"rho": float(rho), "r2": r2}


def load_from_dir(input_dir: str) -> pd.DataFrame:
    """
    指定ディレクトリ内の CSV をすべて読み込み，
    baseline 用カラムを持つものだけを結合して返す。
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
            print(f"[info] skip {path} (missing baseline columns)")
            continue

        print(f"[info] loaded CSV: {path} (rows={len(df)})")
        dfs.append(df)

    if not dfs:
        raise RuntimeError(f"No valid baseline CSVs found in directory: {input_dir}")

    df_all = pd.concat(dfs, ignore_index=True)
    return df_all


def compute_per_user_metrics(df_all: pd.DataFrame, min_items: int = 2) -> pd.DataFrame:
    """
    (model_id, support_set, method, user_id) ごとに per-user metrics を計算。
    min_items 未満のユーザはスキップ。
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


def parse_method(method: str):
    """
    method 文字列から (family, feature_source, layer_idx) を抽出する。

    family:
      - residual_linear
      - direct_linear
      - residual_linear_giaa_gt
      - direct_linear_giaa_gt
      など，接頭辞部分。

    想定形式:
      residual_linear_<source>_L<layer>
      direct_linear_<source>_L<layer>
      residual_linear_giaa_gt_<source>_L<layer>
      direct_linear_giaa_gt_<source>_L<layer>
    """
    # まず giaa_gt 付き
    m = re.match(r"^(residual_linear_giaa_gt|direct_linear_giaa_gt)_(.+)_L(\d+)$", method)
    if m:
        family = m.group(1)
        feature_source = m.group(2)
        layer_idx = int(m.group(3))
        return family, feature_source, layer_idx

    # 通常版
    m = re.match(r"^(residual_linear|direct_linear)_(.+)_L(\d+)$", method)
    if m:
        family = m.group(1)
        feature_source = m.group(2)
        layer_idx = int(m.group(3))
        return family, feature_source, layer_idx

    return None, None, None


# ---------- Plotting ----------

def plot_layerwise_for_model_source_family(
    df_summary: pd.DataFrame,
    model_id: str,
    family: str,
    feature_source: str,
    metric: str,
    out_path: str,
    figsize=(10, 6),
    dpi: int = 160,
) -> bool:
    """
    df_summary: summary rows with columns:
      model_id, family, support_set, feature_source, layer_idx, mean_rho, mean_r2, ...
    For given (model_id, family, feature_source), plots mean metric vs layer_idx,
    with one line per support_set (small / large / etc).
    """
    df_m = df_summary[
        (df_summary["model_id"] == model_id) &
        (df_summary["family"] == family) &
        (df_summary["feature_source"] == feature_source)
    ].copy()
    if df_m.empty:
        return False

    plt.close("all")
    fig, ax = plt.subplots(figsize=figsize)

    support_sets = sorted(df_m["support_set"].unique())
    cmap = plt.get_cmap("tab10")

    for i, sup in enumerate(support_sets):
        df_s = df_m[df_m["support_set"] == sup].copy()
        df_s = df_s.sort_values("layer_idx")
        x = df_s["layer_idx"].to_numpy()
        if metric == "rho":
            y = df_s["mean_rho"].to_numpy()
        else:
            y = df_s["mean_r2"].to_numpy()
        ax.plot(
            x,
            y,
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=f"{sup}",
            color=cmap(i / max(1, len(support_sets))),
        )

    ax.set_xlabel("Layer index")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{model_id} | {family} | {feature_source} | mean {metric}")
    ax.grid(True, linestyle="--", alpha=0.3)

    all_layers = sorted(df_m["layer_idx"].unique())
    ax.set_xticks(all_layers)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] saved: {out_path}")
    return True


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing PIAA baseline CSVs (from residual/direct linear grid scripts).",
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for layerwise line charts.",
    )
    ap.add_argument(
        "--min_items_per_user",
        type=int,
        default=2,
        help="Minimum number of items per user to compute metrics.",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Load CSVs & compute per-user metrics
    print(f"[info] loading baseline CSVs from dir={args.input_dir} ...")
    df_all = load_from_dir(args.input_dir)
    print(f"[info] total rows = {len(df_all)}")

    print("[info] computing per-user metrics...")
    df_user = compute_per_user_metrics(df_all, min_items=args.min_items_per_user)
    print(f"[info] per-user rows = {len(df_user)}")

    # 2) Summary: (model_id, support_set, method) -> family, feature_source, layer_idx, mean_rho, mean_r2
    grouped = df_user.groupby(["model_id", "support_set", "method"])
    rows = []
    for (model_id, support_set, method), g in grouped:
        family, feat_src, layer_idx = parse_method(method)
        if family is None:
            # raw / bias / MLP など grid 以外の手法はここでスキップ
            continue
        mean_rho = g["rho"].mean()
        mean_r2 = g["r2"].mean()
        rows.append(
            {
                "model_id": model_id,
                "family": family,
                "support_set": support_set,
                "method": method,
                "feature_source": feat_src,
                "layer_idx": layer_idx,
                "mean_rho": mean_rho,
                "mean_r2": mean_r2,
            }
        )

    df_summary = pd.DataFrame(rows)
    if df_summary.empty:
        print("[warn] no residual/direct linear grid methods found in df_user; nothing to plot.")
        return

    # 3) For each (model_id, family, feature_source), plot mean rho / r2 vs layer index
    for model_id in df_summary["model_id"].unique():
        df_m = df_summary[df_summary["model_id"] == model_id]
        for family in df_m["family"].unique():
            df_f = df_m[df_m["family"] == family]
            for feat_src in df_f["feature_source"].unique():
                # rho
                out_rho = os.path.join(
                    args.out_dir,
                    f"{sanitize(model_id)}__{sanitize(family)}__{feat_src}__rho_layers.png",
                )
                plot_layerwise_for_model_source_family(
                    df_summary=df_summary,
                    model_id=model_id,
                    family=family,
                    feature_source=feat_src,
                    metric="rho",
                    out_path=out_rho,
                )

                # r2
                out_r2 = os.path.join(
                    args.out_dir,
                    f"{sanitize(model_id)}__{sanitize(family)}__{feat_src}__r2_layers.png",
                )
                plot_layerwise_for_model_source_family(
                    df_summary=df_summary,
                    model_id=model_id,
                    family=family,
                    feature_source=feat_src,
                    metric="r2",
                    out_path=out_r2,
                )

    print("[done] layerwise visualization finished.")


if __name__ == "__main__":
    main()