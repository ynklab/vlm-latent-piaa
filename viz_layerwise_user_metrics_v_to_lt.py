#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Layerwise visualization of PIAA evaluation metrics (per-user rho / r2)
for grid methods, plotted on a concatenated axis: V -> LV -> LT.

Input:
  Directory containing CSVs with columns:
    user_id, image_path, model_id, support_set, method, giaa, piaa_pred, user_score

Method naming:
  residual_linear_<feature_source>_L<layer>
  direct_linear_<feature_source>_L<layer>
  residual_linear_giaa_gt_<feature_source>_L<layer>
  direct_linear_giaa_gt_<feature_source>_L<layer>

We compute per-user metric:
  rho_u = Spearman(user_score, piaa_pred) within each user
  r2_u  = R2(user_score, piaa_pred) within each user

Then for each method (i.e., each source+layer):
  mean_metric = mean_u(metric_u)

Finally, we plot mean_metric along concatenated axis:
  vision -> llm_visual -> llm_text (LT_0 excluded)

Output:
  out_dir/<model>__<family>__<support_set>__<metric>__V_LV_LT.png
"""

import os
import re
import argparse
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import r2_score


# ---------------- utils ----------------
# Colors fixed per source (so across attributes it's consistent)
SOURCE_COLOR = {
    "vision": "#1f77b4",          # blue
    "llm_text": "#d62728",        # red
    "llm_visual": "#ff7f0e",      # orange
}

# marker style per source (optional)
SOURCE_MARKER = {
    "vision": "o",
    "llm_text": "o",
    "llm_visual": "^",
}

def sanitize(s: str) -> str:
    if s is None or not isinstance(s, str) or s.strip() == "":
        return "unknown"
    return re.sub(r"[^0-9A-Za-z._\\-]+", "_", s)

def load_from_dir(input_dir: str) -> pd.DataFrame:
    required = {
        "user_id","image_path","model_id","support_set","method","giaa","piaa_pred","user_score"
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
            continue
        dfs.append(df)
    if not dfs:
        raise RuntimeError(f"No valid baseline CSVs found in directory: {input_dir}")
    return pd.concat(dfs, ignore_index=True)

def parse_method(method: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    m = re.match(r"^(residual_linear_giaa_gt|direct_linear_giaa_gt)_(.+)_L(\d+)$", method)
    if m:
        return m.group(1), m.group(2), int(m.group(3))
    m = re.match(r"^(residual_linear|direct_linear)_(.+)_L(\d+)$", method)
    if m:
        return m.group(1), m.group(2), int(m.group(3))
    return None, None, None

def source_group(feature_source: str) -> Optional[str]:
    if feature_source == "vision":
        return "V"
    if feature_source == "llm_visual":
        return "LV"
    if feature_source == "llm_text":
        return "LT"
    return None

def per_user_metric(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size < 2:
        return float("nan")

    if metric == "rho":
        r = spearmanr(y_true, y_pred).correlation
        if np.isnan(r):
            return float("nan")
        return float(r)

    # r2
    try:
        return float(r2_score(y_true, y_pred))
    except Exception:
        return float("nan")

def build_concat_axis(groups: List[str], group_to_layers: Dict[str, List[int]]) -> Tuple[List[int], List[str], Dict[str, int]]:
    labels = []
    group_start = {}
    for g in groups:
        group_start[g] = len(labels)
        for li in group_to_layers.get(g, []):
            labels.append(f"{g}_{li}")
    x_positions = list(range(len(labels)))
    return x_positions, labels, group_start


# ---------------- plotting core ----------------

def plot_one(
    df: pd.DataFrame,
    model_id: str,
    family: str,
    support_set: str,
    metric: str,
    out_path: str,
    tick_every: int = 5,
    fig_w: float = 10.0,
    fig_h: float = 10.0,
):
    """
    Plot mean per-user metric with two sections:
      [ Vision ] -> [ LLM ]
    LLM section contains two lines: LV and LT.
    """

    # -------- collect per (group, layer) mean metric --------
    rows = []
    for method, g_method in df.groupby("method"):
        fam, feat_src, layer = parse_method(method)
        if fam != family:
            continue
        grp = source_group(feat_src)  # V / LV / LT
        if grp is None:
            continue
        if grp == "LT" and layer == 0:
            continue  # exclude LT_0

        vals = []
        for _, g_user in g_method.groupby("user_id"):
            y_true = g_user["user_score"].to_numpy(dtype=float)
            y_pred = g_user["piaa_pred"].to_numpy(dtype=float)
            v = per_user_metric(y_true, y_pred, metric)
            if np.isfinite(v):
                vals.append(v)

        rows.append((grp, layer, np.mean(vals) if vals else np.nan))

    if not rows:
        return False

    # -------- organize by group --------
    group_to_layers: Dict[str, List[int]] = {}
    group_to_vals: Dict[Tuple[str, int], float] = {}

    for grp, layer, val in rows:
        group_to_layers.setdefault(grp, []).append(layer)
        group_to_vals[(grp, layer)] = val

    for grp in group_to_layers:
        group_to_layers[grp] = sorted(set(group_to_layers[grp]))

    # Vision layers
    v_layers = group_to_layers.get("V", [])
    # LLM layers = union of LV and LT
    l_layers = sorted(
        set(group_to_layers.get("LV", [])) |
        set(group_to_layers.get("LT", []))
    )

    if not v_layers and not l_layers:
        return False

    # -------- build x-axis --------
    x_labels = []
    x_v = []
    x_l = []

    # Vision section
    for i, li in enumerate(v_layers):
        x_v.append(i)
        x_labels.append(f"V_{li}")

    boundary = len(x_labels)

    # LLM section (shared axis)
    for i, li in enumerate(l_layers):
        x_l.append(boundary + i)
        x_labels.append(f"L_{li}")

    x_all = list(range(len(x_labels)))

    # -------- build y values --------
    y_v = [group_to_vals.get(("V", li), np.nan) for li in v_layers]
    y_lv = [group_to_vals.get(("LV", li), np.nan) for li in l_layers]
    y_lt = [group_to_vals.get(("LT", li), np.nan) for li in l_layers]

    # -------- ticks --------
    tick_idx = {i for i in range(len(x_labels)) if i % tick_every == 0}
    if v_layers:
        tick_idx.add(0)
    if l_layers and 1 in l_layers:
        tick_idx.add(boundary + l_layers.index(1))
    tick_idx = sorted(tick_idx)

    # -------- plot --------
    plt.close("all")
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Colors (fixed semantic meaning)
    ax.plot(
        x_v, y_v,
        marker=SOURCE_MARKER["vision"], linewidth=1.6, markersize=3,
        color=SOURCE_COLOR["vision"], label="V"
    )
    ax.plot(
        x_l, y_lv,
        marker=SOURCE_MARKER["llm_visual"], linewidth=1.6, markersize=3,
        color=SOURCE_COLOR["llm_visual"], label="LV"
    )
    ax.plot(
        x_l, y_lt,
        marker=SOURCE_MARKER["llm_text"], linewidth=1.6, markersize=3,
        color=SOURCE_COLOR["llm_text"], label="LT"
    )

    # Boundary
    if v_layers and l_layers:
        ax.axvline(boundary - 0.5, linestyle="--", linewidth=1.0)

    # Axes
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([x_labels[i] for i in tick_idx], rotation=60, ha="right", fontsize=8)
    ax.set_xlim(-0.5, len(x_labels) - 0.5)
    ax.margins(x=0)

    # ax.set_xlabel("Layer axis (Vision → LLM)")
    ax.set_ylabel("Spearman Correlation" if metric == "rho" else "R2 Score", fontsize=18)
    if model_id == "google/gemma-3-4b-it":
        display_model_id = "Gemma 3 4B"
    elif model_id == "Qwen/Qwen3-VL-2B-Instruct":
        display_model_id = "Qwen3-VL 2B"
    else:
        display_model_id = model_id
    
    if "para" in out_path:
        dataset_name = "PARA"
    else:
        dataset_name = "LAPIS"
    ax.set_title(f"{display_model_id} | {dataset_name}", fontsize=18)
    ax.grid(True, linestyle="--", alpha=0.3)

    ax.legend(
        loc="lower right",
        fontsize=18,
        frameon=True,
        framealpha=0.9,
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] saved: {out_path}")
    return True

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--metric", choices=["rho", "r2"], default="rho")
    ap.add_argument("--tick_every", type=int, default=10)
    ap.add_argument("--fig_w", type=float, default=5.0)
    ap.add_argument("--fig_h", type=float, default=4.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df_all = load_from_dir(args.input_dir)

    # parse method info
    fams, srcs, layers = [], [], []
    for m in df_all["method"].astype(str).tolist():
        fam, fs, li = parse_method(m)
        fams.append(fam)
        srcs.append(fs)
        layers.append(li)
    df_all["family"] = fams
    df_all["feature_source"] = srcs
    df_all["layer_idx"] = layers

    df_all = df_all.dropna(subset=["family", "feature_source", "layer_idx"]).copy()
    df_all = df_all[df_all["feature_source"].isin(["vision", "llm_visual", "llm_text"])].copy()
    if df_all.empty:
        print("[warn] no usable grid methods for sources {vision,llm_visual,llm_text}.")
        return

    for (model_id, family, support_set), g in df_all.groupby(["model_id", "family", "support_set"]):
        out_path = os.path.join(
            args.out_dir,
            f"{sanitize(model_id)}__{sanitize(family)}__{sanitize(str(support_set))}__{args.metric}__V_LV_LT.pdf"
        )
        plot_one(
            df=g,
            model_id=str(model_id),
            family=str(family),
            support_set=str(support_set),
            metric=args.metric,
            out_path=out_path,
            tick_every=args.tick_every,
            fig_w=args.fig_w,
            fig_h=args.fig_h,
        )

    print("[done]")


if __name__ == "__main__":
    main()