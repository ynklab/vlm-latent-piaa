#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Augmentation sensitivity visualization (orig/gray/tps) per attribute,
with concatenated axis: Vision (V_*) -> LLM Text (LT_*), LT_0 excluded.

Expected JSON format (probe_image_aug_sensitivity.py output):
{
  "config": { "dataset": ..., "model_id": ..., "prompt_mode": ..., "sources": [...] },
  "sources": {
    "vision": {
      "modes": {
        "orig": { "n_layers": L, "attrs": { "<attr>": { "per_layer": [...] } } },
        "gray": { ... },
        "tps":  { ... }
      }
    },
    "llm_text": {
      "modes": { ... same structure ... }
    }
  }
}

Output:
  out_dir / <dataset> / <prompt_mode> / <model_id> /
    aug__V_to_LT__<attr>__<split>__<metric>.png

Plot per attribute:
  - 3 lines: orig / gray / tps
  - x-axis: V_0..V_{V-1}, then LT_1..LT_{T-1}
  - boundary shown with dashed vertical line
  - LT_0 excluded
"""

import os
import re
import json
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

SPLITS = ["train", "val", "test"]
METRICS = ["rho", "rmse", "r2"]
DEFAULT_MODES = ["orig", "gray", "tps"]


def sanitize(s: str) -> str:
    if s is None or not isinstance(s, str) or s.strip() == "":
        return "unknown"
    return re.sub(r"[^0-9A-Za-z._\\-]+", "_", s)


def load_result_files(inputs: List[str]) -> List[str]:
    files = []
    for p in inputs:
        if os.path.isdir(p):
            for name in os.listdir(p):
                if name.lower().endswith(".json"):
                    files.append(os.path.join(p, name))
        elif os.path.isfile(p) and p.lower().endswith(".json"):
            files.append(p)
    return sorted(files)


def collect_attr_names(results: Dict) -> List[str]:
    """
    Collect attribute names from (vision, orig) if available; otherwise from any source/mode.
    """
    sources = results.get("sources", {})
    # prefer vision->orig
    if "vision" in sources:
        modes = sources["vision"].get("modes", {})
        if "orig" in modes:
            attrs = list(modes["orig"].get("attrs", {}).keys())
            return sorted(attrs)

    # fallback: any source/mode
    for s_entry in sources.values():
        modes = s_entry.get("modes", {})
        for m_entry in modes.values():
            attrs = list(m_entry.get("attrs", {}).keys())
            if attrs:
                return sorted(attrs)
    return []


def collect_layerwise_single_source(
    source_entry: Dict,
    attr: str,
    modes: List[str],
    split: str,
    metric: str,
) -> Tuple[List[int], Dict[str, List[float]]]:
    """
    From a single source entry (vision or llm_text),
    collect per-layer metric values for each mode.

    Returns:
      layers: sorted layer indices
      mode_to_values: dict[mode] -> list aligned with layers
    """
    mode_to_layer_metrics: Dict[str, Dict[int, float]] = {}

    for mode in modes:
        m_entry = source_entry.get("modes", {}).get(mode)
        if not m_entry:
            continue
        attr_entry = m_entry.get("attrs", {}).get(attr)
        if not attr_entry:
            continue
        per_layer = attr_entry.get("per_layer", [])
        d = {}
        for it in per_layer:
            li = int(it.get("layer", 0))
            split_dict = it.get(split, {})
            val = split_dict.get(metric, None)
            d[li] = np.nan if val is None else float(val)
        if d:
            mode_to_layer_metrics[mode] = d

    if not mode_to_layer_metrics:
        return [], {}

    all_layers = sorted({li for d in mode_to_layer_metrics.values() for li in d.keys()})
    mode_to_values = {mode: [d.get(li, np.nan) for li in all_layers] for mode, d in mode_to_layer_metrics.items()}
    return all_layers, mode_to_values


def build_concat_axis(
    v_layers: List[int],
    t_layers: List[int],
) -> Tuple[List[int], List[str], int]:
    """
    Concatenate x-axis: V_* then LT_*.
    Returns:
      x_positions: 0..N-1
      x_labels: ["V_0", ..., "LT_1", ...]
      boundary: index where LT starts
    """
    labels = [f"V_{li}" for li in v_layers]
    boundary = len(labels)
    labels += [f"LT_{li}" for li in t_layers]
    return list(range(len(labels))), labels, boundary


def plot_v_to_lt_per_attr(
    dataset: str,
    prompt_mode: str,
    model_id: str,
    attr: str,
    split: str,
    metric: str,
    v_layers: List[int],
    t_layers: List[int],
    v_mode_to_vals: Dict[str, List[float]],
    t_mode_to_vals: Dict[str, List[float]],
    modes: List[str],
    out_path: str,
    fig_w: float,
    fig_h: float,
) -> None:
    """
    Plot one attribute figure with 3 lines (orig/gray/tps):
      y(mode) = concat(vision values, llm_text values)
    """
    x_pos, x_labels, boundary = build_concat_axis(v_layers, t_layers)

    # xticks: every 5 + must include V_0 and LT_1 (if exists)
    tick_idx = {i for i in range(len(x_labels)) if i % 10 == 0}
    tick_idx.add(0)
    if 1 in t_layers:
        tick_idx.add(boundary + t_layers.index(1))
    else:
        if len(t_layers) > 0:
            tick_idx.add(boundary)
    tick_idx = sorted(tick_idx)

    # fixed colors per mode
    mode_color = {
        "orig": "#1f77b4",
        "gray": "#ff7f0e",
        "tps":  "#2ca02c",
    }
    mode_marker = {"orig": "o", "gray": "s", "tps": "^"}

    plt.close("all")
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))  # fixed width

    for mode in modes:
        if mode not in v_mode_to_vals and mode not in t_mode_to_vals:
            continue

        y = []
        # vision part
        if mode in v_mode_to_vals:
            y += v_mode_to_vals[mode]
        else:
            y += [np.nan] * len(v_layers)

        # llm_text part (LT_0 excluded)  ※ここでは t_layers がそのまま
        if mode in t_mode_to_vals:
            y += t_mode_to_vals[mode]
        else:
            y += [np.nan] * len(t_layers)

        ax.plot(
            x_pos, y,
            marker=mode_marker.get(mode, "o"),
            markersize=3,
            linewidth=1.2,
            label=mode,
            color=mode_color.get(mode, None),
        )

    # boundary line
    if v_layers and t_layers:
        ax.axvline(boundary - 0.5, linestyle="--", linewidth=1.0)

    ax.set_xticks(tick_idx)
    ax.set_xticklabels([x_labels[i] for i in tick_idx], rotation=60, ha="right", fontsize=12)
    ax.set_xlim(-0.5, len(x_labels) - 0.5)
    ax.margins(x=0)

    # ax.set_xlabel("Layer (V_* then LT_*; LT_0 excluded)")
    if metric == "rho":
        ylab = "Spearman Correlation"
    elif metric == "r2":
        ylab = "R2 Score"
    else:
        ylab = metric.upper()
    ax.set_ylabel(ylab, fontsize=18)

    # ax.set_title(f"{model_id} | {attr}")
    ax.set_title(attr, fontsize=18)
    ax.grid(True, linestyle="--", alpha=0.3)
    if attr == 'Object':
        ax.legend(loc="lower right", fontsize=18, frameon=True, framealpha=0.9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="augmentation JSON files or dirs")
    ap.add_argument("--out_dir", default="viz_aug_attr_v_to_lt", help="output root directory")
    ap.add_argument("--split", default="test", choices=SPLITS)
    ap.add_argument("--metric", default="rho", choices=METRICS)
    ap.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    ap.add_argument("--fig_w", type=float, default=5.0, help="fixed figure width")
    ap.add_argument("--fig_h", type=float, default=5.0, help="fixed figure height")
    ap.add_argument("--only_attrs", nargs="*", default=None)
    args = ap.parse_args()

    files = load_result_files(args.inputs)
    if not files:
        print("[viz] No JSON files found.")
        return

    for fp in files:
        try:
            with open(fp, "r") as f:
                results = json.load(f)
        except Exception as e:
            print(f"[viz] skip (load error): {fp} -> {e}")
            continue

        cfg = results.get("config", {})
        dataset = cfg.get("dataset", "unknown_dataset")
        model_id = cfg.get("model_id") or os.path.splitext(os.path.basename(fp))[0]
        prompt_mode = cfg.get("prompt_mode", "unknown_prompt")

        dataset_dir = sanitize(dataset)
        prompt_dir = sanitize(prompt_mode)
        model_dir = sanitize(model_id)

        out_base = os.path.join(args.out_dir, dataset_dir, prompt_dir, model_dir)

        sources = results.get("sources", {})
        if "vision" not in sources or "llm_text" not in sources:
            print(f"[viz] {fp}: requires both sources 'vision' and 'llm_text', skip")
            continue

        # available modes intersection
        v_modes_avail = set(sources["vision"].get("modes", {}).keys())
        t_modes_avail = set(sources["llm_text"].get("modes", {}).keys())
        modes = [m for m in args.modes if (m in v_modes_avail and m in t_modes_avail)]
        if not modes:
            print(f"[viz] {fp}: no requested modes available in both vision & llm_text")
            continue

        # attribute list
        attr_names = collect_attr_names(results)
        if args.only_attrs is not None:
            attr_names = [a for a in attr_names if a in args.only_attrs]
        if not attr_names:
            print(f"[viz] {fp}: no attributes found")
            continue

        # For each attr: collect vision layers & llm_text layers, then plot modes
        for attr in attr_names:
            # vision
            v_layers, v_mode_to_vals = collect_layerwise_single_source(
                sources["vision"], attr, modes, args.split, args.metric
            )
            # llm_text
            t_layers, t_mode_to_vals = collect_layerwise_single_source(
                sources["llm_text"], attr, modes, args.split, args.metric
            )

            # exclude LT_0 if present
            if 0 in t_layers:
                idx0 = t_layers.index(0)
                t_layers = [li for li in t_layers if li != 0]
                # remove corresponding values
                for mode in list(t_mode_to_vals.keys()):
                    ys = t_mode_to_vals[mode]
                    if len(ys) == idx0 + 1 or len(ys) == len(t_layers) + 1:
                        # safest: rebuild dict by layer mapping
                        # rebuild from original per-layer dict
                        pass

            # safer LT_0 removal: re-collect with explicit filtering
            # (rebuild t_layers and t_mode_to_vals from per-layer dict)
            # We'll reconstruct by reading raw per-layer dict for llm_text.
            # If LT_0 doesn't exist, this is still correct.
            t_layers, t_mode_to_vals = collect_layerwise_single_source(
                sources["llm_text"], attr, modes, args.split, args.metric
            )
            t_layers = [li for li in t_layers if li != 0]
            # rebuild values aligned to filtered t_layers
            # by reconstructing per-mode mapping via dict:
            # easiest: build dict layer->val for each mode again
            t_mode_to_vals_filtered = {}
            for mode in modes:
                m_entry = sources["llm_text"].get("modes", {}).get(mode)
                if not m_entry:
                    continue
                attr_entry = m_entry.get("attrs", {}).get(attr)
                if not attr_entry:
                    continue
                per_layer = attr_entry.get("per_layer", [])
                dmap = {}
                for it in per_layer:
                    li = int(it.get("layer", 0))
                    if li == 0:
                        continue
                    split_dict = it.get(args.split, {})
                    val = split_dict.get(args.metric, None)
                    dmap[li] = np.nan if val is None else float(val)
                if dmap:
                    t_mode_to_vals_filtered[mode] = [dmap.get(li, np.nan) for li in t_layers]
            t_mode_to_vals = t_mode_to_vals_filtered

            # plot
            out_path = os.path.join(out_base, f"aug__V_to_LT__{sanitize(attr)}__{args.split}__{args.metric}.png")
            plot_v_to_lt_per_attr(
                dataset=dataset,
                prompt_mode=prompt_mode,
                model_id=model_id,
                attr=attr,
                split=args.split,
                metric=args.metric,
                v_layers=v_layers,
                t_layers=t_layers,
                v_mode_to_vals=v_mode_to_vals,
                t_mode_to_vals=t_mode_to_vals,
                modes=modes,
                out_path=out_path,
                fig_w=args.fig_w,
                fig_h=args.fig_h,
            )


if __name__ == "__main__":
    main()