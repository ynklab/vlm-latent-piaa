#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot per-layer probing performance for each attribute from probing JSON files.

Input:
  - input_dir: directory containing probing result JSON files.
    Expected JSON format:
      {
        "config": {...},
        "attrs": {
          "<attr>": {
            "per_layer": [
              {"source": "vision"|"llm_text"|..., "layer": int, "train": {...}, "val": {...}, "test": {...}},
              ...
            ],
            "best": {...}
          },
          ...
        }
      }

Output:
  - For each JSON file, create one PNG figure in out_dir:
      <stem>__<split>__<metric>__vision_to_llm_text.png

Plot:
  - x-axis: concatenated layers:
      vision layers (sorted) then llm_text layers (sorted, excluding layer 0)
  - y-axis: chosen metric value (rho or r2)
  - one line per attribute
  - vertical divider marking switch from vision to llm_text
  - x tick labels use V_k / LT_k to make the transition obvious

Notes:
  - Excludes LT_0 as requested.
  - If a particular (source, layer, attr) is missing, it is plotted as a gap (NaN).
"""

import os
import re
import json
import argparse
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

ATTR_LABEL = {
    "score": "Overall Score"
}


def _infer_model_name(d: Dict[str, Any], fallback: str) -> str:
    cfg = d.get("config", {}) if isinstance(d.get("config"), dict) else {}
    for k in [
        "model_id",
        "qwen_model_id", "qwen3_model_id",
        "gemma_model_id", "gemma3_model_id",
        "dinov3_model_id", "dino_model_id",
        "backbone_model_id",
        "vlm_model_id",
    ]:
        v = cfg.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return fallback


def _collect_points(d: Dict[str, Any], split: str, metric: str) -> Dict[str, Dict[str, Dict[int, float]]]:
    """
    Return: attr -> source -> layer -> value
    """
    out: Dict[str, Dict[str, Dict[int, float]]] = {}
    attrs = d.get("attrs", {})
    if not isinstance(attrs, dict):
        return out

    for attr, aobj in attrs.items():
        if not isinstance(aobj, dict):
            continue
        per_layer = aobj.get("per_layer", [])
        if not isinstance(per_layer, list):
            continue
        for e in per_layer:
            if not isinstance(e, dict):
                continue
            src = e.get("source")
            layer = e.get("layer")
            sdict = e.get(split)
            if src is None or layer is None or not isinstance(sdict, dict):
                continue
            v = sdict.get(metric)
            if v is None:
                continue
            try:
                li = int(layer)
                fv = float(v)
            except Exception:
                continue
            out.setdefault(str(attr), {}).setdefault(str(src), {})[li] = fv

    return out


def _make_xticks(vision_layers: List[int], text_layers: List[int]) -> Tuple[List[int], List[str], int]:
    """
    Returns:
      x_positions: 0..(n-1)
      x_labels: ["V_0", ..., "LT_1", ...]
      boundary_index: x position where llm_text starts (for vertical line)
    """
    labels = []
    for li in vision_layers:
        labels.append(f"V_{li}")
    boundary = len(labels)  # first LT position
    for li in text_layers:
        labels.append(f"LT_{li}")
    positions = list(range(len(labels)))
    return positions, labels, boundary


def plot_one_json(json_path: str, out_dir: str, split: str, metric: str) -> None:
    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)

    stem = os.path.splitext(os.path.basename(json_path))[0]
    model_name = _infer_model_name(d, stem)

    points = _collect_points(d, split=split, metric=metric)
    if not points:
        print(f"[warn] no usable points in {json_path} (split={split}, metric={metric})")
        return

    # Determine layers for each source
    # We want "vision -> llm_text" only, in that order.
    # LT_0 must be excluded.
    vision_layers_set = set()
    text_layers_set = set()

    for attr, src_map in points.items():
        if "vision" in src_map:
            vision_layers_set.update(src_map["vision"].keys())
        if "llm_text" in src_map:
            for li in src_map["llm_text"].keys():
                if li != 0:  # exclude LT_0
                    text_layers_set.add(li)

    vision_layers = sorted(vision_layers_set)
    text_layers = sorted(text_layers_set)

    if not vision_layers and not text_layers:
        print(f"[warn] {json_path}: no vision/llm_text layers found after filtering (LT_0 removed)")
        return

    x_pos, x_labels, boundary = _make_xticks(vision_layers, text_layers)

    # ---- xticks: every 10 plus always show V_0 and LT_1 ----
    tick_every = 10
    tick_idx = set(i for i in range(len(x_labels)) if i % tick_every == 0)

    # Required: V_0 is index 0, and LT_1 is boundary + index_of_text_layer (=1)
    # LT_0 is already excluded from text_layers, so the first text layer is LT_{text_layers[0]}.
    # Here, "always show LT_1" means always showing layer 1 when it exists.
    tick_idx.add(0)  # V_0
    if 1 in text_layers:
        lt1_pos = boundary + text_layers.index(1)
        tick_idx.add(lt1_pos)
    else:
        # If LT_1 does not exist, fall back to the first LT layer
        if len(text_layers) > 0:
            tick_idx.add(boundary)

    tick_idx = sorted(tick_idx)

    # ---- plot ----
    plt.close("all")
    fig, ax = plt.subplots(figsize=(8, 5.0))   # fixed size

    # ---- color palette: avoid collisions even for >10 attrs ----
    # tab20 + tab20b + tab20c => 60 colors
    cmap_names = ["tab20", "tab20b", "tab20c"]
    color_pool = []
    for cn in cmap_names:
        cmap = plt.get_cmap(cn)
        color_pool.extend([cmap(i) for i in range(cmap.N)])

    # stable color assignment by attribute name
    attrs_sorted = sorted(points.keys())
    attr_to_color = {a: color_pool[i % len(color_pool)] for i, a in enumerate(attrs_sorted)}

    for attr in attrs_sorted:
        src_map = points[attr]
        color = attr_to_color[attr]

        # --- vision part ---
        vmap = src_map.get("vision", {})
        y_v = [vmap.get(li, np.nan) for li in vision_layers]
        x_v = list(range(len(vision_layers)))  # 0..V-1

        # --- llm_text part ---
        tmap = src_map.get("llm_text", {})
        y_t = [tmap.get(li, np.nan) for li in text_layers]
        x_t = list(range(boundary, boundary + len(text_layers)))  # boundary..end

        label = ATTR_LABEL.get(attr, attr)

        # same color for V and LT
        ax.plot(x_v, y_v, marker="o", linewidth=1.2, markersize=2.5, label=label, color=color)
        ax.plot(x_t, y_t, marker="o", linewidth=1.2, markersize=2.5, label=None, color=color)

    # ---- boundary line ----
    if vision_layers and text_layers:
        ax.axvline(boundary - 0.5, linestyle="--", linewidth=1.0)

    # ---- ticks / labels ----
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([x_labels[i] for i in tick_idx], rotation=60, ha="right", fontsize=8)

    ax.set_xlabel("Layer (V_* then LT_*; LT_0 excluded)")
    ax.set_ylabel("Spearman Correlation" if metric == "rho" else "R2 Score")
    # ax.set_title(model_name)
    ax.grid(True, linestyle="--", alpha=0.3)

    # ---- remove left/right whitespace ----
    ax.set_xlim(-0.5, len(x_labels) - 0.5)
    # ---- boundary line ----
    if vision_layers and text_layers:
        ax.axvline(len(vision_layers) - 0.5, linestyle="--", linewidth=1.0)

    # ---- ticks / labels ----
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([x_labels[i] for i in tick_idx], rotation=60, ha="right", fontsize=8)

    ax.set_xlabel("Source Layer")
    ax.set_ylabel("Spearman Correlation")
    # ax.set_title(model_name)
    ax.grid(True, linestyle="--", alpha=0.3)

    # ---- remove left/right whitespace ----
    # x range depends on whether we inserted a boundary NaN point
    max_x = (len(vision_layers) + 1 + len(text_layers) - 1) if (vision_layers and text_layers) else (len(x_labels) - 1)
    ax.set_xlim(-0.5, len(x_labels) - 0.5)
    ax.margins(x=0)

    # legend below the plot (inside figure area)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),   # place below the plot
        ncol=4,                        # number of legend columns in one row
        fontsize=12,
        frameon=False,
        handlelength=1.5,
        columnspacing=0.8,
    )

    # make room at bottom for legend
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.28)   # reserve bottom margin for the legend
    os.makedirs(out_dir, exist_ok=True)

    out_name = f"{stem}__{split}__{metric}__vision_to_llm_text.pdf"
    out_path = os.path.join(out_dir, out_name)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Directory containing probing JSON files.")
    ap.add_argument("--out_dir", required=True, help="Directory to save PNG plots.")
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--metric", choices=["rho", "r2"], default="rho")
    args = ap.parse_args()

    files = [
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.lower().endswith(".json")
    ]
    if not files:
        raise SystemExit(f"No .json files found in {args.input_dir}")

    for p in sorted(files):
        try:
            plot_one_json(p, args.out_dir, split=args.split, metric=args.metric)
        except Exception as e:
            print(f"[warn] failed to plot {p}: {e}")


if __name__ == "__main__":
    main()