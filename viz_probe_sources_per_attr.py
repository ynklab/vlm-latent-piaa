#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Per-attribute layerwise probing visualization with concatenated axes:
  Vision layers (V_*) then LLM layers (L_*).

Additionally overlay:
  - llm_text (L)
  - llm_visual (L_V)

Input JSON format (as provided):
{
  "config": {...},
  "attrs": {
    "<attr>": {
      "per_layer": [
        {
          "source": "vision" | "llm_text" | "llm_visual",
          "layer": int,
          "train": {"rho":..., "r2":...},
          "val":   {...},
          "test":  {...}
        },
        ...
      ]
    },
    ...
  }
}

Output:
  out_dir/<model_name>/<attr>__<split>__<metric>.png
"""

import os
import re
import json
import argparse
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


# ------------ config ------------

# Which sources to include
VISION_SOURCE = "vision"
LLM_SOURCES = ["llm_text",  "llm_visual"]

# Display names in legend
SOURCE_LABEL = {
    "vision": "V",
    "llm_text": "LT",
    "llm_visual": r"LV",
}

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

ATTR_LABEL = {
    "score": "Overall Score"
}

# ------------ helpers ------------

def sanitize(s: str) -> str:
    s = str(s)
    return re.sub(r"[^0-9A-Za-z._\\-]+", "_", s)

def infer_model_name(d: Dict[str, Any], fallback: str) -> str:
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

def collect_attr_points(d: Dict[str, Any], attr: str, split: str, metric: str) -> Dict[str, Dict[int, float]]:
    """
    Return source -> layer -> value for a single attribute.
    """
    out: Dict[str, Dict[int, float]] = {}
    attrs = d.get("attrs", {})
    if not isinstance(attrs, dict) or attr not in attrs:
        return out
    aobj = attrs[attr]
    if not isinstance(aobj, dict):
        return out
    per_layer = aobj.get("per_layer", [])
    if not isinstance(per_layer, list):
        return out

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
        out.setdefault(str(src), {})[li] = fv

    return out

def build_axis_layers(
    vision_layers: List[int],
    llm_layers: List[int],
) -> Tuple[List[int], List[str], int]:
    """
    Build concatenated axis:
      V_0..V_k then L_1..L_n (LT_0 excluded outside)

    Returns:
      x_pos: [0..N-1]
      x_labels: list of strings
      boundary: first L position (= len(vision_layers))
    """
    labels = [f"V_{li}" for li in vision_layers]
    boundary = len(labels)
    labels += [f"L_{li}" for li in llm_layers]
    x_pos = list(range(len(labels)))
    return x_pos, labels, boundary


# ------------ plotting ------------

def plot_attr(
    model_name: str,
    attr: str,
    src_to_layer: Dict[str, Dict[int, float]],
    split: str,
    metric: str,
    out_png: str,
):
    # Determine layer universe
    vision_layers = sorted(src_to_layer.get("vision", {}).keys())

    # L axis layers are based on llm_text layers (for naming L_1..)
    # Exclude LT_0
    llm_layers_set = set()
    for src in LLM_SOURCES:
        for li in src_to_layer.get(src, {}).keys():
            if li != 0:
                llm_layers_set.add(li)
    llm_layers = sorted(llm_layers_set)

    if not vision_layers and not llm_layers:
        print(f"[warn] skip {model_name} attr={attr}: no layers found")
        return

    x_pos, x_labels, boundary = build_axis_layers(vision_layers, llm_layers)

    # xticks: every 5 + always V_0 and L_1 (if exists)
    tick_idx = {i for i in range(len(x_labels)) if i % 10 == 0}
    tick_idx.add(0)
    if 1 in llm_layers:
        tick_idx.add(boundary + llm_layers.index(1))
    else:
        if len(llm_layers) > 0:
            tick_idx.add(boundary)
    tick_idx = sorted(tick_idx)

    # fixed width figure (does not depend on number of layers)
    plt.close("all")
    fig, ax = plt.subplots(figsize=(5.0, 5.0))

    # Plot V on V axis only
    if vision_layers:
        y_v = [src_to_layer.get("vision", {}).get(li, np.nan) for li in vision_layers]
        x_v = list(range(len(vision_layers)))
        ax.plot(
            x_v, y_v,
            color=SOURCE_COLOR["vision"],
            marker=SOURCE_MARKER["vision"],
            # linewidth=1.2,
            markersize=2.5,
            label=SOURCE_LABEL["vision"],
        )

    # Plot each L_* series on L axis only (and keep same color per series)
    for src in LLM_SOURCES:
        layer_map = src_to_layer.get(src, {})
        if not layer_map:
            continue
        # exclude LT_0
        y = [layer_map.get(li, np.nan) for li in llm_layers]
        x = list(range(boundary, boundary + len(llm_layers)))
        ax.plot(
            x, y,
            color=SOURCE_COLOR[src],
            marker=SOURCE_MARKER[src],
            # linewidth=1.2,
            markersize=2.5,
            label=SOURCE_LABEL[src],
        )

    # boundary line
    if vision_layers and llm_layers:
        ax.axvline(boundary - 0.5, linestyle="--", linewidth=1.0)

    # axes labels
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([x_labels[i] for i in tick_idx], rotation=60, ha="right", fontsize=12)

    ax.set_xlim(-0.5, len(x_labels) - 0.5)
    ax.set_ylim(0.3, 0.75)
    ax.margins(x=0)

    # ax.set_xlabel("Layer (V_* then L_*; LT_0 excluded)")
    ax.set_ylabel("Spearman Correlation" if metric == "rho" else "R2 Score", fontsize=18)

    title_attr = ATTR_LABEL.get(attr, attr)
    if model_name == "google/gemma-3-4b-it":
        ax.set_title("Gemma 3 4B", fontsize=18)
    elif model_name == "Qwen/Qwen3-VL-2B-Instruct":
        ax.set_title("Qwen3-VL 2B", fontsize=18)
    else:
        # ax.set_title(f"{model_name} | {title_attr}")
        ax.set_title(title_attr, fontsize=18)

    ax.grid(True, linestyle="--", alpha=0.3)

    # if title_attr == "Object":
    ax.legend(
        loc="lower right",
        fontsize=18,
        frameon=True,        # 枠内なのでON推奨
        framealpha=0.9,      # 少し透過（線が隠れすぎない）
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_png}")


# ------------ main ------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Directory containing probing JSON files.")
    ap.add_argument("--out_dir", required=True, help="Directory to save plots.")
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--metric", choices=["rho", "r2"], default="rho")
    ap.add_argument("--include_score", action="store_true", help="Include attribute 'score' too.")
    args = ap.parse_args()

    files = [
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.lower().endswith(".json")
    ]
    if not files:
        raise SystemExit(f"No json files found in {args.input_dir}")

    os.makedirs(args.out_dir, exist_ok=True)

    for jp in sorted(files):
        try:
            with open(jp, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception as e:
            print(f"[warn] failed to read {jp}: {e}")
            continue

        stem = os.path.splitext(os.path.basename(jp))[0]
        model_name = infer_model_name(d, stem)
        prompt_name = d["config"].get("prompt_mode") or "base"

        model_dir = os.path.join(args.out_dir, sanitize(model_name), sanitize(prompt_name))
        os.makedirs(model_dir, exist_ok=True)

        attrs = d.get("attrs", {})
        if not isinstance(attrs, dict) or not attrs:
            print(f"[warn] {jp}: no attrs")
            continue

        attr_names = sorted(attrs.keys())
        if not args.include_score:
            attr_names = [a for a in attr_names if str(a).lower() != "score"]

        for attr in attr_names:
            src_to_layer = collect_attr_points(d, attr=attr, split=args.split, metric=args.metric)
            out_png = os.path.join(
                model_dir,
                f"{sanitize(attr)}__{args.split}__{args.metric}.png"
            )
            plot_attr(model_name, attr, src_to_layer, args.split, args.metric, out_png)

    print("[done]")


if __name__ == "__main__":
    main()