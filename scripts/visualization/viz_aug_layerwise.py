#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Layerwise visualization for augmentation sensitivity.

Target:
  Output JSON from probe_image_aug_sensitivity.py:
  {
    "config": { ... },
    "sources": {
      "llm_text": {
        "modes": {
          "orig": {
            "n_layers": L,
            "attrs": {
              "<attr>": {
                "per_layer": [
                  { "layer": 0, "train": {...}, "val": {...}, "test": {...} },
                  ...
                ],
                "best": {...}
              },
              ...
            }
          },
          "gray": { ... },
          "tps":  { ... }
        }
      },
      "vision": { ... }
    }
  }

Output:
  out_dir / <dataset> / <prompt_mode> / <model_id> /
    <source>__<attr>__<split>__<metric>.png

Each figure:
  - x-axis: layer index
  - y-axis: metric (rho/rmse/r2)
  - lines: layerwise curves for each image_mode (orig / gray / tps)
"""

import os
import re
import json
import argparse
from typing import Dict, List

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


def collect_attr_names(source_entry: Dict) -> List[str]:
    """
    source_entry: results["sources"][source]
    """
    modes = source_entry.get("modes", {})
    if not modes:
        return []
    # Prefer orig; otherwise use the first available mode
    if "orig" in modes:
        m = modes["orig"]
    else:
        first_mode = list(modes.keys())[0]
        m = modes[first_mode]
    attrs = list(m.get("attrs", {}).keys())
    return sorted(attrs)


def collect_layerwise(
    source_entry: Dict,
    attr: str,
    modes: List[str],
    split: str,
    metric: str,
):
    """
    Get the layerwise metrics for one attribute.
    Returns:
      layers: sorted layer indices
      values: dict[mode] -> list[metric values] (len=layers)
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

    # Take the union of layer indices across all modes
    all_layers = sorted({li for d in mode_to_layer_metrics.values() for li in d.keys()})
    # Arrange each mode's values in this order
    mode_to_values = {}
    for mode, d in mode_to_layer_metrics.items():
        mode_to_values[mode] = [d.get(li, np.nan) for li in all_layers]

    return all_layers, mode_to_values


def plot_layerwise_for_attr(
    model_id: str,
    dataset: str,
    prompt_mode: str,
    source: str,
    attr: str,
    layers: List[int],
    mode_to_values: Dict[str, List[float]],
    split: str,
    metric: str,
    out_path: str,
    figsize=(10, 6),
    dpi=160,
):
    if not layers or not mode_to_values:
        return False

    plt.close("all")
    fig, ax = plt.subplots(figsize=figsize)

    # Use matplotlib's default color cycle
    for mode, ys in mode_to_values.items():
        ax.plot(layers, ys, marker="o", markersize=3, linewidth=1.0, label=mode)

    ax.set_xlabel("Layer index")
    ax.set_ylabel(metric.upper())
    ax.set_title(
        f"{dataset.upper()} | {model_id}\nsource={source}, attr={attr}, split={split}, metric={metric}"
    )
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(title="image_mode")
    ax.set_xticks(layers)

    # Add a small margin to the y-axis range
    all_vals = np.array([v for ys in mode_to_values.values() for v in ys if not np.isnan(v)], dtype=float)
    if all_vals.size > 0:
        ymin, ymax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
        if np.isfinite(ymin) and np.isfinite(ymax):
            if ymin == ymax:
                pad = 0.05 * (abs(ymin) + 1.0)
                ax.set_ylim(ymin - pad, ymax + pad)
            else:
                pad = 0.05 * (ymax - ymin)
                ax.set_ylim(ymin - pad, ymax + pad)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] saved: {out_path}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs", nargs="+", required=True,
        help="augmentation sensitivity JSON files (from probe_image_aug_sensitivity.py)",
    )
    ap.add_argument(
        "--out_dir", default="viz_aug_layers",
        help="output root directory",
    )
    ap.add_argument(
        "--split", default="test", choices=SPLITS,
        help="which split's metrics to visualize (train/val/test)",
    )
    ap.add_argument(
        "--metric", default="rho", choices=METRICS,
        help="which metric to visualize (rho/rmse/r2)",
    )
    ap.add_argument(
        "--modes", nargs="+", default=DEFAULT_MODES,
        help="image modes to include, e.g., orig gray tps",
    )
    ap.add_argument(
        "--fig_w", type=float, default=10.0,
        help="figure width",
    )
    ap.add_argument(
        "--fig_h", type=float, default=6.0,
        help="figure height",
    )
    ap.add_argument(
        "--only_attrs", nargs="*", default=None,
        help="if specified, only visualize these attributes (by name).",
    )
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
        dataset     = cfg.get("dataset", "unknown_dataset")
        model_id    = cfg.get("model_id") or os.path.splitext(os.path.basename(fp))[0]
        prompt_mode = cfg.get("prompt_mode", "unknown_prompt")
        sources     = cfg.get("sources") or list(results.get("sources", {}).keys())

        dataset_dir = sanitize(dataset)
        prompt_dir  = sanitize(prompt_mode)
        model_dir   = sanitize(model_id)

        for source in sources:
            source_entry = results.get("sources", {}).get(source)
            if not source_entry:
                print(f"[viz] no source={source} in {fp}, skip")
                continue

            # Get the attribute list (filtered by only_attrs if provided)
            attr_names = collect_attr_names(source_entry)
            if args.only_attrs is not None:
                attr_names = [a for a in attr_names if a in args.only_attrs]
            if not attr_names:
                print(f"[viz] no attributes to visualize for source={source} in {fp}, skip")
                continue

            # Filter available modes
            available_modes = list(source_entry.get("modes", {}).keys())
            modes = [m for m in args.modes if m in available_modes]
            if not modes:
                print(f"[viz] no modes {args.modes} available for source={source} in {fp}, skip")
                continue

            for attr in attr_names:
                layers, mode_to_values = collect_layerwise(
                    source_entry,
                    attr=attr,
                    modes=modes,
                    split=args.split,
                    metric=args.metric,
                )
                out_path = os.path.join(
                    args.out_dir,
                    dataset_dir,
                    prompt_dir,
                    model_dir,
                    f"{source}__{attr}__{args.split}__{args.metric}.png",
                )
                plot_layerwise_for_attr(
                    model_id=model_id,
                    dataset=dataset,
                    prompt_mode=prompt_mode,
                    source=source,
                    attr=attr,
                    layers=layers,
                    mode_to_values=mode_to_values,
                    split=args.split,
                    metric=args.metric,
                    out_path=out_path,
                    figsize=(args.fig_w, args.fig_h),
                )


if __name__ == "__main__":
    main()