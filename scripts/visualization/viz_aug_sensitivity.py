#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualize augmentation sensitivity results produced by probe_image_aug_sensitivity.py.

Input (JSON):
  {
    "config": {
      "model_id": "...",
      "dataset": "aadb" or "para",
      "prompt_mode": "base" | "format" | "attributes" | "unrelated",
      "sources": ["llm_text", "vision"],
      ...
    },
    "sources": {
      "llm_text": {
        "modes": {
          "orig": {
            "n_layers": ...,
            "attrs": {
              "<attr_name>": {
                "per_layer": [...],
                "best": {
                  "layer": ...,
                  "train": {...},
                  "val":   {...},
                  "test":  {...}
                }
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
    <source>__<split>__<metric>.png

Each figure:
  - x-axis: attributes
  - y-axis: metric (rho/rmse/r2)
  - compare orig/gray/tps side by side for each attribute
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
    return re.sub(r"[^0-9A-Za-z._\-]+", "_", s)


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
    # Prefer attrs from orig; otherwise use the first mode
    mode_keys = list(modes.keys())
    if not mode_keys:
        return []
    if "orig" in modes:
        m = modes["orig"]
    else:
        m = modes[mode_keys[0]]
    attrs = list(m.get("attrs", {}).keys())
    return sorted(attrs)


def collect_best_metric_per_attr(
    source_entry: Dict,
    modes: List[str],
    split: str,
    metric: str,
) -> Dict[str, Dict[str, float]]:
    """
    Returns:
      attr -> {mode -> best[split][metric]}
    """
    out: Dict[str, Dict[str, float]] = {}
    attr_names = collect_attr_names(source_entry)
    for attr in attr_names:
        out[attr] = {}
        for mode in modes:
            mode_entry = source_entry.get("modes", {}).get(mode)
            if not mode_entry:
                out[attr][mode] = np.nan
                continue
            attr_entry = mode_entry.get("attrs", {}).get(attr)
            if not attr_entry:
                out[attr][mode] = np.nan
                continue
            best = attr_entry.get("best", {})
            split_dict = best.get(split, {})
            val = split_dict.get(metric, None)
            out[attr][mode] = np.nan if val is None else float(val)
    return out


def plot_bar_for_source(
    model_id: str,
    dataset: str,
    prompt_mode: str,
    source: str,
    attr_to_mode_values: Dict[str, Dict[str, float]],
    modes: List[str],
    split: str,
    metric: str,
    out_path: str,
    figsize=(12, 6),
    dpi=160,
):
    """
    attr_to_mode_values: attr -> {mode -> value}
    modes: e.g. ["orig", "gray", "tps"]
    """
    attrs = list(attr_to_mode_values.keys())
    if not attrs:
        return False

    # Build the data matrix: rows=attrs, cols=modes
    data = np.array([
        [attr_to_mode_values[a].get(m, np.nan) for m in modes]
        for a in attrs
    ], dtype=float)

    n_attr = len(attrs)
    n_modes = len(modes)
    x = np.arange(n_attr)
    width = 0.7 / max(1, n_modes)  # Adjust bar width per attribute

    plt.close("all")
    fig, ax = plt.subplots(figsize=figsize)

    for j, mode in enumerate(modes):
        xs = x + (j - (n_modes - 1) / 2) * width
        vals = data[:, j]
        ax.bar(xs, vals, width, label=mode)

    ax.set_xticks(x)
    ax.set_xticklabels(attrs, rotation=45, ha="right")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{dataset.upper()} | {model_id} | source={source} | split={split} | metric={metric}")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(title="image_mode")

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
        "--out_dir", default="viz_aug",
        help="output root directory",
    )
    ap.add_argument(
        "--split", default="test", choices=SPLITS,
        help="which split's best metrics to visualize (train/val/test)",
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
        "--fig_w", type=float, default=12.0,
        help="figure width",
    )
    ap.add_argument(
        "--fig_h", type=float, default=6.0,
        help="figure height",
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

            # Filter to available modes (subset of orig/gray/tps)
            available_modes = list(source_entry.get("modes", {}).keys())
            modes = [m for m in args.modes if m in available_modes]
            if not modes:
                print(f"[viz] no modes {args.modes} available for source={source} in {fp}, skip")
                continue

            # Aggregate mode -> metric for each attribute
            attr_to_mode_values = collect_best_metric_per_attr(
                source_entry, modes=modes, split=args.split, metric=args.metric
            )

            out_path = os.path.join(
                args.out_dir,
                dataset_dir,
                prompt_dir,
                model_dir,
                f"{source}__{args.split}__{args.metric}.png",
            )

            plot_bar_for_source(
                model_id=model_id,
                dataset=dataset,
                prompt_mode=prompt_mode,
                source=source,
                attr_to_mode_values=attr_to_mode_values,
                modes=modes,
                split=args.split,
                metric=args.metric,
                out_path=out_path,
                figsize=(args.fig_w, args.fig_h),
            )


if __name__ == "__main__":
    main()