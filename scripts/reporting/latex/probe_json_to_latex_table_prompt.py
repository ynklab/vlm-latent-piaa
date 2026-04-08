#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build ONE LaTeX table:
  rows = attributes
  cols = prompt_mode
  cell = best source-layer + metric value on the chosen split

Assumptions:
  - All JSONs correspond to the SAME model (as you stated).
  - prompt_mode exists under json["config"]["prompt_mode"].
  - Probing JSON format:
      {
        "config": {..., "prompt_mode": "..."},
        "attrs": {
          "<attr>": {
            "per_layer": [
              {"source": "...", "layer": int, "train": {...}, "val": {...}, "test": {...}},
              ...
            ]
          }
        }
      }

Features:
  - source symbols: V / LV / LT / L\\tau (optional if present)
  - Bold: for each attribute row, the best metric among prompt columns is bolded
  - table* + tabularx (fits \\textwidth)
    Requires LaTeX packages: booktabs, tabularx, array
"""

import os
import re
import json
import math
import argparse
from typing import Dict, Any, Tuple, Optional, List

# ---------- source symbols ----------
SOURCE_SYMBOL = {
    "vision": "V",
    "llm_visual": "LV",
    "llm_text": "LT",
    "llm_text_tail": r"L\tau",
}

PROMPT_MODES = [
    ("base", "Base"),
    ("format", "Numeric Format"),
    ("attributes", "Attribute List"),
    ("unrelated", "Unrelated")
]


def fmt_src_layer(src: str, layer: int, bold: bool) -> str:
    sym = SOURCE_SYMBOL.get(src, src)
    if bold:
        return f"$\\mathbf{{{sym}}}_{{\\mathbf{{{layer}}}}}$"
    return f"${sym}_{{{layer}}}$"

def tex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
         .replace("_", "\\_")
         .replace("%", "\\%")
         .replace("&", "\\&")
         .replace("#", "\\#")
         .replace("{", "\\{")
         .replace("}", "\\}")
         .replace("$", "\\$")
         .replace("^", "\\^{}")
         .replace("~", "\\~{}")
    )

def safe_float(x: Any) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return float("nan")
        return v
    except Exception:
        return float("nan")

def infer_model_id(d: Dict[str, Any], fallback: str) -> str:
    cfg = d.get("config", {}) if isinstance(d.get("config"), dict) else {}
    for k in [
        "model_id",
        "backbone_model_id",
        "dinov3_model_id",
        "dino_model_id",
        "qwen_model_id",
        "qwen3_model_id",
        "gemma_model_id",
        "gemma3_model_id",
        "vlm_model_id",
    ]:
        v = cfg.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return fallback

def infer_prompt_mode(d: Dict[str, Any]) -> str:
    cfg = d.get("config", {}) if isinstance(d.get("config"), dict) else {}
    pm = cfg.get("prompt_mode")
    if isinstance(pm, str) and pm.strip():
        return pm.strip()
    return "base"

def parse_best_for_attr(
    attr_obj: Dict[str, Any],
    split: str,
    metric: str,
    source_filter: Optional[str] = None,
    skip_sources: Optional[set] = None,
) -> Optional[Tuple[str, int, float]]:
    """
    Return (source, layer, value) maximizing split[metric] over per_layer entries.
    """
    per_layer = attr_obj.get("per_layer")
    if not isinstance(per_layer, list) or not per_layer:
        return None

    best = None  # (value, source, layer)
    for entry in per_layer:
        if not isinstance(entry, dict):
            continue
        src = entry.get("source")
        if skip_sources and src in skip_sources:
            continue

        layer = entry.get("layer")
        if source_filter is not None and str(src) != source_filter:
            continue
        if src is None or layer is None:
            continue

        split_obj = entry.get(split)
        if not isinstance(split_obj, dict):
            continue
        v = safe_float(split_obj.get(metric))
        if math.isnan(v):
            continue

        try:
            li = int(layer)
        except Exception:
            continue

        if best is None or v > best[0]:
            best = (v, str(src), li)

    if best is None:
        return None
    v, src, li = best
    return (src, li, v)

def render_table_star_tabularx(
    attributes: List[str],
    prompt_modes: List[str],
    cell: Dict[Tuple[str, str], str],   # (attr, prompt) -> cell_str
    caption: str,
    label: str,
) -> str:
    # first column = attribute (left), others = X columns
    col_spec = "l " + f"*{{{len(PROMPT_MODES)}}}{{>{{\\centering\\arraybackslash}}X}}"

    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.1}")
    lines.append(f"\\begin{{tabularx}}{{\\textwidth}}{{{col_spec}}}")
    lines.append("\\toprule")

    # header rows:
    # Row 1: prompt modes
    header = ["Attribute"] + [tex_escape(pm) for (_, pm) in PROMPT_MODES]
    lines.append(" & ".join(header) + " \\\\")
    # Row 2: indicator of what numbers mean
    lines.append(" & ".join([""] + [r"$\rho$ / $R^2$" for _ in PROMPT_MODES]) + " \\\\")
    lines.append("\\midrule")

    for a in attributes:
        row = [tex_escape(a)]
        for (pm, _) in PROMPT_MODES:
            row.append(cell.get((a, pm), "--"))
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabularx}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table*}")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_tex", required=True)
    ap.add_argument("--metric", choices=["rho", "r2"], default="rho",
                    help="Metric used to select best source-layer on the chosen split.")
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--source", default=None,
                    help="If set, restrict selection to this source only (e.g., vision, llm_text).")
    ap.add_argument("--include_score", action="store_true")
    ap.add_argument("--caption", default=None)
    ap.add_argument("--label", default="tab:probing_best_by_prompt")
    ap.add_argument("--skip_sources", nargs="*", default=[],
                    help="Sources to ignore when selecting best (e.g., llm_text_tail llm_visual).")
    args = ap.parse_args()

    files = [
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.lower().endswith(".json")
    ]
    if not files:
        raise SystemExit(f"No .json files found in {args.input_dir}")

    # skip_sources = set(args.skip_sources)
    skip_sources = ["llm_text_tail", "llm_visual", "vision"]

    # prompt_mode -> attr -> (src, layer, val)
    pm_to_best: Dict[str, Dict[str, Tuple[str, int, float]]] = {}
    model_ids = set()

    for path in sorted(files):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[warn] failed to parse {path}: {e}")
            continue

        stem = os.path.splitext(os.path.basename(path))[0]
        model_id = infer_model_id(data, stem)
        model_ids.add(model_id)

        pm = infer_prompt_mode(data)

        attrs = data.get("attrs")
        if not isinstance(attrs, dict) or not attrs:
            print(f"[warn] {path}: no 'attrs' dict, skip")
            continue

        best_map: Dict[str, Tuple[str, int, float]] = {}
        for attr_name, attr_obj in attrs.items():
            if not args.include_score and str(attr_name).lower() == "score":
                continue
            if not isinstance(attr_obj, dict):
                continue
            best = parse_best_for_attr(
                attr_obj,
                split=args.split,
                metric=args.metric,
                source_filter=args.source,
                skip_sources=skip_sources if skip_sources else None,
            )
            if best is None:
                continue
            src, layer, val = best
            best_map[str(attr_name)] = (src, layer, val)

        if not best_map:
            print(f"[warn] {path}: no usable per_layer entries for split={args.split}, metric={args.metric}")
            continue

        pm_to_best[pm] = best_map

    if not pm_to_best:
        raise SystemExit("No usable JSON results found.")

    # sanity: ensure single model (but allow multiple; just show in caption)
    model_id_note = ", ".join(sorted(model_ids))

    prompt_modes = sorted(pm_to_best.keys())
    # union attributes across prompts
    attrs_all = set()
    for pm in prompt_modes:
        attrs_all.update(pm_to_best[pm].keys())
    attributes = sorted(attrs_all)

    # bold best per attribute across prompts
    attr_max: Dict[str, float] = {}
    for a in attributes:
        vals = []
        for pm in prompt_modes:
            if a in pm_to_best[pm]:
                _src, _layer, v = pm_to_best[pm][a]
                if not math.isnan(v):
                    vals.append(v)
        attr_max[a] = max(vals) if vals else float("nan")

    cell: Dict[Tuple[str, str], str] = {}
    for a in attributes:
        vmax = attr_max.get(a, float("nan"))
        for pm in prompt_modes:
            if a not in pm_to_best[pm]:
                cell[(a, pm)] = "--"
                continue
            src, layer, val = pm_to_best[pm][a]
            is_best = (not math.isnan(vmax)) and (abs(val - vmax) <= 1e-12)
            src_layer = fmt_src_layer(src, layer, bold=is_best)
            val_str = f"\\textbf{{{val:.3f}}}" if is_best else f"{val:.3f}"
            cell[(a, pm)] = f"{src_layer} ({val_str})"
            cell[(a, pm)] = f"{val_str} ({layer})"  

    metric_name = "Spearman $\\rho$" if args.metric == "rho" else "$R^2$"
    src_note = f" (source={tex_escape(args.source)})" if args.source else ""
    caption = args.caption or f"Best source-layer (and {metric_name}) on {tex_escape(args.split)} split across prompt modes. Model: {tex_escape(model_id_note)}{src_note}."
    tex = render_table_star_tabularx(attributes, prompt_modes, cell, caption=caption, label=args.label)

    os.makedirs(os.path.dirname(args.out_tex) or ".", exist_ok=True)
    with open(args.out_tex, "w", encoding="utf-8") as f:
        f.write(tex)

    print(f"[save] {args.out_tex}")
    print(f"[info] prompt_modes={prompt_modes}")
    print(f"[info] model_ids={sorted(model_ids)}")

if __name__ == "__main__":
    main()