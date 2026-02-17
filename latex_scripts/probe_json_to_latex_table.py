#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import argparse
from typing import Dict, Any, Tuple, Optional, List

# ---------- source symbols ----------
SOURCE_SYMBOL = {
    "vision": "V",
    "llm_visual": "LV",
    "llm_text": "LT",
}

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

def infer_model_name(d: Dict[str, Any], path: str) -> str:
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
    for k in ["model_id", "model", "model_name", "name"]:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return os.path.splitext(os.path.basename(path))[0]

def parse_best_for_attr(
    attr_obj: Dict[str, Any],
    split: str,
    metric: str,
    source_filter: Optional[str] = None,
    skip_vision=True,
) -> Optional[Tuple[str, int, float]]:
    per_layer = attr_obj.get("per_layer")
    if not isinstance(per_layer, list) or not per_layer:
        return None

    best = None  # (value, source, layer)
    for entry in per_layer:
        if not isinstance(entry, dict):
            continue
        src = entry.get("source")
        if src != 'vision':
            continue
        # if src == 'llm_text_tail':
        #     continue
        # if src == 'llm_visual':
        #     continue
        # if src == 'vision' and skip_vision:
        #     continue
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

# ---------- model ordering / header grouping ----------
MODEL_META = {
    "Qwen/Qwen3-VL-2B-Instruct":  ("Qwen3-VL", "2B"),
    "Qwen/Qwen3-VL-4B-Instruct":  ("Qwen3-VL", "4B"),
    "Qwen/Qwen3-VL-8B-Instruct":  ("Qwen3-VL", "8B"),
    "google/gemma-3-4b-it":       ("Gemma 3",  "4B"),
    "google/gemma-3-12b-it":      ("Gemma 3",  "12B"),
    "facebook/dinov3-vitb16-pretrain-lvd1689m": ("DINOv3", "ViT-B/16"),
    "facebook/dinov3-vitl16-pretrain-lvd1689m": ("DINOv3", "ViT-L/16"),
}

PREFERRED_MODEL_ORDER = [
    "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "facebook/dinov3-vitl16-pretrain-lvd1689m",
]

def sort_models_present(models_present: List[str]) -> List[str]:
    s = set(models_present)
    ordered = [m for m in PREFERRED_MODEL_ORDER if m in s]
    extra = sorted([m for m in models_present if m not in set(ordered)])
    return ordered + extra

def build_group_headers(models: List[str]) -> Tuple[str, str]:
    fams = [MODEL_META.get(m, ("Other", tex_escape(m)))[0] for m in models]
    sizes = [MODEL_META.get(m, ("Other", tex_escape(m)))[1] for m in models]

    spans = []
    i = 0
    while i < len(models):
        fam = fams[i]
        j = i
        while j < len(models) and fams[j] == fam:
            j += 1
        spans.append((fam, i, j))
        i = j

    parts1 = ["Attribute"]
    for fam, i, j in spans:
        parts1.append(f"\\multicolumn{{{j-i}}}{{c}}{{{tex_escape(fam)}}}")
    header1 = " & ".join(parts1) + " \\\\"

    parts2 = [""] + [tex_escape(s) for s in sizes]
    header2 = " & ".join(parts2) + " \\\\"
    return header1, header2

# --- render 部分だけ差し替えればOKです（他は v5 と同じ） ---

def render_latex_table_star_tabularx(
    attributes: List[str],
    models: List[str],
    cell: Dict[Tuple[str, str], str],
    caption: str,
    label: str,
) -> str:
    """
    Render a two-column wide table that fits exactly within \\textwidth using tabularx.
    Requires LaTeX packages: booktabs, tabularx, array
    """
    header1, header2 = build_group_headers(models)

    # Column spec:
    #   first column: l (attribute)
    #   remaining: X columns that stretch to fit textwidth, centered
    # Example: l *{6}{>{\centering\arraybackslash}X}
    col_spec = "l " + f"*{{{len(models)}}}{{>{{\\centering\\arraybackslash}}X}}"

    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")  # tighter columns (optional)
    lines.append("\\renewcommand{\\arraystretch}{1.1}")  # slightly taller rows (optional)

    lines.append(f"\\begin{{tabularx}}{{\\textwidth}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append(header1)
    lines.append(header2)
    lines.append("\\midrule")
    for a in attributes:
        row = [tex_escape(a)]
        for m in models:
            row.append(cell.get((a, m), "--"))
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabularx}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table*}")

    return "\n".join(lines)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_tex", required=True)
    ap.add_argument("--metric", choices=["rho", "r2"], default="rho")
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--source", default=None)
    ap.add_argument("--caption", default=None)
    ap.add_argument("--label", default="tab:probing_best_layers")
    ap.add_argument("--include_score", action="store_true")
    args = ap.parse_args()

    files = [
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.lower().endswith(".json")
    ]
    if not files:
        raise SystemExit(f"No .json files found in {args.input_dir}")

    # model -> attr -> (src, layer, val)
    model_to_best: Dict[str, Dict[str, Tuple[str, int, float]]] = {}

    for path in sorted(files):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[warn] failed to parse {path}: {e}")
            continue

        model = infer_model_name(data, path)

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
            best = parse_best_for_attr(attr_obj, split=args.split, metric=args.metric, source_filter=args.source, skip_vision=not model.startswith('facebook'))
            if best is None:
                continue
            src, layer, val = best
            best_map[str(attr_name)] = (src, layer, val)

        if best_map:
            model_to_best[model] = best_map
        else:
            print(f"[warn] {path}: no usable per_layer entries for split={args.split}, metric={args.metric}")

    if not model_to_best:
        raise SystemExit("No usable JSON results found (nothing parsed).")

    models = sort_models_present(list(model_to_best.keys()))

    # union of attributes
    attrs_all = set()
    for m in models:
        attrs_all.update(model_to_best[m].keys())
    attributes = sorted(attrs_all)

    # Precompute max value per attribute across models (for bold)
    attr_max: Dict[str, float] = {}
    for a in attributes:
        vals = []
        for m in models:
            if a in model_to_best[m]:
                _src, _layer, v = model_to_best[m][a]
                if not math.isnan(v):
                    vals.append(v)
        attr_max[a] = max(vals) if vals else float("nan")

    # build cell strings (bold max per row)
    cell: Dict[Tuple[str, str], str] = {}
    for a in attributes:
        vmax = attr_max.get(a, float("nan"))
        for m in models:
            if a in model_to_best[m]:
                src, layer, val = model_to_best[m][a]
                is_best = (not math.isnan(vmax)) and (abs(val - vmax) <= 1e-12)
                src_layer = fmt_src_layer(src, layer, bold=is_best)
                val_str = f"\\textbf{{{val:.3f}}}" if is_best else f"{val:.3f}"
                cell[(a, m)] = f"{src_layer} ({val_str})"
                cell[(a, m)] = f"{val_str} ({layer})"
            else:
                cell[(a, m)] = "--"

    metric_name = "Spearman $\\rho$" if args.metric == "rho" else "$R^2$"
    src_note = f" (restricted to source={tex_escape(args.source)})" if args.source else ""
    caption = args.caption or f"Best source-layer (and {metric_name}) on {tex_escape(args.split)} split{src_note}."
    tex = render_latex_table_star_tabularx(attributes, models, cell, caption=caption, label=args.label)

    os.makedirs(os.path.dirname(args.out_tex) or ".", exist_ok=True)
    with open(args.out_tex, "w", encoding="utf-8") as f:
        f.write(tex)

    print(f"[save] wrote LaTeX table to: {args.out_tex}")
    print(f"[info] models={len(models)}, attrs={len(attributes)}, metric={args.metric}, split={args.split}")

if __name__ == "__main__":
    main()