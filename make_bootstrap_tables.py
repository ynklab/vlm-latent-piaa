#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# --------------------- CONFIG ---------------------

MODEL_META = {
    "qwen3vl-2b": ("Qwen3VL", "2B"),
    "qwen3vl-4b": ("Qwen3VL", "4B"),
    "qwen3vl-8b": ("Qwen3VL", "8B"),
    "gemma3-4b":  ("Gemma 3", "4B"),
    "gemma3-12b": ("Gemma 3", "12B"),
}
MODEL_ORDER = ["qwen3vl-2b", "qwen3vl-4b", "qwen3vl-8b", "gemma3-4b", "gemma3-12b"]

MODEL_FOLDER_TO_KEY = {
    "qwen3vl-2b": "qwen3vl-2b",
    "qwen3vl-4b": "qwen3vl-4b",
    "qwen3vl-8b": "qwen3vl-8b",
    "gemma3-4b": "gemma3-4b",
    "gemma3-12b": "gemma3-12b",
    # aliases
    "qwe3vl-2b": "qwen3vl-2b",
    "qwe3vl-4b": "qwen3vl-4b",
    "qwe3vl-8b": "qwen3vl-8b",
}

SUPPORT_DISPLAY = {
    "small": "10-shot",
    "large": "100-shot",
    "-": "",
}

@dataclass(frozen=True)
class RowSpec:
    method: str
    support_set: str   # "small" | "large" | "-"
    display_name: str  # shown in table

ROW_SPECS: List[RowSpec] = [
    RowSpec("raw", "small", "Raw Text"),
    # RowSpec("bias", "small", "Adjust-Bias"),
    RowSpec("bias", "large", "Adjust-Bias"),
    RowSpec("vlm_fewshot_small", "small", "Few-shot"),
    # RowSpec("vlm_fewshot_large", "large", "Few-shot"),
    # RowSpec("lora_per_user_small", "small", "LoRA"),
    RowSpec("lora_per_user_large", "large", "LoRA"),
    # RowSpec("direct_linear_llm_text_L15", "small", "Linear-Hidden"),
    RowSpec("direct_linear_llm_text_L15", "large", "Linear-Hidden"),
    # RowSpec("direct_linear_giaa_gt_llm_text_L15", "small", "Linear-Hidden (GIAA)"),
    RowSpec("direct_linear_giaa_gt_llm_text_L15", "large", "Linear-Hidden (GIAA)"),
    # RowSpec("hidden_attr_linear_llm_text_L15", "small", "Linear-Hidden (Reduce)"),
    RowSpec("hidden_attr_linear_llm_text_L15", "large", "Linear-Hidden (Reduce)"),
]


# --------------------- helpers ---------------------

def latex_escape(s: str) -> str:
    return (
        str(s).replace("\\", "\\textbackslash{}")
              .replace("_", "\\_")
              .replace("%", "\\%")
              .replace("&", "\\&")
              .replace("#", "\\#")
              .replace("{", "\\{")
              .replace("}", "\\}")
              .replace("$", "\\$")
    )

def sanitize(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z._\\-]+", "_", str(s))

def build_headers(col_models: List[str]) -> Tuple[str, str]:
    fams = [MODEL_META[m][0] for m in col_models]
    sizes = [MODEL_META[m][1] for m in col_models]

    spans = []
    i = 0
    while i < len(col_models):
        fam = fams[i]
        j = i
        while j < len(col_models) and fams[j] == fam:
            j += 1
        spans.append((fam, i, j))
        i = j

    r1 = ["Method", "Support"]
    for fam, i, j in spans:
        r1.append(f"\\multicolumn{{{j-i}}}{{c}}{{{latex_escape(fam)}}}")
    row1 = " & ".join(r1) + " \\\\"

    r2 = ["", ""] + [latex_escape(s) for s in sizes]
    row2 = " & ".join(r2) + " \\\\"
    return row1, row2

def rows_sorted_for_table() -> List[RowSpec]:
    support_order = {"-": 0, "small": 1, "large": 2}
    return sorted(ROW_SPECS, key=lambda r: (support_order.get(r.support_set, 99), r.display_name, r.method))

def format_rho_ci(mean: float, lo: float, hi: float, digits: int) -> str:
    if not (np.isfinite(mean) and np.isfinite(lo) and np.isfinite(hi)):
        return "--"
    fmt = f"{{:.{digits}f}}"
    return f"[{fmt.format(lo)}, {fmt.format(hi)}]"
    # return f"{fmt.format(mean)} [{fmt.format(lo)}, {fmt.format(hi)}]"

def format_p(p: float, digits: int, mode: str, hi_thr: float, lo_thr: float) -> str:
    """
    mode:
      - "numeric": show p
      - "flag": show ✓ / ✗ / ~ based on thresholds
    """
    if not np.isfinite(p):
        return "--"
    if mode == "numeric":
        return f"{p:.{digits}f}"
    # flag mode
    if p >= hi_thr:
        return r"\checkmark"
    if p <= lo_thr:
        return r"\times"
    return r"\sim"


# --------------------- loading bootstrap CSVs ---------------------

def load_bootstrap_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {
        "model_id","metric","support_set","method",
        "mean","ci_low","ci_high",
        "baseline_support_set","baseline_method",
        "p_greater",
    }
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns in {path}: {miss}")
    # filter metric=rho only (tables are rho-based)
    df = df[df["metric"].astype(str) == "rho"].copy()
    # normalize types
    df["support_set"] = df["support_set"].astype(str)
    df["method"] = df["method"].astype(str)
    for c in ["mean","ci_low","ci_high","p_greater"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def build_cell_maps(df_boot: pd.DataFrame, baseline_support: str, baseline_method: str) -> Tuple[Dict[Tuple[str,str], Tuple[float,float,float]], Dict[Tuple[str,str], float]]:
    """
    Returns:
      rho_ci_map[(method, support_set)] = (mean, lo, hi)
      p_map[(method, support_set)] = p_greater  (method - baseline)
    """
    # keep only rows that match the chosen baseline combo (safety)
    df = df_boot[
        (df_boot["baseline_support_set"].astype(str) == baseline_support) &
        (df_boot["baseline_method"].astype(str) == baseline_method)
    ].copy()

    rho_ci_map = {}
    p_map = {}
    for _, r in df.iterrows():
        key = (str(r["method"]), str(r["support_set"]))
        rho_ci_map[key] = (float(r["mean"]), float(r["ci_low"]), float(r["ci_high"]))
        p_map[key] = float(r["p_greater"])
    return rho_ci_map, p_map


# --------------------- rendering ---------------------

def render_latex_table(dataset: str, title: str, col_models: List[str], cell: Dict[Tuple[str,str,str], str], caption: str, label: str) -> str:
    rows = rows_sorted_for_table()
    h1, h2 = build_headers(col_models)

    col_spec = "l l " + " ".join(["c"] * len(col_models))
    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{3.5pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.15}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append(h1)
    lines.append(h2)
    lines.append("\\midrule")

    prev_disp = None
    prev_support = None
    for r in rows:
        if prev_support is not None and r.support_set != prev_support:
            lines.append("\\midrule")
            prev_disp = None
        prev_support = r.support_set

        method_cell = "" if (prev_disp == r.display_name) else r.display_name
        prev_disp = r.display_name

        support_cell = SUPPORT_DISPLAY.get(r.support_set, r.support_set)

        row_cells = [latex_escape(method_cell), latex_escape(support_cell)]
        for mk in col_models:
            row_cells.append(cell.get((r.method, r.support_set, mk), "--"))
        lines.append(" & ".join(row_cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table*}")
    return "\n".join(lines)

def render_markdown_table(col_models: List[str], cell: Dict[Tuple[str,str,str], str]) -> str:
    rows = rows_sorted_for_table()
    headers = ["Method", "Support"] + col_models
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    prev_disp = None
    prev_support = None
    for r in rows:
        if prev_support is not None and r.support_set != prev_support:
            # blank line between support groups
            lines.append("| " + " | ".join([""] * len(headers)) + " |")
            prev_disp = None
        prev_support = r.support_set

        method_cell = "" if (prev_disp == r.display_name) else r.display_name
        prev_disp = r.display_name
        support_cell = SUPPORT_DISPLAY.get(r.support_set, r.support_set)

        row = [method_cell, support_cell]
        for mk in col_models:
            row.append(cell.get((r.method, r.support_set, mk), "--"))
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


# --------------------- main ---------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=True,
                    help="Root dir containing bootstrap_vs/{para,lapis}/*.csv (one file per model key).")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dataset_names", nargs="+", default=["para", "lapis"])
    ap.add_argument("--baseline_method", default="direct_linear_llm_text_L15")
    ap.add_argument("--baseline_support_set", default="large")
    ap.add_argument("--digits_ci", type=int, default=3)
    ap.add_argument("--digits_p", type=int, default=2)
    ap.add_argument("--p_mode", choices=["numeric","flag"], default="numeric",
                    help="How to render p_greater: numeric or flag(✓/×/~).")
    ap.add_argument("--p_hi", type=float, default=0.95, help="flag threshold for ✓")
    ap.add_argument("--p_lo", type=float, default=0.05, help="flag threshold for ×")
    ap.add_argument("--markdown", action="store_true", help="Also output Markdown tables.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for dataset in args.dataset_names:
        ds_dir = os.path.join(args.root_dir, dataset)
        if not os.path.isdir(ds_dir):
            print(f"[warn] skip missing dataset dir: {ds_dir}")
            continue

        # load bootstrap dfs per model
        model_key_to_df: Dict[str, pd.DataFrame] = {}
        for fn in os.listdir(ds_dir):
            if not fn.lower().endswith(".csv"):
                continue
            stem = os.path.splitext(fn)[0]
            mk = MODEL_FOLDER_TO_KEY.get(stem)
            if mk is None:
                continue
            p = os.path.join(ds_dir, fn)
            try:
                model_key_to_df[mk] = load_bootstrap_csv(p)
            except Exception as e:
                print(f"[warn] skip {p}: {e}")

        col_models = [m for m in MODEL_ORDER if m in model_key_to_df]
        if not col_models:
            print(f"[warn] no model bootstrap csvs found under {ds_dir}")
            continue

        # build cells for CI table and p table
        ci_cell: Dict[Tuple[str,str,str], str] = {}
        p_cell: Dict[Tuple[str,str,str], str] = {}

        for mk in col_models:
            df_boot = model_key_to_df[mk]
            rho_ci_map, p_map = build_cell_maps(
                df_boot,
                baseline_support=args.baseline_support_set,
                baseline_method=args.baseline_method,
            )

            for r in ROW_SPECS:
                key = (r.method, r.support_set)
                if key in rho_ci_map:
                    mean, lo, hi = rho_ci_map[key]
                    ci_cell[(r.method, r.support_set, mk)] = format_rho_ci(mean, lo, hi, args.digits_ci)
                else:
                    ci_cell[(r.method, r.support_set, mk)] = "--"

                if key in p_map:
                    p = p_map[key]
                    p_cell[(r.method, r.support_set, mk)] = format_p(p, args.digits_p, args.p_mode, args.p_hi, args.p_lo)
                else:
                    p_cell[(r.method, r.support_set, mk)] = "--"

        # LaTeX CI table
        caption_ci = f"User-average Spearman $\\rho$ with {int(100*0.95)}\\% bootstrap CI on {dataset.upper()}."
        label_ci = f"tab:{dataset}_rho_ci"
        tex_ci = render_latex_table(dataset, "rho_ci", col_models, ci_cell, caption_ci, label_ci)
        out_ci = os.path.join(args.out_dir, f"{dataset}_rho_ci.tex")
        with open(out_ci, "w", encoding="utf-8") as f:
            f.write(tex_ci)
        print(f"[save] {out_ci}")

        # LaTeX p table
        caption_p = (f"Bootstrap probability $P(\\Delta>0)$ where $\\Delta=\\rho(\\text{{method}})-\\rho(\\text{{baseline}})$, "
                     f"baseline = {latex_escape(args.baseline_method)} ({latex_escape(args.baseline_support_set)}), on {dataset.upper()}.")
        label_p = f"tab:{dataset}_p_greater"
        tex_p = render_latex_table(dataset, "p_greater", col_models, p_cell, caption_p, label_p)
        out_p = os.path.join(args.out_dir, f"{dataset}_p_greater.tex")
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(tex_p)
        print(f"[save] {out_p}")

        # Markdown
        if args.markdown:
            md_ci = render_markdown_table(col_models, ci_cell)
            md_p = render_markdown_table(col_models, p_cell)
            with open(os.path.join(args.out_dir, f"{dataset}_rho_ci.md"), "w", encoding="utf-8") as f:
                f.write(md_ci)
            with open(os.path.join(args.out_dir, f"{dataset}_p_greater.md"), "w", encoding="utf-8") as f:
                f.write(md_p)
            print(f"[save] markdown tables for {dataset}")

    print("[done]")


if __name__ == "__main__":
    main()