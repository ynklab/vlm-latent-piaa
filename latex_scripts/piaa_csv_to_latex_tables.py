#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score


# --------------------- CONFIG ---------------------

# Column order + 2-level header
MODEL_META = {
    "qwen3vl-2b": ("Qwen3VL", "2B"),
    "qwen3vl-4b": ("Qwen3VL", "4B"),
    "qwen3vl-8b": ("Qwen3VL", "8B"),
    "gemma3-4b":  ("Gemma 3", "4B"),
    "gemma3-12b": ("Gemma 3", "12B"),
}
MODEL_ORDER = ["qwen3vl-2b", "qwen3vl-4b", "qwen3vl-8b", "gemma3-4b", "gemma3-12b"]

# Map folder name -> column key
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
    display_name: str  # shown in LaTeX "Method" column

# Your rows (+ display names you can tweak)
ROW_SPECS: List[RowSpec] = [
    RowSpec("raw", "-", "Raw"),
    RowSpec("bias", "small", "Bias"),
    RowSpec("bias", "large", "Bias"),
    RowSpec("vlm_fewshot_small", "small", "Few-shot"),
    RowSpec("lora_per_user_small", "small", "LoRA"),
    RowSpec("lora_per_user_large", "large", "LoRA"),
    RowSpec("direct_linear_llm_text_L15", "small", "Linear-Hidden"),
    RowSpec("direct_linear_llm_text_L15", "large", "Linear-Hidden"),
    RowSpec("direct_linear_giaa_gt_llm_text_L15", "small", "Linear-Hidden (GIAA)"),
    RowSpec("direct_linear_giaa_gt_llm_text_L15", "large", "Linear-Hidden (GIAA)"),
    RowSpec("hidden_attr_linear_llm_text_L15", "small", "Linear-Hidden (Reduce)"),
    RowSpec("hidden_attr_linear_llm_text_L15", "large", "Linear-Hidden (Reduce)"),
]


# --------------------- helpers ---------------------

def sanitize(s: str) -> str:
    if s is None:
        return "unknown"
    s = str(s)
    return re.sub(r"[^0-9A-Za-z._\\-]+", "_", s)

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

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"user_id","image_path","model_id","support_set","method","piaa_pred","user_score"}
    if not need.issubset(df.columns):
        raise ValueError(f"Missing columns in {path}: {need - set(df.columns)}")
    return df

def load_all_csvs_for_model(model_dir: str) -> pd.DataFrame:
    if not os.path.isdir(model_dir):
        return pd.DataFrame()
    dfs = []
    for name in os.listdir(model_dir):
        if not name.lower().endswith(".csv"):
            continue
        p = os.path.join(model_dir, name)
        try:
            dfs.append(load_csv(p))
        except Exception as e:
            print(f"[warn] skip csv (load error): {p} -> {e}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

def per_user_mean_metrics(df: pd.DataFrame) -> Tuple[float, float]:
    """
    Compute:
      rho_u = Spearman(user_score, piaa_pred) per user
      r2_u  = R2(user_score, piaa_pred) per user
    then average over users (skipping undefined).
    """
    rhos = []
    r2s = []
    for uid, g in df.groupby("user_id"):
        y_true = g["user_score"].to_numpy(dtype=float)
        y_pred = g["piaa_pred"].to_numpy(dtype=float)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        if y_true.size < 2:
            continue

        rho = spearmanr(y_true, y_pred).correlation
        if np.isfinite(rho):
            rhos.append(float(rho))

        try:
            r2 = float(r2_score(y_true, y_pred))
        except Exception:
            r2 = float("nan")
        if np.isfinite(r2):
            r2s.append(float(r2))

    mean_rho = float(np.mean(rhos)) if rhos else float("nan")
    mean_r2  = float(np.mean(r2s))  if r2s else float("nan")
    return mean_rho, mean_r2

def subset_for_row(df_model_all: pd.DataFrame, row: RowSpec) -> pd.DataFrame:
    """
    support_set == "-" means:
      - Use either small or large (prefer small if exists), discard the other.
      - If only one exists, use it.
    """
    if df_model_all.empty:
        return df_model_all

    if row.support_set != "-":
        return df_model_all[(df_model_all["method"] == row.method) &
                            (df_model_all["support_set"] == row.support_set)].copy()

    # support_set == "-"
    df_m = df_model_all[df_model_all["method"] == row.method].copy()
    if df_m.empty:
        return df_m

    has_small = (df_m["support_set"] == "small").any()
    has_large = (df_m["support_set"] == "large").any()

    if has_small:
        return df_m[df_m["support_set"] == "small"].copy()
    if has_large:
        return df_m[df_m["support_set"] == "large"].copy()

    # fallback: if neither small/large exists (rare), just take all
    return df_m.copy()

def format_cell(rho: float, r2: float, bold_rho: bool, bold_r2: bool) -> str:
    """
    Single-line: "0.123 / 0.456" (ρ / R^2)
    Bold each component independently.
    """
    if not np.isfinite(rho) and not np.isfinite(r2):
        return "--"

    rho_s = f"{rho:.3f}" if np.isfinite(rho) else "--"
    r2_s  = f"{r2:.3f}"  if np.isfinite(r2)  else "--"

    if bold_rho and rho_s != "--":
        rho_s = f"\\textbf{{{rho_s}}}"
    if bold_r2 and r2_s != "--":
        r2_s = f"\\textbf{{{r2_s}}}"

    return f"{rho_s} / {r2_s}"

def build_headers(col_models: List[str]) -> Tuple[str, str, str]:
    """
    Return 3 header rows:
      1) family row with multicolumn: Qwen3VL / Gemma 3
      2) size row: 2B/4B/8B/4B/12B
      3) metric row under each model column: "$\\rho$ / $R^2$"
    """
    fams = [MODEL_META[m][0] for m in col_models]
    sizes = [MODEL_META[m][1] for m in col_models]

    # spans by family (contiguous)
    spans = []
    i = 0
    while i < len(col_models):
        fam = fams[i]
        j = i
        while j < len(col_models) and fams[j] == fam:
            j += 1
        spans.append((fam, i, j))
        i = j

    # row1: Method | Support | multicolumn...
    r1 = ["Method", "Support"]
    for fam, i, j in spans:
        r1.append(f"\\multicolumn{{{j-i}}}{{c}}{{{latex_escape(fam)}}}")
    row1 = " & ".join(r1) + " \\\\"

    # row2: empty | empty | sizes...
    r2 = ["", ""] + [latex_escape(s) for s in sizes]
    row2 = " & ".join(r2) + " \\\\"

    # row3: empty | empty | rho/r2 indicator...
    r3 = ["", ""] + [r"$\rho$ / $R^2$"] * len(col_models)
    row3 = " & ".join(r3) + " \\\\"

    return row1, row2, row3

def render_table(dataset_name: str,
                 col_models: List[str],
                 metrics: Dict[Tuple[str,str,str], Tuple[float,float]],
                 bold_mask: Dict[Tuple[str,str,str], Tuple[bool,bool]]) -> str:
    """
    key: (method, support_set, model_key) -> (rho, r2)
    """

    # --- reorder rows by support group: "-" -> "small" -> "large" ---
    support_order = {"-": 0, "small": 1, "large": 2}
    rows_sorted = sorted(
        ROW_SPECS,
        key=lambda r: (support_order.get(r.support_set, 99), r.display_name, r.method)
    )

    col_spec = "l l " + " ".join(["c"] * len(col_models))
    h1, h2, h3 = build_headers(col_models)

    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{User-average metrics on {dataset_name.upper()}. Best per-column $\\rho$ and $R^2$ are bolded.}}")
    lines.append(f"\\label{{tab:{dataset_name}_useravg_metrics}}")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.15}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append(h1)
    lines.append(h2)
    lines.append(h3)
    lines.append("\\midrule")

    prev_disp = None
    prev_support = None

    for row in rows_sorted:
        # insert a midrule when support group changes (except for the first group)
        if prev_support is not None and row.support_set != prev_support:
            lines.append("\\midrule")
            prev_disp = None  # reset "same method suppression" within each support group
        prev_support = row.support_set

        disp = row.display_name
        method_cell = "" if (prev_disp == disp) else disp
        prev_disp = disp

        support_cell = SUPPORT_DISPLAY.get(row.support_set, row.support_set)

        row_cells = [latex_escape(method_cell), latex_escape(support_cell)]
        for mk in col_models:
            rho, r2 = metrics.get((row.method, row.support_set, mk), (float("nan"), float("nan")))
            br, b2  = bold_mask.get((row.method, row.support_set, mk), (False, False))
            row_cells.append(format_cell(rho, r2, br, b2))

        lines.append(" & ".join(row_cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines)

# --------------------- main ---------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=True,
                    help="Root directory containing para/ and lapis/ subdirectories.")
    ap.add_argument("--out_dir", required=True,
                    help="Output directory to write para_metrics.tex and lapis_metrics.tex")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for dataset in ["para", "lapis"]:
        dataset_dir = os.path.join(args.root_dir, dataset)
        if not os.path.isdir(dataset_dir):
            print(f"[warn] dataset dir not found, skip: {dataset_dir}")
            continue

        # load all model csvs into memory
        model_key_to_df: Dict[str, pd.DataFrame] = {}
        for folder in os.listdir(dataset_dir):
            p = os.path.join(dataset_dir, folder)
            if not os.path.isdir(p):
                continue
            mk = MODEL_FOLDER_TO_KEY.get(folder)
            if mk is None:
                continue
            df_all = load_all_csvs_for_model(p)
            if df_all.empty:
                continue
            model_key_to_df[mk] = df_all

        col_models = [m for m in MODEL_ORDER if m in model_key_to_df]
        if not col_models:
            print(f"[warn] no recognized model folders for dataset={dataset} under {dataset_dir}")
            continue

        # compute metrics for each row x model
        metrics: Dict[Tuple[str,str,str], Tuple[float,float]] = {}

        for mk in col_models:
            df_model_all = model_key_to_df[mk]
            for row in ROW_SPECS:
                df_sub = subset_for_row(df_model_all, row)
                if df_sub.empty:
                    metrics[(row.method, row.support_set, mk)] = (float("nan"), float("nan"))
                    continue
                rho, r2 = per_user_mean_metrics(df_sub)
                metrics[(row.method, row.support_set, mk)] = (rho, r2)

        # bold masks per model column (best rho and best r2 independently)
        bold_mask: Dict[Tuple[str,str,str], Tuple[bool,bool]] = {}
        for mk in col_models:
            rho_vals = []
            r2_vals = []
            for row in ROW_SPECS:
                rho, r2 = metrics.get((row.method, row.support_set, mk), (float("nan"), float("nan")))
                if np.isfinite(rho):
                    rho_vals.append(rho)
                if np.isfinite(r2):
                    r2_vals.append(r2)

            best_rho = max(rho_vals) if rho_vals else float("nan")
            best_r2  = max(r2_vals)  if r2_vals else float("nan")

            for row in ROW_SPECS:
                rho, r2 = metrics.get((row.method, row.support_set, mk), (float("nan"), float("nan")))
                br = np.isfinite(best_rho) and np.isfinite(rho) and abs(rho - best_rho) <= 1e-12
                b2 = np.isfinite(best_r2)  and np.isfinite(r2)  and abs(r2  - best_r2)  <= 1e-12
                bold_mask[(row.method, row.support_set, mk)] = (br, b2)

        tex = render_table(dataset, col_models, metrics, bold_mask)
        out_tex = os.path.join(args.out_dir, f"{dataset}_metrics.tex")
        with open(out_tex, "w", encoding="utf-8") as f:
            f.write(tex)
        print(f"[save] {out_tex}")

    print("[done] generated LaTeX tables.")


if __name__ == "__main__":
    main()