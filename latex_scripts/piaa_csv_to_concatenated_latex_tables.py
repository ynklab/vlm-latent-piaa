#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score


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
    "-": "",   # raw support is blank
}

@dataclass(frozen=True)
class RowSpec:
    method: str
    support_set: str   # "small" | "large" | "-"
    display_name: str  # shown in LaTeX "Method" column

ROW_SPECS: List[RowSpec] = [
    RowSpec("raw", "-", "Raw"),
    # RowSpec("bias", "small", "Bias"),
    # RowSpec("vlm_fewshot_small", "small", "Few-shot"),
    # RowSpec("lora_per_user_small", "small", "LoRA"),
    # RowSpec("direct_linear_llm_text_L15", "small", "Direct"),
    # RowSpec("direct_linear_giaa_gt_llm_text_L15", "small", "Direct (GIAA target)"),
    # RowSpec("hidden_attr_linear_llm_text_L15", "small", "HiddenAttr"),
    # RowSpec("bias", "large", "Bias"),
    # RowSpec("vlm_fewshot", "large", "Few-shot"),
    # RowSpec("lora_per_user_large", "large", "LoRA"),
    # RowSpec("direct_linear_llm_text_L15", "small", "Linear-Hidden"),
    RowSpec("direct_linear_llm_text_L15", "large", "Linear-Hidden"),
    # RowSpec("direct_linear_giaa_gt_llm_text_L15", "small", "Linear-Hidden (GIAA)"),
    RowSpec("direct_linear_giaa_gt_llm_text_L15", "large", "Linear-Hidden (GIAA)"),
    # RowSpec("hidden_attr_linear_llm_text_L15", "small", "Linear-Hidden (Reduce)"),
    RowSpec("hidden_attr_linear_llm_text_L15", "large", "Linear-Hidden (Reduce)"),
]

# ここで “supportグループごとの並び順” を指定
ORDER_BY_SUPPORT = {
    "-": ["raw"],
    "small": [
        "bias",
        "vlm_fewshot",
        "lora_per_user_small",
        "direct_linear_llm_text_L15",
        "direct_linear_giaa_gt_llm_text_L15",
        "hidden_attr_linear_llm_text_L15",
    ],
    "large": [
        "bias",
        "vlm_fewshot",
        "lora_per_user_large",
        "direct_linear_llm_text_L15",
        "direct_linear_giaa_gt_llm_text_L15",
        "hidden_attr_linear_llm_text_L15",
    ],
}


# --------------------- helpers ---------------------
# --------------------- helpers ---------------------

def sanitize(s: str) -> str:
    if s is None:
        return "unknown"
    return re.sub(r"[^0-9A-Za-z._\\-]+", "_", str(s))

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
    rhos, r2s = [], []
    for _, g in df.groupby("user_id"):
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
    return (float(np.mean(rhos)) if rhos else float("nan"),
            float(np.mean(r2s)) if r2s else float("nan"))

def subset_for_row(df_model_all: pd.DataFrame, row: RowSpec) -> pd.DataFrame:
    if df_model_all.empty:
        return df_model_all

    if row.support_set != "-":
        return df_model_all[(df_model_all["method"] == row.method) &
                            (df_model_all["support_set"] == row.support_set)].copy()

    # support_set == "-" => take only small if exists else large else all
    df_m = df_model_all[df_model_all["method"] == row.method].copy()
    if df_m.empty:
        return df_m
    if (df_m["support_set"] == "small").any():
        return df_m[df_m["support_set"] == "small"].copy()
    if (df_m["support_set"] == "large").any():
        return df_m[df_m["support_set"] == "large"].copy()
    return df_m.copy()

def format_cell(rho: float, r2: float, bold_rho: bool, bold_r2: bool) -> str:
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
    row2 = " & ".join(["", ""] + [latex_escape(s) for s in sizes]) + " \\\\"
    row3 = " & ".join(["", ""] + [r"$\rho$ / $R^2$"] * len(col_models)) + " \\\\"
    return row1, row2, row3

def order_rows_for_support(rows: List[RowSpec], support: str) -> List[RowSpec]:
    preferred = ORDER_BY_SUPPORT.get(support, [])
    rs = [r for r in rows if r.support_set == support]
    idx = {m: i for i, m in enumerate(preferred)}
    return sorted(rs, key=lambda r: (idx.get(r.method, 10_000), r.display_name, r.method))

def rows_in_final_order(rows: List[RowSpec]) -> List[RowSpec]:
    ordered = []
    for sup in ["-", "small", "large"]:
        ordered.extend(order_rows_for_support(rows, sup))
    return ordered

def compute_metrics_for_dataset(dataset_root: str) -> Tuple[List[str], Dict[str, pd.DataFrame]]:
    model_key_to_df: Dict[str, pd.DataFrame] = {}
    for folder in os.listdir(dataset_root):
        p = os.path.join(dataset_root, folder)
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
    return col_models, model_key_to_df

def build_metrics_and_bold(model_key_to_df: Dict[str, pd.DataFrame],
                           col_models: List[str]) -> Tuple[
                               Dict[Tuple[str,str,str], Tuple[float,float]],
                               Dict[Tuple[str,str,str], Tuple[bool,bool]]
                           ]:
    metrics: Dict[Tuple[str,str,str], Tuple[float,float]] = {}
    for mk in col_models:
        df_model_all = model_key_to_df[mk]
        for row in ROW_SPECS:
            df_sub = subset_for_row(df_model_all, row)
            if df_sub.empty:
                metrics[(row.method, row.support_set, mk)] = (float("nan"), float("nan"))
            else:
                metrics[(row.method, row.support_set, mk)] = per_user_mean_metrics(df_sub)

    bold: Dict[Tuple[str,str,str], Tuple[bool,bool]] = {}
    for mk in col_models:
        rho_vals, r2_vals = [], []
        for row in ROW_SPECS:
            rho, r2 = metrics.get((row.method, row.support_set, mk), (float("nan"), float("nan")))
            if np.isfinite(rho): rho_vals.append(rho)
            if np.isfinite(r2):  r2_vals.append(r2)
        best_rho = max(rho_vals) if rho_vals else float("nan")
        best_r2  = max(r2_vals)  if r2_vals else float("nan")

        for row in ROW_SPECS:
            rho, r2 = metrics.get((row.method, row.support_set, mk), (float("nan"), float("nan")))
            br = np.isfinite(best_rho) and np.isfinite(rho) and abs(rho - best_rho) <= 1e-12
            b2 = np.isfinite(best_r2)  and np.isfinite(r2)  and abs(r2  - best_r2)  <= 1e-12
            bold[(row.method, row.support_set, mk)] = (br, b2)
    return metrics, bold


# --------------------- rendering combined table ---------------------

def render_combined_table(
    datasets: List[Tuple[str, str]],  # [(dir_name, display_name), ...]
    col_models: List[str],
    ds_to_metrics: Dict[str, Dict[Tuple[str,str,str], Tuple[float,float]]],
    ds_to_bold: Dict[str, Dict[Tuple[str,str,str], Tuple[bool,bool]]],
) -> str:
    col_spec = "l l " + " ".join(["c"] * len(col_models))
    h1, h2, h3 = build_headers(col_models)
    ordered_rows = rows_in_final_order(ROW_SPECS)

    def render_block(display_name: str,
                     metrics: Dict[Tuple[str,str,str], Tuple[float,float]],
                     bold: Dict[Tuple[str,str,str], Tuple[bool,bool]]) -> List[str]:
        lines = []
        span = 2 + len(col_models)
        lines.append(f"\\multicolumn{{{span}}}{{l}}{{\\textbf{{{latex_escape(display_name)}}}}} \\\\")
        lines.append("\\midrule")

        prev_disp = None
        prev_support = None

        for row in ordered_rows:
            if prev_support is not None and row.support_set != prev_support:
                lines.append("\\midrule")
                prev_disp = None
            prev_support = row.support_set

            method_cell = "" if (prev_disp == row.display_name) else row.display_name
            prev_disp = row.display_name
            support_cell = SUPPORT_DISPLAY.get(row.support_set, row.support_set)

            cells = [latex_escape(method_cell), latex_escape(support_cell)]
            for mk in col_models:
                rho, r2 = metrics.get((row.method, row.support_set, mk), (float("nan"), float("nan")))
                br, b2 = bold.get((row.method, row.support_set, mk), (False, False))
                cells.append(format_cell(rho, r2, br, b2))
            lines.append(" & ".join(cells) + " \\\\")
        return lines

    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{User-average metrics across datasets. Each cell shows $\\rho$ / $R^2$. Best per-column values are bolded.}")
    lines.append("\\label{tab:multi_dataset_useravg_metrics}")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\renewcommand{\\arraystretch}{1.15}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append(h1)
    lines.append(h2)
    lines.append(h3)
    lines.append("\\midrule")

    for i, (dir_name, disp_name) in enumerate(datasets):
        lines.extend(render_block(disp_name, ds_to_metrics[dir_name], ds_to_bold[dir_name]))
        if i != len(datasets) - 1:
            lines.append("\\midrule\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


# --------------------- main ---------------------

def parse_datasets_arg(ds_args: List[str]) -> List[Tuple[str, str]]:
    """
    Parse ["para:PARA", "lapis:LAPIS", ...] into [(dir_name, display_name), ...]
    """
    out = []
    for s in ds_args:
        if ":" not in s:
            raise ValueError(f"--datasets item must be like dir:DisplayName, got: {s}")
        dir_name, disp = s.split(":", 1)
        out.append((dir_name.strip(), disp.strip()))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=True, help="Root directory containing dataset subdirs.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_tex", default="combined_metrics.tex")
    ap.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="List like: para:PARA lapis:LAPIS (order defines table order).",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    datasets = parse_datasets_arg(args.datasets)

    # Load each dataset and compute metrics/bold
    ds_to_cols: Dict[str, List[str]] = {}
    ds_to_model_df: Dict[str, Dict[str, pd.DataFrame]] = {}

    for dir_name, disp in datasets:
        ds_root = os.path.join(args.root_dir, dir_name)
        if not os.path.isdir(ds_root):
            raise SystemExit(f"Dataset directory not found: {ds_root}")
        cols, model_df = compute_metrics_for_dataset(ds_root)
        ds_to_cols[dir_name] = cols
        ds_to_model_df[dir_name] = model_df

    # Use intersection of columns so header matches all datasets
    col_models = [m for m in MODEL_ORDER if all(m in ds_to_cols[dn] for dn, _ in datasets)]
    if not col_models:
        raise SystemExit("No common model columns across all datasets. (intersection is empty)")

    ds_to_metrics: Dict[str, Dict[Tuple[str,str,str], Tuple[float,float]]] = {}
    ds_to_bold: Dict[str, Dict[Tuple[str,str,str], Tuple[bool,bool]]] = {}

    for dir_name, _disp in datasets:
        metrics, bold = build_metrics_and_bold(ds_to_model_df[dir_name], col_models)
        ds_to_metrics[dir_name] = metrics
        ds_to_bold[dir_name] = bold

    tex = render_combined_table(datasets, col_models, ds_to_metrics, ds_to_bold)

    out_path = os.path.join(args.out_dir, args.out_tex)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(tex)
    print(f"[save] {out_path}")

if __name__ == "__main__":
    main()