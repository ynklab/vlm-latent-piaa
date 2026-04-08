#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


MODEL_META = {
    "qwen3vl-2b": ("Qwen3VL", "2B"),
    "qwen3vl-4b": ("Qwen3VL", "4B"),
    "qwen3vl-8b": ("Qwen3VL", "8B"),
    "gemma3-4b": ("Gemma 3", "4B"),
    "gemma3-12b": ("Gemma 3", "12B"),
}
MODEL_ORDER = ["qwen3vl-2b", "qwen3vl-4b", "qwen3vl-8b", "gemma3-4b", "gemma3-12b"]

MODEL_FOLDER_TO_KEY = {
    "qwen3vl-2b": "qwen3vl-2b",
    "qwen3vl-4b": "qwen3vl-4b",
    "qwen3vl-8b": "qwen3vl-8b",
    "gemma3-4b": "gemma3-4b",
    "gemma3-12b": "gemma3-12b",
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
    support_set: str
    display_name: str


ROW_SPECS: List[RowSpec] = [
    RowSpec("raw", "small", "Raw Text"),
    RowSpec("raw", "large", "Raw Text"),
    RowSpec("bias", "small", "Adjust-Bias"),
    RowSpec("bias", "large", "Adjust-Bias"),
    RowSpec("vlm_fewshot_small", "small", "Few-shot"),
    RowSpec("vlm_fewshot_large", "large", "Few-shot"),
    RowSpec("lora_per_user_small", "small", "LoRA"),
    RowSpec("lora_per_user_large", "large", "LoRA"),
    RowSpec("direct_linear_llm_text_L15", "small", "Linear-Hidden"),
    RowSpec("direct_linear_llm_text_L15", "large", "Linear-Hidden"),
    RowSpec("direct_linear_giaa_gt_llm_text_L15", "small", "Linear-Hidden (GIAA)"),
    RowSpec("direct_linear_giaa_gt_llm_text_L15", "large", "Linear-Hidden (GIAA)"),
    RowSpec("hidden_attr_linear_llm_text_L15", "small", "Linear-Hidden (Reduce)"),
    RowSpec("hidden_attr_linear_llm_text_L15", "large", "Linear-Hidden (Reduce)"),
]


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

    row1 = ["Method", "Support"]
    for fam, i, j in spans:
        row1.append(f"\\multicolumn{{{j - i}}}{{c}}{{{latex_escape(fam)}}}")

    row2 = ["", ""] + [latex_escape(s) for s in sizes]
    return " & ".join(row1) + " \\\\", " & ".join(row2) + " \\\\"


def rows_sorted_for_table() -> List[RowSpec]:
    support_order = {"-": 0, "small": 1, "large": 2}
    return sorted(ROW_SPECS, key=lambda r: (support_order.get(r.support_set, 99), r.display_name, r.method))


def load_nan_count_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {
        "support_set",
        "method",
        "n_users_rho_nan",
        "frac_users_rho_nan",
        "mean_rho_fill_zero",
        "mean_rho_drop_nan",
    }
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns in {path}: {miss}")

    df = df.copy()
    df["support_set"] = df["support_set"].astype(str)
    df["method"] = df["method"].astype(str)
    for c in ["n_users_rho_nan", "frac_users_rho_nan", "mean_rho_fill_zero", "mean_rho_drop_nan"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def build_cell_maps(
    df_nan: pd.DataFrame,
    digits_count_frac: int,
    digits_rho: int,
) -> Tuple[Dict[Tuple[str, str], str], Dict[Tuple[str, str], str]]:
    nan_cell = {}
    rho_cell = {}

    for _, r in df_nan.iterrows():
        key = (str(r["method"]), str(r["support_set"]))
        n_nan = r["n_users_rho_nan"]
        frac_nan = r["frac_users_rho_nan"]
        rho_fill = r["mean_rho_fill_zero"]
        rho_drop = r["mean_rho_drop_nan"]

        if np.isfinite(n_nan) and np.isfinite(frac_nan):
            nan_cell[key] = f"{int(n_nan)} ({100.0 * float(frac_nan):.{digits_count_frac}f}%)"
        else:
            nan_cell[key] = "--"

        if np.isfinite(rho_fill) and np.isfinite(rho_drop):
            rho_cell[key] = f"{float(rho_fill):.{digits_rho}f} / {float(rho_drop):.{digits_rho}f}"
        else:
            rho_cell[key] = "--"

    return nan_cell, rho_cell


def build_nan_value_map(df_nan: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    value_map = {}
    for _, r in df_nan.iterrows():
        key = (str(r["method"]), str(r["support_set"]))
        value_map[key] = float(r["n_users_rho_nan"]) if np.isfinite(r["n_users_rho_nan"]) else float("nan")
    return value_map


def filter_rows_with_any_nan(
    rows: List[RowSpec],
    nan_value_maps: List[Dict[Tuple[str, str], float]],
) -> List[RowSpec]:
    filtered = []
    for row in rows:
        key = (row.method, row.support_set)
        has_nonzero = False
        for value_map in nan_value_maps:
            val = value_map.get(key, float("nan"))
            if np.isfinite(val) and val > 0:
                has_nonzero = True
                break
        if has_nonzero:
            filtered.append(row)
    return filtered


def render_latex_rows(
    rows: List[RowSpec],
    col_models: List[str],
    cell: Dict[Tuple[str, str, str], str],
) -> List[str]:
    lines: List[str] = []
    prev_disp = None
    prev_support = None
    for r in rows:
        if prev_support is not None and r.support_set != prev_support:
            lines.append("\\midrule")
            prev_disp = None
        prev_support = r.support_set

        method_cell = "" if prev_disp == r.display_name else r.display_name
        prev_disp = r.display_name
        support_cell = SUPPORT_DISPLAY.get(r.support_set, r.support_set)

        row_cells = [latex_escape(method_cell), latex_escape(support_cell)]
        for mk in col_models:
            row_cells.append(latex_escape(cell.get((r.method, r.support_set, mk), "--")))
        lines.append(" & ".join(row_cells) + " \\\\")
    return lines


def render_latex_table(
    rows: List[RowSpec],
    col_models: List[str],
    cell: Dict[Tuple[str, str, str], str],
    caption: str,
    label: str,
) -> str:
    h1, h2 = build_headers(col_models)

    col_spec = "l l " + " ".join(["c"] * len(col_models))
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\renewcommand{\\arraystretch}{1.15}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        h1,
        h2,
        "\\midrule",
    ]
    lines.extend(render_latex_rows(rows, col_models, cell))

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\end{table*}",
        ]
    )
    return "\n".join(lines)


def render_markdown_table(rows: List[RowSpec], col_models: List[str], cell: Dict[Tuple[str, str, str], str]) -> str:
    headers = ["Method", "Support"] + col_models
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]

    prev_disp = None
    prev_support = None
    for r in rows:
        if prev_support is not None and r.support_set != prev_support:
            lines.append("| " + " | ".join([""] * len(headers)) + " |")
            prev_disp = None
        prev_support = r.support_set

        method_cell = "" if prev_disp == r.display_name else r.display_name
        prev_disp = r.display_name
        support_cell = SUPPORT_DISPLAY.get(r.support_set, r.support_set)

        row = [method_cell, support_cell]
        for mk in col_models:
            row.append(cell.get((r.method, r.support_set, mk), "--"))
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def render_combined_latex_table(
    datasets: List[Tuple[str, str]],
    col_models: List[str],
    ds_to_rows: Dict[str, List[RowSpec]],
    ds_to_cell: Dict[str, Dict[Tuple[str, str, str], str]],
    caption: str,
    label: str,
) -> str:
    h1, h2 = build_headers(col_models)
    col_spec = "l l " + " ".join(["c"] * len(col_models))
    span = 2 + len(col_models)

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\small",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\renewcommand{\\arraystretch}{1.15}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        h1,
        h2,
        "\\midrule",
    ]

    for idx, (dataset_key, dataset_display) in enumerate(datasets):
        rows = ds_to_rows[dataset_key]
        lines.append(f"\\multicolumn{{{span}}}{{l}}{{\\textbf{{{latex_escape(dataset_display)}}}}} \\\\")
        lines.append("\\midrule")
        lines.extend(render_latex_rows(rows, col_models, ds_to_cell[dataset_key]))
        if idx != len(datasets) - 1:
            lines.append("\\midrule\\midrule")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root_dir",
        required=True,
        help="Root dir containing nan_count/{para,lapis}/*.csv (one file per model key).",
    )
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dataset_names", nargs="+", default=["para", "lapis"])
    ap.add_argument("--combined_datasets", nargs="+", default=["para:PARA", "lapis:LAPIS"])
    ap.add_argument("--digits_count_frac", type=int, default=1)
    ap.add_argument("--digits_rho", type=int, default=3)
    ap.add_argument("--markdown", action="store_true", help="Also output Markdown tables.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    combined_datasets: List[Tuple[str, str]] = []
    for s in args.combined_datasets:
        if ":" not in s:
            raise ValueError(f"--combined_datasets item must be like dir:DisplayName, got: {s}")
        dataset_key, display_name = s.split(":", 1)
        combined_datasets.append((dataset_key.strip(), display_name.strip()))

    ds_to_rows: Dict[str, List[RowSpec]] = {}
    ds_to_nan_cell: Dict[str, Dict[Tuple[str, str, str], str]] = {}
    ds_to_rho_cell: Dict[str, Dict[Tuple[str, str, str], str]] = {}
    ds_to_cols: Dict[str, List[str]] = {}

    for dataset in args.dataset_names:
        ds_dir = os.path.join(args.root_dir, dataset)
        if not os.path.isdir(ds_dir):
            print(f"[warn] skip missing dataset dir: {ds_dir}")
            continue

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
                model_key_to_df[mk] = load_nan_count_csv(p)
            except Exception as e:
                print(f"[warn] skip {p}: {e}")

        col_models = [m for m in MODEL_ORDER if m in model_key_to_df]
        if not col_models:
            print(f"[warn] no model nan_count csvs found under {ds_dir}")
            continue

        ds_to_cols[dataset] = col_models

        nan_cell: Dict[Tuple[str, str, str], str] = {}
        rho_cell: Dict[Tuple[str, str, str], str] = {}
        nan_value_maps: List[Dict[Tuple[str, str], float]] = []

        for mk in col_models:
            df_nan = model_key_to_df[mk]
            nan_value_maps.append(build_nan_value_map(df_nan))
            per_model_nan_cell, per_model_rho_cell = build_cell_maps(
                df_nan,
                digits_count_frac=args.digits_count_frac,
                digits_rho=args.digits_rho,
            )

            for r in ROW_SPECS:
                key = (r.method, r.support_set)
                nan_cell[(r.method, r.support_set, mk)] = per_model_nan_cell.get(key, "--")
                rho_cell[(r.method, r.support_set, mk)] = per_model_rho_cell.get(key, "--")

        visible_rows = filter_rows_with_any_nan(rows_sorted_for_table(), nan_value_maps)
        ds_to_rows[dataset] = visible_rows
        ds_to_nan_cell[dataset] = nan_cell
        ds_to_rho_cell[dataset] = rho_cell

        caption_nan = (
            f"Number of users with undefined Spearman $\\rho$ per method on {dataset.upper()}. "
            f"Each cell shows count and fraction among eligible users."
        )
        label_nan = f"tab:{sanitize(dataset)}_rho_nan_count"
        tex_nan = render_latex_table(visible_rows, col_models, nan_cell, caption_nan, label_nan)
        out_nan = os.path.join(args.out_dir, f"{dataset}_rho_nan_count.tex")
        with open(out_nan, "w", encoding="utf-8") as f:
            f.write(tex_nan)
        print(f"[save] {out_nan}")

        caption_rho = (
            f"Mean user-level Spearman $\\rho$ on {dataset.upper()}. "
            f"Each cell shows fill-0 mean / drop-NaN mean."
        )
        label_rho = f"tab:{sanitize(dataset)}_rho_fill_vs_drop"
        tex_rho = render_latex_table(visible_rows, col_models, rho_cell, caption_rho, label_rho)
        out_rho = os.path.join(args.out_dir, f"{dataset}_rho_fill_vs_drop.tex")
        with open(out_rho, "w", encoding="utf-8") as f:
            f.write(tex_rho)
        print(f"[save] {out_rho}")

        if args.markdown:
            md_nan = render_markdown_table(visible_rows, col_models, nan_cell)
            md_rho = render_markdown_table(visible_rows, col_models, rho_cell)
            with open(os.path.join(args.out_dir, f"{dataset}_rho_nan_count.md"), "w", encoding="utf-8") as f:
                f.write(md_nan)
            with open(os.path.join(args.out_dir, f"{dataset}_rho_fill_vs_drop.md"), "w", encoding="utf-8") as f:
                f.write(md_rho)
            print(f"[save] markdown tables for {dataset}")

    available_combined = [(k, disp) for k, disp in combined_datasets if k in ds_to_rows]
    if available_combined:
        common_models = [m for m in MODEL_ORDER if all(m in ds_to_cols[k] for k, _ in available_combined)]
        if common_models:
            caption_nan = (
                "Number of users with undefined Spearman $\\rho$ across datasets. "
                "Rows with zero NaN users for every model are omitted within each dataset block."
            )
            tex_nan_combined = render_combined_latex_table(
                available_combined,
                common_models,
                ds_to_rows,
                ds_to_nan_cell,
                caption_nan,
                "tab:combined_rho_nan_count",
            )
            out_nan_combined = os.path.join(args.out_dir, "combined_rho_nan_count.tex")
            with open(out_nan_combined, "w", encoding="utf-8") as f:
                f.write(tex_nan_combined)
            print(f"[save] {out_nan_combined}")

            caption_rho = (
                "Mean user-level Spearman $\\rho$ across datasets. "
                "Each cell shows fill-0 mean / drop-NaN mean, and rows with zero NaN users for every model are omitted within each dataset block."
            )
            tex_rho_combined = render_combined_latex_table(
                available_combined,
                common_models,
                ds_to_rows,
                ds_to_rho_cell,
                caption_rho,
                "tab:combined_rho_fill_vs_drop",
            )
            out_rho_combined = os.path.join(args.out_dir, "combined_rho_fill_vs_drop.tex")
            with open(out_rho_combined, "w", encoding="utf-8") as f:
                f.write(tex_rho_combined)
            print(f"[save] {out_rho_combined}")
        else:
            print("[warn] skipped combined tables because common model columns are empty")

    print("[done]")


if __name__ == "__main__":
    main()
