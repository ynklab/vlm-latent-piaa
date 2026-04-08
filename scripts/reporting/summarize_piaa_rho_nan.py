#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summarize per-file NaN Spearman rho behavior for PIAA result CSVs.

Input:
  - A directory containing CSVs with columns:
      user_id, image_path, model_id, support_set, method, giaa, piaa_pred, user_score

For each valid CSV file in --input_dir:
  1. Group rows by user_id.
  2. Keep users with at least --min_items_per_user rows, matching
     bootstrap_piaa_significance.py.
  3. Compute raw Spearman rho per user without replacing NaN.
  4. Report:
       - how many eligible users have rho = NaN
       - mean rho if NaN is filled with 0
       - mean rho if NaN users are excluded from the denominator

Example:
  python -m scripts.reporting.summarize_piaa_rho_nan \
    --input_dir outputs/piaa/para/gemma3-4b \
    --out_csv outputs/viz/rho_nan_summary/para_gemma3-4b.csv
"""

import argparse
import os
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


REQUIRED_COLS = {
    "user_id",
    "image_path",
    "model_id",
    "support_set",
    "method",
    "giaa",
    "piaa_pred",
    "user_score",
}


def summarize_unique(values: pd.Series) -> str:
    uniq = sorted({str(v) for v in values.dropna().tolist()})
    if not uniq:
        return ""
    return uniq[0] if len(uniq) == 1 else " | ".join(uniq)


def compute_raw_rho(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return float("nan")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho = spearmanr(y_true, y_pred).correlation
    return float(rho)


def summarize_file(path: Path, min_items_per_user: int) -> Dict[str, object]:
    df = pd.read_csv(path)
    groups = df.groupby("user_id")

    rho_values: List[float] = []
    skipped_users = 0

    for _, g in groups:
        y_true = g["user_score"].to_numpy(dtype=np.float32)
        y_pred = g["piaa_pred"].to_numpy(dtype=np.float32)

        if len(y_true) < min_items_per_user:
            skipped_users += 1
            continue

        rho_values.append(compute_raw_rho(y_true, y_pred))

    rho_arr = np.array(rho_values, dtype=float)
    nan_mask = np.isnan(rho_arr)
    n_users_eligible = int(rho_arr.size)
    n_users_rho_nan = int(nan_mask.sum())
    n_users_rho_non_nan = int((~nan_mask).sum())

    if n_users_eligible == 0:
        mean_rho_fill_zero = float("nan")
    else:
        mean_rho_fill_zero = float(np.nan_to_num(rho_arr, nan=0.0).mean())

    if n_users_rho_non_nan == 0:
        mean_rho_drop_nan = float("nan")
    else:
        mean_rho_drop_nan = float(rho_arr[~nan_mask].mean())

    return {
        "file_name": path.name,
        "file_path": str(path),
        "n_rows": int(len(df)),
        "n_users_total": int(df["user_id"].nunique()),
        "n_users_skipped_min_items": int(skipped_users),
        "n_users_eligible": n_users_eligible,
        "n_users_rho_nan": n_users_rho_nan,
        "n_users_rho_non_nan": n_users_rho_non_nan,
        "frac_users_rho_nan": (
            float(n_users_rho_nan / n_users_eligible) if n_users_eligible > 0 else float("nan")
        ),
        "mean_rho_fill_zero": mean_rho_fill_zero,
        "mean_rho_drop_nan": mean_rho_drop_nan,
        "mean_rho_delta_fill_zero_minus_drop_nan": (
            float(mean_rho_fill_zero - mean_rho_drop_nan)
            if np.isfinite(mean_rho_fill_zero) and np.isfinite(mean_rho_drop_nan)
            else float("nan")
        ),
        "model_id": summarize_unique(df["model_id"]),
        "support_set": summarize_unique(df["support_set"]),
        "method": summarize_unique(df["method"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing PIAA result CSVs.",
    )
    ap.add_argument(
        "--out_csv",
        default="",
        help="Optional path to save the per-file summary CSV.",
    )
    ap.add_argument(
        "--min_items_per_user",
        type=int,
        default=2,
        help="Minimum number of rows per user to include, matching bootstrap_piaa_significance.py.",
    )
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise RuntimeError(f"input_dir is not a directory: {input_dir}")

    rows: List[Dict[str, object]] = []
    for path in sorted(input_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".csv":
            continue

        try:
            df_head = pd.read_csv(path, nrows=5)
        except Exception as e:
            print(f"[warn] failed to read {path}: {e}, skip")
            continue

        if not REQUIRED_COLS.issubset(df_head.columns):
            print(f"[info] skip {path} (missing required baseline columns)")
            continue

        print(f"[info] summarize {path}")
        rows.append(summarize_file(path, min_items_per_user=args.min_items_per_user))

    if not rows:
        raise RuntimeError(f"No valid baseline CSVs found in directory: {input_dir}")

    df_out = pd.DataFrame(rows).sort_values(
        by=["support_set", "method", "file_name"], na_position="last"
    )

    if args.out_csv:
        out_path = Path(args.out_csv)
        os.makedirs(out_path.parent or Path("."), exist_ok=True)
        df_out.to_csv(out_path, index=False)
        print(f"[save] {out_path} (rows={len(df_out)})")

    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df_out.to_string(index=False))


if __name__ == "__main__":
    main()
