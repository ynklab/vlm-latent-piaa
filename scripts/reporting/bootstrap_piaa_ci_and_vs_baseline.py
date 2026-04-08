#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bootstrap CI for each method + bootstrap significance vs a fixed baseline combo.

Input:
  - input_dir: directory containing PIAA result CSVs with columns:
      user_id, image_path, model_id, support_set, method, giaa, piaa_pred, user_score

Process:
  1) Load all valid CSVs from input_dir
  2) Compute per-user metrics for each (model_id, support_set, method, user_id):
        rho: Spearman(user_score, piaa_pred)
        r2 : sklearn r2_score(user_score, piaa_pred)
  3) For each model_id and each combo=(support_set, method):
        A) Bootstrap CI of mean(metric) over users:
            - resample users with replacement
            - compute mean metric
            - report mean, CI, and bootstrap std
        B) Bootstrap paired difference vs baseline combo (support_set_baseline, method_baseline):
            - use users that have BOTH combo and baseline
            - diff_i = metric_i(combo) - metric_i(baseline)
            - bootstrap mean(diff_i)
            - report delta_mean, CI, p_greater = P(mean(diff)>0)

Output CSV (one row per model_id × combo × metric):
  model_id, metric, support_set, method,
  n_users,
  mean, ci_low, ci_high, boot_std,
  baseline_support_set, baseline_method,
  n_users_paired,
  delta_mean, delta_ci_low, delta_ci_high, p_greater
"""

import os
import re
import argparse
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score


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


def sanitize(s: str) -> str:
    if s is None:
        return "unknown"
    return re.sub(r"[^0-9A-Za-z._\\-]+", "_", str(s))


def load_from_dir(input_dir: str) -> pd.DataFrame:
    if not os.path.isdir(input_dir):
        raise RuntimeError(f"input_dir is not a directory: {input_dir}")

    dfs = []
    for name in sorted(os.listdir(input_dir)):
        if not name.lower().endswith(".csv"):
            continue
        path = os.path.join(input_dir, name)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[warn] failed to read {path}: {e}, skip")
            continue
        if not REQUIRED_COLS.issubset(df.columns):
            print(f"[info] skip {path} (missing required baseline columns)")
            continue
        dfs.append(df)

    if not dfs:
        raise RuntimeError(f"No valid baseline CSVs found in {input_dir}")

    df_all = pd.concat(dfs, ignore_index=True)
    return df_all


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size < 2:
        return {"rho": float("nan"), "r2": float("nan")}

    rho = spearmanr(y_true, y_pred).correlation  # keep NaN as-is
    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = float("nan")

    return {"rho": float(rho), "r2": r2}


def compute_per_user_metrics(df_all: pd.DataFrame, min_items: int = 2) -> pd.DataFrame:
    groups = df_all.groupby(["model_id", "support_set", "method", "user_id"])
    rows = []
    for (model_id, support_set, method, user_id), g in groups:
        y_true = g["user_score"].to_numpy(dtype=np.float32)
        y_pred = g["piaa_pred"].to_numpy(dtype=np.float32)
        if len(y_true) < min_items:
            continue

        m = _metrics(y_true, y_pred)
        rows.append({
            "model_id": model_id,
            "support_set": str(support_set),
            "method": str(method),
            "user_id": str(user_id),
            "n_items": int(len(y_true)),
            "rho": m["rho"],
            "r2": m["r2"],
        })
    return pd.DataFrame(rows)


def bootstrap_mean_ci(
    values: np.ndarray,
    n_bootstrap: int,
    confidence: float,
    rng: np.random.RandomState,
) -> Tuple[float, float, float, float]:
    """
    Bootstrap CI of mean(values). Returns:
      mean_est, ci_low, ci_high, boot_std
    """
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    n = values.size
    boot_means = np.empty((n_bootstrap,), dtype=float)
    for b in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_means[b] = float(np.mean(values[idx]))
    alpha = 1.0 - confidence
    ci_low = float(np.percentile(boot_means, 100 * (alpha / 2.0)))
    ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2.0)))
    return float(np.mean(values)), ci_low, ci_high, float(np.std(boot_means))


def bootstrap_delta_vs_baseline(
    vals_method: np.ndarray,
    vals_base: np.ndarray,
    n_bootstrap: int,
    confidence: float,
    rng: np.random.RandomState,
) -> Tuple[float, float, float, float]:
    """
    Paired bootstrap for mean(delta)=mean(method-base) over users.
    Returns:
      delta_mean, ci_low, ci_high, p_greater
    """
    mask = np.isfinite(vals_method) & np.isfinite(vals_base)
    a = vals_method[mask]
    b = vals_base[mask]
    if a.size < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    diff = a - b
    n = diff.size
    boot_means = np.empty((n_bootstrap,), dtype=float)
    for k in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_means[k] = float(np.mean(diff[idx]))
    alpha = 1.0 - confidence
    ci_low = float(np.percentile(boot_means, 100 * (alpha / 2.0)))
    ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2.0)))
    p_greater = float(np.mean(boot_means > 0.0))
    return float(np.mean(diff)), ci_low, ci_high, p_greater


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--metrics", nargs="+", default=["rho"], choices=["rho", "r2"])
    ap.add_argument("--baseline_method", required=True, help="e.g., direct_linear_llm_text_L15")
    ap.add_argument("--baseline_support_set", required=True, help="e.g., large")
    ap.add_argument("--min_items_per_user", type=int, default=2)
    ap.add_argument("--n_bootstrap", type=int, default=2000)
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)

    print(f"[info] loading CSVs from {args.input_dir} ...")
    df_all = load_from_dir(args.input_dir)
    print(f"[info] total rows={len(df_all)}")

    print("[info] computing per-user metrics ...")
    df_user = compute_per_user_metrics(df_all, min_items=args.min_items_per_user)
    print(f"[info] per-user rows={len(df_user)}")

    if df_user.empty:
        raise RuntimeError("No per-user metrics computed. Check input_dir / columns / min_items_per_user.")

    out_rows = []

    baseline_support = str(args.baseline_support_set)
    baseline_method = str(args.baseline_method)

    for model_id in sorted(df_user["model_id"].unique()):
        df_m = df_user[df_user["model_id"] == model_id].copy()
        df_m["combo"] = df_m["support_set"] + "::" + df_m["method"]

        baseline_combo = baseline_support + "::" + baseline_method
        if baseline_combo not in set(df_m["combo"].unique()):
            print(f"[warn] model_id={model_id}: baseline combo not found: {baseline_combo} (skip baseline comparison)")
            baseline_pivot = None
        else:
            baseline_pivot = df_m[df_m["combo"] == baseline_combo].set_index("user_id")

        combos = sorted(df_m["combo"].unique())

        for combo in combos:
            support_set, method = combo.split("::", 1)
            df_c = df_m[df_m["combo"] == combo].set_index("user_id")

            for metric in args.metrics:
                vals = df_c[metric].to_numpy(dtype=float)
                # CI for this method
                mean_est, ci_low, ci_high, boot_std = bootstrap_mean_ci(
                    values=vals,
                    n_bootstrap=args.n_bootstrap,
                    confidence=args.confidence,
                    rng=rng,
                )
                n_users = int(np.isfinite(vals).sum())

                # delta vs baseline
                if baseline_pivot is None or metric not in baseline_pivot.columns:
                    delta_mean = delta_ci_low = delta_ci_high = p_greater = float("nan")
                    n_users_paired = 0
                else:
                    # align by common users
                    common = df_c.index.intersection(baseline_pivot.index)
                    if len(common) < 2:
                        delta_mean = delta_ci_low = delta_ci_high = p_greater = float("nan")
                        n_users_paired = int(len(common))
                    else:
                        a = df_c.loc[common, metric].to_numpy(dtype=float)
                        b = baseline_pivot.loc[common, metric].to_numpy(dtype=float)
                        delta_mean, delta_ci_low, delta_ci_high, p_greater = bootstrap_delta_vs_baseline(
                            vals_method=a,
                            vals_base=b,
                            n_bootstrap=args.n_bootstrap,
                            confidence=args.confidence,
                            rng=rng,
                        )
                        n_users_paired = int(np.isfinite(a).sum() if np.isfinite(a).sum() < np.isfinite(b).sum() else np.isfinite(b).sum())

                out_rows.append({
                    "model_id": model_id,
                    "metric": metric,
                    "support_set": support_set,
                    "method": method,
                    "n_users": n_users,
                    "mean": mean_est,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "boot_std": boot_std,
                    "baseline_support_set": baseline_support,
                    "baseline_method": baseline_method,
                    "n_users_paired": n_users_paired,
                    "delta_mean": delta_mean,
                    "delta_ci_low": delta_ci_low,
                    "delta_ci_high": delta_ci_high,
                    "p_greater": p_greater,
                })

    df_out = pd.DataFrame(out_rows)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df_out.to_csv(args.out_csv, index=False)
    print(f"[save] {args.out_csv} (rows={len(df_out)})")


if __name__ == "__main__":
    main()
