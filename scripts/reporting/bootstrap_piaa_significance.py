#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bootstrap-based significance testing for PIAA methods.

Input:
  - A directory containing PIAA prediction CSVs
    (same as scripts.piaa.eval_piaa_baselines expects), i.e. files with columns:

      user_id, image_path, model_id, support_set, method, giaa, piaa_pred, user_score

Pipeline:
  1. Load all valid baseline CSVs from --input_dir.
  2. Compute per-user metrics for each (model_id, support_set, method, user_id):
       - Spearman rho (piaa_pred vs user_score)
       - R^2 (sklearn.metrics.r2_score)
  3. For each model_id, and for each pair of method-combos
     (comboA, comboB), where combo = (support_set, method):
       - Collect users who have metrics for both A and B.
       - Let m_i^A, m_i^B (per-user metric), define Δ_i = m_i^B - m_i^A
       - Bootstrap over users:
           * resample users with replacement
           * compute mean(Δ_i) for each bootstrap sample
       - From the bootstrap distribution of mean Δ:
           * delta_mean:      mean of bootstrap means
           * ci_low, ci_high: percentile-based CI (e.g., 2.5% and 97.5%)
           * p_greater:       fraction of bootstrap samples where mean(Δ) > 0
  4. Save results to CSV.

Example:

  python -m scripts.reporting.bootstrap_piaa_significance \
    --input_dir runs/piaa_baseline_gemma3_small/ \
    --out_csv runs/bootstrap_significance_gemma3_small.csv \
    --metrics rho r2 \
    --n_bootstrap 2000
"""

import os
import re
import math
import argparse
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.metrics import r2_score


# ---------- Utils ----------

def sanitize(s: str) -> str:
    if s is None or not isinstance(s, str) or s.strip() == "":
        return "unknown"
    return re.sub(r"[^0-9A-Za-z._\\-]+", "_", s)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Spearman ρ と R² を計算。NaN を含むペアはドロップ。
    R² は scikit-learn の r2_score を使用。
    """
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return {"rho": 0.0, "r2": float("nan")}

    rho = spearmanr(y_true, y_pred).correlation
    if np.isnan(rho):
        rho = 0.0

    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = float("nan")

    return {"rho": float(rho), "r2": r2}


def load_from_dir(input_dir: str) -> pd.DataFrame:
    """
    指定ディレクトリ内の CSV をすべて読み込み，
    必要なカラムを持つものだけを結合して返す。
    """
    required = {
        "user_id",
        "image_path",
        "model_id",
        "support_set",
        "method",
        "giaa",
        "piaa_pred",
        "user_score",
    }
    dfs: List[pd.DataFrame] = []

    if not os.path.isdir(input_dir):
        raise RuntimeError(f"input_dir is not a directory: {input_dir}")

    for name in os.listdir(input_dir):
        if not name.lower().endswith(".csv"):
            continue
        path = os.path.join(input_dir, name)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[warn] failed to read {path}: {e}, skip")
            continue

        if not required.issubset(df.columns):
            # per_user_metrics.csv, summary_metrics.csv などはここで弾かれる
            print(f"[info] skip {path} (missing required baseline columns)")
            continue

        print(f"[info] loaded baseline CSV: {path} (rows={len(df)})")
        dfs.append(df)

    if not dfs:
        raise RuntimeError(f"No valid baseline CSVs found in directory: {input_dir}")

    df_all = pd.concat(dfs, ignore_index=True)
    return df_all


def compute_per_user_metrics(df_all: pd.DataFrame, min_items: int = 2) -> pd.DataFrame:
    """
    (model_id, support_set, method, user_id) ごとに per-user metrics を計算。
    min_items 未満のユーザはスキップ。
    """
    groups = df_all.groupby(["model_id", "support_set", "method", "user_id"])
    rows = []

    for (model_id, support_set, method, user_id), g in groups:
        y_true = g["user_score"].to_numpy(dtype=np.float32)
        y_pred = g["piaa_pred"].to_numpy(dtype=np.float32)

        if len(y_true) < min_items:
            continue

        m = _metrics(y_true, y_pred)
        rows.append(
            {
                "model_id": model_id,
                "support_set": support_set,
                "method": method,
                "user_id": user_id,
                "n_items": len(y_true),
                "rho": m["rho"],
                "r2": m["r2"],
            }
        )

    df_user = pd.DataFrame(rows)
    return df_user


# ---------- Bootstrap ----------

def bootstrap_pairwise(
    df_user: pd.DataFrame,
    metric: str,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> pd.DataFrame:
    """
    df_user: per-user metrics (model_id, support_set, method, user_id, rho, r2, ...)
    metric : "rho" or "r2"

    比較単位は (support_set, method) の組。すなわち:

      comboA = (support_set_A, method_A)
      comboB = (support_set_B, method_B)

    としたとき、各 model_id ごとに comboA vs comboB の性能差をブートストラップで評価する。
    """
    rng = np.random.RandomState(seed)
    results = []

    # 各 model_id ごとに処理
    for model_id in df_user["model_id"].unique():
        df_m = df_user[df_user["model_id"] == model_id].copy()
        if df_m.empty:
            continue

        # コンボキー: "support_set::method"
        df_m["combo"] = df_m["support_set"].astype(str) + "::" + df_m["method"].astype(str)

        combos = sorted(df_m["combo"].unique())
        if len(combos) < 2:
            continue

        # pivot: rows=user_id, columns=combo, values=metric
        pivot = df_m.pivot_table(
            index="user_id",
            columns="combo",
            values=metric,
            aggfunc="mean",
        )

        # ペアワイズに比較
        for i in range(len(combos)):
            for j in range(i + 1, len(combos)):
                cA = combos[i]
                cB = combos[j]

                if cA not in pivot.columns or cB not in pivot.columns:
                    continue

                sub = pivot[[cA, cB]].dropna()
                if sub.empty:
                    continue

                valsA = sub[cA].to_numpy(dtype=float)
                valsB = sub[cB].to_numpy(dtype=float)
                n_users = len(valsA)
                if n_users < 2:
                    continue

                diff = valsB - valsA  # Δ_i = B - A

                boot_means = []
                for _ in range(n_bootstrap):
                    idx = rng.randint(0, n_users, size=n_users)
                    boot_means.append(float(np.mean(diff[idx])))
                boot_means = np.array(boot_means, dtype=float)

                delta_mean = float(boot_means.mean())
                alpha = 1.0 - confidence
                ci_low = float(np.percentile(boot_means, 100 * (alpha / 2.0)))
                ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2.0)))
                p_greater = float(np.mean(boot_means > 0.0))

                # combo から (support_set, method) を復元
                support_a, method_a = cA.split("::", 1)
                support_b, method_b = cB.split("::", 1)

                results.append(
                    {
                        "model_id": model_id,
                        "metric": metric,
                        "support_set_a": support_a,
                        "method_a": method_a,
                        "support_set_b": support_b,
                        "method_b": method_b,
                        "delta_mean": delta_mean,   # B - A
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "p_greater": p_greater,     # P(mean(B-A)>0)
                        "n_users": n_users,
                        "n_bootstrap": n_bootstrap,
                    }
                )

    return pd.DataFrame(results)


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing PIAA baseline CSVs (from piaa_from_giaa_para.py).",
    )
    ap.add_argument(
        "--out_csv",
        required=True,
        help="Path to output CSV summarizing bootstrap significance results.",
    )
    ap.add_argument(
        "--metrics",
        nargs="+",
        default=["rho"],
        choices=["rho", "r2"],
        help="Which metrics to test (rho, r2 or both).",
    )
    ap.add_argument(
        "--min_items_per_user",
        type=int,
        default=2,
        help="Minimum number of items per user to compute metrics.",
    )
    ap.add_argument(
        "--n_bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap samples.",
    )
    ap.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for CI (e.g., 0.95 for 95%% CI).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap.",
    )
    args = ap.parse_args()

    # 1) Load baseline CSVs and compute per-user metrics
    print(f"[info] loading baseline CSVs from dir={args.input_dir} ...")
    df_all = load_from_dir(args.input_dir)
    print(f"[info] total rows = {len(df_all)}")

    print("[info] computing per-user metrics...")
    df_user = compute_per_user_metrics(df_all, min_items=args.min_items_per_user)
    print(f"[info] per-user rows = {len(df_user)}")

    # 2) Bootstrap for each metric
    all_boot = []
    for metric in args.metrics:
        print(f"[bootstrap] metric={metric} ...")
        df_boot = bootstrap_pairwise(
            df_user=df_user,
            metric=metric,
            n_bootstrap=args.n_bootstrap,
            confidence=args.confidence,
            seed=args.seed,
        )
        all_boot.append(df_boot)

    if all_boot:
        df_out = pd.concat(all_boot, ignore_index=True)
    else:
        df_out = pd.DataFrame()

    # 3) Save
    out_path = args.out_csv
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"[save] bootstrap results -> {out_path} (rows={len(df_out)})")


if __name__ == "__main__":
    main()
