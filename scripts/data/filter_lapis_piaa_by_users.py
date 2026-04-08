#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Filter PIAA experiment CSVs for LAPIS so that they only contain users
present in get_personalized_lapis_dataset(seed, dataset_dir).

Usage:
  python -m scripts.data.filter_lapis_piaa_by_users \
    --input_dir runs/piaa_results_lapis \
    --output_dir runs/piaa_results_lapis_filtered \
    --dataset_dir datasets/LAPIS \
    --seed 42

What it does:
  - Calls get_personalized_lapis_dataset(seed, dataset_dir) to obtain
    the set of valid user_ids.
  - Scans all .csv files in --input_dir.
  - For each CSV that contains a 'user_id' column, keeps only rows
    whose user_id is in that valid set.
  - Writes the filtered CSV with the same file name into --output_dir.

Files that don't have a 'user_id' column are skipped (e.g. summary CSVs).
"""

import os
import argparse
import pandas as pd
from tqdm import tqdm

from utils.lapis import get_personalized_lapis_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing original PIAA experiment CSVs for LAPIS.",
    )
    ap.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write filtered CSVs (only users in get_personalized_lapis_dataset).",
    )
    ap.add_argument(
        "--dataset_dir",
        default="datasets/LAPIS",
        help="Root of LAPIS dataset (where annotation/ and images/ exist).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for get_personalized_lapis_dataset (must match experiments).",
    )
    args = ap.parse_args()

    # 1) Personalized LAPIS から有効ユーザの集合を取得
    print(f"[info] loading personalized LAPIS users from seed={args.seed}, dataset_dir={args.dataset_dir}")
    personalized = get_personalized_lapis_dataset(seed=args.seed, dataset_dir=args.dataset_dir)
    valid_user_ids = set(personalized.keys())
    print(f"[info] num valid users = {len(valid_user_ids)}")

    # 2) 入出力ディレクトリの準備
    if not os.path.isdir(args.input_dir):
        raise RuntimeError(f"input_dir is not a directory: {args.input_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    # 3) input_dir 内の CSV をすべて処理
    files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(".csv")]
    print(f"[info] found {len(files)} CSV files in {args.input_dir}")

    for name in tqdm(files, desc="Filtering CSVs"):
        in_path = os.path.join(args.input_dir, name)
        try:
            df = pd.read_csv(in_path)
        except Exception as e:
            print(f"[warn] failed to read {in_path}: {e}, skip")
            continue

        if "user_id" not in df.columns:
            # 集計系 (summary_metrics.csv 等) は user_id を持たないのでスキップ
            print(f"[info] skip {in_path} (no 'user_id' column)")
            continue

        before = len(df)
        df_f = df[df["user_id"].isin(valid_user_ids)].copy()
        after = len(df_f)

        out_path = os.path.join(args.output_dir, name)
        df_f.to_csv(out_path, index=False)
        print(f"[info] {name}: {before} -> {after} rows (filtered)")

    print("[done] filtering complete.")


if __name__ == "__main__":
    main()
