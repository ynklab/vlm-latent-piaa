#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build simple PIAA baselines from pre-computed GIAA predictions
for PARA / LAPIS personalized datasets.

Baselines (per user u, per image i):

  Baseline 1 (raw):
    PIAA_pred(u,i) = GIAA_pred(i)

  Baseline 2 (bias-corrected):
    On support set S_u:
      bias_u = mean_{j in S_u} ( GIAA_pred(j) - user_score_j )
    On test set T_u:
      PIAA_pred(u,i) = GIAA_pred(i) - bias_u

Input:
  - GIAA prediction CSV (from vlm_giaa.py):
      model_id, dataset, split, image_path, giaa, raw_output

  - Personalized dataset:
      dataset="para"  -> utils.para.get_personalized_para_dataset(seed, dataset_dir)
      dataset="lapis" -> utils.lapis.get_personalized_lapis_dataset(seed, dataset_dir)

Output:
  - <out_prefix>_raw.csv
  - <out_prefix>_bias.csv

Each CSV has rows:
  user_id, image_path, model_id, support_set, method, giaa, piaa_pred, user_score
"""

import os
import csv
import math
import argparse
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from utils.para import get_personalized_para_dataset
from utils.lapis import get_personalized_lapis_dataset


# ---------- GIAA loader ----------

def load_giaa_map(
    giaa_csv: str,
    model_id_filter: str | None = None,
) -> Tuple[Dict[str, float], str]:
    """
    GIAA CSV を読み込み，image_path -> giaa の辞書を作る。

    期待する列 (vlm_giaa.py の出力):
      model_id, dataset, split, image_path, giaa, raw_output

    model_id_filter が指定されていない場合:
      - CSV 内に 1 種類の model_id しか無いことを要求
    """
    image_to_giaa: Dict[str, float] = {}
    model_ids: set[str] = set()

    with open(giaa_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not {"model_id", "image_path", "giaa"}.issubset(reader.fieldnames or []):
            raise ValueError("GIAA CSV must contain columns: model_id, image_path, giaa")

        for row in reader:
            mid = row["model_id"]
            path = row["image_path"]
            try:
                score = float(row["giaa"])
            except Exception:
                score = math.nan
            model_ids.add(mid)

            if model_id_filter is not None and mid != model_id_filter:
                continue

            image_to_giaa[path] = score

    if model_id_filter is not None:
        model_id_used = model_id_filter
    else:
        if len(model_ids) != 1:
            raise ValueError(
                f"GIAA CSV contains multiple model_ids: {model_ids}. "
                f"Please specify --model_id_filter."
            )
        model_id_used = next(iter(model_ids))

    return image_to_giaa, model_id_used


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        required=True,
        choices=["para", "lapis"],
        help="Dataset to use for PIAA baselines (para or lapis).",
    )
    ap.add_argument(
        "--giaa_csv",
        required=True,
        help="Path to GIAA prediction CSV (from vlm_giaa.py).",
    )
    ap.add_argument(
        "--dataset_dir",
        default=None,
        help="Dataset root directory. If None, uses datasets/PARA or datasets/LAPIS.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for personalized split (must match when generating splits).",
    )
    ap.add_argument(
        "--support_set",
        default="small",
        choices=["small", "large"],
        help="Which support set to use for bias estimation: small(10) or large(100).",
    )
    ap.add_argument(
        "--model_id_filter",
        default=None,
        help="If GIAA CSV contains multiple model_ids, specify which model_id to use.",
    )
    ap.add_argument(
        "--out_prefix",
        required=True,
        help="Output prefix. Two files will be produced: <prefix>_raw.csv, <prefix>_bias.csv",
    )
    args = ap.parse_args()

    # dataset_dir デフォルト
    if args.dataset_dir is None:
        if args.dataset == "para":
            args.dataset_dir = "datasets/PARA"
        else:
            args.dataset_dir = "datasets/LAPIS"

    # 1) GIAA 読み込み
    image_to_giaa, model_id_used = load_giaa_map(args.giaa_csv, args.model_id_filter)
    print(f"[info] using model_id={model_id_used} with {len(image_to_giaa)} GIAA entries")

    # 2) Personalized データ読み込み
    print(f"[info] loading personalized {args.dataset.upper()} dataset...")
    if args.dataset == "para":
        personalized_data = get_personalized_para_dataset(seed=args.seed, dataset_dir=args.dataset_dir)
    else:
        personalized_data = get_personalized_lapis_dataset(seed=args.seed, dataset_dir=args.dataset_dir)
    print(f"[info] num users in personalized dataset: {len(personalized_data)}")

    raw_rows: List[dict] = []
    bias_rows: List[dict] = []

    for user_id, pdata in tqdm(personalized_data.items(), desc="Users"):
        # support/test の取り出し
        if args.support_set == "small":
            support_items = pdata.support_small
        else:
            support_items = pdata.support_large
        test_items = pdata.test

        # --- Baseline 1: raw (= GIAAそのまま) ---
        for item in test_items:
            path = item.image_path
            # If GIAA is missing, raise a Warning and use nan
            if path not in image_to_giaa:
                print(f"[warning] GIAA missing for image_path={path}, user_id={user_id}")
            giaa = image_to_giaa.get(path, math.nan)
            raw_rows.append(
                {
                    "user_id": user_id,
                    "image_path": path,
                    "model_id": model_id_used,
                    "support_set": args.support_set,
                    "method": "raw",
                    "giaa": giaa,
                    "piaa_pred": giaa,
                    "user_score": float(item.score),
                }
            )

        # --- Baseline 2: bias-corrected ---
        diffs = []
        for item in support_items:
            path = item.image_path
            giaa = image_to_giaa.get(path, math.nan)
            if math.isnan(giaa):
                continue
            diffs.append(giaa - float(item.score))
        if len(diffs) == 0:
            bias = 0.0
        else:
            bias = float(np.mean(diffs))

        for item in test_items:
            path = item.image_path
            giaa = image_to_giaa.get(path, math.nan)
            if math.isnan(giaa):
                piaa = math.nan
            else:
                piaa = giaa - bias
            bias_rows.append(
                {
                    "user_id": user_id,
                    "image_path": path,
                    "model_id": model_id_used,
                    "support_set": args.support_set,
                    "method": "bias",
                    "giaa": giaa,
                    "piaa_pred": piaa,
                    "user_score": float(item.score),
                }
            )

    # 3) CSV 保存
    def _write_csv(path: str, rows: List[dict]):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fieldnames = [
            "user_id",
            "image_path",
            "model_id",
            "support_set",
            "method",
            "giaa",
            "piaa_pred",
            "user_score",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"[save] {path} (rows={len(rows)})")

    raw_path = f"{args.out_prefix}_raw.csv"
    bias_path = f"{args.out_prefix}_bias.csv"

    _write_csv(raw_path, raw_rows)
    _write_csv(bias_path, bias_rows)

    print("[done]")


if __name__ == "__main__":
    main()