#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
From pre-computed GIAA predictions on PARA images, build simple PIAA baselines:

Baseline 1 (raw):
  PIAA_pred(user, image) = GIAA_pred(image)

Baseline 2 (bias-corrected):
  For each user u:
    bias_u = mean_{i in support}( GIAA_pred(image_i) - user_score_i )
    PIAA_pred(user, image_j in test) = GIAA_pred(image_j) - bias_u

入力:
  - GIAA 予測結果 CSV (e.g. from vlm_giaa_para.py)
      columns: model_id, split, image_path, giaa, raw_output

  - PARA データセット (utils/para.py)
      get_personalized_para_dataset(seed, dataset_dir) を用いて
      user ごとの support_small / support_large / test を取得

出力:
  - <out_prefix>_raw.csv
      columns: user_id, image_path, model_id, support_set, giaa, piaa_pred, user_score, method
      (method="raw")

  - <out_prefix>_bias.csv
      columns: user_id, image_path, model_id, support_set, giaa, piaa_pred, user_score, method
      (method="bias")

例:

  python piaa_from_giaa_para.py \
    --giaa_csv runs/giaa_gemma3_4b_para.csv \
    --dataset_dir datasets/PARA \
    --support_set small \
    --out_prefix runs/piaa_baseline_gemma3_small

  python piaa_from_giaa_para.py \
    --giaa_csv runs/giaa_qwen3vl2b_para.csv \
    --dataset_dir datasets/PARA \
    --support_set large \
    --out_prefix runs/piaa_baseline_qwen3_large
"""

import os
import csv
import math
import argparse
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from utils.para import get_personalized_para_dataset


def load_giaa_map(
    giaa_csv: str,
    model_id_filter: str | None = None,
) -> Tuple[Dict[str, float], str]:
    """
    GIAA CSV を読み込み，image_path -> giaa の辞書を作る。
    CSVフォーマットは vlm_giaa_para.py の出力を想定:

      model_id, split, image_path, giaa, raw_output

    model_id_filter が指定されていなければ:
      - CSV 内に 1 種類の model_id しか無い場合はそれを使う
      - 複数モデルが混ざっている場合はエラー

    戻り値:
      (image_to_giaa, model_id_used)
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
            image_to_giaa.setdefault(mid, {})  # temporary if we wanted per-model dict

    # 修正: 上の実装は per-model dict を想定しているが，シンプルにするため書き直す
    image_to_giaa = {}
    model_ids = set()
    with open(giaa_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = row["model_id"]
            path = row["image_path"]
            try:
                score = float(row["giaa"])
            except Exception:
                score = math.nan
            model_ids.add(mid)
            # モデルフィルタがある場合はそれ以外をスキップ
            if model_id_filter is not None and mid != model_id_filter:
                continue
            if model_id_filter is None and len(model_ids) > 1:
                # 複数モデル混在でフィルタなしの場合はエラー
                raise ValueError(
                    f"GIAA CSV contains multiple model_ids: {model_ids}. "
                    f"Please specify --model_id_filter."
                )
            image_to_giaa[path] = score

    if model_id_filter is not None:
        model_id_used = model_id_filter
    else:
        # CSV 内に 1 種類しかないはず
        if len(model_ids) != 1:
            raise ValueError(f"Unexpected number of model_ids in CSV: {model_ids}")
        model_id_used = next(iter(model_ids))

    return image_to_giaa, model_id_used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--giaa_csv",
        required=True,
        help="Path to GIAA prediction CSV (from vlm_giaa_para.py)",
    )
    ap.add_argument(
        "--dataset_dir",
        default="datasets/PARA",
        help="Path to PARA dataset root",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used in get_personalized_para_dataset (must match).",
    )
    ap.add_argument(
        "--support_set",
        default="small",
        choices=["small", "large"],
        help="Which support set to use for bias estimation: small(10) or large(100)",
    )
    ap.add_argument(
        "--model_id_filter",
        default=None,
        help="If GIAA CSV contains multiple models, specify which model_id to use.",
    )
    ap.add_argument(
        "--out_prefix",
        required=True,
        help="Output prefix. Two files will be produced: <prefix>_raw.csv, <prefix>_bias.csv",
    )
    args = ap.parse_args()

    # 1) GIAA 読み込み (image_path -> giaa)
    image_to_giaa, model_id_used = load_giaa_map(args.giaa_csv, args.model_id_filter)
    print(f"[info] using model_id={model_id_used} with {len(image_to_giaa)} GIAA entries")

    # 2) Personalized PARA 読み込み
    print("[info] loading personalized PARA dataset...")
    personalized_data = get_personalized_para_dataset(seed=args.seed, dataset_dir=args.dataset_dir)
    print(f"[info] num users in personalized dataset: {len(personalized_data)}")

    # 3) ベースライン予測を構築
    raw_rows: List[dict] = []
    bias_rows: List[dict] = []

    for user_id, pdata in tqdm(personalized_data.items(), desc="Users"):
        # サポートセットの選択
        if args.support_set == "small":
            support_items = pdata.support_small
        else:
            support_items = pdata.support_large
        test_items = pdata.test

        # --- Baseline1: raw (= giaaそのまま) ---
        # test セットについて: PIAA_raw = GIAA(image)
        for item in test_items:
            giaa = image_to_giaa.get(item.image_path, math.nan)
            raw_rows.append(
                {
                    "user_id": user_id,
                    "image_path": item.image_path,
                    "model_id": model_id_used,
                    "support_set": args.support_set,
                    "method": "raw",
                    "giaa": giaa,
                    "piaa_pred": giaa,
                    "user_score": item.score,
                }
            )

        # --- Baseline2: bias-corrected ---
        # support上で GIAA - user_score の平均を求める
        diffs = []
        for item in support_items:
            giaa = image_to_giaa.get(item.image_path, math.nan)
            if math.isnan(giaa):
                continue
            diffs.append(giaa - item.score)
        if len(diffs) == 0:
            bias = 0.0
        else:
            bias = float(np.mean(diffs))

        # test セットの予測から bias を引く
        for item in test_items:
            giaa = image_to_giaa.get(item.image_path, math.nan)
            if math.isnan(giaa):
                piaa = math.nan
            else:
                piaa = giaa - bias
            bias_rows.append(
                {
                    "user_id": user_id,
                    "image_path": item.image_path,
                    "model_id": model_id_used,
                    "support_set": args.support_set,
                    "method": "bias",
                    "giaa": giaa,
                    "piaa_pred": piaa,
                    "user_score": item.score,
                }
            )

    # 4) CSV 保存
    def _write_csv(path: str, rows: List[dict]):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fieldnames = ["user_id", "image_path", "model_id", "support_set", "method",
                      "giaa", "piaa_pred", "user_score"]
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