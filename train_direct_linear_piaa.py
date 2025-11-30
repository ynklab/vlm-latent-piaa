#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train per-user direct linear models (Ridge) for PIAA on PARA/LAPIS using mm_embed features.

For each user u:
  - We have support set S_u (support_small or support_large) and test set T_u
    from get_personalized_*_dataset.
  - For each image i:
      feature z_i    : mm_embed feature from a chosen source/layer.
      user_score_i   : user-specific aesthetic score (PIAA).
      giaa_gt_i      : dataset-level GIAA mean (optional, if available).
  - Depending on --target_score:

    target_score = "piaa"   : use user_score_i as ground truth.
    target_score = "giaa_gt": use dataset-level GIAA mean as ground truth.

    We train a Ridge regression:
      target_score ~ z

  - For evaluation and output, we always log:
      user_score_i (PIAA)
      piaa_pred = model(z_i)   # predicted target_score (piaa or giaa_gt)

Outputs a CSV with one row per user × test image:
  user_id, image_path, model_id, support_set, method, giaa, piaa_pred, user_score

method 名:
  - target_score = piaa   : direct_linear_<source>_L<layer>
  - target_score = giaa_gt: direct_linear_giaa_gt_<source>_L<layer>
"""

import os
import csv
import math
import argparse
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from utils.para import get_personalized_para_dataset, get_para_dataset
from utils.lapis import get_personalized_lapis_dataset, get_lapis_dataset
from utils.mm_embed import load_mm_model, build_inputs, extract_all_pools


# ---------- Prompt (for mm_embed features) ----------

AESTHETIC_ATTRS_FOR_PROMPT = [
    "BalancingElements", "ColorHarmony", "Content", "DoF",
    "Light", "MotionBlur", "Object", "Repetition",
    "RuleOfThirds", "Symmetry", "VividColor",
]

def make_prompt(mode: str) -> str:
    if mode == "base":
        return "Assess the aesthetics of this image."
    elif mode == "format":
        return (
            "Assess the overall aesthetic quality of this image. "
            "Please rate it on a scale from 1 to 5. "
            "Output only the numeric score, and do not output any other text."
        )
    elif mode == "attributes":
        attrs = ", ".join(AESTHETIC_ATTRS_FOR_PROMPT)
        return (
            "Assess the aesthetics of this image with respect to the following attributes: "
            f"{attrs}. "
            "You do not need to output the attributes explicitly; just use them as internal criteria."
        )
    elif mode == "unrelated":
        return "Describe the weather today in one sentence."
    else:
        raise ValueError(f"Unknown prompt_mode: {mode}")


# ---------- Feature helper ----------

def extract_feature_vector(
    pools,
    source: str,
    layer_idx: int,
) -> np.ndarray:
    """
    Given AllPools from mm_embed.extract_all_pools and a source/layer,
    return a 1D numpy feature vector.
    """
    if source == "llm_text":
        vec = pools.llm_text[layer_idx]
    elif source == "llm_visual":
        vec = pools.llm_visual[layer_idx]
    elif source == "llm_text_tail":
        vec = pools.llm_text_tail[layer_idx]
    elif source == "vision":
        if pools.vision_layers is None:
            raise RuntimeError("vision_layers is None; vision source not available for this model.")
        vec = pools.vision_layers[layer_idx]
    elif source == "bridge_text":
        vec = pools.bridge_text[0]
    elif source == "bridge_visual":
        vec = pools.bridge_visual[0]
    else:
        raise ValueError(f"Unknown feature_source: {source}")

    return vec.astype(np.float32)


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        required=True,
        choices=["para", "lapis"],
        help="Dataset to use (para or lapis).",
    )
    ap.add_argument(
        "--dataset_dir",
        default=None,
        help="Dataset root directory. If None, uses datasets/PARA or datasets/LAPIS.",
    )
    ap.add_argument(
        "--model_id",
        required=True,
        help="Multimodal model id for mm_embed (e.g. Qwen/Qwen3-VL-2B-Instruct, google/gemma-3-4b-it).",
    )
    ap.add_argument(
        "--feature_source",
        required=True,
        choices=[
            "llm_text",
            "llm_visual",
            "llm_text_tail",
            "vision",
            "bridge_text",
            "bridge_visual",
        ],
        help="Which feature source from mm_embed.AllPools to use.",
    )
    ap.add_argument(
        "--feature_layer",
        type=int,
        required=True,
        help="Layer index (0-based) for the chosen feature_source.",
    )
    ap.add_argument(
        "--support_set",
        default="small",
        choices=["small", "large"],
        help="Which support set from personalized dataset to use.",
    )
    ap.add_argument(
        "--prompt_mode",
        default="base",
        choices=["base", "format", "attributes", "unrelated"],
        help="Prompt preset used when extracting features with mm_embed.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for personalized split (must match when generating splits).",
    )
    ap.add_argument(
        "--target_score",
        default="piaa",
        choices=["piaa", "giaa_gt"],
        help=(
            "Which score to use as ground truth for direct regression: "
            "'piaa' (user_score) or 'giaa_gt' (dataset-level GIAA mean)."
        ),
    )
    ap.add_argument(
        "--quick",
        type=int,
        default=None,
        help="If set, limit to at most N users (for debugging). Use 1 for a single-user check.",
    )
    ap.add_argument(
        "--out_csv",
        required=True,
        help="Path to output CSV for per-user test predictions.",
    )
    args = ap.parse_args()

    # dataset_dir デフォルト
    if args.dataset_dir is None:
        if args.dataset == "para":
            args.dataset_dir = "datasets/PARA"
        else:
            args.dataset_dir = "datasets/LAPIS"

    # 1) Personalized データ読み込み
    print(f"[info] loading personalized {args.dataset.upper()} dataset...")
    if args.dataset == "para":
        personalized = get_personalized_para_dataset(seed=args.seed, dataset_dir=args.dataset_dir)
    else:
        personalized = get_personalized_lapis_dataset(seed=args.seed, dataset_dir=args.dataset_dir)

    all_user_ids = sorted(personalized.keys())
    print(f"[info] num users in personalized dataset: {len(all_user_ids)}")

    # quick: ユーザ数を制限
    if args.quick is not None and args.quick < len(all_user_ids):
        user_ids = all_user_ids[:args.quick]
        print(f"[info] quick mode: using first {len(user_ids)} users out of {len(all_user_ids)}")
    else:
        user_ids = all_user_ids

    # 2) dataset-level GIAA ground truth を読み込み（必要なら）
    image_to_giaa_gt: Dict[str, float] = {}
    if args.target_score == "giaa_gt":
        print(f"[info] loading dataset-level GIAA ground truth for {args.dataset.upper()} ...")
        if args.dataset == "para":
            gt_items = get_para_dataset(None, dataset_dir=args.dataset_dir)
        else:
            gt_items = get_lapis_dataset(None, dataset_dir=args.dataset_dir)
        for it in gt_items:
            image_to_giaa_gt[it.image_path] = float(it.score)
        print(f"[info] GIAA ground truth entries: {len(image_to_giaa_gt)}")

    # 3) 対象ユーザに出現する全画像パスを収集
    all_paths = set()
    for user_id in user_ids:
        pdata = personalized[user_id]
        for item in pdata.support_small + pdata.support_large + pdata.test:
            all_paths.add(item.image_path)
    print(f"[info] total unique images in selected users' splits: {len(all_paths)}")

    # 4) mm_embed 用の VLM ロード
    print(f"[info] loading VLM for features: {args.model_id}")
    model, processor = load_mm_model(args.model_id, dtype="auto", device_map="auto", attn_impl=None)
    model.eval()
    device = model.device
    prompt = make_prompt(args.prompt_mode)
    print(f"[info] prompt_mode={args.prompt_mode}, prompt={prompt!r}")

    # 5) AllPools を全画像についてキャッシュ
    pools_cache: Dict[str, object] = {}
    print("[info] extracting AllPools for all selected images...")
    for path in tqdm(sorted(all_paths), desc="Embed"):
        img = Image.open(path).convert("RGB")
        inputs = build_inputs(processor, img, prompt)
        pools = extract_all_pools(model, inputs, processor=processor)
        pools_cache[path] = pools
    print(f"[info] pools cache size: {len(pools_cache)}")

    if not pools_cache:
        raise RuntimeError("No features extracted. Check dataset_dir / seed / model.")

    # 6) ユーザごとに RidgeCV を train し test に適用
    rows: List[dict] = []
    if args.target_score == "piaa":
        method_name = f"direct_linear_{args.feature_source}_L{args.feature_layer}"
    else:
        method_name = f"direct_linear_giaa_gt_{args.feature_source}_L{args.feature_layer}"

    for user_id in tqdm(user_ids, desc="Users"):
        pdata = personalized[user_id]
        if args.support_set == "small":
            support_items = pdata.support_small
        else:
            support_items = pdata.support_large
        test_items = pdata.test

        # support セットの作成
        X_support = []
        y_support = []
        for item in support_items:
            path = item.image_path
            if path not in pools_cache:
                continue
            pools = pools_cache[path]
            try:
                feat = extract_feature_vector(pools, args.feature_source, args.feature_layer)
            except IndexError:
                raise IndexError(
                    f"feature_layer={args.feature_layer} is out of range for source={args.feature_source} "
                    f"(check number of layers for this model/source)."
                )

            user_score = float(item.score)
            if args.target_score == "piaa":
                target = user_score
            else:
                if path not in image_to_giaa_gt:
                    # データセット平均がない画像はスキップ
                    continue
                target = image_to_giaa_gt[path]

            X_support.append(feat)
            y_support.append(target)

        X_support = np.array(X_support, dtype=np.float32)
        y_support = np.array(y_support, dtype=np.float32)

        if len(y_support) < 2:
            # 学習に十分なサンプルがないユーザはスキップ
            continue

        pipe = make_pipeline(
            StandardScaler(with_std=True),
            RidgeCV(alphas=np.logspace(-3, 3, 13))
        )
        pipe.fit(X_support, y_support)

        # test セット予測
        for item in test_items:
            path = item.image_path
            if path not in pools_cache:
                continue
            pools = pools_cache[path]
            feat = extract_feature_vector(pools, args.feature_source, args.feature_layer)
            z = feat[None, :]  # [1,D]
            piaa_pred = pipe.predict(z)[0]
            # 評価用に always user_score をログしておく
            user_score = float(item.score)
            rows.append(
                {
                    "user_id": user_id,
                    "image_path": path,
                    "model_id": args.model_id,
                    "support_set": args.support_set,
                    "method": method_name,
                    "giaa": math.nan,        # direct モデルなので GIAA予測は使わない
                    "piaa_pred": piaa_pred,
                    "user_score": user_score,
                }
            )

    # 7) CSV 保存
    out_path = args.out_csv
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
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
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[done] wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()