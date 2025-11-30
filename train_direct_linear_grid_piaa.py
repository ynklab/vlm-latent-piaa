#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train per-user direct linear models (Ridge) for PIAA on PARA/LAPIS using mm_embed features,
for ALL (feature_source, feature_layer) combinations in a single run.

目的:
  - GIAA 予測を一切使わず，VLMの埋め込み z から直接ユーザのスコア y (PIAA) を予測する。
  - 各 feature_source × layer_idx について per-user Ridge を学習し，
    test セットに対する予測をCSVに吐き出す。

対応データセット:
  - PARA  : utils.para.get_personalized_para_dataset
  - LAPIS : utils.lapis.get_personalized_lapis_dataset

出力:
  - --out_dir の中に feature_source × layer ごとに1ファイル:
      direct_linear_<source>_L<layer>.csv
    CSV形式:
      user_id, image_path, model_id, support_set, method, giaa, piaa_pred, user_score
    ※ giaa はここでは使わないので NaN を入れる。
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

from utils.para import get_personalized_para_dataset
from utils.lapis import get_personalized_lapis_dataset
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


# ---------- Feature helpers ----------

def extract_feature_vector_from_pools(
    pools,
    source: str,
    layer_idx: int,
) -> np.ndarray:
    """
    mm_embed.extract_all_pools の AllPools から source/layer を指定して [D] ベクトルを取り出す。
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
        vec = pools.bridge_text[0]  # layer_idx は無視
    elif source == "bridge_visual":
        vec = pools.bridge_visual[0]
    else:
        raise ValueError(f"Unknown feature_source: {source}")

    return vec.astype(np.float32)


def get_layer_range_for_source(example_pools, source: str) -> List[int]:
    """
    source ごとに利用可能な layer index のリストを返す。
    bridge_* は [0] のみ。
    """
    if source == "llm_text":
        return list(range(len(example_pools.llm_text)))
    elif source == "llm_visual":
        return list(range(len(example_pools.llm_visual)))
    elif source == "llm_text_tail":
        return list(range(len(example_pools.llm_text_tail)))
    elif source == "vision":
        if example_pools.vision_layers is None:
            return []
        return list(range(len(example_pools.vision_layers)))
    elif source in ("bridge_text", "bridge_visual"):
        return [0]
    else:
        raise ValueError(f"Unknown feature_source: {source}")


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
        "--feature_sources",
        nargs="+",
        default=["llm_text"],
        choices=[
            "llm_text",
            "llm_visual",
            "llm_text_tail",
            "vision",
            "bridge_text",
            "bridge_visual",
        ],
        help="Feature sources from mm_embed.AllPools to use. "
             "All layers for each source will be probed.",
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
        "--quick",
        type=int,
        default=None,
        help="If set, limit to at most N users (for debugging). Use 1 for a single-user check.",
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Output directory. One CSV per (source, layer) combination will be created.",
    )
    args = ap.parse_args()

    # dataset_dir デフォルト
    os.makedirs(args.out_dir, exist_ok=True)
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

    # 2) 対象ユーザに出現する全画像パスを収集
    all_paths = set()
    for user_id in user_ids:
        pdata = personalized[user_id]
        for item in pdata.support_small + pdata.support_large + pdata.test:
            all_paths.add(item.image_path)
    print(f"[info] total unique images in selected users' splits: {len(all_paths)}")

    # 3) mm_embed 用モデルロード
    print(f"[info] loading VLM for features: {args.model_id}")
    model, processor = load_mm_model(args.model_id, dtype="auto", device_map="auto", attn_impl=None)
    model.eval()
    device = model.device
    prompt = make_prompt(args.prompt_mode)
    print(f"[info] prompt_mode={args.prompt_mode}, prompt={prompt!r}")

    # 4) AllPools をキャッシュ
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

    # 5) sourceごとの layer index 範囲を決める（例として1つの AllPools を使う）
    example_pools = next(iter(pools_cache.values()))
    source_to_layers: Dict[str, List[int]] = {}
    for src in args.feature_sources:
        layer_indices = get_layer_range_for_source(example_pools, src)
        if not layer_indices:
            print(f"[warn] feature_source={src} has no layers (e.g., vision_layers=None), skipping")
            continue
        source_to_layers[src] = layer_indices
        print(f"[info] feature_source={src}: layers={layer_indices[0]}..{layer_indices[-1]} (total {len(layer_indices)})")

    if not source_to_layers:
        raise RuntimeError("No valid feature sources with layers found.")

    # Helper: ユーザごとの support/test を source/layer に応じて組み立てる
    def build_support_and_test(user_id, source: str, layer_idx: int) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, float]]]:
        """
        戻り値:
          X_support: [Ns, D], y_support: [Ns]
          test_rows: List[(image_path, user_score)] for test items
        """
        pdata = personalized[user_id]
        if args.support_set == "small":
            support_items = pdata.support_small
        else:
            support_items = pdata.support_large
        test_items = pdata.test

        X_support = []
        y_support = []
        for item in support_items:
            path = item.image_path
            if path not in pools_cache:
                continue
            pools = pools_cache[path]
            feat = extract_feature_vector_from_pools(pools, source, layer_idx)
            user_score = float(item.score)
            X_support.append(feat)
            y_support.append(user_score)

        X_support = np.array(X_support, dtype=np.float32)
        y_support = np.array(y_support, dtype=np.float32)

        test_rows = []
        for item in test_items:
            path = item.image_path
            if path not in pools_cache:
                continue
            user_score = float(item.score)
            test_rows.append((path, user_score))

        return X_support, y_support, test_rows

    # 6) 各 (source, layer) について per-user Ridge を学習＆CSV出力
    for source, layers in source_to_layers.items():
        for layer_idx in layers:
            method_name = f"direct_linear_{source}_L{layer_idx}"
            print(f"[grid] Training direct_linear for source={source}, layer={layer_idx} ...")
            rows: List[dict] = []

            for user_id in tqdm(user_ids, desc=f"Users[{source}_L{layer_idx}]"):
                X_support, y_support, test_rows = build_support_and_test(user_id, source, layer_idx)
                if len(y_support) < 2:
                    continue

                # y ~ z を RidgeCV で学習
                pipe = make_pipeline(
                    StandardScaler(with_std=True),
                    RidgeCV(alphas=np.logspace(-3, 3, 13))
                )
                pipe.fit(X_support, y_support)

                # test 予測
                for path, user_score in test_rows:
                    pools = pools_cache[path]
                    feat = extract_feature_vector_from_pools(pools, source, layer_idx)
                    z = feat[None, :]
                    piaa_pred = pipe.predict(z)[0]
                    rows.append(
                        {
                            "user_id": user_id,
                            "image_path": path,
                            "model_id": args.model_id,
                            "support_set": args.support_set,
                            "method": method_name,
                            "giaa": math.nan,        # direct なので GIAA は未使用
                            "piaa_pred": piaa_pred,
                            "user_score": user_score,
                        }
                    )

            # CSV 保存
            if not rows:
                print(f"[grid] No rows for source={source}, layer={layer_idx}, skip saving.")
                continue

            out_name = f"{method_name}.csv"
            out_path = os.path.join(args.out_dir, out_name)
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

            print(f"[grid] wrote {len(rows)} rows to {out_path}")

    print("[done] direct_linear grid training finished.")


if __name__ == "__main__":
    main()