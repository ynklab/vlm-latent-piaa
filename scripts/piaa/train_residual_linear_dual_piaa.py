#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train per-user residual linear models (Ridge) for PIAA on PARA using
concatenated mm_embed features from TWO sources/layers.

For each user u:
  - We have support set S_u (support_small or support_large) and test set T_u
    from get_personalized_para_dataset.
  - For each image i:
      GIAA_pred(i)   : precomputed GIAA prediction (from vlm_giaa_para.py).
      user_score_i   : user-specific aesthetic score.
      feature z_i    : concat( z_i^{(A)}, z_i^{(B)} ), where each z_i^{(X)} comes
                       from a chosen mm_embed source & layer.
  - Target residual:
      r_i = user_score_i - GIAA_pred(i)
  - We fit a Ridge regression r ~ concat(z^{(A)}, z^{(B)}) on the support set.
  - On the test set, PIAA_pred(i) = GIAA_pred(i) + Ridge(concat(z_i^{(A)}, z_i^{(B)})).

Outputs a CSV with one row per user × test image:
  user_id, image_path, model_id, support_set, method, giaa, piaa_pred, user_score

Method name:
  residual_linear_<source_a>_L<layer_a>__plus__<source_b>_L<layer_b>
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


# ---------- GIAA loader ----------

def load_giaa_map(
    giaa_csv: str,
    model_id_filter: str | None = None,
) -> Tuple[Dict[str, float], str]:
    """
    Read GIAA CSV and build a dict: image_path -> giaa (float).

    Expected columns in CSV (from vlm_giaa_para.py):
      model_id, split, image_path, giaa, raw_output

    If model_id_filter is given, only that model_id is used.
    If not, CSV must contain exactly one model_id.
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


# ---------- Feature helper ----------

def extract_single_feature(
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
        vec = pools.bridge_text[0]  # layer_idx is ignored
    elif source == "bridge_visual":
        vec = pools.bridge_visual[0]
    else:
        raise ValueError(f"Unknown feature_source: {source}")

    return vec.astype(np.float32)


def extract_dual_feature(
    pools,
    source_a: str,
    layer_a: int,
    source_b: str,
    layer_b: int,
) -> np.ndarray:
    """
    Extract vectors from two source/layer pairs, concatenate them, and return the result.
    """
    va = extract_single_feature(pools, source_a, layer_a)
    vb = extract_single_feature(pools, source_b, layer_b)
    return np.concatenate([va, vb], axis=-1).astype(np.float32)


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--giaa_csv",
        required=True,
        help="Path to GIAA prediction CSV (from vlm_giaa_para.py).",
    )
    ap.add_argument(
        "--dataset_dir",
        default="datasets/PARA",
        help="Path to PARA dataset root.",
    )
    ap.add_argument(
        "--model_id",
        required=True,
        help="Multimodal model id for mm_embed (e.g. Qwen/Qwen3-VL-2B-Instruct, google/gemma-3-4b-it).",
    )
    ap.add_argument(
        "--model_id_filter",
        default=None,
        help="If giaa_csv contains multiple model_ids, specify which one to use.",
    )
    ap.add_argument(
        "--feature_source_a",
        required=True,
        choices=[
            "llm_text",
            "llm_visual",
            "llm_text_tail",
            "vision",
            "bridge_text",
            "bridge_visual",
        ],
        help="First feature source from mm_embed.AllPools.",
    )
    ap.add_argument(
        "--feature_layer_a",
        type=int,
        required=True,
        help="Layer index (0-based) for the first feature source.",
    )
    ap.add_argument(
        "--feature_source_b",
        required=True,
        choices=[
            "llm_text",
            "llm_visual",
            "llm_text_tail",
            "vision",
            "bridge_text",
            "bridge_visual",
        ],
        help="Second feature source from mm_embed.AllPools.",
    )
    ap.add_argument(
        "--feature_layer_b",
        type=int,
        required=True,
        help="Layer index (0-based) for the second feature source.",
    )
    ap.add_argument(
        "--support_set",
        default="small",
        choices=["small", "large"],
        help="Which support set from get_personalized_para_dataset to use.",
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
        help="Random seed for get_personalized_para_dataset (must match when generating splits).",
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

    # 1) Load GIAA map
    image_to_giaa, giaa_model_id = load_giaa_map(args.giaa_csv, args.model_id_filter)
    print(f"[info] using GIAA model_id={giaa_model_id}, entries={len(image_to_giaa)}")

    # 2) Load personalized PARA dataset
    print("[info] loading personalized PARA dataset...")
    personalized = get_personalized_para_dataset(seed=args.seed, dataset_dir=args.dataset_dir)
    all_user_ids = sorted(personalized.keys())
    print(f"[info] num users in personalized dataset: {len(all_user_ids)}")

    # quick: limit number of users
    if args.quick is not None and args.quick < len(all_user_ids):
        user_ids = all_user_ids[:args.quick]
        print(f"[info] quick mode: using first {len(user_ids)} users out of {len(all_user_ids)}")
    else:
        user_ids = all_user_ids

    # 3) Collect all image_paths for selected users
    all_paths = set()
    for user_id in user_ids:
        pdata = personalized[user_id]
        for item in pdata.support_small + pdata.support_large + pdata.test:
            all_paths.add(item.image_path)
    print(f"[info] total unique images in selected users' splits: {len(all_paths)}")

    # 4) Load VLM for mm_embed features
    print(f"[info] loading VLM for features: {args.model_id}")
    model, processor = load_mm_model(args.model_id, dtype="auto", device_map="auto", attn_impl=None)
    model.eval()
    device = model.device
    prompt = make_prompt(args.prompt_mode)
    print(f"[info] prompt_mode={args.prompt_mode}, prompt={prompt!r}")

    # 5) Precompute AllPools for all relevant images
    pools_cache: Dict[str, object] = {}
    print("[info] extracting AllPools for all selected images...")
    for path in tqdm(sorted(all_paths), desc="Embed"):
        if path not in image_to_giaa:
            continue
        img = Image.open(path).convert("RGB")
        inputs = build_inputs(processor, img, prompt)
        pools = extract_all_pools(model, inputs, processor=processor)
        pools_cache[path] = pools
    print(f"[info] pools cache size: {len(pools_cache)} (images with both GIAA and features)")

    if not pools_cache:
        raise RuntimeError("No images with both GIAA and features. Check giaa_csv / dataset_dir / seed.")

    # 6) Per-user training (Ridge) and prediction with concatenated features
    rows: List[dict] = []
    method_name = (
        f"residual_linear_{args.feature_source_a}_L{args.feature_layer_a}"
        f"__plus__{args.feature_source_b}_L{args.feature_layer_b}"
    )

    for user_id in tqdm(user_ids, desc="Users"):
        pdata = personalized[user_id]
        if args.support_set == "small":
            support_items = pdata.support_small
        else:
            support_items = pdata.support_large
        test_items = pdata.test

        # Build support set
        X_support = []
        y_support = []
        for item in support_items:
            path = item.image_path
            if path not in image_to_giaa or path not in pools_cache:
                continue
            giaa = image_to_giaa[path]
            user_score = float(item.score)
            residual = user_score - giaa
            pools = pools_cache[path]
            try:
                feat = extract_dual_feature(
                    pools,
                    args.feature_source_a,
                    args.feature_layer_a,
                    args.feature_source_b,
                    args.feature_layer_b,
                )
            except IndexError:
                raise IndexError(
                    f"Layer index out of range for source A or B. "
                    f"A={args.feature_source_a}[{args.feature_layer_a}], "
                    f"B={args.feature_source_b}[{args.feature_layer_b}]"
                )
            X_support.append(feat)
            y_support.append(residual)

        X_support = np.array(X_support, dtype=np.float32)
        y_support = np.array(y_support, dtype=np.float32)

        if len(y_support) < 2:
            # not enough support points to train; skip this user
            continue

        # Fit RidgeCV on support
        pipe = make_pipeline(
            StandardScaler(with_std=True),
            RidgeCV(alphas=np.logspace(-3, 3, 13))
        )
        pipe.fit(X_support, y_support)

        # Predict on test set
        for item in test_items:
            path = item.image_path
            if path not in image_to_giaa or path not in pools_cache:
                continue
            giaa = image_to_giaa[path]
            pools = pools_cache[path]
            feat = extract_dual_feature(
                pools,
                args.feature_source_a,
                args.feature_layer_a,
                args.feature_source_b,
                args.feature_layer_b,
            )
            z = feat[None, :]  # [1,Dab]
            residual_pred = pipe.predict(z)[0]
            piaa_pred = giaa + residual_pred
            rows.append(
                {
                    "user_id": user_id,
                    "image_path": path,
                    "model_id": args.model_id,
                    "support_set": args.support_set,
                    "method": method_name,
                    "giaa": giaa,
                    "piaa_pred": piaa_pred,
                    "user_score": float(item.score),
                }
            )

    # 7) Save CSV
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