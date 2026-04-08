#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train per-user residual linear models (Ridge) for PIAA on PARA using mm_embed features.

Residual definition:
  Let `target_score` be one of:
    - `piaa`: the user score (personalized PIAA)
    - `giaa_gt`: the dataset-level mean GIAA score (`aestheticScore_mean`)

    residual = target_score - GIAA_pred

We fit Ridge regression on this residual target, and at test time:
    PIAA_pred = GIAA_pred + residual_pred
to obtain the final prediction.

This allows the same script to compare:
  - `target_score = piaa`: residual learning for personalized preference
  - `target_score = giaa_gt`: residual learning that corrects global GIAA

Output:
  user_id, image_path, model_id, support_set, method, giaa, piaa_pred, user_score

Method name:
  - `piaa`: `residual_linear_<source>_L<layer>` (same as before)
  - `giaa_gt`: `residual_linear_giaa_gt_<source>_L<layer>`
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

    Expected columns in CSV (from vlm_giaa.py):
      model_id, dataset, split, image_path, giaa, raw_output

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
        "--giaa_csv",
        required=True,
        help="Path to GIAA prediction CSV (from vlm_giaa.py).",
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
        "--target_score",
        default="piaa",
        choices=["piaa", "giaa_gt"],
        help=(
            "Which score to use as ground truth for residual: "
            "'piaa' (user_score) or 'giaa_gt' (dataset-level mean aestheticScore_mean)."
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

    # 1) Load GIAA predicted map
    image_to_giaa, giaa_model_id = load_giaa_map(args.giaa_csv, args.model_id_filter)
    print(f"[info] using GIAA model_id={giaa_model_id}, entries={len(image_to_giaa)}")

    # 2) Load dataset-level GIAA ground truth if needed
    image_to_giaa_gt: Dict[str, float] = {}
    if args.target_score == "giaa_gt":
        print("[info] loading dataset-level GIAA ground truth from PARA-Giaa*.csv ...")
        gt_items = get_para_dataset(None, dataset_dir=args.dataset_dir)
        for it in gt_items:
            image_to_giaa_gt[it.image_path] = float(it.score)
        print(f"[info] GIAA ground truth entries: {len(image_to_giaa_gt)}")

    # 3) Load personalized PARA dataset
    print("[info] loading personalized PARA dataset...")
    personalized = get_personalized_para_dataset(seed=args.seed, dataset_dir=args.dataset_dir)
    all_user_ids = sorted(personalized.keys())
    print(f"[info] num users in personalized dataset: {len(all_user_ids)}")

    # quick: limit users
    if args.quick is not None and args.quick < len(all_user_ids):
        user_ids = all_user_ids[:args.quick]
        print(f"[info] quick mode: using first {len(user_ids)} users out of {len(all_user_ids)}")
    else:
        user_ids = all_user_ids

    # 4) Collect all image_paths for selected users
    all_paths = set()
    for user_id in user_ids:
        pdata = personalized[user_id]
        for item in pdata.support_small + pdata.support_large + pdata.test:
            all_paths.add(item.image_path)
    print(f"[info] total unique images in selected users' splits: {len(all_paths)}")

    # 5) Load VLM for mm_embed features
    print(f"[info] loading VLM for features: {args.model_id}")
    model, processor = load_mm_model(args.model_id, dtype="auto", device_map="auto", attn_impl=None)
    model.eval()
    device = model.device
    prompt = make_prompt(args.prompt_mode)
    print(f"[info] prompt_mode={args.prompt_mode}, prompt={prompt!r}")

    # 6) Precompute AllPools
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

    # 7) Per-user RidgeCV training and test prediction
    rows: List[dict] = []
    if args.target_score == "piaa":
        method_name = f"residual_linear_{args.feature_source}_L{args.feature_layer}"
    else:  # giaa_gt
        method_name = f"residual_linear_giaa_gt_{args.feature_source}_L{args.feature_layer}"

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
            giaa_pred = image_to_giaa[path]
            user_score = float(item.score)

            if args.target_score == "piaa":
                target_score = user_score
            else:  # giaa_gt
                if path not in image_to_giaa_gt:
                    continue
                target_score = image_to_giaa_gt[path]

            residual = target_score - giaa_pred

            pools = pools_cache[path]
            try:
                feat = extract_feature_vector(pools, args.feature_source, args.feature_layer)
            except IndexError:
                raise IndexError(
                    f"feature_layer={args.feature_layer} is out of range for source={args.feature_source} "
                    f"(check number of layers for this model/source)."
                )

            X_support.append(feat)
            y_support.append(residual)

        X_support = np.array(X_support, dtype=np.float32)
        y_support = np.array(y_support, dtype=np.float32)

        if len(y_support) < 2:
            # Not enough support points; skip this user
            continue

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
            giaa_pred = image_to_giaa[path]
            user_score = float(item.score)

            pools = pools_cache[path]
            feat = extract_feature_vector(pools, args.feature_source, args.feature_layer)
            z = feat[None, :]
            residual_pred = pipe.predict(z)[0]
            piaa_pred = giaa_pred + residual_pred

            rows.append(
                {
                    "user_id": user_id,
                    "image_path": path,
                    "model_id": args.model_id,
                    "support_set": args.support_set,
                    "method": method_name,
                    "giaa": giaa_pred,
                    "piaa_pred": piaa_pred,
                    "user_score": user_score,
                }
            )

    # 8) Save CSV
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
