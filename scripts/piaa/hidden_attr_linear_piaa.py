#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PIAA optimization on PARA / LAPIS using aesthetic attributes predicted from VLM hidden features.

Workflow:
  - Use a pre-trained projection W,b (from train_attr_projection_aadb.py) that maps
    hidden_vec -> attr_vec (AADB attribute space, excluding 'score').
  - On a personalized dataset (PARA or LAPIS):
      For each image:
        1) Extract hidden_vec for chosen feature_source / feature_layer via mm_embed.
        2) Apply projection W,b (with scaler) to obtain attr_vec_pred.
      For each user u:
        3) Train per-user Ridge regression:
             user_score ~ attr_vec_pred  on support_small / support_large.
        4) Evaluate on test set.

Supported datasets:
  - PARA  : utils.para.get_personalized_para_dataset
  - LAPIS : utils.lapis.get_personalized_lapis_dataset

Output CSV columns:
  user_id, image_path, model_id, support_set, method, giaa, piaa_pred, user_score

  - model_id: underlying VLM model id (for grouping in eval scripts).
  - method : "hidden_attr_linear_<source>_L<layer>" (derived from projection file).
  - giaa   : NaN (we do not use dataset-level GIAA here; focus is on PIAA).
"""

import os
import csv
import math
import argparse
from typing import Dict, List

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from utils.para import get_personalized_para_dataset
from utils.lapis import get_personalized_lapis_dataset
from utils.para_hard_images import get_personalized_para_hard_dataset
from utils.para_hard_users import get_personalized_para_hard_users_dataset
from utils.mm_embed import load_mm_model, build_inputs, extract_all_pools


def make_prompt(prompt_mode: str) -> str:
    if prompt_mode == "base":
        return "Assess the aesthetics of this image."
    elif prompt_mode == "format":
        return (
            "Assess the overall aesthetic quality of this image. "
            "Please rate it on a scale from 1 to 5. "
            "Output only the numeric score, and do not output any other text."
        )
    elif prompt_mode == "attributes":
        return "Describe the aesthetic properties of this image."
    elif prompt_mode == "unrelated":
        return "Describe the weather today in one sentence."
    else:
        return "Assess the aesthetics of this image."


def extract_feature_vector(pools, source: str, layer_idx: int) -> np.ndarray:
    """
    Extract the 1D feature vector for the specified source/layer from mm_embed.AllPools.
    """
    if source == "llm_text":
        vec = pools.llm_text[layer_idx]
    elif source == "llm_visual":
        vec = pools.llm_visual[layer_idx]
    elif source == "llm_text_tail":
        vec = pools.llm_text_tail[layer_idx]
    elif source == "vision":
        if pools.vision_layers is None:
            raise RuntimeError("vision_layers is None; vision source not available.")
        vec = pools.vision_layers[layer_idx]
    elif source == "bridge_text":
        vec = pools.bridge_text[0]
    elif source == "bridge_visual":
        vec = pools.bridge_visual[0]
    else:
        raise ValueError(f"Unknown feature_source: {source}")
    return vec.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        required=True,
        choices=["para", "para_hard_images", "para_hard_users", "lapis"],
        help="Dataset to use (para or lapis).",
    )
    ap.add_argument(
        "--proj_file",
        required=True,
        help="Projection .npz file trained on AADB (from train_attr_projection_aadb.py).",
    )
    ap.add_argument(
        "--dataset_dir",
        default=None,
        help="Path to dataset root. If None, uses datasets/PARA or datasets/LAPIS.",
    )
    ap.add_argument(
        "--model_id",
        required=True,
        help="VLM model id (should match the one used for projection).",
    )
    ap.add_argument(
        "--support_set",
        default="small",
        choices=["small", "large"],
        help="Which support set from personalized dataset to use.",
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
        help="If set, limit to at most N users (for debugging).",
    )
    ap.add_argument(
        "--out_csv",
        required=True,
        help="Path to output CSV for per-user test predictions.",
    )
    args = ap.parse_args()

    # Default dataset_dir
    if args.dataset_dir is None:
        if args.dataset in ["para", "para_hard_images", "para_hard_users"]:
            args.dataset_dir = "datasets/PARA"
        else:
            args.dataset_dir = "datasets/LAPIS"

    # 1) Load the projection
    proj = np.load(args.proj_file, allow_pickle=True)
    proj_model_id = proj["model_id"].item()
    feature_source = proj["feature_source"].item()
    feature_layer = int(proj["feature_layer"].item())
    prompt_mode = proj["prompt_mode"].item()
    attr_names = proj["attr_names"].tolist()
    coef = proj["coef"]              # [K, D]
    intercept = proj["intercept"]    # [K]
    scaler_mean = proj["scaler_mean"]
    scaler_scale = proj["scaler_scale"]

    print(f"[info] loaded projection from {args.proj_file}")
    print(f"        proj_model_id = {proj_model_id}")
    print(f"        feature_source= {feature_source}")
    print(f"        feature_layer = {feature_layer}")
    print(f"        prompt_mode   = {prompt_mode}")
    print(f"        attr_names    = {attr_names}")
    print(f"        coef shape    = {coef.shape}")

    # 2) Load personalized data (PARA / LAPIS)
    print(f"[info] loading personalized {args.dataset.upper()} dataset...")
    if args.dataset == "para":
        personalized = get_personalized_para_dataset(seed=args.seed, dataset_dir=args.dataset_dir)
    elif args.dataset == "para_hard_images":
        personalized = get_personalized_para_hard_dataset(seed=args.seed, dataset_dir=args.dataset_dir)
    elif args.dataset == "para_hard_users":
        personalized, _ = get_personalized_para_hard_users_dataset(seed=args.seed, dataset_dir=args.dataset_dir)
    else:
        personalized = get_personalized_lapis_dataset(seed=args.seed, dataset_dir=args.dataset_dir)

    all_user_ids = sorted(personalized.keys())
    print(f"[info] num users in personalized dataset: {len(all_user_ids)}")

    if args.quick is not None and args.quick < len(all_user_ids):
        user_ids = all_user_ids[: args.quick]
        print(f"[info] quick mode: using first {len(user_ids)} users out of {len(all_user_ids)}")
    else:
        user_ids = all_user_ids

    # 3) Collect all image paths that appear for the selected users
    all_paths = set()
    for user_id in user_ids:
        pdata = personalized[user_id]
        for item in pdata.support_small + pdata.support_large + pdata.test:
            all_paths.add(item.image_path)
    print(f"[info] total unique images in selected users' splits: {len(all_paths)}")

    # 4) Load the VLM
    print(f"[info] loading VLM: {args.model_id}")
    model, processor = load_mm_model(args.model_id, dtype="auto", device_map="auto", attn_impl=None)
    model.eval()
    device = model.device
    prompt = make_prompt(prompt_mode)
    print(f"[info] prompt_mode (from proj) = {prompt_mode!r}, prompt={prompt!r}")

    # 5) Projection function from hidden vectors to attribute vectors
    def hidden_to_attr_vec(h: np.ndarray) -> np.ndarray:
        xs = (h - scaler_mean) / (scaler_scale + 1e-12)  # [D]
        return xs @ coef.T + intercept  # [K]

    # 6) Cache hidden -> attr_vec_pred for all images
    attr_cache: Dict[str, np.ndarray] = {}
    print("[info] extracting hidden features and projecting to attributes...")
    for path in tqdm(sorted(all_paths), desc="Embed+Project"):
        img = Image.open(path).convert("RGB")
        inputs = build_inputs(processor, img, prompt)
        pools = extract_all_pools(model, inputs, processor=processor)
        try:
            h = extract_feature_vector(pools, feature_source, feature_layer)
        except IndexError:
            raise IndexError(
                f"feature_layer={feature_layer} is out of range for feature_source={feature_source}"
            )
        a_pred = hidden_to_attr_vec(h)
        attr_cache[path] = a_pred.astype(np.float32)

    print(f"[info] attr_cache size={len(attr_cache)}")

    # 7) User-specific linear model (attr_vec_pred -> PIAA)
    rows: List[dict] = []
    method_name = f"hidden_attr_linear_{feature_source}_L{feature_layer}"

    for user_id in tqdm(user_ids, desc="Users(hidden-attr-linear)"):
        pdata = personalized[user_id]
        if args.support_set == "small":
            support_items = pdata.support_small
        else:
            support_items = pdata.support_large
        test_items = pdata.test

        # support set
        X_support = []
        y_support = []
        for it in support_items:
            path = it.image_path
            if path not in attr_cache:
                continue
            X_support.append(attr_cache[path])
            y_support.append(float(it.score))  # user-specific PIAA

        X_support = np.array(X_support, dtype=np.float32)
        y_support = np.array(y_support, dtype=np.float32)

        if len(y_support) < 2:
            continue

        pipe = make_pipeline(
            StandardScaler(with_std=True),
            RidgeCV(alphas=np.logspace(-3, 3, 13)),
        )
        pipe.fit(X_support, y_support)

        # test set
        for it in test_items:
            path = it.image_path
            if path not in attr_cache:
                continue
            x = attr_cache[path][None, :]  # [1,K]
            piaa_pred = pipe.predict(x)[0]
            user_score = float(it.score)

            rows.append(
                {
                    "user_id": user_id,
                    "image_path": path,
                    "model_id": args.model_id,
                    "support_set": args.support_set,
                    "method": method_name,
                    "giaa": math.nan,          # dataset-level GIAA is unused here
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