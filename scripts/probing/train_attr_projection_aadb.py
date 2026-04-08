#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train a linear projection from VLM hidden features to aesthetic attributes on AADB.

Given:
  - AADB dataset (train split)
  - A VLM model (mm_embed-compatible) and a chosen feature_source / feature_layer
  - AADB's AESTHETIC_ATTRIBUTES (excluding 'score')

We learn a matrix W and bias b such that:

    attr_vec ≈ W * hidden_vec + b

where:
  - hidden_vec: feature from mm_embed for a given image (1D, size D)
  - attr_vec  : vector of aesthetic attributes (size K), excluding 'score'

Implementation details:
  - Features X: [N, D]
  - Targets  Y: [N, K]
  - Model    : StandardScaler on X + RidgeCV (multioutput)

The learned projection (scaler + coef + intercept + attr_names + metadata) is saved to a .npz file.

Usage example:

  python -m scripts.probing.train_attr_projection_aadb \
    --model_id Qwen/Qwen3-VL-2B-Instruct \
    --feature_source llm_text \
    --feature_layer 20 \
    --prompt_mode base \
    --aadb_dir datasets/aadb \
    --proj_out proj/qwen_llm_text_L20_aadb_attr_proj.npz
"""

import os
import argparse
from typing import Dict, List

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from utils.aadb import get_aadb_dataset, AESTHETIC_ATTRIBUTES as AADB_ATTRS
from utils.mm_embed import load_mm_model, build_inputs, extract_all_pools


def extract_feature_vector(pools, source: str, layer_idx: int) -> np.ndarray:
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
        return "Describe the aesthetic properties of this image."
    elif mode == "unrelated":
        return "Describe the weather today in one sentence."
    else:
        raise ValueError(f"Unknown prompt_mode: {mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True,
                    help="VLM model id compatible with mm_embed (e.g. Qwen/Qwen3-VL-2B-Instruct).")
    ap.add_argument("--feature_source", required=True,
                    choices=["llm_text", "llm_visual", "llm_text_tail", "vision", "bridge_text", "bridge_visual"],
                    help="Feature source from mm_embed.AllPools.")
    ap.add_argument("--feature_layer", type=int, required=True,
                    help="Layer index (0-based) for the chosen feature_source.")
    ap.add_argument("--prompt_mode", default="base",
                    choices=["base", "format", "attributes", "unrelated"],
                    help="Prompt mode when extracting features.")
    ap.add_argument("--aadb_dir", default="datasets/aadb",
                    help="Path to AADB dataset root.")
    ap.add_argument("--split", default="train",
                    choices=["train", "validation", "test"],
                    help="Which AADB split to use for training the projection.")
    ap.add_argument("--quick", type=int, default=None,
                    help="If set, use at most N images (for debugging).")
    ap.add_argument("--proj_out", required=True,
                    help="Path to save the projection (.npz).")
    args = ap.parse_args()

    # 1) Load AADB items
    items = get_aadb_dataset(args.split, dataset_dir=args.aadb_dir)
    if args.quick is not None and args.quick < len(items):
        items = items[:args.quick]
    print(f"[info] AADB split={args.split}, N={len(items)}")

    # 2) Attribute names (excluding 'score')
    attr_names = [a for a in AADB_ATTRS if a != "score"]
    print(f"[info] attributes used for projection (excluding 'score'): {attr_names}")
    K = len(attr_names)

    # 3) Load VLM via mm_embed
    print(f"[info] loading VLM: {args.model_id}")
    model, processor = load_mm_model(args.model_id, dtype="auto", device_map="auto", attn_impl=None)
    model.eval()
    device = model.device
    prompt = make_prompt(args.prompt_mode)
    print(f"[info] prompt_mode={args.prompt_mode}, prompt={prompt!r}")

    # 4) Extract features & targets
    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []

    for it in tqdm(items, desc="Extract AADB features"):
        img = Image.open(it.image_path).convert("RGB")
        inputs = build_inputs(processor, img, prompt)
        pools = extract_all_pools(model, inputs, processor=processor)

        try:
            feat = extract_feature_vector(pools, args.feature_source, args.feature_layer)
        except IndexError:
            raise IndexError(
                f"feature_layer={args.feature_layer} is out of range for source={args.feature_source}"
            )

        # Target attribute vector
        attr_vec = np.array([float(it.attributes[a]) for a in attr_names], dtype=np.float32)

        X_list.append(feat)
        Y_list.append(attr_vec)

    X = np.stack(X_list, axis=0)  # [N,D]
    Y = np.stack(Y_list, axis=0)  # [N,K]
    print(f"[info] X.shape={X.shape}, Y.shape={Y.shape}")

    # 5) Train Scaler + RidgeCV (multi-output)
    pipe = make_pipeline(
        StandardScaler(with_std=True),
        RidgeCV(alphas=np.logspace(-3, 3, 13)),
    )
    pipe.fit(X, Y)

    scaler: StandardScaler = pipe.named_steps["standardscaler"]
    ridge: RidgeCV = pipe.named_steps["ridgecv"]

    # coef_: shape (K, D), intercept_: shape (K,)
    coef = ridge.coef_.astype(np.float32)
    intercept = ridge.intercept_.astype(np.float32)
    scaler_mean = scaler.mean_.astype(np.float32)
    scaler_scale = scaler.scale_.astype(np.float32)

    # 6) Save projection
    proj_out = args.proj_out
    os.makedirs(os.path.dirname(proj_out) or ".", exist_ok=True)
    np.savez(
        proj_out,
        model_id=args.model_id,
        feature_source=args.feature_source,
        feature_layer=args.feature_layer,
        prompt_mode=args.prompt_mode,
        attr_names=np.array(attr_names),
        coef=coef,
        intercept=intercept,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
    )
    print(f"[save] projection -> {proj_out}")
    print("[done]")


if __name__ == "__main__":
    main()
