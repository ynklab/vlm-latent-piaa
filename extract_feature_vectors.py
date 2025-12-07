#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extract hidden feature vectors for a given (dataset, projection) and save them as a .npz file.

前提:
  - --proj_file には AADB で学習した射影ファイル (train_attr_projection_aadb.py の出力) が渡される。
    そこから以下を読み取る:
      * model_id
      * feature_source
      * feature_layer
      * prompt_mode

対応データセット:
  - PARA  : utils.para
  - LAPIS : utils.lapis
  - AADB  : utils.aadb

mode:
  - all          : データセット中の全画像 (GIAA 観点)
  - personalized : personalized split (PIAA 観点) に現れる画像だけ

出力:
  - --out_npz に以下を保存:

      dataset       : "para" / "lapis" / "aadb"
      dataset_dir   : パス
      mode          : "all" / "personalized"
      seed          : personalized split 用 seed
      proj_file     : 射影ファイルのパス

      model_id      : VLM モデルID (proj_file から取得)
      feature_source: "llm_text" など (proj_file から取得)
      feature_layer : int         (proj_file から取得)
      prompt_mode   : "base" 等   (proj_file から取得)

      image_paths   : [N] (str array)
      features      : [N, D] (float32 array)

Example usage:

  python extract_feature_vectors.py \
    --dataset para \
    --dataset_dir datasets/PARA \
    --mode all \
    --seed 42 \
    --proj_file proj/qwen_llm_text_L20_aadb_attr_proj.npz \
    --quick 1000 \
    --out_npz features/para_qwen_llm_text_L20_all_quick.npz
"""

import os
import argparse
from typing import List, Set

import numpy as np
from PIL import Image
from tqdm import tqdm

from utils.para import (
    get_para_dataset,
    get_personalized_para_dataset,
)
from utils.lapis import (
    get_lapis_dataset,
    get_personalized_lapis_dataset,
)
from utils.aadb import get_aadb_dataset
from utils.mm_embed import load_mm_model, build_inputs, extract_all_pools


# ---------- Helpers ----------

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


def extract_feature_from_pools(pools, source: str, layer_idx: int) -> np.ndarray:
    """
    mm_embed.extract_all_pools の AllPools から source/layer に対応する 1D feature を取り出す。
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
        choices=["para", "lapis", "aadb"],
        help="Dataset to use (para, lapis, or aadb).",
    )
    ap.add_argument(
        "--dataset_dir",
        default=None,
        help="Dataset root directory. If None, uses default path based on dataset.",
    )
    ap.add_argument(
        "--mode",
        default="all",
        choices=["all", "personalized"],
        help="Which subset to extract features for: all images or only personalized images.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for personalized splits (PARA / LAPIS).",
    )
    ap.add_argument(
        "--proj_file",
        required=True,
        help="Projection .npz file from train_attr_projection_aadb.py "
             "(used only to read model_id, feature_source, feature_layer, prompt_mode).",
    )
    ap.add_argument(
        "--quick",
        type=int,
        default=None,
        help="If set, limit to at most N images (for debugging).",
    )
    ap.add_argument(
        "--out_npz",
        required=True,
        help="Path to output .npz file.",
    )
    args = ap.parse_args()

    # dataset_dir default
    if args.dataset_dir is None:
        if args.dataset == "para":
            args.dataset_dir = "datasets/PARA"
        elif args.dataset == "lapis":
            args.dataset_dir = "datasets/LAPIS"
        else:
            args.dataset_dir = "datasets/aadb"

    os.makedirs(os.path.dirname(args.out_npz) or ".", exist_ok=True)

    # ---------- 1) projectionファイルから model / feature 情報を読み出す ----------

    proj = np.load(args.proj_file, allow_pickle=True)
    model_id = proj["model_id"].item()
    feature_source = proj["feature_source"].item()
    feature_layer = int(proj["feature_layer"].item())
    prompt_mode = proj["prompt_mode"].item()

    print(f"[info] proj_file={args.proj_file}")
    print(f"       model_id      = {model_id}")
    print(f"       feature_source= {feature_source}")
    print(f"       feature_layer = {feature_layer}")
    print(f"       prompt_mode   = {prompt_mode}")

    # ---------- 2) 対象となる image_paths の集合を集める ----------

    image_paths: Set[str] = set()

    if args.dataset == "para":
        if args.mode == "all":
            items = get_para_dataset(None, dataset_dir=args.dataset_dir)
            for it in items:
                image_paths.add(it.image_path)
        else:
            personalized = get_personalized_para_dataset(seed=args.seed, dataset_dir=args.dataset_dir)
            for _, pdata in personalized.items():
                for it in pdata.support_small + pdata.support_large + pdata.test:
                    image_paths.add(it.image_path)

    elif args.dataset == "lapis":
        if args.mode == "all":
            items = get_lapis_dataset(None, dataset_dir=args.dataset_dir)
            for it in items:
                image_paths.add(it.image_path)
        else:
            personalized = get_personalized_lapis_dataset(seed=args.seed, dataset_dir=args.dataset_dir)
            for _, pdata in personalized.items():
                for it in pdata.support_small + pdata.support_large + pdata.test:
                    image_paths.add(it.image_path)

    else:  # aadb
        # AADB は personalized split がないので mode は実質 all
        splits = ["train", "validation", "test"]
        for split in splits:
            items = get_aadb_dataset(split, dataset_dir=args.dataset_dir)
            for it in items:
                image_paths.add(it.image_path)

    image_paths = sorted(image_paths)
    if args.quick is not None and args.quick < len(image_paths):
        image_paths = image_paths[: args.quick]

    print(f"[info] dataset={args.dataset}, mode={args.mode}, num_images={len(image_paths)}")

    # ---------- 3) VLM をロードして特徴抽出 ----------

    print(f"[info] loading VLM for features: {model_id}")
    model, processor = load_mm_model(model_id, dtype="auto", device_map="auto", attn_impl=None)
    model.eval()
    device = model.device
    prompt = make_prompt(prompt_mode)
    print(f"[info] mm_embed prompt_mode={prompt_mode}, prompt={prompt!r}")

    feats: List[np.ndarray] = []
    used_paths: List[str] = []

    for path in tqdm(image_paths, desc="Extract features"):
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[warn] failed to open image {path}: {e}, skip")
            continue

        inputs = build_inputs(processor, img, prompt)
        pools = extract_all_pools(model, inputs, processor=processor)
        try:
            vec = extract_feature_from_pools(pools, feature_source, feature_layer)
        except IndexError:
            raise IndexError(
                f"feature_layer={feature_layer} is out of range for feature_source={feature_source}"
            )

        feats.append(vec)
        used_paths.append(path)

    if not feats:
        raise RuntimeError("No features extracted; check dataset_dir / proj_file / feature_layer etc.")

    features = np.stack(feats, axis=0).astype(np.float32)
    image_paths_arr = np.array(used_paths)

    print(f"[info] features shape = {features.shape}, num_images_used = {len(used_paths)}")

    # ---------- 4) .npz 保存 ----------

    np.savez(
        args.out_npz,
        dataset=args.dataset,
        dataset_dir=args.dataset_dir,
        mode=args.mode,
        seed=args.seed,
        proj_file=args.proj_file,
        model_id=model_id,
        feature_source=feature_source,
        feature_layer=feature_layer,
        prompt_mode=prompt_mode,
        image_paths=image_paths_arr,
        features=features,
    )
    print(f"[save] features -> {args.out_npz}")
    print("[done]")


if __name__ == "__main__":
    main()