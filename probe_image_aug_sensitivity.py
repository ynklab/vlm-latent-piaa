#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image pre-processing sensitivity for linear probing.

Compare probing performance under different image modes:
  - orig : normal RGB
  - gray : grayscale (converted to 3-channel)
  - tps  : Albumentations ThinPlateSpline augmentation

Supported datasets:
  - AADB (utils/aadb.py)
  - PARA (utils/para.py)

Supported models:
  - any multimodal model supported by mm_embed.load_mm_model
    (e.g., Qwen/Qwen3-VL-2B-Instruct, google/gemma-3-4b-it, etc.)

Now supports probing multiple sources in a single job:
  --sources llm_text vision

Example:

  python probe_image_aug_sensitivity.py \
      --dataset aadb \
      --dataset_dir datasets/aadb \
      --model_id Qwen/Qwen3-VL-2B-Instruct \
      --sources llm_text vision \
      --prompt_mode base \
      --out_json runs/qwen3vl_aadb_aug_sensitivity_llm_text_vision.json
"""

import os
import json
import math
import argparse
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import albumentations as A

# Dataset loaders
from utils.aadb import get_aadb_dataset, AESTHETIC_ATTRIBUTES as AADB_ATTRS
from utils.para import get_para_dataset, AESTHETIC_ATTRIBUTES as PARA_ATTRS

# Model + embedding extractor
from utils.mm_embed import load_mm_model, build_inputs, extract_all_pools

# --------- prompts (英語) ---------

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

# --------- metrics & utils ---------

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rho = spearmanr(y_true, y_pred).correlation
    if np.isnan(rho):
        rho = 0.0
    mse = float(np.mean((y_true - y_pred) ** 2))
    var = float(np.var(y_true)) + 1e-12
    r2 = 1.0 - mse / var
    rmse = math.sqrt(mse)
    return {"rho": float(rho), "rmse": float(rmse), "r2": float(r2)}

def _rng_choice(seq, n, seed=0):
    if n is None or n >= len(seq):
        return list(seq)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(seq), size=n, replace=False)
    return [seq[i] for i in idx]

def _items_to_paths_and_targets(items, attrs: List[str]):
    paths = [it.image_path for it in items]
    targets = {attr: [it.attributes[attr] for it in items] for attr in attrs}
    return paths, {k: np.array(v, dtype=np.float32) for k, v in targets.items()}

def _fit_eval_one_layer(Xtr, ytr, Xval, yval, Xte, yte):
    pipe = make_pipeline(StandardScaler(with_std=True), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    pipe.fit(Xtr, ytr)
    yhat_tr = pipe.predict(Xtr)
    yhat_va = pipe.predict(Xval)
    yhat_te = pipe.predict(Xte)
    return _metrics(ytr, yhat_tr), _metrics(yval, yhat_va), _metrics(yte, yhat_te)

# --------- image transforms ---------

def build_tps_transform():
    return A.Compose([
        A.ThinPlateSpline(p=1.0)
    ])

def apply_image_mode(img: Image.Image, mode: str, tps_transform=None) -> Image.Image:
    """
    img: PIL.Image (RGB)
    mode: "orig", "gray", "tps"
    """
    if mode == "orig":
        return img
    elif mode == "gray":
        g = img.convert("L")
        return g.convert("RGB")
    elif mode == "tps":
        if tps_transform is None:
            tps_transform = build_tps_transform()
        img_np = np.array(img)
        aug = tps_transform(image=img_np)["image"]
        return Image.fromarray(aug)
    else:
        raise ValueError(f"Unknown image_mode: {mode}")

# --------- main probing logic ---------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="aadb", choices=["aadb", "para"],
                    help="Dataset to use (aadb or para).")
    ap.add_argument("--dataset_dir", default=None,
                    help="Dataset root. If None, use defaults: datasets/aadb or datasets/PARA.")
    ap.add_argument("--model_id", required=True,
                    help="Multimodal model id (for mm_embed.load_mm_model).")

    # ★ 複数 source を受け取るよう変更
    ap.add_argument("--sources", nargs="+", default=["llm_text"],
                    choices=["llm_text", "vision"],
                    help="Which sources to probe: llm_text, vision (you can specify both).")

    ap.add_argument("--prompt_mode", default="base",
                    choices=["base", "format", "attributes", "unrelated"],
                    help="Prompt preset for building inputs.")
    ap.add_argument("--train_split", default=None,
                    help="Train split name (default depends on dataset).")
    ap.add_argument("--val_split", default=None,
                    help="Val   split name (default depends on dataset).")
    ap.add_argument("--test_split", default=None,
                    help="Test  split name (default depends on dataset).")
    ap.add_argument("--quick", type=int, default=None,
                    help="Use at most N examples per split (for quick debugging).")
    ap.add_argument("--out_json", default="runs/aug_sensitivity.json",
                    help="Path to save JSON results.")
    args = ap.parse_args()

    # sanity: unique sources
    sources = sorted(set(args.sources))
    print(f"[info] sources to probe: {sources}")

    # ----- dataset setup -----
    if args.dataset_dir is None:
        if args.dataset == "aadb":
            args.dataset_dir = "datasets/aadb"
        else:
            args.dataset_dir = "datasets/PARA"

    if args.dataset == "aadb":
        get_dataset = get_aadb_dataset
        attrs = list(AADB_ATTRS)
        train_split = args.train_split or "train"
        val_split   = args.val_split   or "validation"
        test_split  = args.test_split  or "test"
    else:  # para
        get_dataset = get_para_dataset
        attrs = list(PARA_ATTRS)
        train_split = args.train_split or "train"
        val_split   = args.val_split   or "test"
        test_split  = args.test_split  or "test"

    print(f"[info] dataset={args.dataset}, dir={args.dataset_dir}")
    print(f"[info] splits: train={train_split}, val={val_split}, test={test_split}")
    print(f"[info] attributes: {attrs}")

    # ----- load dataset items -----
    tr_items = get_dataset(train_split, dataset_dir=args.dataset_dir)
    va_items = get_dataset(val_split,   dataset_dir=args.dataset_dir)
    te_items = get_dataset(test_split,  dataset_dir=args.dataset_dir)

    if args.quick is not None:
        tr_items = _rng_choice(tr_items, args.quick, 0)
        va_items = _rng_choice(va_items, args.quick, 1)
        te_items = _rng_choice(te_items, args.quick, 2)

    tr_paths, tr_targets = _items_to_paths_and_targets(tr_items, attrs)
    va_paths, va_targets = _items_to_paths_and_targets(va_items, attrs)
    te_paths, te_targets = _items_to_paths_and_targets(te_items, attrs)

    print(f"[info] N train={len(tr_paths)}, val={len(va_paths)}, test={len(te_paths)}")

    # ----- load model & processor -----
    model, processor = load_mm_model(args.model_id, dtype="auto", device_map="auto", attn_impl=None)
    model.eval()
    device = model.device
    print(f"[info] loaded model on device={device}")

    prompt = make_prompt(args.prompt_mode)
    print(f"[info] prompt_mode={args.prompt_mode}, prompt={prompt!r}")

    # ----- define image modes -----
    image_modes = ["orig", "gray", "tps"]
    tps_transform = build_tps_transform()

    def extract_features_for_mode_and_source(image_paths: List[str], image_mode: str, source: str):
        feats_per_layer: List[List[np.ndarray]] = None
        for p in tqdm(image_paths, desc=f"Embed[{image_mode}][{source}]"):
            img = Image.open(p).convert("RGB")
            img_ = apply_image_mode(img, image_mode, tps_transform)

            inputs = build_inputs(processor, img_, prompt)
            pools = extract_all_pools(model, inputs, processor=processor)

            if source == "llm_text":
                vecs = pools.llm_text  # list of [D]
            elif source == "vision":
                if pools.vision_layers is None:
                    return None
                vecs = pools.vision_layers
            else:
                raise ValueError(f"Unsupported source: {source}")

            if feats_per_layer is None:
                feats_per_layer = [[] for _ in range(len(vecs))]
            for li, v in enumerate(vecs):
                feats_per_layer[li].append(v.astype(np.float32))

        if feats_per_layer is None:
            return None
        return [np.stack(lst, axis=0) for lst in feats_per_layer]  # List[L] of [N,D]

    results = {
        "config": {
            "model_id": args.model_id,
            "dataset": args.dataset,
            "dataset_dir": args.dataset_dir,
            "train_split": train_split,
            "val_split": val_split,
            "test_split": test_split,
            "prompt_mode": args.prompt_mode,
            "sources": sources,
            "quick": args.quick,
        },
        "sources": {},  # source -> mode -> ...
    }

    for source in sources:
        print(f"[source] {source}")
        source_entry = {"modes": {}}

        for mode in image_modes:
            print(f"[source={source}] mode={mode} — extracting features for train/val/test")
            Xtr_layers = extract_features_for_mode_and_source(tr_paths, mode, source)
            Xva_layers = extract_features_for_mode_and_source(va_paths, mode, source)
            Xte_layers = extract_features_for_mode_and_source(te_paths, mode, source)

            if Xtr_layers is None or Xva_layers is None or Xte_layers is None:
                print(f"[warn] No features for source={source}, mode={mode}, skipping this mode.")
                continue

            n_layers = len(Xtr_layers)
            mode_entry = {"attrs": {}, "n_layers": n_layers}
            print(f"[source={source}] mode={mode}: n_layers={n_layers}")

            for attr in attrs:
                print(f"[source={source}][mode={mode}] attr={attr}")
                ytr = tr_targets[attr]
                yva = va_targets[attr]
                yte = te_targets[attr]

                best = {"layer": None,
                        "train": None,
                        "val": {"rho": -1, "rmse": 1e9, "r2": -1},
                        "test": None}
                per_layer = []

                for li in range(n_layers):
                    Xtr = Xtr_layers[li]
                    Xva = Xva_layers[li]
                    Xte = Xte_layers[li]

                    train_m, val_m, test_m = _fit_eval_one_layer(Xtr, ytr, Xva, yva, Xte, yte)
                    per_layer.append({
                        "layer": li,
                        "train": train_m,
                        "val": val_m,
                        "test": test_m,
                    })
                    if val_m["rho"] > best["val"]["rho"]:
                        best = {
                            "layer": li,
                            "train": train_m,
                            "val": val_m,
                            "test": test_m,
                        }

                mode_entry["attrs"][attr] = {
                    "per_layer": per_layer,
                    "best": best,
                }

            source_entry["modes"][mode] = mode_entry

        results["sources"][source] = source_entry

    # ----- save JSON -----
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[save] {args.out_json}")


if __name__ == "__main__":
    main()