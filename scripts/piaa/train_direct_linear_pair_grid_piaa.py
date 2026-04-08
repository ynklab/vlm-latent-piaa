#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate whether combining two (source, layer) representations improves PIAA prediction.

- Extract AllPools ONCE per image (mm_embed.extract_all_pools)
- Consider sources among: llm_text, llm_visual, vision
- Choose two sources (with replacement)
- Choose one layer per source where layer % 5 == 0
- Concatenate the two feature vectors and train per-user Ridge on support set, evaluate on test set
- Output:
    out_dir/summary.csv  (one row per combo)
  Optional:
    out_dir/preds/<combo>.csv (predictions per test item) if --save_preds is set

Assumes your personalized dataset loaders:
  - utils.para.get_personalized_para_dataset
  - utils.lapis.get_personalized_lapis_dataset
"""

import os
import re
import csv
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

from utils.para import get_personalized_para_dataset
from utils.lapis import get_personalized_lapis_dataset
from utils.mm_embed import load_mm_model, build_inputs, extract_all_pools


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


def sanitize(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z._\\-]+", "_", str(s))


def extract_feature_from_pools(pools, source: str, layer_idx: int) -> np.ndarray:
    if source == "llm_text":
        return pools.llm_text[layer_idx].astype(np.float32)
    if source == "llm_visual":
        return pools.llm_visual[layer_idx].astype(np.float32)
    if source == "vision":
        if pools.vision_layers is None:
            raise RuntimeError("vision_layers is None (vision source not available).")
        return pools.vision_layers[layer_idx].astype(np.float32)
    raise ValueError(f"Unknown source: {source}")


def get_layers_mod5(example_pools, source: str, exclude_lt0: bool) -> List[int]:
    if source == "llm_text":
        L = len(example_pools.llm_text)
        layers = [i for i in range(L) if i % 5 == 0]
        if exclude_lt0:
            layers = [i for i in layers if i != 0]
        return layers
    if source == "llm_visual":
        L = len(example_pools.llm_visual)
        return [i for i in range(L) if i % 5 == 0]
    if source == "vision":
        if example_pools.vision_layers is None:
            return []
        L = len(example_pools.vision_layers)
        return [i for i in range(L) if i % 5 == 0]
    raise ValueError(source)


def metrics_per_user(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    # drop NaNs
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size < 2:
        return 0.0, float("nan")

    rho = spearmanr(y_true, y_pred).correlation
    if np.isnan(rho):
        rho = 0.0
    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = float("nan")
    return float(rho), r2


@dataclass
class Combo:
    s1: str
    l1: int
    s2: str
    l2: int


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["para", "lapis"])
    ap.add_argument("--dataset_dir", default=None)
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--support_set", default="small", choices=["small", "large"])
    ap.add_argument("--prompt_mode", default="base", choices=["base", "format", "attributes", "unrelated"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick_users", type=int, default=None, help="debug: limit users (e.g. 1, 5, 20)")
    ap.add_argument("--exclude_lt0", action="store_true", help="exclude llm_text layer 0 from candidates")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--save_preds", action="store_true", help="save per-combo prediction CSVs (can be large)")
    ap.add_argument("--alpha_grid", type=int, default=13, help="RidgeCV alpha grid size (logspace -3..3)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.dataset_dir is None:
        args.dataset_dir = "datasets/PARA" if args.dataset == "para" else "datasets/LAPIS"

    # 1) load personalized data
    if args.dataset == "para":
        personalized = get_personalized_para_dataset(seed=args.seed, dataset_dir=args.dataset_dir)
    else:
        personalized = get_personalized_lapis_dataset(seed=args.seed, dataset_dir=args.dataset_dir)

    user_ids = sorted(list(personalized.keys()))
    if args.quick_users is not None and args.quick_users < len(user_ids):
        user_ids = user_ids[:args.quick_users]

    # 2) collect needed image paths (support chosen + test only)
    needed_paths = set()
    for uid in user_ids:
        pdata = personalized[uid]
        support_items = pdata.support_small if args.support_set == "small" else pdata.support_large
        for it in support_items + pdata.test:
            needed_paths.add(it.image_path)
    needed_paths = sorted(needed_paths)
    path_to_idx = {p: i for i, p in enumerate(needed_paths)}
    print(f"[info] users={len(user_ids)} unique_images={len(needed_paths)} support_set={args.support_set}")

    # 3) load VLM + extract pools once per image, but store only (source, layer%5==0) vectors
    print(f"[info] loading model for mm_embed: {args.model_id}")
    model, processor = load_mm_model(args.model_id, dtype="auto", device_map="auto", attn_impl=None)
    model.eval()
    device = next(model.parameters()).device
    prompt = make_prompt(args.prompt_mode)

    # extract one pools for layer list discovery
    print("[info] probing one image to determine layer candidates...")
    probe_img = Image.open(needed_paths[0]).convert("RGB")
    probe_inputs = build_inputs(processor, probe_img, prompt)
    probe_pools = extract_all_pools(model, probe_inputs)

    sources = ["llm_text", "llm_visual", "vision"]
    source_to_layers: Dict[str, List[int]] = {}
    for s in sources:
        layers = get_layers_mod5(probe_pools, s, exclude_lt0=args.exclude_lt0)
        if len(layers) == 0:
            print(f"[warn] source={s} has no candidate layers; will skip")
            continue
        source_to_layers[s] = layers
        print(f"[info] {s}: candidates={layers}")

    if not source_to_layers:
        raise RuntimeError("No valid sources with mod-5 layers found.")

    # feature bank: src -> layer -> [N,D]
    feature_bank: Dict[str, Dict[int, List[np.ndarray]]] = {s: {li: [] for li in ls} for s, ls in source_to_layers.items()}

    print("[info] extracting AllPools (once per image) and filling feature_bank...")
    for p in tqdm(needed_paths, desc="Embed"):
        img = Image.open(p).convert("RGB")
        inputs = build_inputs(processor, img, prompt)
        pools = extract_all_pools(model, inputs)

        for s, layers in source_to_layers.items():
            for li in layers:
                vec = extract_feature_from_pools(pools, s, li)
                feature_bank[s][li].append(vec)

    # stack
    bank_np: Dict[str, Dict[int, np.ndarray]] = {}
    for s, layers_dict in feature_bank.items():
        bank_np[s] = {}
        for li, vecs in layers_dict.items():
            bank_np[s][li] = np.stack(vecs, axis=0).astype(np.float32)  # [N,D]

    # 4) enumerate combos (two sources with replacement; two layers each)
    combos: List[Combo] = []
    src_list = sorted(bank_np.keys())
    for s1 in src_list:
        for s2 in src_list:
            for l1 in source_to_layers[s1]:
                for l2 in source_to_layers[s2]:
                    combos.append(Combo(s1, l1, s2, l2))
    print(f"[info] total combos = {len(combos)}")

    # 5) evaluate combos
    summary_rows = []
    preds_dir = os.path.join(args.out_dir, "preds")
    if args.save_preds:
        os.makedirs(preds_dir, exist_ok=True)

    alphas = np.logspace(-3, 3, args.alpha_grid)

    for combo in tqdm(combos, desc="Combos"):
        method_name = f"direct_pair_{combo.s1}_L{combo.l1}__{combo.s2}_L{combo.l2}"
        per_user_rho = []
        per_user_r2 = []
        n_users_used = 0

        pred_rows = []  # optional huge

        for uid in user_ids:
            pdata = personalized[uid]
            support_items = pdata.support_small if args.support_set == "small" else pdata.support_large
            test_items = pdata.test

            # build support matrices
            Xs = []
            ys = []
            for it in support_items:
                idx = path_to_idx.get(it.image_path)
                if idx is None:
                    continue
                v1 = bank_np[combo.s1][combo.l1][idx]
                v2 = bank_np[combo.s2][combo.l2][idx]
                Xs.append(np.concatenate([v1, v2], axis=0))
                ys.append(float(it.score))
            if len(ys) < 2:
                continue

            Xs = np.stack(Xs, axis=0).astype(np.float32)
            ys = np.array(ys, dtype=np.float32)

            pipe = make_pipeline(
                StandardScaler(with_std=True),
                RidgeCV(alphas=alphas)
            )
            pipe.fit(Xs, ys)

            # test predict
            y_true = []
            y_pred = []
            for it in test_items:
                idx = path_to_idx.get(it.image_path)
                if idx is None:
                    continue
                v1 = bank_np[combo.s1][combo.l1][idx]
                v2 = bank_np[combo.s2][combo.l2][idx]
                Xt = np.concatenate([v1, v2], axis=0)[None, :]
                pred = float(pipe.predict(Xt)[0])
                gt = float(it.score)

                y_true.append(gt)
                y_pred.append(pred)

                if args.save_preds:
                    pred_rows.append({
                        "user_id": uid,
                        "image_path": it.image_path,
                        "model_id": args.model_id,
                        "support_set": args.support_set,
                        "method": method_name,
                        "giaa": math.nan,
                        "piaa_pred": pred,
                        "user_score": gt,
                        "prompt_mode": args.prompt_mode,
                    })

            rho_u, r2_u = metrics_per_user(np.array(y_true), np.array(y_pred))
            per_user_rho.append(rho_u)
            per_user_r2.append(r2_u)
            n_users_used += 1

        mean_rho = float(np.mean(per_user_rho)) if per_user_rho else float("nan")
        mean_r2 = float(np.mean([x for x in per_user_r2 if np.isfinite(x)])) if any(np.isfinite(per_user_r2)) else float("nan")

        summary_rows.append({
            "dataset": args.dataset,
            "model_id": args.model_id,
            "support_set": args.support_set,
            "prompt_mode": args.prompt_mode,
            "s1": combo.s1, "l1": combo.l1,
            "s2": combo.s2, "l2": combo.l2,
            "n_users": n_users_used,
            "mean_rho": mean_rho,
            "mean_r2": mean_r2,
        })

        if args.save_preds and pred_rows:
            out_path = os.path.join(preds_dir, f"{sanitize(method_name)}.csv")
            fieldnames = ["user_id","image_path","model_id","support_set","method","giaa","piaa_pred","user_score","prompt_mode"]
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for r in pred_rows:
                    w.writerow(r)

    # 6) write summary
    summary_path = os.path.join(args.out_dir, "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["dataset","model_id","support_set","prompt_mode","s1","l1","s2","l2","n_users","mean_rho","mean_r2"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    print(f"[save] summary -> {summary_path}")
    print("[done]")


if __name__ == "__main__":
    main()