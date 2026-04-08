#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Linear PIAA baseline on PARA using only General Aesthetic Attributes (GAA).

For each user u (from get_personalized_para_dataset):
  - Support set S_u (support_small or support_large)
  - Test set T_u

  Features x_i:
    - General Aesthetic Attributes from PARA-Giaa*.csv
    - We use AESTHETIC_ATTRIBUTES from utils.para, EXCLUDING "score".
      e.g. [composition, color, dof, light, content]

  Target score (controlled by --target_score):
    - piaa   : user_score_i (PIAA, per-user personalized score, 1~5)
    - giaa_gt: dataset-level GIAA mean (aestheticScore_mean, 1~5)

  Per-user model:
    - Train a Ridge regression y_i ~ x_i on S_u.
    - Predict on T_u and always evaluate against user_score (PIAA).

This baseline tells us how much of PIAA can be explained by General Aesthetic Attributes,
and allows comparing "trained on PIAA" vs "trained on GIAA mean".

Output CSV columns:
  user_id, image_path, model_id, support_set, method, giaa, piaa_pred, user_score

  - model_id: label for this baseline (e.g., "ga_attr_linear_small")
  - method  :
      * "ga_attr_linear"              (target_score=piaa)
      * "ga_attr_linear_giaa_gt"      (target_score=giaa_gt)
  - giaa    : dataset-level mean aestheticScore_mean for that image (GIAA ground truth)
"""

import os
import csv
import math
import argparse
from typing import Dict, List

import numpy as np
from tqdm import tqdm
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from utils.para import (
    get_para_dataset,
    get_personalized_para_dataset,
    AESTHETIC_ATTRIBUTES as PARA_ATTRS,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset_dir",
        default="datasets/PARA",
        help="Path to PARA dataset root (where annotation/ and imgs/ exist).",
    )
    ap.add_argument(
        "--support_set",
        default="small",
        choices=["small", "large"],
        help="Which support set from get_personalized_para_dataset to use.",
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
        "--model_id",
        default="ga_attr_linear",
        help="Model ID label to write into CSV (for grouping in eval scripts).",
    )
    ap.add_argument(
        "--target_score",
        default="piaa",
        choices=["piaa", "giaa_gt"],
        help="Which score to use as ground truth in training: "
             "'piaa' (user_score) or 'giaa_gt' (dataset-level GIAA mean).",
    )
    ap.add_argument(
        "--out_csv",
        required=True,
        help="Path to output CSV for per-user test predictions.",
    )
    args = ap.parse_args()

    # 1) Load General Aesthetic Attributes (PARA-GiaaTrain/Val/Test)
    print("[info] loading general aesthetic attributes from PARA-Giaa*.csv ...")
    ga_items = get_para_dataset(None, dataset_dir=args.dataset_dir)

    # Use only attributes from AESTHETIC_ATTRIBUTES excluding "score"
    attr_names = [a for a in PARA_ATTRS if a not in ['quality', 'score']]
    print(f"[info] using attributes (excluding 'quality', 'score'): {attr_names}")

    # Build a dictionary mapping image_path -> (ga_score, attr_vector)
    path_to_ga: Dict[str, float] = {}
    path_to_attrs: Dict[str, np.ndarray] = {}

    for it in ga_items:
        path_to_ga[it.image_path] = float(it.score)  # dataset-level GIAA mean
        vec = np.array([float(it.attributes[a]) for a in attr_names], dtype=np.float32)
        path_to_attrs[it.image_path] = vec

    print(f"[info] total GA entries: {len(path_to_attrs)}")

    # 2) Load personalized data
    print("[info] loading personalized PARA dataset...")
    personalized = get_personalized_para_dataset(seed=args.seed, dataset_dir=args.dataset_dir)
    all_user_ids = sorted(personalized.keys())
    print(f"[info] num users in personalized dataset: {len(all_user_ids)}")

    # quick: limit the number of users
    if args.quick is not None and args.quick < len(all_user_ids):
        user_ids = all_user_ids[:args.quick]
        print(f"[info] quick mode: using first {len(user_ids)} users out of {len(all_user_ids)}")
    else:
        user_ids = all_user_ids

    rows: List[dict] = []
    if args.target_score == "piaa":
        method_name = "ga_attr_linear"
    else:
        method_name = "ga_attr_linear_giaa_gt"

    # 3) Train RidgeCV for each user
    for user_id in tqdm(user_ids, desc="Users(GA-linear)"):
        pdata = personalized[user_id]
        if args.support_set == "small":
            support_items = pdata.support_small
        else:
            support_items = pdata.support_large
        test_items = pdata.test

        # Build the support set
        X_support = []
        y_support = []
        for it in support_items:
            path = it.image_path
            if path not in path_to_attrs:
                continue
            # Features
            X_support.append(path_to_attrs[path])
            # Target (selected according to target_score)
            user_score = float(it.score)
            if args.target_score == "piaa":
                target = user_score
            else:  # giaa_gt
                if path not in path_to_ga:
                    continue
                target = path_to_ga[path]
            y_support.append(target)

        X_support = np.array(X_support, dtype=np.float32)
        y_support = np.array(y_support, dtype=np.float32)

        if len(y_support) < 2:
            # Skip users without enough support samples for training
            continue

        # RidgeCV + StandardScaler pipeline
        pipe = make_pipeline(
            StandardScaler(with_std=True),
            RidgeCV(alphas=np.logspace(-3, 3, 13)),
        )
        pipe.fit(X_support, y_support)

        # Predict on the test set (evaluation is always against user_score)
        for it in test_items:
            path = it.image_path
            if path not in path_to_attrs:
                continue
            x = path_to_attrs[path][None, :]  # [1, D]
            piaa_pred = pipe.predict(x)[0]
            user_score = float(it.score)
            ga_score = path_to_ga.get(path, math.nan)  # dataset-level GIAA mean

            rows.append(
                {
                    "user_id": user_id,
                    "image_path": path,
                    "model_id": args.model_id,
                    "support_set": args.support_set,
                    "method": method_name,
                    "giaa": ga_score,        # GIAA ground truth (mean aestheticScore_mean)
                    "piaa_pred": piaa_pred,  # predicted score (PIAA or GIAA-driven, depending on target_score)
                    "user_score": user_score,
                }
            )

    # 4) Save CSV
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
