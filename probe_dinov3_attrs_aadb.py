#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DINOv3-B/16 (facebook/dinov3-vitb16-pretrain-lvd1689m) の全レイヤー特徴から，
AADB の美的属性 (AESTHETIC_ATTRIBUTES) に対する layer-wise linear probing を行い，
train/val/test の rho / RMSE / R^2 を JSON に保存するスクリプト。

出力フォーマットは Qwen/Gemma 用 probe_attrs_aadb.py と同じ形:
{
  "config": {
    "dinov3_model_id": "...",
    "dataset": "aadb",
    "train_split": "...",
    "val_split": "...",
    "test_split": "...",
    "quick": 100,
    "sources": ["vision"]
  },
  "attrs": {
    "<AttrName>": {
      "per_layer": [
        {"source": "vision", "layer": 0, "train": {...}, "val": {...}, "test": {...}},
        ...
      ],
      "best": {
        "source": "vision",
        "layer": k,
        "train": {...},
        "val": {...},
        "test": {...}
      }
    },
    ...
  }
}
"""

import os
import json
import math
import argparse
from typing import List, Dict, Tuple

import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# あなたの AADB ローダに合わせて調整してください
from utils.aadb import get_aadb_dataset, AESTHETIC_ATTRIBUTES
from utils.mm_embed import load_dinov3_model, extract_dinov3_all_layer_features


# ---------- 評価ユーティリティ ----------

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rho = spearmanr(y_true, y_pred).correlation
    if np.isnan(rho):
        rho = 0.0
    mse = float(np.mean((y_true - y_pred) ** 2))
    var = float(np.var(y_true)) + 1e-12
    r2 = 1.0 - mse / var
    rmse = math.sqrt(mse)
    return {"rho": float(rho), "r2": float(r2), "rmse": float(rmse)}


def _fit_eval_one_layer(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xval: np.ndarray,
    yval: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
) -> Tuple[Dict, Dict, Dict]:
    """
    1つのレイヤの特徴 (Xtr, Xval, Xte) に対して Ridge 回帰を行い，
    train/val/test の metrics を返す。
    """
    pipe = make_pipeline(
        StandardScaler(with_std=True),
        RidgeCV(alphas=np.logspace(-3, 3, 13))
    )
    pipe.fit(Xtr, ytr)
    yhat_tr = pipe.predict(Xtr)
    yhat_va = pipe.predict(Xval)
    yhat_te = pipe.predict(Xte)
    return (
        _metrics(ytr, yhat_tr),
        _metrics(yval, yhat_va),
        _metrics(yte,  yhat_te),
    )


def _rng_choice(seq, n, seed=0):
    if n is None or n >= len(seq):
        return list(seq)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(seq), size=n, replace=False)
    return [seq[i] for i in idx]


def _items_to_paths_and_targets(items) -> Tuple[List[str], Dict[str, List[float]]]:
    paths = [it.image_path for it in items]
    targets = {attr: [it.attributes[attr] for it in items] for attr in AESTHETIC_ATTRIBUTES}
    return paths, targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dinov3_model_id",
        default="facebook/dinov3-vitb16-pretrain-lvd1689m",
        help="DINOv3 Vision model ID",
    )
    ap.add_argument(
        "--dataset_dir",
        default="datasets/aadb",
        help="Path to AADB dataset directory",
    )
    ap.add_argument(
        "--train_split",
        default="train",
        help="AADB train split name",
    )
    ap.add_argument(
        "--val_split",
        default="validation",
        help="AADB validation split name",
    )
    ap.add_argument(
        "--test_split",
        default="test",
        help="AADB test split name",
    )
    ap.add_argument(
        "--out_json",
        default="runs/dinov3_vitb16_aadb_layer_attrs.json",
        help="Output JSON path for layer-wise attribute metrics",
    )
    ap.add_argument(
        "--quick",
        type=int,
        default=None,
        help="Limit the number of samples per split (for quick debugging)",
    )
    args = ap.parse_args()

    # 1) DINOv3 Vision モデルロード
    device = "cuda" if hasattr(__import__("torch"), "cuda") and __import__("torch").cuda.is_available() else "cpu"
    model, image_processor, device = load_dinov3_model(
        model_id=args.dinov3_model_id,
        dtype="auto",
        device=device,
    )

    # 2) AADB 読み込み
    tr_items = get_aadb_dataset(args.train_split, dataset_dir=args.dataset_dir)
    va_items = get_aadb_dataset(args.val_split,   dataset_dir=args.dataset_dir)
    te_items = get_aadb_dataset(args.test_split,  dataset_dir=args.dataset_dir)

    tr_items = _rng_choice(tr_items, args.quick, seed=0) if args.quick else tr_items
    va_items = _rng_choice(va_items, args.quick, seed=1) if args.quick else va_items
    te_items = _rng_choice(te_items, args.quick, seed=2) if args.quick else te_items

    tr_paths, tr_targets = _items_to_paths_and_targets(tr_items)
    va_paths, va_targets = _items_to_paths_and_targets(va_items)
    te_paths, te_targets = _items_to_paths_and_targets(te_items)

    print(f"[info] train: {len(tr_paths)}, val: {len(va_paths)}, test: {len(te_paths)}")

    # 3) DINOv3 全レイヤー特徴を抽出
    print("[info] extracting DINOv3 all-layer features (train)")
    Xtr_layers = extract_dinov3_all_layer_features(model, image_processor, device, tr_paths)
    print("[info] extracting DINOv3 all-layer features (val)")
    Xva_layers = extract_dinov3_all_layer_features(model, image_processor, device, va_paths)
    print("[info] extracting DINOv3 all-layer features (test)")
    Xte_layers = extract_dinov3_all_layer_features(model, image_processor, device, te_paths)

    n_layers = len(Xtr_layers)
    print(f"[info] #layers (including embedding layer) = {n_layers}")

    # 4) 属性ごとに layer-wise probing
    results = {
        "config": {
            "dinov3_model_id": args.dinov3_model_id,
            "dataset": "aadb",
            "dataset_dir": args.dataset_dir,
            "train_split": args.train_split,
            "val_split": args.val_split,
            "test_split": args.test_split,
            "quick": args.quick,
            "sources": ["vision"],
        },
        "attrs": {},
    }

    for attr in AESTHETIC_ATTRIBUTES:
        print(f"[attr] {attr}")
        ytr = np.array(tr_targets[attr], dtype=np.float32)
        yva = np.array(va_targets[attr], dtype=np.float32)
        yte = np.array(te_targets[attr], dtype=np.float32)
        # 上の2行はご自分の AADB ローダに合わせて:
        #   yva = np.array(va_targets[attr], dtype=np.float32)
        #   yte = np.array(te_targets[attr], dtype=np.float32)
        # にしてください。

        per_layer = []
        best = {"source": "vision", "layer": None,
                "train": None, "val": {"rho": -1, "r2": -1, "rmse": 1e9}, "test": None}

        for li in range(n_layers):
            Xtr = Xtr_layers[li]
            Xva = Xva_layers[li]
            Xte = Xte_layers[li]

            train_m, val_m, test_m = _fit_eval_one_layer(Xtr, ytr, Xva, yva, Xte, yte)
            item = {
                "source": "vision",
                "layer": li,
                "train": train_m,
                "val":   val_m,
                "test":  test_m,
            }
            per_layer.append(item)

            if val_m["rho"] > best["val"]["rho"]:
                best = {
                    "source": "vision",
                    "layer": li,
                    "train": train_m,
                    "val":   val_m,
                    "test":  test_m,
                }

        results["attrs"][attr] = {
            "per_layer": per_layer,
            "best": best,
        }

    # 5) JSON 保存
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[save] {args.out_json}")


if __name__ == "__main__":
    main()