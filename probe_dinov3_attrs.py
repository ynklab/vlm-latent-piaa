#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DINOv3-B/16 (facebook/dinov3-vitb16-pretrain-lvd1689m) の全レイヤー特徴から，
AADB / PARA の美的属性に対して layer-wise linear probing を行うスクリプト。

出力 JSON フォーマットは Qwen/Gemma 用の probe_attrs_* とほぼ同じ:
{
  "config": {
    "dinov3_model_id": "...",
    "dataset": "aadb" or "para",
    "dataset_dir": "...",
    "train_split": "...",
    "val_split": "...",
    "test_split": "...",
    "quick": 200,
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
        "val":   {...},
        "test":  {...}
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

import torch

# データセットローダ
from utils.aadb import get_aadb_dataset, AESTHETIC_ATTRIBUTES as AADB_ATTRS
from utils.para import get_para_dataset, AESTHETIC_ATTRIBUTES as PARA_ATTRS

# DINOv3 ローダ＆全レイヤー特徴抽出
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

def _finite_report(X: np.ndarray, name: str):
    X = X.astype(np.float32, copy=False)
    bad = ~np.isfinite(X)
    if bad.any():
        n_bad = int(bad.sum())
        max_abs = float(np.nanmax(np.abs(X[np.isfinite(X)]))) if np.isfinite(X).any() else float("nan")
        print(f"[warn] {name}: non-finite={n_bad}, max_abs_finite={max_abs}")
        # どの行（画像）が壊れているか
        bad_rows = np.where(~np.isfinite(X).all(axis=1))[0]
        print(f"       bad_rows (first 10)={bad_rows[:10].tolist()}")
        return True
    return False


def _fit_eval_one_layer(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
) -> Tuple[Dict, Dict, Dict]:
    """
    1レイヤ分の特徴 (Xtr, Xva, Xte) に対して Ridge 回帰を行い，
    train / val / test の metrics を返す。
    """
    pipe = make_pipeline(
        StandardScaler(with_std=True),
        RidgeCV(alphas=np.logspace(-3, 3, 13))
    )
    pipe.fit(Xtr, ytr)
    yhat_tr = pipe.predict(Xtr)
    yhat_va = pipe.predict(Xva)
    yhat_te = pipe.predict(Xte)
    return (
        _metrics(ytr, yhat_tr),
        _metrics(yva, yhat_va),
        _metrics(yte,  yhat_te),
    )


def _rng_choice(seq, n, seed=0):
    if n is None or n >= len(seq):
        return list(seq)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(seq), size=n, replace=False)
    return [seq[i] for i in idx]


def _items_to_paths_and_targets(items, attrs: List[str]) -> Tuple[List[str], Dict[str, np.ndarray]]:
    paths = [it.image_path for it in items]
    targets: Dict[str, List[float]] = {a: [] for a in attrs}
    for it in items:
        for a in attrs:
            # attributes dict に入っている前提（AADB / PARA とも）
            targets[a].append(float(it.attributes[a]))
    return paths, {k: np.array(v, dtype=np.float32) for k, v in targets.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dinov3_model_id",
        default="facebook/dinov3-vitb16-pretrain-lvd1689m",
        help="DINOv3 Vision model ID",
    )
    ap.add_argument(
        "--dataset",
        default="aadb",
        choices=["aadb", "para"],
        help="Which dataset to use: aadb or para",
    )
    ap.add_argument(
        "--dataset_dir",
        default=None,
        help="Path to dataset root. If None, uses datasets/aadb or datasets/PARA.",
    )
    ap.add_argument(
        "--train_split",
        default=None,
        help="Train split name (default depends on dataset).",
    )
    ap.add_argument(
        "--val_split",
        default=None,
        help="Val split name (default depends on dataset).",
    )
    ap.add_argument(
        "--test_split",
        default=None,
        help="Test split name (default depends on dataset).",
    )
    ap.add_argument(
        "--out_json",
        default="runs/dinov3_vitb16_attrs.json",
        help="Output JSON path for layer-wise attribute metrics",
    )
    ap.add_argument(
        "--quick",
        type=int,
        default=None,
        help="Limit the number of samples per split (for quick debugging)",
    )
    args = ap.parse_args()

    # ----- dataset 設定 -----
    if args.dataset_dir is None:
        args.dataset_dir = "datasets/aadb" if args.dataset == "aadb" else "datasets/PARA"

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
        # PARA は train/test のみなので、とりあえず val=test として共有（必要なら内部splitで改善）
        val_split   = args.val_split   or "test"
        test_split  = args.test_split  or "test"

    print(f"[info] dataset={args.dataset}, dir={args.dataset_dir}")
    print(f"[info] splits: train={train_split}, val={val_split}, test={test_split}")
    print(f"[info] attributes: {attrs}")

    # ----- DINOv3 Vision モデルロード -----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, image_processor, device = load_dinov3_model(
        model_id=args.dinov3_model_id,
        dtype="auto",
        device=device,
    )

    # ----- データ読み込み -----
    tr_items = get_dataset(train_split, dataset_dir=args.dataset_dir)
    va_items = get_dataset(val_split,   dataset_dir=args.dataset_dir)
    te_items = get_dataset(test_split,  dataset_dir=args.dataset_dir)

    if args.quick is not None:
        tr_items = _rng_choice(tr_items, args.quick, seed=0)
        va_items = _rng_choice(va_items, args.quick, seed=1)
        te_items = _rng_choice(te_items, args.quick, seed=2)

    tr_paths, tr_targets = _items_to_paths_and_targets(tr_items, attrs)
    va_paths, va_targets = _items_to_paths_and_targets(va_items, attrs)
    te_paths, te_targets = _items_to_paths_and_targets(te_items, attrs)

    print(f"[info] N train={len(tr_paths)}, val={len(va_paths)}, test={len(te_paths)}")

    # ----- DINOv3 全レイヤー特徴抽出 -----
    print("[info] extracting DINOv3 all-layer features (train)")
    Xtr_layers = extract_dinov3_all_layer_features(model, image_processor, device, tr_paths)
    print("[info] extracting DINOv3 all-layer features (val)")
    Xva_layers = extract_dinov3_all_layer_features(model, image_processor, device, va_paths)
    print("[info] extracting DINOv3 all-layer features (test)")
    Xte_layers = extract_dinov3_all_layer_features(model, image_processor, device, te_paths)

    n_layers = len(Xtr_layers)
    print(f"[info] #layers (including embedding layer) = {n_layers}")

    # ----- 属性ごとに layer-wise probing -----
    results = {
        "config": {
            "dinov3_model_id": args.dinov3_model_id,
            "dataset": args.dataset,
            "dataset_dir": args.dataset_dir,
            "train_split": train_split,
            "val_split": val_split,
            "test_split": test_split,
            "quick": args.quick,
            "sources": ["vision"],
        },
        "attrs": {},
    }

    for attr in attrs:
        print(f"[attr] {attr}")
        ytr = tr_targets[attr]
        yva = va_targets[attr]
        yte = te_targets[attr]

        per_layer = []
        best = {
            "source": "vision",
            "layer": None,
            "train": None,
            "val":   {"rho": -1, "r2": -1, "rmse": 1e9},
            "test":  None,
        }

        for li in range(n_layers):
            Xtr = Xtr_layers[li]
            Xva = Xva_layers[li]
            Xte = Xte_layers[li]
            assert not _finite_report(Xtr, f"Xtr L{li}")

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

    # ----- JSON 保存 -----
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[save] {args.out_json}")


if __name__ == "__main__":
    main()