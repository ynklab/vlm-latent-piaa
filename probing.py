# probe_attrs_aadb.py (v3)
# 変更点:
#  - --quick N: 各 split から最大N枚だけを使って高速評価
#  - 指標に RMSE を追加
#  - LLMの text_pool / visual_pool を常時 全層評価
#  - Vision Encoder 各層 / VLブリッジ (LLM直前) も評価対象に追加
#
# 出力:
#   results["attrs"][attr]["per_layer"] に source/layer ごとの {val:{rho,r2,rmse}, test:{...}}
#   source ∈ {"llm_text","llm_visual","vision","bridge_text","bridge_visual"}

import os, json, argparse, math, random
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# あなたのAADBローダに合わせて修正
from utils.aadb import get_aadb_dataset, AESTHETIC_ATTRIBUTES

from utils.qwen3vl_embed import load_qwen3vl, build_inputs, extract_all_pools


def _rng_choice(seq, n, seed=0):
    if n is None or n >= len(seq):
        return list(seq)
    rng = random.Random(seed)
    idx = list(range(len(seq)))
    rng.shuffle(idx)
    idx = idx[:n]
    return [seq[i] for i in idx]


def _items_to_paths_and_targets(items) -> Tuple[List[str], Dict[str, List[float]]]:
    paths = [it.image_path for it in items]
    targets = {
        attr: [it.attributes[attr] for it in items] for attr in AESTHETIC_ATTRIBUTES
    }
    return paths, targets


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rho = spearmanr(y_true, y_pred).correlation
    if np.isnan(rho):
        rho = 0.0
    mse = float(np.mean((y_true - y_pred) ** 2))
    var = float(np.var(y_true)) + 1e-12
    r2 = 1.0 - mse / var
    rmse = math.sqrt(mse)
    return {"rho": float(rho), "r2": float(r2), "rmse": float(rmse)}


def _fit_eval_one_layer(Xtr, ytr, Xval, yval, Xte, yte, refit_trainval: bool):
    pipe = make_pipeline(
        StandardScaler(with_std=True), RidgeCV(alphas=np.logspace(-3, 3, 13))
    )
    pipe.fit(Xtr, ytr)
    val_m = _metrics(yval, pipe.predict(Xval))
    if refit_trainval:
        Xtrval = np.concatenate([Xtr, Xval], axis=0)
        ytrval = np.concatenate([ytr, yval], axis=0)
        pipe = make_pipeline(
            StandardScaler(with_std=True), RidgeCV(alphas=np.logspace(-3, 3, 13))
        )
        pipe.fit(Xtrval, ytrval)
    test_m = _metrics(yte, pipe.predict(Xte))
    return val_m, test_m


def _concat_sources(
    feature_bank_tr, feature_bank_val, feature_bank_te, ytr, yval, yte, refit_trainval
):
    """
    feature_bank_*: dict[str -> List[np.ndarray]]  例:
      {
        "llm_text":   [X^0, X^1, ..., X^L],   # 各 [N, D_text]
        "llm_visual": [ ... ],
        "vision":     [V^0, ..., V^Lv],       # 各 [N, D_vis] (無い場合は None)
        "bridge_text":[B_text],               # len=1
        "bridge_visual":[B_vis],              # len=1
      }
    戻り値:
      per_layer(list[dict]), best_global(dict)
    """
    per_layer = []
    best = {
        "source": None,
        "layer": None,
        "val": {"rho": -1, "r2": -1, "rmse": 1e9},
        "test": None,
    }
    for src in ["llm_text", "llm_visual", "vision", "bridge_text", "bridge_visual"]:
        if feature_bank_tr.get(src) is None:
            continue
        for li, (Xtr, Xval, Xte) in enumerate(
            zip(feature_bank_tr[src], feature_bank_val[src], feature_bank_te[src])
        ):
            val_m, test_m = _fit_eval_one_layer(
                Xtr, ytr, Xval, yval, Xte, yte, refit_trainval
            )
            item = {"source": src, "layer": li, "val": val_m, "test": test_m}
            per_layer.append(item)
            if val_m["rho"] > best["val"]["rho"]:
                best = {"source": src, "layer": li, "val": val_m, "test": test_m}
    return per_layer, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default="datasets/aadb")
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--val_split", default="validation")
    ap.add_argument("--test_split", default="test")
    ap.add_argument(
        "--quick",
        type=int,
        default=None,
        help="各splitのサンプルを最大N枚に制限（例: --quick 10）",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model_id", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--attn_impl", default=None)
    ap.add_argument("--prompt", default="Assess the aesthetics of this image.")
    ap.add_argument("--out_json", default="probe_aadb_results.json")
    ap.add_argument(
        "--refit_trainval", action="store_true", help="Test評価前に Train+Val で再学習"
    )
    args = ap.parse_args()

    # 1) データ読み込み + クイックサンプル
    tr_items = get_aadb_dataset(args.train_split, dataset_dir=args.dataset_dir)
    va_items = get_aadb_dataset(args.val_split, dataset_dir=args.dataset_dir)
    te_items = get_aadb_dataset(args.test_split, dataset_dir=args.dataset_dir)
    if args.quick is not None:
        tr_items = _rng_choice(tr_items, args.quick, args.seed)
        va_items = _rng_choice(va_items, args.quick, args.seed + 1)
        te_items = _rng_choice(te_items, args.quick, args.seed + 2)

    tr_paths, tr_targets = _items_to_paths_and_targets(tr_items)
    va_paths, va_targets = _items_to_paths_and_targets(va_items)
    te_paths, te_targets = _items_to_paths_and_targets(te_items)

    # 2) モデル
    model, processor = load_qwen3vl(
        args.model_id, args.dtype, args.device_map, args.attn_impl
    )
    model.eval()

    # 3) 特徴抽出（splitごと・画像ごとに AllPools を集約）
    # 構造: feature_bank_tr["llm_text"][li] -> [N,D]
    def _empty_bank():
        return {
            "llm_text": [],
            "llm_visual": [],
            "vision": [],
            "bridge_text": [],
            "bridge_visual": [],
        }

    feature_bank_tr = _empty_bank()
    feature_bank_va = _empty_bank()
    feature_bank_te = _empty_bank()

    def _accumulate(paths, bank):
        all_pools_list = []
        for p in tqdm(paths, desc="Extract"):
            img = Image.open(p).convert("RGB")
            inputs = build_inputs(processor, img, args.prompt)
            pools = extract_all_pools(model, inputs)
            all_pools_list.append(pools)

        # LLM: 層数合わせ
        L = len(all_pools_list[0].llm_text)
        for li in range(L):
            X_text = [ap.llm_text[li] for ap in all_pools_list]  # List[[D]] -> [N,D]
            X_vis = [ap.llm_visual[li] for ap in all_pools_list]
            bank["llm_text"].append(np.stack(X_text, axis=0))
            bank["llm_visual"].append(np.stack(X_vis, axis=0))

        # Vision: ない場合もある
        if all_pools_list[0].vision_layers is not None:
            Lv = len(all_pools_list[0].vision_layers)
            for li in range(Lv):
                Xv = [ap.vision_layers[li] for ap in all_pools_list]
                bank["vision"].append(np.stack(Xv, axis=0))
        else:
            bank["vision"] = None

        # Bridge: len=1
        X_bt = [ap.bridge_text[0] for ap in all_pools_list]
        X_bv = [ap.bridge_visual[0] for ap in all_pools_list]
        bank["bridge_text"].append(np.stack(X_bt, axis=0))
        bank["bridge_visual"].append(np.stack(X_bv, axis=0))

    _accumulate(tr_paths, feature_bank_tr)
    _accumulate(va_paths, feature_bank_va)
    _accumulate(te_paths, feature_bank_te)

    # 4) 属性ごとに学習・評価（すべてのsource × すべての層）
    results = {
        "config": {
            "model_id": args.model_id,
            "train_split": args.train_split,
            "val_split": args.val_split,
            "test_split": args.test_split,
            "prompt": args.prompt,
            "refit_trainval": bool(args.refit_trainval),
            "quick": args.quick,
            "seed": args.seed,
        },
        "attrs": {},
    }

    for attr in AESTHETIC_ATTRIBUTES:
        ytr = np.array(tr_targets[attr], dtype=np.float32)
        yva = np.array(va_targets[attr], dtype=np.float32)
        yte = np.array(te_targets[attr], dtype=np.float32)

        per_layer, best = _concat_sources(
            feature_bank_tr,
            feature_bank_va,
            feature_bank_te,
            ytr,
            yva,
            yte,
            args.refit_trainval,
        )
        results["attrs"][attr] = {
            "per_layer": per_layer,  # [{source, layer, val:{rho,r2,rmse}, test:{...}}, ...]
            "best": best,  # validation rho 最大の組合せ
        }

    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print("== saved:", args.out_json)


if __name__ == "__main__":
    main()
