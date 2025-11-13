# probe_attrs_aadb.py (v8: ConstantInputWarning を抑制 + 定数入力のロギング追加)
import os, json, argparse, math, random, warnings
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import spearmanr
# --- SciPyの警告クラスを堅牢に取得
try:
    from scipy.stats import ConstantInputWarning
except Exception:
    try:
        from scipy.stats._stats_py import ConstantInputWarning  # 古いSciPy
    except Exception:
        class ConstantInputWarning(UserWarning): ...
# 既定でこの警告は非表示に
warnings.filterwarnings("ignore", category=ConstantInputWarning)

from utils.aadb import get_aadb_dataset, AESTHETIC_ATTRIBUTES
from utils.qwen3vl_embed import load_qwen3vl, build_inputs, extract_all_pools

def _rng_choice(seq, n, seed=0):
    if n is None or n >= len(seq): return list(seq)
    rng = random.Random(seed)
    idx = list(range(len(seq))); rng.shuffle(idx); idx = idx[:n]
    return [seq[i] for i in idx]

def _items_to_paths_and_targets(items) -> Tuple[List[str], Dict[str, List[float]]]:
    paths = [it.image_path for it in items]
    targets = {attr: [it.attributes[attr] for it in items] for attr in AESTHETIC_ATTRIBUTES}
    return paths, targets

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    # Spearmanは定数入力でNaNになるので、警告は抑制済み。NaNなら0扱いにする。
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        rho = spearmanr(y_true, y_pred).correlation
    if np.isnan(rho): rho = 0.0
    mse = float(np.mean((y_true - y_pred) ** 2))
    var = float(np.var(y_true)) + 1e-12
    r2 = 1.0 - mse / var
    rmse = math.sqrt(mse)
    return {"rho": float(rho), "r2": float(r2), "rmse": float(rmse)}

def _append_jsonl(path: str, record: dict):
    if path is None: 
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def _const_flags(arr: np.ndarray, tol: float) -> Tuple[bool, float, float, int, float]:
    # 近似定数判定: 標準偏差≦tol
    arr = np.asarray(arr, dtype=float)
    std = float(np.nanstd(arr))
    is_const = bool(std <= tol)
    vmin = float(np.nanmin(arr)) if arr.size else float("nan")
    vmax = float(np.nanmax(arr)) if arr.size else float("nan")
    uniq = int(np.unique(np.round(arr, 8)).size) if arr.size else 0
    return is_const, vmin, vmax, uniq, std

def _maybe_log_constant(log_path: str, ctx: dict, split: str, y_true, y_pred, X=None, tol=1e-8, debug_feature_var=False):
    # y_true / y_pred の定数チェック
    yt_flag, yt_min, yt_max, yt_uniq, yt_std = _const_flags(y_true, tol)
    yp_flag, yp_min, yp_max, yp_uniq, yp_std = _const_flags(y_pred, tol)

    need = yt_flag or yp_flag
    rec = None
    if need:
        rec = {
            "model_id": ctx.get("model_id"),
            "attr": ctx.get("attr"),
            "source": ctx.get("source"),
            "layer": int(ctx.get("layer", -1)),
            "split": split,
            "n_samples": int(len(y_true)),
            "y_true_const": yt_flag,
            "y_true_min": yt_min, "y_true_max": yt_max, "y_true_unique": yt_uniq, "y_true_std": yt_std,
            "y_pred_const": yp_flag,
            "y_pred_min": yp_min, "y_pred_max": yp_max, "y_pred_unique": yp_uniq, "y_pred_std": yp_std,
        }
        if debug_feature_var and X is not None:
            try:
                # 特徴の分散合計（0なら「全特徴が定数」）
                rec["X_var_sum"] = float(np.var(X, axis=0).sum())
                rec["X_n"] = int(X.shape[0]); rec["X_d"] = int(X.shape[1])
            except Exception:
                pass
        _append_jsonl(log_path, rec)
    return rec  # 使用しないが返しておく

def _fit_eval_one_layer(Xtr, ytr, Xval, yval, Xte, yte,
                        ctx: dict, log_constant_path: str, tol: float, debug_feature_var: bool):
    pipe = make_pipeline(StandardScaler(with_std=True), RidgeCV(alphas=np.logspace(-3,3,13)))
    pipe.fit(Xtr, ytr)

    yhat_tr = pipe.predict(Xtr); train_m = _metrics(ytr, yhat_tr)
    yhat_va = pipe.predict(Xval); val_m   = _metrics(yval, yhat_va)
    yhat_te = pipe.predict(Xte);  test_m  = _metrics(yte,  yhat_te)

    # 必要なら定数ケースのログ（train/val/test それぞれ）
    _maybe_log_constant(log_constant_path, ctx, "train", ytr, yhat_tr, Xtr, tol, debug_feature_var)
    _maybe_log_constant(log_constant_path, ctx, "val",   yval, yhat_va, Xval, tol, debug_feature_var)
    _maybe_log_constant(log_constant_path, ctx, "test",  yte,  yhat_te, Xte, tol, debug_feature_var)

    return train_m, val_m, test_m

def _count_total_layers(feature_bank_tr: Dict[str, List[np.ndarray]]) -> int:
    total = 0
    for src in ["llm_text","llm_text_tail","llm_visual","vision","bridge_text","bridge_visual"]:
        lst = feature_bank_tr.get(src)
        if lst is None:
            continue
        total += len(lst)
    return total

def _concat_sources(feature_bank_tr, feature_bank_val, feature_bank_te, ytr, yval, yte,
                    attr_name: str, model_id: str, log_constant_path: str, tol: float, debug_feature_var: bool):
    per_layer = []
    best = {"source": None, "layer": None,
            "train": None, "val": {"rho": -1, "r2": -1, "rmse": 1e9}, "test": None}

    total = _count_total_layers(feature_bank_tr)
    pbar = tqdm(total=total, desc=f"Regress[{attr_name}]", leave=False)

    try:
        for src in ["llm_text","llm_text_tail","llm_visual","vision","bridge_text","bridge_visual"]:
            bank_tr = feature_bank_tr.get(src)
            bank_va = feature_bank_val.get(src)
            bank_te = feature_bank_te.get(src)
            if bank_tr is None:
                continue
            for li, (Xtr, Xval, Xte) in enumerate(zip(bank_tr, bank_va, bank_te)):
                ctx = {"model_id": model_id, "attr": attr_name, "source": src, "layer": li}
                train_m, val_m, test_m = _fit_eval_one_layer(
                    Xtr, ytr, Xval, yval, Xte, yte,
                    ctx=ctx, log_constant_path=log_constant_path, tol=tol, debug_feature_var=debug_feature_var
                )
                item = {"source": src, "layer": li, "train": train_m, "val": val_m, "test": test_m}
                per_layer.append(item)
                if val_m["rho"] > best["val"]["rho"]:
                    best = {"source": src, "layer": li, "train": train_m, "val": val_m, "test": test_m}
                pbar.update(1)
    finally:
        pbar.close()

    return per_layer, best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default="datasets/aadb")
    ap.add_argument("--train_split", default="train")
    ap.add_argument("--val_split", default="validation")
    ap.add_argument("--test_split", default="test")
    ap.add_argument("--quick", type=int, default=None, help="各splitのサンプルを最大N枚に制限（例: --quick 10）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model_id", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--attn_impl", default=None)
    ap.add_argument("--prompt", default="Assess the aesthetics of this image.")
    ap.add_argument("--out_json", default="probe_aadb_results.json")
    # ← 追加：定数入力のロギング
    ap.add_argument("--log_constant", default=None, help="定数入力を検知したとき JSONL に追記するパス（省略時はロギング無し）")
    ap.add_argument("--constant_tol", type=float, default=1e-8, help="定数判定の標準偏差しきい値")
    ap.add_argument("--debug_feature_var", action="store_true", help="定数発生時に特徴量の分散合計も記録")
    args = ap.parse_args()

    # 1) データ
    tr_items = get_aadb_dataset(args.train_split, dataset_dir=args.dataset_dir)
    va_items = get_aadb_dataset(args.val_split,   dataset_dir=args.dataset_dir)
    te_items = get_aadb_dataset(args.test_split,  dataset_dir=args.dataset_dir)
    if args.quick is not None:
        tr_items = _rng_choice(tr_items, args.quick, args.seed)
        va_items = _rng_choice(va_items, args.quick, args.seed+1)
        te_items = _rng_choice(te_items, args.quick, args.seed+2)

    tr_paths, tr_targets = _items_to_paths_and_targets(tr_items)
    va_paths, va_targets = _items_to_paths_and_targets(va_items)
    te_paths, te_targets = _items_to_paths_and_targets(te_items)

    # 2) モデル
    model, processor = load_qwen3vl(args.model_id, args.dtype, args.device_map, args.attn_impl)
    model.eval()

    # 3) 特徴抽出
    def _empty_bank():
        return {
            "llm_text": [], "llm_text_tail": [], "llm_visual": [],
            "vision": [], "bridge_text": [], "bridge_visual": []
        }
    feature_bank_tr = _empty_bank()
    feature_bank_va = _empty_bank()
    feature_bank_te = _empty_bank()

    def _accumulate(paths, bank, split_name):
        all_pools = []
        for p in tqdm(paths, desc=f"Extract[{split_name}]", leave=False):
            img = Image.open(p).convert("RGB")
            inputs = build_inputs(processor, img, args.prompt)
            pools = extract_all_pools(model, inputs)
            all_pools.append(pools)

        # LLM: 層数合わせ
        L = len(all_pools[0].llm_text)
        for li in range(L):
            bank["llm_text"].append(np.stack([ap.llm_text[li] for ap in all_pools], axis=0))
            bank["llm_text_tail"].append(np.stack([ap.llm_text_tail[li] for ap in all_pools], axis=0))
            bank["llm_visual"].append(np.stack([ap.llm_visual[li] for ap in all_pools], axis=0))

        # Vision
        if all_pools[0].vision_layers is not None:
            Lv = len(all_pools[0].vision_layers)
            for li in range(Lv):
                bank["vision"].append(np.stack([ap.vision_layers[li] for ap in all_pools], axis=0))
        else:
            bank["vision"] = None

        # Bridge（len=1）
        bank["bridge_text"].append(np.stack([ap.bridge_text[0] for ap in all_pools], axis=0))
        bank["bridge_visual"].append(np.stack([ap.bridge_visual[0] for ap in all_pools], axis=0))

    _accumulate(tr_paths, feature_bank_tr, "train")
    _accumulate(va_paths, feature_bank_va, "val")
    _accumulate(te_paths, feature_bank_te, "test")

    # 4) 属性ごとに学習・評価（回帰に tqdm + 定数ログ）
    results = {
        "config": {
            "model_id": args.model_id,
            "train_split": args.train_split,
            "val_split": args.val_split,
            "test_split": args.test_split,
            "prompt": args.prompt,
            "quick": args.quick,
            "seed": args.seed,
        },
        "attrs": {},
    }

    for attr in tqdm(AESTHETIC_ATTRIBUTES, desc="Attributes"):
        ytr = np.array(tr_targets[attr], dtype=np.float32)
        yva = np.array(va_targets[attr], dtype=np.float32)
        yte = np.array(te_targets[attr], dtype=np.float32)

        per_layer, best = _concat_sources(
            feature_bank_tr, feature_bank_va, feature_bank_te,
            ytr, yva, yte,
            attr_name=attr, model_id=args.model_id,
            log_constant_path=args.log_constant, tol=args.constant_tol,
            debug_feature_var=args.debug_feature_var
        )

        results["attrs"][attr] = {
            "per_layer": per_layer,   # [{source, layer, train:{...}, val:{...}, test:{...}}, ...]
            "best": best,             # 選択は val.rho 最大
        }

    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print("== saved:", args.out_json)

if __name__ == "__main__":
    main()
