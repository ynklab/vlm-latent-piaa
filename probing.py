import os, json, argparse, math, random, warnings, shutil, tempfile
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import spearmanr
import torch

# --- SciPyの警告クラスを堅牢に取得
try:
    from scipy.stats import ConstantInputWarning
except Exception:
    try:
        from scipy.stats._stats_py import ConstantInputWarning
    except Exception:
        class ConstantInputWarning(UserWarning): ...
warnings.filterwarnings("ignore", category=ConstantInputWarning)

from utils.para import get_para_dataset, AESTHETIC_ATTRIBUTES as PARA_ATTRS
from utils.aadb import get_aadb_dataset, AESTHETIC_ATTRIBUTES as AADB_ATTRS
from utils.mm_embed import load_mm_model, build_inputs, extract_all_pools

# === OOM対策: ディスクへ特徴量を退避させるクラス ===
class DiskFeatureBank:
    """
    特徴量をメモリに溜めず、一時ディレクトリ内のバイナリファイルに書き出す。
    読み出し時は np.memmap を使用してメモリ消費を抑える。
    """
    def __init__(self, temp_dir):
        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)
        # file_handles: { "source_layer": binary_file_handle }
        self.file_handles = {}
        # metadata: { "source_layer": {"count": int, "dim": int} }
        self.metadata = {}

    def append(self, source: str, layer: int, vector):
        key = f"{source}_{layer}"
        
        # Tensorならnumpyへ変換 (GPUメモリ解放のため必須)
        if hasattr(vector, "detach"):
            vector = vector.detach().cpu().numpy()
        
        # 初回: ファイル作成とメタデータ記録
        if key not in self.file_handles:
            path = os.path.join(self.temp_dir, f"{key}.bin")
            self.file_handles[key] = open(path, "wb")
            self.metadata[key] = {"count": 0, "dim": vector.shape[-1]}
        
        # float32にして書き込み
        data = vector.astype(np.float32)
        self.file_handles[key].write(data.tobytes())
        self.metadata[key]["count"] += 1

    def close(self):
        """書き込み終了。ファイルを閉じる"""
        for f in self.file_handles.values():
            f.close()
        self.file_handles = {}

    def get(self, source: str, layer: int):
        """
        指定された source, layer のデータを np.memmap (ReadOnly) として取得。
        """
        key = f"{source}_{layer}"
        if key not in self.metadata:
            return None
        
        meta = self.metadata[key]
        path = os.path.join(self.temp_dir, f"{key}.bin")
        # 形状: (サンプル数, 次元数)
        shape = (meta["count"], meta["dim"])
        
        # メモリにロードせず、ディスク上のファイルを配列として扱う
        try:
            return np.memmap(path, dtype=np.float32, mode="r", shape=shape)
        except FileNotFoundError:
            return None

def make_prompt(mode: str, attrs: list[str]) -> str:
    if mode == "base":
        return "Assess the aesthetics of this image."
    elif mode == "format":
        return "Assess the aesthetics of this image. Please rate it on a scale from 1 to 5. Output only the numeric score, and do not output any other text."
    elif mode == "attributes":
        attrs_str = ", ".join(attrs)
        return f"Assess the aesthetics of this image with respect to the following attributes: {attrs_str}. You do not need to output the attributes explicitly; just use them as internal criteria."
    elif mode == "unrelated":
        return "Describe the weather today in one sentence."
    else:
        raise ValueError(f"Unknown prompt_mode: {mode}")

def _rng_choice(seq, n, seed=0):
    if n is None or n >= len(seq): return list(seq)
    rng = random.Random(seed)
    idx = list(range(len(seq))); rng.shuffle(idx); idx = idx[:n]
    return [seq[i] for i in idx]

def _items_to_paths_and_targets(items, attrs) -> Tuple[List[str], Dict[str, List[float]]]:
    paths = [it.image_path for it in items]
    targets = {attr: [it.attributes[attr] for it in items] for attr in attrs}
    return paths, targets

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
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
    arr = np.asarray(arr, dtype=float)
    std = float(np.nanstd(arr))
    is_const = bool(std <= tol)
    vmin = float(np.nanmin(arr)) if arr.size else float("nan")
    vmax = float(np.nanmax(arr)) if arr.size else float("nan")
    uniq = int(np.unique(np.round(arr, 8)).size) if arr.size else 0
    return is_const, vmin, vmax, uniq, std

def _maybe_log_constant(log_path: str, ctx: dict, split: str, y_true, y_pred, X=None, tol=1e-8, debug_feature_var=False):
    yt_flag, yt_min, yt_max, yt_uniq, yt_std = _const_flags(y_true, tol)
    yp_flag, yp_min, yp_max, yp_uniq, yp_std = _const_flags(y_pred, tol)
    need = yt_flag or yp_flag
    rec = None
    if need:
        rec = {
            "model_id": ctx.get("model_id"), "attr": ctx.get("attr"),
            "source": ctx.get("source"), "layer": int(ctx.get("layer", -1)),
            "split": split, "n_samples": int(len(y_true)),
            "y_true_const": yt_flag, "y_true_min": yt_min, "y_true_max": yt_max, "y_true_unique": yt_uniq, "y_true_std": yt_std,
            "y_pred_const": yp_flag, "y_pred_min": yp_min, "y_pred_max": yp_max, "y_pred_unique": yp_uniq, "y_pred_std": yp_std,
        }
        if debug_feature_var and X is not None:
            try:
                rec["X_var_sum"] = float(np.var(X, axis=0).sum())
                rec["X_n"] = int(X.shape[0]); rec["X_d"] = int(X.shape[1])
            except Exception: pass
        _append_jsonl(log_path, rec)
    return rec

def _fit_eval_one_layer(Xtr, ytr, Xval, yval, Xte, yte,
                        ctx: dict, log_constant_path: str, tol: float, debug_feature_var: bool):
    # RidgeCV: メモリ効率のため svd ではなく cholesky などを使う手もあるが、デフォルトでOK
    pipe = make_pipeline(StandardScaler(with_std=True), RidgeCV(alphas=np.logspace(-3,3,13)))
    pipe.fit(Xtr, ytr)
    yhat_tr = pipe.predict(Xtr); train_m = _metrics(ytr, yhat_tr)
    yhat_va = pipe.predict(Xval); val_m   = _metrics(yval, yhat_va)
    yhat_te = pipe.predict(Xte);  test_m  = _metrics(yte,  yhat_te)

    _maybe_log_constant(log_constant_path, ctx, "train", ytr, yhat_tr, Xtr, tol, debug_feature_var)
    _maybe_log_constant(log_constant_path, ctx, "val",   yval, yhat_va, Xval, tol, debug_feature_var)
    _maybe_log_constant(log_constant_path, ctx, "test",  yte,  yhat_te, Xte, tol, debug_feature_var)
    return train_m, val_m, test_m

# === メモリ節約版 accumulate ===
def _accumulate_to_disk(paths, bank: DiskFeatureBank, split_name: str, processor, model, args, ATTRS):
    for p in tqdm(paths, desc=f"Extract[{split_name}]", leave=False):
        try:
            img = Image.open(p).convert("RGB")
            prompt = make_prompt(args.prompt_mode, ATTRS)
            inputs = build_inputs(processor, img, prompt)
            
            # 勾配不要、推論モード
            with torch.no_grad():
                pools = extract_all_pools(model, inputs)

            # 即座に書き込んでメモリから捨てる
            # LLM
            for li, vec in enumerate(pools.llm_text):
                bank.append("llm_text", li, vec)
            for li, vec in enumerate(pools.llm_text_tail):
                bank.append("llm_text_tail", li, vec)
            for li, vec in enumerate(pools.llm_visual):
                bank.append("llm_visual", li, vec)
            
            # Vision
            if pools.vision_layers is not None:
                for li, vec in enumerate(pools.vision_layers):
                    bank.append("vision", li, vec)
            
            # Bridge
            for li, vec in enumerate(pools.bridge_text):
                bank.append("bridge_text", li, vec)
            for li, vec in enumerate(pools.bridge_visual):
                bank.append("bridge_visual", li, vec)

        except Exception as e:
            print(f"Warning: Failed to process {p}: {e}")
            # エラー時もスキップして続行（必要に応じてraise）
            continue
        
        # 明示的に削除
        del inputs, pools, img
    
    # 書き込み完了
    bank.close()

def _concat_sources_disk(feature_bank_tr, feature_bank_val, feature_bank_te, ytr, yval, yte,
                    attr_name: str, model_id: str, log_constant_path: str, tol: float, debug_feature_var: bool):
    per_layer = []
    best = {"source": None, "layer": None,
            "train": None, "val": {"rho": -1, "r2": -1, "rmse": 1e9}, "test": None}

    sources = ["llm_text", "llm_text_tail", "llm_visual", "vision", "bridge_text", "bridge_visual"]
    
    # 層の総数を概算（train bankのメタデータから）
    total_layers = len(feature_bank_tr.metadata)
    pbar = tqdm(total=total_layers, desc=f"Regress[{attr_name}]", leave=False)

    try:
        for src in sources:
            li = 0
            while True:
                # np.memmap を取得
                Xtr = feature_bank_tr.get(src, li)
                if Xtr is None:
                    break # このソースの層は終了
                
                Xval = feature_bank_val.get(src, li)
                Xte = feature_bank_te.get(src, li)

                if Xval is not None and Xte is not None:
                    ctx = {"model_id": model_id, "attr": attr_name, "source": src, "layer": li}
                    train_m, val_m, test_m = _fit_eval_one_layer(
                        Xtr, ytr, Xval, yval, Xte, yte,
                        ctx=ctx, log_constant_path=log_constant_path, tol=tol, debug_feature_var=debug_feature_var
                    )
                    item = {"source": src, "layer": li, "train": train_m, "val": val_m, "test": test_m}
                    per_layer.append(item)
                    if val_m["rho"] > best["val"]["rho"]:
                        best = {"source": src, "layer": li, "train": train_m, "val": val_m, "test": test_m}
                
                # 参照を切る
                del Xtr, Xval, Xte
                li += 1
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
    ap.add_argument("--quick", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model_id", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--attn_impl", default=None)
    ap.add_argument("--prompt_mode", default="base", choices=["base", "format", "attributes", "unrelated"])
    ap.add_argument("--out_json", default="probe_results.json")
    ap.add_argument("--log_constant", default=None)
    ap.add_argument("--constant_tol", type=float, default=1e-8)
    ap.add_argument("--debug_feature_var", action="store_true")
    ap.add_argument("--dataset", default="aadb", choices=["aadb", "para"])
    args = ap.parse_args()

    if args.dataset == "aadb":
        get_dataset = get_aadb_dataset; ATTRS = AADB_ATTRS
    elif args.dataset == "para":
        get_dataset = get_para_dataset; ATTRS = PARA_ATTRS
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # 1) データ
    tr_items = get_dataset(args.train_split, dataset_dir=args.dataset_dir)
    va_items = get_dataset(args.val_split,   dataset_dir=args.dataset_dir)
    te_items = get_dataset(args.test_split,  dataset_dir=args.dataset_dir)
    if args.quick is not None:
        tr_items = _rng_choice(tr_items, args.quick, args.seed)
        va_items = _rng_choice(va_items, args.quick, args.seed+1)
        te_items = _rng_choice(te_items, args.quick, args.seed+2)

    tr_paths, tr_targets = _items_to_paths_and_targets(tr_items, ATTRS)
    va_paths, va_targets = _items_to_paths_and_targets(va_items, ATTRS)
    te_paths, te_targets = _items_to_paths_and_targets(te_items, ATTRS)

    # 2) モデル
    model, processor = load_mm_model(args.model_id, args.dtype, args.device_map, args.attn_impl)
    model.eval()

    # 3) 特徴抽出 (Disk Offloading)
    # 一時フォルダを作成 (終了後削除)
    work_dir = tempfile.mkdtemp(prefix="probe_feats_")
    print(f"Temporary storage: {work_dir}")

    try:
        feature_bank_tr = DiskFeatureBank(os.path.join(work_dir, "train"))
        feature_bank_va = DiskFeatureBank(os.path.join(work_dir, "val"))
        feature_bank_te = DiskFeatureBank(os.path.join(work_dir, "test"))

        # 抽出 & ディスク書き込み
        _accumulate_to_disk(tr_paths, feature_bank_tr, "train", processor, model, args, ATTRS)
        _accumulate_to_disk(va_paths, feature_bank_va, "val",   processor, model, args, ATTRS)
        _accumulate_to_disk(te_paths, feature_bank_te, "test",  processor, model, args, ATTRS)
        
        # モデルはもう不要なので、可能ならVRAM/RAM解放
        del model, processor
        torch.cuda.empty_cache()

        # 4) 学習・評価
        results = {
            "config": vars(args),
            "attrs": {},
        }

        for attr in tqdm(ATTRS, desc="Attributes"):
            ytr = np.array(tr_targets[attr], dtype=np.float32)
            yva = np.array(va_targets[attr], dtype=np.float32)
            yte = np.array(te_targets[attr], dtype=np.float32)

            per_layer, best = _concat_sources_disk(
                feature_bank_tr, feature_bank_va, feature_bank_te,
                ytr, yva, yte,
                attr_name=attr, model_id=args.model_id,
                log_constant_path=args.log_constant, tol=args.constant_tol,
                debug_feature_var=args.debug_feature_var
            )
            results["attrs"][attr] = {"per_layer": per_layer, "best": best}

        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print("== saved:", args.out_json)

    finally:
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
            print("Cleaned up temporary files.")

if __name__ == "__main__":
    main()