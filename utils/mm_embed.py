# mm_embed.py (hotfix: robust 1D conversion; remove .squeeze(0) uses)

import torch, inspect
from PIL import Image
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np
from transformers import AutoProcessor, AutoModelForCausalLM, AutoImageProcessor, AutoModel
from tqdm import tqdm
try:
    from transformers import Qwen3VLForConditionalGeneration as _QWEN_CLASS
except Exception:
    _QWEN_CLASS = None

@dataclass
class AllPools:
    llm_text: List[np.ndarray]
    llm_text_tail: List[np.ndarray]
    llm_visual: List[np.ndarray]
    vision_layers: Optional[List[np.ndarray]]
    bridge_text: List[np.ndarray]
    bridge_visual: List[np.ndarray]

def load_mm_model(model_id: str, dtype="auto", device_map="auto", attn_impl=None):
    kw = dict(dtype=dtype, device_map=device_map)
    if attn_impl is not None:
        kw["attn_implementation"] = attn_impl
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    except Exception:
        if _QWEN_CLASS is not None:
            model = _QWEN_CLASS.from_pretrained(model_id, **kw)
        else:
            kw2 = dict(kw); kw2["trust_remote_code"] = True
            model = AutoModelForCausalLM.from_pretrained(model_id, **kw2)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor

def build_inputs(processor, image: Image.Image, prompt: str):
    messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        return_dict=True, return_tensors="pt"
    )
    inputs.pop("token_type_ids", None)
    return inputs

# ---------- helpers ----------

def _masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
    return (h * mask.unsqueeze(-1)).sum(dim=1) / denom  # [B,D]

def _to_numpy_f32(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float32).cpu().numpy()

def _to_numpy_1d(t: torch.Tensor) -> np.ndarray:
    """[D] はそのまま, [B,D] はバッチ平均して常に [D] を返す（dtypeはfloat32）。"""
    if t.dim() == 2:
        t = t.mean(dim=0)
    elif t.dim() > 2:
        # 念のため（来ない想定）：先頭軸で平均
        t = t.mean(dim=0)
    return _to_numpy_f32(t)

def _coerce_first_tensor(x: Any) -> Optional[torch.Tensor]:
    if x is None: return None
    if isinstance(x, torch.Tensor): return x
    if isinstance(x, (list, tuple)) and len(x) > 0: return _coerce_first_tensor(x[0])
    for k in ("image_embeds","image_features","last_hidden_state","hidden_states"):
        if isinstance(x, dict) and k in x and isinstance(x[k], torch.Tensor): return x[k]
        if hasattr(x, k):
            v = getattr(x, k)
            if isinstance(v, torch.Tensor): return v
    return None

def _pool_tokenish(x: torch.Tensor) -> Optional[torch.Tensor]:
    if x is None or not torch.is_tensor(x): return None
    t = x
    if t.dim() == 5:
        t = t.movedim(1, -1).reshape(t.size(0), -1, t.size(-1)); vec = t.mean(dim=1).mean(dim=0)
    elif t.dim() == 4:
        t = t.movedim(1, -1).reshape(t.size(0), -1, t.size(-1)); vec = t.mean(dim=1).mean(dim=0)
    elif t.dim() == 3:
        # 3Dは [B, N, D] 想定
        vec = t.mean(dim=1).mean(dim=0)  # トークン平均→バッチ平均
    elif t.dim() == 2:
        vec = t.mean(dim=0)
    else:
        return None
    return vec

def _first_tensor_from_output(o: Any) -> Optional[torch.Tensor]:
    if isinstance(o, torch.Tensor): return o
    if isinstance(o, (list, tuple)):
        for v in o:
            ft = _first_tensor_from_output(v)
            if isinstance(ft, torch.Tensor): return ft
        return None
    if isinstance(o, dict):
        for v in o.values():
            ft = _first_tensor_from_output(v)
            if isinstance(ft, torch.Tensor): return ft
    if hasattr(o, "last_hidden_state"):
        v = getattr(o, "last_hidden_state")
        if isinstance(v, torch.Tensor): return v
    return None

def _guess_image_token_id(model, processor) -> Optional[int]:
    for obj in (getattr(model, "config", None), getattr(model, "generation_config", None)):
        if obj is not None and hasattr(obj, "image_token_id"):
            return getattr(obj, "image_token_id")
    tok = getattr(processor, "tokenizer", None) or getattr(processor, "tokenizer_", None)
    if tok is not None:
        for cand in ["<image>", "<image_1>", "<image_token>", "<imagepad>"]:
            try:
                ids = tok.encode(cand, add_special_tokens=False)
                if isinstance(ids, list) and len(ids) == 1:
                    return int(ids[0])
            except Exception:
                pass
    return None

def _get_root(model):
    return getattr(model, "model", model)

def _has_qwen_visual(root) -> bool:
    return hasattr(root, "visual")

def _has_gemma_vision(root) -> bool:
    return hasattr(root, "vision_tower")

# ---------- main extractor ----------

@torch.inference_mode()
def extract_all_pools(model, inputs: Dict[str, torch.Tensor], processor=None) -> AllPools:
    device = model.device
    root = _get_root(model)
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

    input_ids: torch.Tensor = inputs["input_ids"]
    attention_mask: Optional[torch.Tensor] = inputs.get("attention_mask")
    pixel_values: Optional[torch.Tensor] = inputs.get("pixel_values")
    image_grid_thw: Optional[torch.Tensor] = inputs.get("image_grid_thw")  # Qwenのみ

    image_token_id = _guess_image_token_id(model, processor)
    image_mask = (input_ids == image_token_id) if image_token_id is not None \
                 else torch.zeros_like(input_ids, dtype=torch.bool)

    # ---- Bridge（LLM入力直前）
    bridge_text_vec = bridge_visual_vec = None
    try:
        word_emb = root.get_input_embeddings()(input_ids)  # [B,S,D]
        image_embeds = None
        if pixel_values is not None and hasattr(root, "get_image_features"):
            try:
                if "grid_thw" in inspect.signature(root.get_image_features).parameters:
                    raw = root.get_image_features(pixel_values, image_grid_thw)
                else:
                    raw = root.get_image_features(pixel_values)
                image_embeds = _coerce_first_tensor(raw)
            except Exception:
                image_embeds = None
        if image_embeds is not None and image_mask.any():
            image_embeds = image_embeds.to(word_emb.device, word_emb.dtype)
            num = int(image_mask.sum().item())
            emb2d = image_embeds.reshape(-1, word_emb.size(-1))
            if emb2d.size(0) < num:
                rep = (num + emb2d.size(0) - 1) // emb2d.size(0)
                emb2d = emb2d.repeat(rep, 1)
            we = word_emb.clone()
            we[image_mask] = emb2d[:num]
            pad_mask = attention_mask.bool() if attention_mask is not None \
                       else torch.ones_like(input_ids, dtype=torch.bool)
            bridge_visual_mask = image_mask & pad_mask
            bridge_text_mask   = (~image_mask) & pad_mask
            bridge_text_vec  = _to_numpy_1d(_masked_mean(we, bridge_text_mask))
            bridge_visual_vec = _to_numpy_1d(_masked_mean(we, bridge_visual_mask))
    except Exception:
        pass

    # ---- LLM層 hidden_states（fallbackでBridgeも計算）
    outputs = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
    hiddens = outputs.hidden_states  # tuple of [B,S,D]
    pad_mask = attention_mask.bool() if attention_mask is not None else torch.ones_like(input_ids, dtype=torch.bool)
    visual_mask = image_mask & pad_mask
    text_mask   = (~image_mask) & pad_mask

    if bridge_text_vec is None or bridge_visual_vec is None:
        h0 = hiddens[0]
        if bridge_text_vec is None:
            bridge_text_vec = _to_numpy_1d(_masked_mean(h0, text_mask))
        if bridge_visual_vec is None:
            bridge_visual_vec = _to_numpy_1d(_masked_mean(h0, visual_mask))

    # LLM: text / visual / tail（常に1D化）
    B, S = input_ids.shape
    arange_idx = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
    text_idx = torch.where(text_mask, arange_idx, torch.full_like(arange_idx, -1))
    last_text_idx = text_idx.max(dim=1).values
    last_text_idx = torch.where(last_text_idx < 0, torch.zeros_like(last_text_idx), last_text_idx)
    b_index = torch.arange(B, device=device)

    llm_text, llm_text_tail, llm_visual = [], [], []
    for h in hiddens:
        llm_text.append(_to_numpy_1d(_masked_mean(h, text_mask)))
        llm_visual.append(_to_numpy_1d(_masked_mean(h, visual_mask)))
        tail_vec = h[b_index, last_text_idx]           # [B,D]
        llm_text_tail.append(_to_numpy_1d(tail_vec.mean(dim=0)))  # [D]

    # ---- Vision blocks only
    vision_layers = None
    if pixel_values is not None:
        vecs: List[np.ndarray] = []
        if _has_qwen_visual(root):
            vis = root.visual
            hooks = []; captured = {}
            def make_hook_block(i):
                def _hook(_, __, out):
                    captured[i] = _first_tensor_from_output(out)
                return _hook
            try:
                if hasattr(vis, "blocks") and isinstance(vis.blocks, (list, torch.nn.ModuleList)):
                    for i, blk in enumerate(list(vis.blocks)):
                        hooks.append(blk.register_forward_hook(make_hook_block(i)))
                try:
                    _ = vis(pixel_values.to(device), grid_thw=image_grid_thw, output_hidden_states=False)
                except TypeError:
                    _ = vis(pixel_values.to(device))
            finally:
                for h in hooks:
                    try: h.remove()
                    except Exception: pass
            if captured:
                for i in sorted(captured.keys()):
                    v = _pool_tokenish(captured[i])
                    if v is not None:
                        vecs.append(_to_numpy_f32(v))
        elif _has_gemma_vision(root):
            vt = root.vision_tower
            vis = getattr(vt, "vision_model", vt)
            enc = getattr(vis, "encoder", None)
            layers = getattr(enc, "layers", None)
            hooks = []; captured = {}
            def make_hook_layer(i):
                def _hook(_, __, out):
                    captured[i] = _first_tensor_from_output(out)
                return _hook
            try:
                if isinstance(layers, (list, torch.nn.ModuleList)):
                    for i, lyr in enumerate(list(layers)):
                        hooks.append(lyr.register_forward_hook(make_hook_layer(i)))
                _ = vt(pixel_values.to(device))
            finally:
                for h in hooks:
                    try: h.remove()
                    except Exception: pass
            if captured:
                for i in sorted(captured.keys()):
                    v = _pool_tokenish(captured[i])
                    if v is not None:
                        vecs.append(_to_numpy_f32(v))
        vision_layers = vecs if len(vecs) > 0 else None

    return AllPools(
        llm_text=llm_text,
        llm_text_tail=llm_text_tail,
        llm_visual=llm_visual,
        vision_layers=vision_layers,
        bridge_text=[bridge_text_vec],
        bridge_visual=[bridge_visual_vec],
    )

# ============================================================
# DINOv3 Vision-only モデル用ヘルパ（例: facebook/dinov3-vitb16-pretrain-lvd1689m）
# ============================================================

def load_dinov3_model(
    model_id: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
    dtype: str = "auto",
    device: Optional[str] = None,
):
    """
    DINOv3-B/16 などの Vision-only モデルをロードするヘルパ。
    戻り値: (model, image_processor, device)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if dtype == "auto":
        if torch.cuda.is_available():
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
    else:
        # dtype に 'float32' / 'float16' などを渡せるように
        torch_dtype = getattr(torch, dtype)

    # Vision-only モデルなので AutoModel でOK
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    image_processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, image_processor, device


@torch.inference_mode()
def extract_dinov3_pooler_features(
    model,
    image_processor,
    device: str,
    image_paths: List[str],
) -> np.ndarray:
    """
    DINOv3 の pooler_output（なければ last_hidden_state の global average）を
    [N, D] の numpy 配列として返す。
    """
    feats = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        inputs = image_processor(images=img, return_tensors="pt")
        # AutoImageProcessor の戻り値は通常 {"pixel_values": [B,3,H,W]}
        if "pixel_values" not in inputs:
            raise RuntimeError(f"Processor outputs have no 'pixel_values' key for image: {p}")
        pixel_values = inputs["pixel_values"].to(device)

        outputs = model(pixel_values=pixel_values)

        # DINOv3Model は通常 BaseModelOutputWithPooling を返し、pooler_output を持つ
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            emb = outputs.pooler_output  # [B, D]
        else:
            # 念のためフォールバック: last_hidden_state を global average
            if not hasattr(outputs, "last_hidden_state") or outputs.last_hidden_state is None:
                raise RuntimeError("DINOv3 model outputs have neither pooler_output nor last_hidden_state")
            h = outputs.last_hidden_state  # [B, N, D] or [B, D]
            if h.dim() == 3:
                emb = h.mean(dim=1)  # [B, D]
            else:
                emb = h
        emb = emb.detach().to(torch.float32).cpu().numpy()  # [B,D]
        feats.append(emb[0])

    return np.stack(feats, axis=0)  # [N, D]

# すでにあるインポートに AutoImageProcessor, AutoModel を追加済みと仮定
# from transformers import AutoProcessor, AutoModelForCausalLM, AutoImageProcessor, AutoModel
# ・・・
# load_dinov3_model / extract_dinov3_pooler_features は既に定義済みとする

@torch.inference_mode()
def extract_dinov3_all_layer_features(
    model,
    image_processor,
    device: str,
    image_paths: List[str],
) -> List[np.ndarray]:
    """
    DINOv3 (例: facebook/dinov3-vitb16-pretrain-lvd1689m) の全レイヤー hidden_states を取得し，
    各レイヤーごとに global pooling（パッチ平均→バッチ平均）した [N, D] の特徴行列を返す。

    戻り値:
      feats_per_layer: List[np.ndarray] で長さ = #layers+1（embedding層＋各Encoder層）
                       feats_per_layer[i] の shape は [N, D]
    """
    feats_per_layer = None  # List[List[np.ndarray]]
    for p in tqdm(image_paths, desc="Extract DINOv3 all layers"):
        img = Image.open(p).convert("RGB")
        inputs = image_processor(images=img, return_tensors="pt")
        if "pixel_values" not in inputs:
            raise RuntimeError(f"DINOv3 processor outputs have no 'pixel_values' key for {p}")
        pixel_values = inputs["pixel_values"].to(device)

        # 全レイヤーの hidden states を取得
        outputs = model(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
        if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
            raise RuntimeError("DINOv3 model did not return hidden_states. Set output_hidden_states=True.")

        hs = outputs.hidden_states  # tuple(len = n_layers+1), 各 [B, N, D] or [B, D]
        per_image_vecs = []
        for h in hs:
            if h.dim() == 3:
                # [B, N, D] -> パッチ平均 → [B, D]
                v = h.mean(dim=1)
            elif h.dim() == 2:
                # [B, D]
                v = h
            else:
                # 想定外の形状は落としてしまう
                v = h.view(h.size(0), -1)
            # バッチ平均（通常 B=1 だが一般化しておく）
            v = v.mean(dim=0)  # [D]
            per_image_vecs.append(v.detach().to(torch.float32).cpu().numpy())

        if feats_per_layer is None:
            feats_per_layer = [[] for _ in range(len(per_image_vecs))]
        for li, vec in enumerate(per_image_vecs):
            feats_per_layer[li].append(vec)

    if feats_per_layer is None:
        raise RuntimeError("No features were extracted from DINOv3 model. Check inputs.")

    # [num_images, D] にまとめる
    feats_per_layer = [np.stack(layer_list, axis=0) for layer_list in feats_per_layer]
    return feats_per_layer