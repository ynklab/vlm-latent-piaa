# qwen3vl_embed.py (v7: vision_layers を "blocks のみ" に限定)
import torch
from PIL import Image
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

@dataclass
class AllPools:
    llm_text: List[np.ndarray]        # 各層: テキスト領域 平均
    llm_text_tail: List[np.ndarray]   # 各層: テキスト領域 末尾トークン
    llm_visual: List[np.ndarray]      # 各層: 視覚領域 平均
    vision_layers: Optional[List[np.ndarray]]  # Vision: [Block0, Block1, ...] のみ
    bridge_text: List[np.ndarray]     # LLM直前: テキスト領域 平均
    bridge_visual: List[np.ndarray]   # LLM直前: 視覚領域 平均

def load_qwen3vl(model_id="Qwen/Qwen3-VL-2B-Instruct", dtype="auto", device_map="auto", attn_impl=None):
    kw = dict(dtype=dtype, device_map=device_map)
    if attn_impl is not None:
        kw["attn_implementation"] = attn_impl
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **kw)
    processor = AutoProcessor.from_pretrained(model_id)
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

def _masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
    return (h * mask.unsqueeze(-1)).sum(dim=1) / denom

def _to_numpy_f32(t: torch.Tensor) -> np.ndarray:
    # NumPy が bfloat16 を扱えないため、float32 に変換して返す
    return t.detach().to(torch.float32).cpu().numpy()

def _coerce_first_tensor(x: Any) -> Optional[torch.Tensor]:
    # nested tuple/dict/ModelOutput から先頭の Tensor を取り出す
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (list, tuple)) and len(x) > 0:
        return _coerce_first_tensor(x[0])
    for k in ("image_embeds", "image_features", "last_hidden_state", "hidden_states"):
        if isinstance(x, dict) and k in x and isinstance(x[k], torch.Tensor):
            return x[k]
        if hasattr(x, k):
            v = getattr(x, k)
            if isinstance(v, torch.Tensor):
                return v
    return None

def _pool_tokenish(x: torch.Tensor) -> Optional[torch.Tensor]:
    """
    Vision出力を 1ベクトルにプール（パッチ平均→バッチ平均）。
    支持形状: [B,C,T,H,W] / [B,C,H,W] / [B,N,D] / [N,D] / [B,C,N]
    """
    if x is None or not torch.is_tensor(x):
        return None
    t = x
    if t.dim() == 5:         # [B,C,T,H,W]
        t = t.movedim(1, -1).reshape(t.size(0), -1, t.size(-1))  # [B,Np,C]
        vec = t.mean(dim=1).mean(dim=0)
    elif t.dim() == 4:       # [B,C,H,W]
        t = t.movedim(1, -1).reshape(t.size(0), -1, t.size(-1))  # [B,Np,C]
        vec = t.mean(dim=1).mean(dim=0)
    elif t.dim() == 3:       # [B,N,D] or [B,C,N]
        if t.size(-1) < t.size(1):    # [B,C,N] っぽい
            t = t.movedim(1, -1)      # -> [B,N,C]
        vec = t.mean(dim=1).mean(dim=0)
    elif t.dim() == 2:       # [N,D]
        vec = t.mean(dim=0)
    else:
        return None
    return vec

def _first_tensor_from_output(o: Any) -> Optional[torch.Tensor]:
    # forward hook の出力が tuple/dict でも最初の Tensor を拾う
    if isinstance(o, torch.Tensor):
        return o
    if isinstance(o, (list, tuple)):
        for v in o:
            ft = _first_tensor_from_output(v)
            if isinstance(ft, torch.Tensor):
                return ft
        return None
    if isinstance(o, dict):
        for v in o.values():
            ft = _first_tensor_from_output(v)
            if isinstance(ft, torch.Tensor):
                return ft
    if hasattr(o, "last_hidden_state"):
        v = getattr(o, "last_hidden_state")
        if isinstance(v, torch.Tensor):
            return v
    return None

@torch.inference_mode()
def extract_all_pools(model, inputs: Dict[str, torch.Tensor]) -> AllPools:
    device = model.device
    cfg = model.config
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

    input_ids: torch.Tensor = inputs["input_ids"]
    attention_mask: Optional[torch.Tensor] = inputs.get("attention_mask")
    pixel_values: Optional[torch.Tensor] = inputs.get("pixel_values")
    image_grid_thw: Optional[torch.Tensor] = inputs.get("image_grid_thw")

    # --- (A) VLブリッジ（LLM直前: 入力埋め込み）
    word_emb = model.model.get_input_embeddings()(input_ids)  # [B,S,D_text]
    image_embeds = None
    if pixel_values is not None:
        raw = model.model.get_image_features(pixel_values, image_grid_thw)
        image_embeds = _coerce_first_tensor(raw)
        if image_embeds is not None:
            image_embeds = image_embeds.to(word_emb.device, word_emb.dtype)

    image_token_id = getattr(cfg, "image_token_id", None)
    image_mask = (input_ids == image_token_id) if (image_token_id is not None) \
                 else torch.zeros_like(input_ids, dtype=torch.bool)

    if image_embeds is not None and image_mask.any():
        num = int(image_mask.sum().item())
        emb2d = image_embeds.reshape(-1, word_emb.size(-1))
        if emb2d.size(0) < num:
            rep = (num + emb2d.size(0) - 1) // emb2d.size(0)
            emb2d = emb2d.repeat(rep, 1)
        word_emb = word_emb.clone()
        word_emb[image_mask] = emb2d[:num]

    pad_mask = attention_mask.bool() if attention_mask is not None \
               else torch.ones_like(input_ids, dtype=torch.bool)
    bridge_visual_mask = image_mask & pad_mask
    bridge_text_mask   = (~image_mask) & pad_mask
    bridge_text_vec  = _to_numpy_f32(_masked_mean(word_emb, bridge_text_mask)).squeeze(0)
    bridge_visual_vec = _to_numpy_f32(_masked_mean(word_emb, bridge_visual_mask)).squeeze(0)

    # --- (B) LLM層 hidden_states（text/visual/tail）
    outputs = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
    hiddens = outputs.hidden_states  # tuple(len = n_layers+1) of [B,S,D_text]

    visual_mask = image_mask & pad_mask
    text_mask   = (~image_mask) & pad_mask

    B, S = input_ids.shape
    arange_idx = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
    text_idx = torch.where(text_mask, arange_idx, torch.full_like(arange_idx, -1))
    last_text_idx = text_idx.max(dim=1).values
    last_text_idx = torch.where(last_text_idx < 0, torch.zeros_like(last_text_idx), last_text_idx)
    b_index = torch.arange(B, device=device)

    llm_text, llm_text_tail, llm_visual = [], [], []
    for h in hiddens:
        llm_text.append(_to_numpy_f32(_masked_mean(h, text_mask)).squeeze(0))
        llm_visual.append(_to_numpy_f32(_masked_mean(h, visual_mask)).squeeze(0))
        tail_vec = h[b_index, last_text_idx]                  # [B,D]
        llm_text_tail.append(_to_numpy_f32(tail_vec.mean(dim=0)))  # [D]

    # --- (C) Vision Encoder: 各 Block 出力のみ（merger/deepstack/last は収集しない）
    vision_layers = None
    if pixel_values is not None:
        vis = model.model.visual
        hooks = []
        captured_blocks: Dict[int, torch.Tensor] = {}

        def make_hook_block(i):
            def _hook(_, __, out):
                captured_blocks[i] = _first_tensor_from_output(out)
            return _hook

        try:
            if hasattr(vis, "blocks") and isinstance(vis.blocks, (list, torch.nn.ModuleList)):
                for i, blk in enumerate(list(vis.blocks)):
                    hooks.append(blk.register_forward_hook(make_hook_block(i)))

            # forward 実行（フックで各Blockの出力を捕捉）
            _ = vis(
                pixel_values.to(device),
                grid_thw=image_grid_thw,
                output_hidden_states=False
            )
        finally:
            for h in hooks:
                try: h.remove()
                except Exception: pass

        if captured_blocks:
            vecs: List[np.ndarray] = []
            for i in sorted(captured_blocks.keys()):
                v = _pool_tokenish(captured_blocks[i])
                if v is not None:
                    vecs.append(_to_numpy_f32(v))
            vision_layers = vecs if len(vecs) > 0 else None
        else:
            vision_layers = None
    
    return AllPools(
        llm_text=llm_text,
        llm_text_tail=llm_text_tail,
        llm_visual=llm_visual,
        vision_layers=vision_layers,  # ← blocks のみ
        bridge_text=[bridge_text_vec],
        bridge_visual=[bridge_visual_vec],
    )
