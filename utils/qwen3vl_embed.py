# qwen3vl_embed.py (v2)
import torch
from PIL import Image
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


@dataclass
class AllPools:
    # LLM層（言語モデル）: 各層で text/visual を別々に平均プール
    llm_text: List[np.ndarray]  # len = n_llm_layers+1, 各 [D_text]
    llm_visual: List[np.ndarray]  # len = n_llm_layers+1, 各 [D_text]
    # Vision Encoder層（ViT側）
    vision_layers: Optional[List[np.ndarray]]  # len = n_vision_layers+1, 各 [D_vision]
    # VLブリッジ（LLM直前: 入力埋め込み。image_token 部分は視覚特徴に差し替え済み）
    bridge_text: List[np.ndarray]  # len=1, [D_text]
    bridge_visual: List[np.ndarray]  # len=1, [D_text]


def load_qwen3vl(
    model_id="Qwen/Qwen3-VL-2B-Instruct",
    dtype="auto",
    device_map="auto",
    attn_impl=None,
):
    kw = dict(dtype=dtype, device_map=device_map)
    if attn_impl is not None:
        kw["attn_implementation"] = attn_impl
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **kw)
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def build_inputs(processor, image: Image.Image, prompt: str):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    return inputs


def _masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # h: [B,S,D], mask: [B,S]
    denom = mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
    return (h * mask.unsqueeze(-1)).sum(dim=1) / denom


@torch.inference_mode()
def extract_all_pools(model, inputs: Dict[str, torch.Tensor]) -> AllPools:
    """
    1) LLM層 hidden_states を text/visual でプール
    2) Vision Encoder の hidden_states をパッチ平均でプール
    3) VLブリッジ（inputs_embeds 成形後）の text/visual をプール
    """
    device = model.device
    cfg = model.config
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

    input_ids: torch.Tensor = inputs["input_ids"]  # [B,S]
    attention_mask: Optional[torch.Tensor] = inputs.get(
        "attention_mask"
    )  # [B,S] or None
    pixel_values: Optional[torch.Tensor] = inputs.get("pixel_values")  # [Nimg, 3, H, W]
    image_grid_thw: Optional[torch.Tensor] = inputs.get(
        "image_grid_thw"
    )  # [Nimg,3] (T,H,W)

    # --- (A) VLブリッジ（LLM直前: 入力埋め込み）
    #    input_ids==image_token_id の位置に image_embeds を差し替えた埋め込みを構築（Qwen系の実装に準拠）
    #    参照: configの image_token_id / vision_{start,end}_token_id【HF公式】、
    #          get_image_featuresで得た埋め込みを input_embeds 上の image_token 部に masked_scatter で置換【実装】
    word_emb = model.model.get_input_embeddings()(input_ids)  # [B,S,D_text]
    image_embeds = None
    if pixel_values is not None:
        # Qwenの get_image_features は、LLMへ渡す連続埋め込み（D_text次元）を返す
        image_embeds = model.model.get_image_features(
            pixel_values, image_grid_thw
        )  # [N_img_tokens, D_text]
        image_embeds = image_embeds.to(word_emb.device, word_emb.dtype)
    image_token_id = getattr(cfg, "image_token_id", None)
    image_mask = (
        (input_ids == image_token_id)
        if (image_token_id is not None)
        else torch.zeros_like(input_ids, dtype=torch.bool)
    )
    if image_embeds is not None:
        # 置換: image_mask位置に image_embeds を埋め込む（順序はHF実装と同一）
        mask_expanded = image_mask.unsqueeze(-1).expand_as(word_emb)
        # N_img_tokens が一致しない場合は ValueError にすべきだが、ここでは安全側で min に切る
        ne = int(mask_expanded.sum().item() // word_emb.shape[-1])
        if ne > 0 and image_embeds.shape[0] >= ne:
            word_emb = word_emb.clone()
            word_emb[mask_expanded] = image_embeds[:ne].reshape(-1)
    # ブリッジ段階のマスク
    pad_mask = (
        attention_mask.bool()
        if attention_mask is not None
        else torch.ones_like(input_ids, dtype=torch.bool)
    )
    bridge_visual_mask = image_mask & pad_mask
    bridge_text_mask = (~image_mask) & pad_mask
    bridge_text_vec = (
        _masked_mean(word_emb, bridge_text_mask).detach().cpu().numpy()
    )  # [B,D_text] -> [B,D]
    bridge_visual_vec = (
        _masked_mean(word_emb, bridge_visual_mask).detach().cpu().numpy()
    )

    # --- (B) LLM層 hidden_states（text/visualでプール）
    outputs = model(
        **inputs, output_hidden_states=True, use_cache=False, return_dict=True
    )
    hiddens = outputs.hidden_states  # tuple(len = n_layers+1) of [B,S,D_text]
    # LLM側でも image_token_id を用いた visual/text マスクが使える（Qwen系は image_token をLLM列に直接埋め込む）
    if attention_mask is not None and attention_mask.shape[-1] == input_ids.shape[-1]:
        pad_mask = attention_mask.bool()
    else:
        pad_mask = torch.ones_like(input_ids, dtype=torch.bool)
    visual_mask = image_mask & pad_mask
    text_mask = (~image_mask) & pad_mask

    llm_text, llm_visual = [], []
    for h in hiddens:
        llm_text.append(_masked_mean(h, text_mask).detach().cpu().numpy().squeeze(0))
        llm_visual.append(
            _masked_mean(h, visual_mask).detach().cpu().numpy().squeeze(0)
        )

    # --- (C) Vision Encoder hidden_states（各層パッチ平均）
    vision_layers = None
    if pixel_values is not None:
        try:
            vouts = model.model.visual(
                pixel_values.to(device),
                grid_thw=image_grid_thw,
                output_hidden_states=True,
                return_dict=True,
            )
            v_hiddens = (
                vouts.hidden_states
            )  # tuple(len = n_vis_layers+1) of [N_tokens, D_vision] or [B,N_tokens,D]
            vision_layers = []
            for vh in v_hiddens:
                # 次元にBがあればB方向、その後はトークン方向平均。なければトークン方向平均だけ。
                if vh.dim() == 3:  # [B, Np, D]
                    vec = vh.mean(dim=1).mean(dim=0)  # B平均→パッチ平均
                else:  # [Np, D]
                    vec = vh.mean(dim=0)
                vision_layers.append(vec.detach().cpu().numpy())
        except TypeError:
            # 古いTransformersで hidden_states を返さない場合は未対応扱い（必要ならforward hookで拡張）
            vision_layers = None

    return AllPools(
        llm_text=llm_text,
        llm_visual=llm_visual,
        vision_layers=vision_layers,
        bridge_text=[bridge_text_vec.squeeze(0)],
        bridge_visual=[bridge_visual_vec.squeeze(0)],
    )
