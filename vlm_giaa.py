#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compute General Image Aesthetic Assessment (GIAA) scores for image datasets
using multimodal VLMs (Gemma3, Qwen3-VL).

Supported datasets:
  - PARA  (utils.para.get_para_dataset)
  - LAPIS (utils.lapis.get_lapis_dataset)

For each image:
  - Ask the VLM to rate its overall aesthetics from 1 to 5 (may be decimal).
  - Parse the numeric score from the output.
  - Save results to a CSV:

      model_id, dataset, split, image_path, giaa, raw_output

Notes:
  - split is "all" in both datasets (we load all splits at once).
  - `giaa` is float, can be e.g. 3.5.

Example usage:

  # PARA, Gemma3
  python vlm_giaa.py \
    --dataset para \
    --gemma_model_id google/gemma-3-4b-it \
    --dataset_dir datasets/PARA \
    --out_csv runs/giaa_gemma3_4b_para.csv

  # LAPIS, Qwen3-VL
  python vlm_giaa.py \
    --dataset lapis \
    --qwen_model_id Qwen/Qwen3-VL-2B-Instruct \
    --dataset_dir datasets/LAPIS \
    --out_csv runs/giaa_qwen3vl2b_lapis.csv \
    --quick 100
"""

import os
import re
import csv
import math
import random
import argparse
from typing import List

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import Qwen3VLForConditionalGeneration

from utils.para import get_para_dataset
from utils.lapis import get_lapis_dataset


# ---------- FORMAT Prompt（統一） ----------

FORMAT_PROMPT = (
    "Assess the overall aesthetic quality of this image. "
    "Please rate it on a scale from 1 to 5. "
    "Output only the numeric score, and do not output any other text."
)


# ---------- Helpers ----------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_img(path: str, resize: bool, max_side: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if not resize:
        return img

    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img

    scale = max_side / float(m)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), Image.BICUBIC)


def build_inputs(processor, image: Image.Image, prompt: str, device: torch.device):
    """
    画像＋テキストを Processor の chat_template でまとめてモデル入力を作る。

    想定:
      - Gemma3 (google/gemma-3-*-it)
      - Qwen3-VL (Qwen/Qwen3-VL-*-Instruct)
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    if not hasattr(processor, "apply_chat_template"):
        raise RuntimeError(
            "Processor does not support `apply_chat_template`. "
            "This script assumes a chat-style multimodal model (Gemma3/Qwen3-VL)."
        )

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,  # assistant ターンを入れる
        return_tensors="pt",
        return_dict=True,
    )

    model_inputs = {k: v.to(device) for k, v in inputs.items()}
    model_inputs.pop("token_type_ids", None)  # Qwen系で付いてくる場合は削除
    return model_inputs


def parse_float_from_text(text: str) -> float:
    """
    出力テキストから最初の float っぽい数字をパースする。
    "3.5", "4", "I rate it 2.0." などに対応。
    パース失敗時は NaN を返す。
    """
    m = re.search(r"[-+]?\d+(\.\d+)?", text)
    if not m:
        return math.nan
    try:
        return float(m.group(0))
    except Exception:
        return math.nan


def run_giaa_for_items(
    model,
    processor,
    device,
    items,
    prompt: str,
    quick: int | None = None,
    resize_image: bool = False,
    max_side: int = 1024,
):
    """
    items: list of objects with .image_path attribute (PARAItem / LAPISItem)
    Returns: list of dict {image_path, giaa, raw_output}
    """
    if quick is not None and quick < len(items):
        rng = np.random.RandomState(123)
        idx = rng.choice(len(items), size=quick, replace=False)
        items = [items[i] for i in idx]
        print(f"[info] quick mode: using {len(items)} samples out of {len(idx)}")

    rows = []

    for item in tqdm(items, desc="GIAA[all]"):
        img = load_img(item.image_path, resize_image, max_side)
        inputs = build_inputs(processor, img, prompt, device)

        with torch.inference_mode():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                num_beams=1,
            )

        input_len = inputs["input_ids"].shape[-1]
        gen_tokens = gen_ids[:, input_len:]
        text = processor.tokenizer.decode(
            gen_tokens[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

        score = parse_float_from_text(text)
        rows.append(
            {
                "image_path": item.image_path,
                "giaa": score,
                "raw_output": text,
            }
        )

    return rows


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        required=True,
        choices=["para", "lapis"],
        help="Dataset to run on (para or lapis).",
    )
    ap.add_argument(
        "--gemma_model_id",
        help="Gemma 3 model id (e.g. google/gemma-3-4b-it).",
    )
    ap.add_argument(
        "--qwen_model_id",
        help="Qwen3-VL model id (e.g. Qwen/Qwen3-VL-2B-Instruct).",
    )
    ap.add_argument(
        "--dataset_dir",
        default=None,
        help="Path to dataset root. If None, use defaults (datasets/PARA or datasets/LAPIS).",
    )
    ap.add_argument(
        "--out_csv",
        required=True,
        help="Path to output CSV.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling in quick mode.",
    )
    ap.add_argument(
        "--quick",
        type=int,
        default=None,
        help="If set, evaluate at most N images (all splits combined) for each model (debug mode).",
    )
    ap.add_argument(
        "--resize_image",
        action="store_true",
        help="If set, resize images so that max(width,height) <= --max_side before feeding to the VLM.",
    )
    ap.add_argument(
        "--max_side",
        type=int,
        default=1024,
        help="Max side length for image resizing when --resize_image is enabled.",
    )
    
    args = ap.parse_args()

    set_seed(args.seed)

    # デフォルトの dataset_dir を決める
    if args.dataset_dir is None:
        if args.dataset == "para":
            args.dataset_dir = "datasets/PARA"
        else:
            args.dataset_dir = "datasets/LAPIS"

    # データ読み込み (all splits)
    if args.dataset == "para":
        items = get_para_dataset(None, dataset_dir=args.dataset_dir)
    else:
        items = get_lapis_dataset(None, dataset_dir=args.dataset_dir)

    print(f"[info] dataset={args.dataset}, dir={args.dataset_dir}, total_items={len(items)}")

    # どのモデルを走らせるか
    model_ids: List[tuple[str, str]] = []
    if args.gemma_model_id:
        model_ids.append(("gemma", args.gemma_model_id))
    if args.qwen_model_id:
        model_ids.append(("qwen", args.qwen_model_id))

    if not model_ids:
        raise ValueError("Please specify at least one of --gemma_model_id or --qwen_model_id")

    all_rows = []

    for family, mid in model_ids:
        print(f"[info] loading model: {mid}")
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

        if family == "qwen":
            # Qwen3-VL は専用クラスでロード
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                mid,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                mid,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True,
            )

        processor = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
        model.eval()
        device = model.device if hasattr(model, "device") else torch.device(device_str)

        rows = run_giaa_for_items(
            model=model,
            processor=processor,
            device=device,
            items=items,
            prompt=FORMAT_PROMPT,
            quick=args.quick,
            resize_image=args.resize_image,
            max_side=args.max_side,
        )

        for r in rows:
            all_rows.append(
                {
                    "model_id": mid,
                    "dataset": args.dataset,
                    "split": "all",
                    "image_path": r["image_path"],
                    "giaa": r["giaa"],
                    "raw_output": r["raw_output"],
                }
            )

    # CSV 保存
    out_path = args.out_csv
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fieldnames = ["model_id", "dataset", "split", "image_path", "giaa", "raw_output"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in all_rows:
            w.writerow(row)

    print(f"[done] wrote {len(all_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()