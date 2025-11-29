#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Use a multimodal LLM (Gemma 3 or Qwen3-VL) to compute General Image Aesthetic Assessment (GIAA)
scores for all PARA images (train + test).

- Prompt: format prompt

    "Assess the overall aesthetic quality of this image. "
    "Please rate it on a scale from 1 to 5. "
    "Output only the numeric score, and do not output any other text."

- 出力: CSV with columns: model_id, split, image_path, giaa, raw_output
  ※ split は PARA 側で train/test が消えているので "all" 固定。

- モデル:
    --gemma_model_id (例: google/gemma-3-4b-it)
    --qwen_model_id  (例: Qwen/Qwen3-VL-2B-Instruct)

- 決定的な出力:
    do_sample=False, temperature=0.0, num_beams=1 で Greedy デコード。

- Quick モード:
    --quick N で、全画像のうち最大 N 枚だけ評価する（train/test 合算）。

Usage examples:

  python vlm_giaa_para.py \
    --gemma_model_id google/gemma-3-4b-it \
    --dataset_dir datasets/PARA \
    --out_csv runs/giaa_gemma3_4b_para.csv

  python vlm_giaa_para.py \
    --qwen_model_id Qwen/Qwen3-VL-2B-Instruct \
    --dataset_dir datasets/PARA \
    --out_csv runs/giaa_qwen3vl2b_para.csv

  python vlm_giaa_para.py \
    --gemma_model_id google/gemma-3-4b-it \
    --qwen_model_id Qwen/Qwen3-VL-2B-Instruct \
    --dataset_dir datasets/PARA \
    --out_csv runs/giaa_gemma_qwen_para.csv
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
from transformers import AutoProcessor, AutoModelForCausalLM, Qwen3VLForConditionalGeneration

from utils.para import get_para_dataset  # utils/para.py 前提


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


def build_inputs(processor, image: Image.Image, prompt: str, device: torch.device):
    """
    Build chat-style inputs with image + text using processor.apply_chat_template.

    想定:
      - Gemma 3 (google/gemma-3-*-it)
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
        add_generation_prompt=True,  # assistant ターンを挿入
        return_tensors="pt",
        return_dict=True,
    )

    model_inputs = {k: v.to(device) for k, v in inputs.items()}
    model_inputs.pop("token_type_ids", None)
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


def run_giaa_for_all(
    model,
    processor,
    device,
    dataset_dir: str,
    prompt: str,
    quick: int | None = None,
):
    """
    PARA の train/test をまとめて (get_para_dataset(None)) 読み込み，
    全画像に対して GIAA を実行する。

    戻り値:
      rows: List[dict] with keys: image_path, split="all", giaa, raw_output
    """
    items = get_para_dataset(None, dataset_dir=dataset_dir)  # train+test 全件
    print(f"[info] total PARA items (train+test) = {len(items)}")

    if quick is not None and quick < len(items):
        rng = np.random.RandomState(123)
        idx = rng.choice(len(items), size=quick, replace=False)
        items = [items[i] for i in idx]
        print(f"[info] quick mode: using {len(items)} samples out of all images")

    rows = []
    for item in tqdm(items, desc="GIAA [all]"):
        img = Image.open(item.image_path).convert("RGB")
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
                "split": "all",  # train/test の区別は get_para_dataset(None) では失われている
                "giaa": score,
                "raw_output": text,
            }
        )

    return rows


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gemma_model_id",
        help="Gemma 3 model id (e.g. google/gemma-3-4b-it)",
    )
    ap.add_argument(
        "--qwen_model_id",
        help="Qwen3-VL model id (e.g. Qwen/Qwen3-VL-2B-Instruct)",
    )
    ap.add_argument(
        "--dataset_dir",
        default="datasets/PARA",
        help="Path to PARA dataset root",
    )
    ap.add_argument(
        "--out_csv",
        required=True,
        help="Path to output CSV",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    ap.add_argument(
        "--quick",
        type=int,
        default=None,
        help="If set, evaluate at most N images (train+test combined) for each model (debug mode).",
    )
    args = ap.parse_args()

    set_seed(args.seed)

    model_ids: List[tuple[str, str]] = []
    if args.gemma_model_id:
        model_ids.append(("gemma", args.gemma_model_id))
    if args.qwen_model_id:
        model_ids.append(("qwen", args.qwen_model_id))

    if not model_ids:
        raise ValueError("Please specify at least one of --gemma_model_id or --qwen_model_id")

    all_rows: List[dict] = []

    for family, mid in model_ids:
        print(f"[info] loading model: {mid}")
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

        if family == "qwen":
            # Qwen3-VL は専用クラスでロードする必要がある
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                mid,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            # Gemma3 など他の CausalLM 系は AutoModelForCausalLM でOK
            model = AutoModelForCausalLM.from_pretrained(
                mid,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True,
            )

        processor = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
        model.eval()

        device = model.device if hasattr(model, "device") else torch.device(device_str)

        rows = run_giaa_for_all(
            model=model,
            processor=processor,
            device=device,
            dataset_dir=args.dataset_dir,
            prompt=FORMAT_PROMPT,
            quick=args.quick,
        )
        for row in rows:
            row["model_id"] = mid
            all_rows.append(row)
    # CSV 保存
    out_path = args.out_csv
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fieldnames = ["model_id", "split", "image_path", "giaa", "raw_output"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"[done] wrote {len(all_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()