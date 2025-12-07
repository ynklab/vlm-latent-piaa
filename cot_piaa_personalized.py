#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CoT-style PIAA prediction on personalized PARA / LAPIS using a multimodal VLM (Gemma3 / Qwen3-VL).

- Attributes: use AADB's aesthetic attributes (AESTHETIC_ATTRIBUTES) excluding "score".
- Target data: personalized datasets (PARA / LAPIS), i.e.
    - PARA  : utils.para.get_personalized_para_dataset
    - LAPIS : utils.lapis.get_personalized_lapis_dataset
- Support setは無視し、personalizedデータに含まれる全画像
  (support_small + support_large + test) をそのまま PIAA 評価対象とする。

For each (user, image):
  - Prompt the model to:
      1) List each aesthetic attribute on 1..5:
           AttributeName: X
      2) Then output 'Overall: Y' for that user.
  - Parse the output to get:
      - predicted attributes
      - predicted overall score

Output format (PIAA scriptsとほぼ同じ + raw_output):
  user_id, image_path, model_id, support_set, method, giaa, piaa_pred, user_score, raw_output

  - method: "cot_attr" (CoT-based attribute reasoning baseline)
  - support_set: "none"（support setを使っていないため固定値）
  - giaa : NaN（GIAAはここでは使わない）
  - piaa_pred : parsed Overall score
  - user_score: personalized ground truth score
  - raw_output: モデルの生の応答テキスト
"""

import os
import re
import csv
import math
import argparse
from typing import Dict, List

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import Qwen3VLForConditionalGeneration

from utils.aadb import AESTHETIC_ATTRIBUTES as AADB_ATTRS
from utils.para import get_personalized_para_dataset
from utils.lapis import get_personalized_lapis_dataset


# ---------- Prompt ----------

def build_prompt(attr_names: List[str]) -> str:
    attr_list = ", ".join(attr_names)
    text = (
        "You are an expert judge of image aesthetics.\n"
        "For the given image, please assess this particular user's personal aesthetic preferences.\n"
        "First, for this user, assess the following aesthetic attributes on a scale from 1 to 5:\n"
        f"{attr_list}.\n"
        "For each attribute, output a line in the form 'AttributeName: X', one attribute per line.\n"
        "After listing all the attributes, on the last line output 'Overall: Y', "
        "where Y is your overall aesthetic rating for this user from 1 to 5.\n"
        "Do not output anything else.\n"
        "Now assess this image for this user."
    )
    return text


# ---------- Helpers ----------

def parse_attributes_from_output(text: str, attr_names: List[str]) -> Dict[str, float]:
    """
    モデルの出力テキストから、各 AttributeName: X の値をパースする。
    最後に Overall: Y を探す。見つからなければ、最後の数値を Overall とみなす。
    """
    result: Dict[str, float] = {}
    # 各属性
    for attr in attr_names:
        # AttributeName: X または AttributeName = X の形を探す
        m = re.search(
            rf"{re.escape(attr)}\s*[:=]\s*([-+]?\d+(\.\d+)?)",
            text,
        )
        if m:
            try:
                result[attr] = float(m.group(1))
            except Exception:
                result[attr] = math.nan
        else:
            result[attr] = math.nan

    # Overall
    overall = math.nan
    m_over = re.search(r"Overall\s*[:=]\s*([-+]?\d+(\.\d+)?)", text, flags=re.IGNORECASE)
    if m_over:
        try:
            overall = float(m_over.group(1))
        except Exception:
            overall = math.nan
    else:
        # fallback: テキスト中の最後の数値を Overall として拾う
        all_nums = list(re.finditer(r"[-+]?\d+(\.\d+)?", text))
        if all_nums:
            try:
                overall = float(all_nums[-1].group(0))
            except Exception:
                overall = math.nan

    result["Overall"] = overall
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        required=True,
        choices=["para", "lapis"],
        help="Personalized dataset to use (para or lapis).",
    )
    ap.add_argument(
        "--dataset_dir",
        default=None,
        help="Path to dataset root. If None, uses datasets/PARA or datasets/LAPIS.",
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
        "--seed",
        type=int,
        default=42,
        help="Seed used for get_personalized_*_dataset (must match previous PIAA experiments).",
    )
    ap.add_argument(
        "--quick_users",
        type=int,
        default=None,
        help="If set, limit to at most N users (for debugging).",
    )
    ap.add_argument(
        "--quick_items_per_user",
        type=int,
        default=None,
        help="If set, limit to at most M images per user (for debugging).",
    )
    ap.add_argument(
        "--out_csv",
        required=True,
        help="Path to output CSV (PIAA format + raw_output).",
    )
    args = ap.parse_args()

    # dataset_dir デフォルト
    if args.dataset_dir is None:
        if args.dataset == "para":
            args.dataset_dir = "datasets/PARA"
        else:
            args.dataset_dir = "datasets/LAPIS"

    # 1) Personalized データ読み込み
    if args.dataset == "para":
        print(f"[info] loading personalized PARA (seed={args.seed})...")
        personalized = get_personalized_para_dataset(seed=args.seed, dataset_dir=args.dataset_dir)
    else:
        print(f"[info] loading personalized LAPIS (seed={args.seed})...")
        personalized = get_personalized_lapis_dataset(seed=args.seed, dataset_dir=args.dataset_dir)

    all_user_ids = sorted(personalized.keys())
    print(f"[info] num users in personalized {args.dataset.upper()} dataset: {len(all_user_ids)}")

    if args.quick_users is not None and args.quick_users < len(all_user_ids):
        user_ids = all_user_ids[: args.quick_users]
        print(f"[info] quick_users: using first {len(user_ids)} users out of {len(all_user_ids)}")
    else:
        user_ids = all_user_ids

    # 2) AADB属性リスト（score を除く）
    attr_names = [a for a in AADB_ATTRS if a != "score"]
    print(f"[info] attributes used in CoT prompt (from AADB): {attr_names}")

    prompt_text = build_prompt(attr_names)

    # 3) モデルロード（Gemma or Qwen のどちらか1つ）
    model_specs = []
    if args.gemma_model_id:
        model_specs.append(("gemma", args.gemma_model_id))
    if args.qwen_model_id:
        model_specs.append(("qwen", args.qwen_model_id))

    if len(model_specs) != 1:
        raise ValueError("Please specify exactly one of --gemma_model_id or --qwen_model_id.")

    family, mid = model_specs[0]
    print(f"[info] loading model: {mid}")
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    if family == "qwen":
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

    rows: List[dict] = []
    method_name = "cot_attr"
    support_set_value = "none"  # Support Setは使わない

    # 4) ユーザごとにtest画像を評価
    for user_id in tqdm(user_ids, desc=f"CoT PIAA [{mid}] users"):
        pdata = personalized[user_id]
        items = pdata.test
        if args.quick_items_per_user is not None and args.quick_items_per_user < len(items):
            items = items[: args.quick_items_per_user]

        for it in tqdm(items, desc=f"  user={user_id}", leave=False):
            img = Image.open(it.image_path).convert("RGB")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]

            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            model_inputs = {k: v.to(device) for k, v in inputs.items()}
            model_inputs.pop("token_type_ids", None)

            with torch.inference_mode():
                gen_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    num_beams=1,
                )

            input_len = model_inputs["input_ids"].shape[-1]
            gen_tokens = gen_ids[:, input_len:]
            text = processor.tokenizer.decode(
                gen_tokens[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()

            parsed = parse_attributes_from_output(text, attr_names)
            pred_overall = parsed.get("Overall", math.nan)

            rows.append(
                {
                    "user_id": user_id,
                    "image_path": it.image_path,
                    "model_id": mid,
                    "support_set": support_set_value,
                    "method": method_name,
                    "giaa": math.nan,               # GIAAは使わない
                    "piaa_pred": pred_overall,      # OverallをPIAA予測として扱う
                    "user_score": float(it.score),  # Personalized ground truth
                    "raw_output": text,            # 生の応答を保存
                }
            )

    # 5) CSV 保存（PIAAフォーマット＋raw_output）
    out_path = args.out_csv
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fieldnames = [
        "user_id",
        "image_path",
        "model_id",
        "support_set",
        "method",
        "giaa",
        "piaa_pred",
        "user_score",
        "raw_output",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[done] wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()