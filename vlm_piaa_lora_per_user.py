#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Per-user LoRA fine-tuning baseline for PIAA on PARA / LAPIS using multimodal VLMs (Gemma3 / Qwen3-VL).

For each user u:
  - Support set S_u (support_small or support_large) and test set T_u from
    get_personalized_*_dataset.
  - Train a user-specific LoRA-augmented VLM on S_u to output that user's score (1~5)
    for each image.
  - Evaluate the LoRA model on T_u by generating a rating for each test image.

This is an intentionally simple per-user LoRA baseline; it is not optimized
for speed or memory, but is good for understanding the upper bound of personalization.

Output CSV columns (one row per user × test image):
  user_id, image_path, model_id, support_set, method, giaa, piaa_pred, user_score

  - method: "lora_per_user_<support_set>"
  - giaa  : NaN (GIAA is not used here; the model directly predicts PIAA)
"""

import os
import re
import csv
import math
import random
import argparse
from typing import List, Dict

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoProcessor, AutoModelForCausalLM, get_linear_schedule_with_warmup
from transformers import Qwen3VLForConditionalGeneration

from peft import LoraConfig, get_peft_model

from utils.para import get_personalized_para_dataset
from utils.lapis import get_personalized_lapis_dataset


# ---------- Prompt & Parsing ----------

PROMPT_TEMPLATE = (
    "You are an expert judge of image aesthetics.\n"
    "For each image, rate its overall visual aesthetic quality on a scale from 1.0 to 5.0.\n"
    "Respond with a single number (you may use a decimal like 3.5) and nothing else.\n"
    "Rate this image now."
)


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

def parse_float_from_text(text: str) -> float:
    m = re.search(r"[-+]?\d+(\.\d+)?", text)
    if not m:
        return math.nan
    try:
        return float(m.group(0))
    except Exception:
        return math.nan


# ---------- Per-user Dataset ----------

class UserPIAALoRADataset(Dataset):
    """
    1ユーザー分の support set を LoRA 学習用の Dataset に落とし込む。

    - items: List[{"image_path": ..., "score": ...}] (score in [1,5])
    - __getitem__ で (input_ids, attention_mask, labels) を返す。
    """

    def __init__(self, items, processor, device, resize_image: bool, max_side: int):
        self.items = items
        self.processor = processor
        self.device = device
        self.resize_image = resize_image
        self.max_side = max_side

    def __len__(self):
        return len(self.items)

    def _build_messages(self, image: Image.Image, score: float):
        # ユーザ: 画像＋質問
        # アシスタント: スコアをテキストで返す
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT_TEMPLATE},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"{score:.2f}"},
                ],
            },
        ]
        return messages

    def __getitem__(self, idx):
        item = self.items[idx]
        img = load_img(item["image_path"], self.resize_image, self.max_side)
        score = float(item["score"])

        messages = self._build_messages(img, score)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


def collate_fn(batch):
    """
    左詰めpaddingのためのcollate。labels は input_ids と同一（prompt+回答全部）。
    """
    input_ids = [b["input_ids"] for b in batch]
    attention_mask = [b["attention_mask"] for b in batch]

    max_len = max(x.size(0) for x in input_ids)
    padded_ids = []
    padded_mask = []

    for ids, mask in zip(input_ids, attention_mask):
        pad_len = max_len - ids.size(0)
        padded_ids.append(
            torch.cat([ids, torch.full((pad_len,), fill_value=0, dtype=ids.dtype)])
        )
        padded_mask.append(
            torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])
        )

    input_ids = torch.stack(padded_ids, dim=0)
    attention_mask = torch.stack(padded_mask, dim=0)
    labels = input_ids.clone()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        required=True,
        choices=["para", "lapis"],
        help="Dataset to use (para or lapis).",
    )
    ap.add_argument(
        "--dataset_dir",
        default=None,
        help="Dataset root directory. If None, uses datasets/PARA or datasets/LAPIS.",
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
        "--support_set",
        default="small",
        choices=["small", "large"],
        help="Which support set from personalized dataset to use for per-user LoRA training.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for personalized split and training shuffling.",
    )
    ap.add_argument(
        "--quick_users",
        type=int,
        default=None,
        help="If set, limit to at most N users (for training & eval). Use 1 for a single-user check.",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs for per-user LoRA.",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for per-user LoRA training.",
    )
    ap.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for per-user LoRA training.",
    )
    ap.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA rank.",
    )
    ap.add_argument(
        "--lora_alpha",
        type=float,
        default=16.0,
        help="LoRA alpha.",
    )
    ap.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout.",
    )
    ap.add_argument(
        "--out_csv",
        required=True,
        help="Path to output CSV for per-user test predictions.",
    )
    ap.add_argument(
        "--resize_image",
        action="store_true",
        help="If set, resize images so that max(width,height) <= --max_side (to reduce image tokens / VRAM).",
    )
    ap.add_argument(
        "--max_side",
        type=int,
        default=1024,
        help="Max side length used when --resize_image is enabled.",
    )
    args = ap.parse_args()

    set_seed(args.seed)

    # dataset_dir デフォルト
    if args.dataset_dir is None:
        if args.dataset == "para":
            args.dataset_dir = "datasets/PARA"
        else:
            args.dataset_dir = "datasets/LAPIS"

    # Personalized データ読み込み
    print(f"[info] loading personalized {args.dataset.upper()} dataset...")
    if args.dataset == "para":
        personalized = get_personalized_para_dataset(seed=args.seed, dataset_dir=args.dataset_dir)
    else:
        personalized = get_personalized_lapis_dataset(seed=args.seed, dataset_dir=args.dataset_dir)

    all_user_ids = sorted(personalized.keys())
    print(f"[info] num users in personalized dataset: {len(all_user_ids)}")

    if args.quick_users is not None and args.quick_users < len(all_user_ids):
        user_ids = all_user_ids[:args.quick_users]
        print(f"[info] quick_users mode: using first {len(user_ids)} users out of {len(all_user_ids)}")
    else:
        user_ids = all_user_ids

    # モデル指定（Gemma or Qwenのどちらか1つ）
    model_specs = []
    if args.gemma_model_id:
        model_specs.append(("gemma", args.gemma_model_id))
    if args.qwen_model_id:
        model_specs.append(("qwen", args.qwen_model_id))

    if len(model_specs) != 1:
        raise ValueError("For per-user LoRA, please specify exactly one of --gemma_model_id or --qwen_model_id.")

    family, mid = model_specs[0]
    print(f"[info] loading base model: {mid}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if family == "qwen":
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            mid,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            mid,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

    processor = AutoProcessor.from_pretrained(mid, trust_remote_code=True)

    # LoRA設定（テキスト側Attention/MLP用の典型的ターゲット）
    lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    rows: List[dict] = []
    method_name = f"lora_per_user_{args.support_set}"

    # ユーザごとに LoRA モデルを作って train & eval
    for user_id in tqdm(user_ids, desc="Per-user LoRA"):
        pdata = personalized[user_id]
        if args.support_set == "small":
            support_items = pdata.support_small
        else:
            support_items = pdata.support_large
        test_items = pdata.test

        if len(support_items) < 1:
            continue

        # 学習データ （user専用）
        train_examples = [
            {"image_path": it.image_path, "score": float(it.score)}
            for it in support_items
        ]
        # 評価データ
        eval_items = [
            {"image_path": it.image_path, "score": float(it.score)}
            for it in test_items
        ]

        # ベースモデルからLoRA付きモデルを作る（簡易実装：毎ユーザごとに get_peft_model）
        # ※効率面では後述の注意を参照
        model = get_peft_model(
            base_model.__class__.from_pretrained(
                mid,
                torch_dtype=base_model.dtype,
                device_map="auto",
                trust_remote_code=True,
            ),
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=lora_targets,
            ),
        )
        model.to(device)
        model.train()

        train_dataset = UserPIAALoRADataset(
            train_examples,
            processor,
            device,
            resize_image=args.resize_image,
            max_side=args.max_side,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        total_steps = args.epochs * max(1, len(train_loader))
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=min(10, total_steps // 10),
            num_training_steps=total_steps,
        )

        # --- LoRA training for this user ---
        for epoch in range(args.epochs):
            model.train()
            epoch_losses = []
            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                epoch_losses.append(loss.item())

            print(f"[train][user={user_id}] epoch {epoch+1}/{args.epochs}, loss={np.mean(epoch_losses):.4f}")

        # --- Evaluation for this user ---
        model.eval()
        for item in tqdm(eval_items, desc=f"  Eval user={user_id}", leave=False):
            img = load_img(item["image_path"], args.resize_image, args.max_side)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": PROMPT_TEMPLATE},
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
                    max_new_tokens=16,
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

            score_pred = parse_float_from_text(text)
            user_score = float(item["score"])

            rows.append(
                {
                    "user_id": user_id,
                    "image_path": item["image_path"],
                    "model_id": mid,
                    "support_set": args.support_set,
                    "method": method_name,
                    "giaa": math.nan,
                    "piaa_pred": score_pred,
                    "user_score": user_score,
                }
            )

        # LoRAモデルを解放（次のユーザのために）
        del model
        torch.cuda.empty_cache()

    # CSV 保存
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
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[done] wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()