#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Token-level LoRA baseline for PIAA on PARA / LAPIS using multimodal VLMs (Gemma3 / Qwen3-VL).

- Treat the supervision signal as the user score (1-5) cast to an integer class,
  and apply CrossEntropy loss only to the answer token (the single numeric token).
- Do not apply loss to the prompt or system-message tokens.
- Apply LoRA only to the text decoder Attention/MLP modules.

Training data:
  - Union of all users' support_small or support_large examples from get_personalized_*_dataset.
  - Each sample uses (image_path, user_score), where the score is rounded to an integer class in [1..5].

Evaluation:
  - For each user u, generate scores on the user's test set with the LoRA model,
    then measure per-user rho / R^2 with eval_piaa_baselines.py or similar tools.

Output CSV:
  user_id, image_path, model_id, support_set, method, giaa, piaa_pred, user_score

  - method: "lora_tokencls_<support_set>" (Examples: lora_tokencls_small)
  - giaa : NaN (GIAA is unused here)
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

from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from transformers import Qwen3VLForConditionalGeneration

from peft import LoraConfig, get_peft_model

from utils.para import get_personalized_para_dataset
from utils.lapis import get_personalized_lapis_dataset


# ---------- Prompt & Parsing ----------

PROMPT_TEMPLATE = (
    "You are an expert judge of image aesthetics.\n"
    "For each image, rate its overall visual aesthetic quality on a scale from 1 to 5.\n"
    "Respond with a single integer (1, 2, 3, 4, or 5) and nothing else.\n"
    "Rate this image now."
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_int_score_from_text(text: str) -> float:
    """
    Find the first integer-like number in the text, clip it to 1-5, and return it.
    Return NaN if nothing is found.
    """
    m = re.search(r"\b[1-5]\b", text)
    if not m:
        # fallback: find any integer and clip it to 1-5
        m2 = re.search(r"[-+]?\d+", text)
        if not m2:
            return math.nan
        try:
            v = int(m2.group(0))
        except Exception:
            return math.nan
        return float(max(1, min(5, v)))
    try:
        v = int(m.group(0))
    except Exception:
        return math.nan
    return float(max(1, min(5, v)))


def score_to_class(score: float) -> int:
    """
    Convert a user score (real-valued 1-5) into a class label in [0..4].
    Use a simple round-and-clip rule.
    """
    v = round(score)
    v = max(1, min(5, v))
    return int(v - 1)


# ---------- Dataset for token-level classification ----------

class TokenClsDataset(Dataset):
    """
    Each sample:
      - image_path: str
      - score_int: int in {1,2,3,4,5} (class labels are 0-4)

    In __getitem__:
      - tokenize a chat-form conversation (user + assistant) with apply_chat_template.
      - keep only the token positions corresponding to the assistant's numeric answer ("1"-"5") as labels,
        and mask all other positions with -100.
    """

    def __init__(self, items, processor, device, tokenizer):
        """
        items: List[{"image_path": ..., "score_int": ..., "score_cls": ...}]
        """
        self.items = items
        self.processor = processor
        self.device = device
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.items)

    def _build_messages(self, image: Image.Image, score_int: int):
        """
        score_int: 1〜5
        """
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
                    {"type": "text", "text": str(score_int)},
                ],
            },
        ]
        return messages

    def __getitem__(self, idx):
        item = self.items[idx]
        img = Image.open(item["image_path"]).convert("RGB")
        score_int = int(item["score_int"])  # 1〜5

        messages = self._build_messages(img, score_int)

        # Tokenize the full conversation (user + assistant)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = inputs["input_ids"][0]         # [L]
        attention_mask = inputs["attention_mask"][0]

        # Get the token sequence for the assistant's numeric answer (1-5)
        ans_text = str(score_int)
        ans_ids = self.tokenizer(ans_text, add_special_tokens=False).input_ids
        if not ans_ids:
            raise RuntimeError(f"Tokenization for answer '{ans_text}' returned empty ids.")

        # Search for a suffix in input_ids that matches ans_ids
        # Usually it is near the end, so scan backward
        L = input_ids.size(0)
        ans_len = len(ans_ids)
        start_idx = None
        for pos in range(L - ans_len, -1, -1):
            if torch.equal(input_ids[pos : pos + ans_len], torch.tensor(ans_ids, dtype=input_ids.dtype)):
                start_idx = pos
                break

        if start_idx is None:
            # Fallback: treat only the last token as the answer
            start_idx = L - 1
            ans_len = 1
            ans_ids = [int(input_ids[start_idx].item())]

        labels = torch.full_like(input_ids, fill_value=-100)
        # Set ans_ids as the labels
        for i in range(ans_len):
            labels[start_idx + i] = input_ids[start_idx + i]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def collate_fn(batch):
    """
    Collate function for left-aligned padding.
    `labels` has the same shape as `input_ids`, but non-answer tokens are masked with -100.
    """
    input_ids = [b["input_ids"] for b in batch]
    attention_mask = [b["attention_mask"] for b in batch]
    labels = [b["labels"] for b in batch]

    max_len = max(x.size(0) for x in input_ids)
    padded_ids = []
    padded_mask = []
    padded_labels = []

    for ids, mask, lab in zip(input_ids, attention_mask, labels):
        pad_len = max_len - ids.size(0)
        padded_ids.append(
            torch.cat([ids, torch.full((pad_len,), 0, dtype=ids.dtype)])
        )
        padded_mask.append(
            torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])
        )
        padded_labels.append(
            torch.cat([lab, torch.full((pad_len,), -100, dtype=lab.dtype)])
        )

    input_ids = torch.stack(padded_ids, dim=0)
    attention_mask = torch.stack(padded_mask, dim=0)
    labels = torch.stack(padded_labels, dim=0)

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
        help="Which support set from personalized dataset to use for LoRA training.",
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
        "--max_train_samples",
        type=int,
        default=None,
        help="If set, use at most N training examples (support items across users).",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs for LoRA.",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for LoRA training.",
    )
    ap.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for LoRA training.",
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
    args = ap.parse_args()

    set_seed(args.seed)

    # Default dataset_dir
    if args.dataset_dir is None:
        if args.dataset == "para":
            args.dataset_dir = "datasets/PARA"
        else:
            args.dataset_dir = "datasets/LAPIS"

    # 1) Load personalized data
    print(f"[info] loading personalized {args.dataset.upper()} dataset...")
    if args.dataset == "para":
        personalized = get_personalized_para_dataset(seed=args.seed, dataset_dir=args.dataset_dir)
    else:
        personalized = get_personalized_lapis_dataset(seed=args.seed, dataset_dir=args.dataset_dir)

    all_user_ids = sorted(personalized.keys())
    print(f"[info] num users in personalized dataset: {len(all_user_ids)}")

    if args.quick_users is not None and args.quick_users < len(all_user_ids):
        user_ids = all_user_ids[: args.quick_users]
        print(f"[info] quick_users mode: using first {len(user_ids)} users out of {len(all_user_ids)}")
    else:
        user_ids = all_user_ids

    # 2) Build training (support) and evaluation (test) data
    train_examples = []
    eval_items = []

    for user_id in user_ids:
        pdata = personalized[user_id]
        if args.support_set == "small":
            support_items = pdata.support_small
        else:
            support_items = pdata.support_large
        test_items = pdata.test

        for it in support_items:
            score_int = score_to_class(float(it.score)) + 1  # 1〜5
            train_examples.append(
                {
                    "user_id": user_id,
                    "image_path": it.image_path,
                    "score_int": score_int,
                }
            )

        for it in test_items:
            eval_items.append(
                {
                    "user_id": user_id,
                    "image_path": it.image_path,
                    "score": float(it.score),
                }
            )

    print(f"[info] total training examples (support) = {len(train_examples)}")
    print(f"[info] total eval examples (test)      = {len(eval_items)}")

    if args.max_train_samples is not None and args.max_train_samples < len(train_examples):
        train_examples = train_examples[: args.max_train_samples]
        print(f"[info] max_train_samples: using first {len(train_examples)} training examples")

    # 3) Select one model (Gemma or Qwen)
    model_specs = []
    if args.gemma_model_id:
        model_specs.append(("gemma", args.gemma_model_id))
    if args.qwen_model_id:
        model_specs.append(("qwen", args.qwen_model_id))

    if len(model_specs) != 1:
        raise ValueError("For token-level LoRA, please specify exactly one of --gemma_model_id or --qwen_model_id.")

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
    tokenizer = processor.tokenizer

    # LoRA configuration (text-side Attention/MLP)
    lora_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_targets,
    )
    model = get_peft_model(base_model, lora_config)
    model.to(device)
    model.train()
    model.print_trainable_parameters()

    # 4) Dataset & DataLoader
    train_dataset = TokenClsDataset(train_examples, processor=processor, device=device, tokenizer=tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    num_training_steps = args.epochs * max(1, len(train_loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(100, num_training_steps // 10),
        num_training_steps=num_training_steps,
    )

    print(f"[train] epochs={args.epochs}, steps={num_training_steps}, batch_size={args.batch_size}")

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        for batch in tqdm(train_loader, desc=f"LoRA tokencls epoch {epoch+1}/{args.epochs}"):
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

        print(f"[train] epoch {epoch+1}: loss = {np.mean(epoch_losses):.4f}")

    # 5) Evaluation: generate outputs on each user's test set
    model.eval()
    rows = []
    method_name = f"lora_tokencls_{args.support_set}"

    for item in tqdm(eval_items, desc="LoRA tokencls eval"):
        img = Image.open(item["image_path"]).convert("RGB")
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
                max_new_tokens=8,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                num_beams=1,
            )

        input_len = model_inputs["input_ids"].shape[-1]
        gen_tokens = gen_ids[:, input_len:]
        text = tokenizer.decode(
            gen_tokens[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()

        score_pred = parse_int_score_from_text(text)
        user_score = float(item["score"])
        user_id = item["user_id"]

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

    # 6) Save CSV
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