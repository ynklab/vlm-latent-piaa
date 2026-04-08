#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Per-user CoT-style instruction tuning for PIAA on PARA / LAPIS using a multimodal VLM (Gemma3 / Qwen3-VL),
with pseudo aesthetic attributes obtained from an AADB-trained hidden->attribute projection.

Workflow (per user u):

  1. Use get_personalized_*_dataset(seed) to obtain:
       - support_small or support_large (chosen by --support_set)
       - test set T_u

  2. For each image i in support:
       - Extract hidden feature h_i via mm_embed for (feature_source, feature_layer) from proj_file.
       - Apply projection W,b (with scaler) to obtain attr_vec_i (AADB-style attribute vector).
       - Decide target score s_i depending on --target_score:
           * piaa   : s_i = user_score_i   (personalized PIAA)
           * giaa_gt: s_i = dataset-level GIAA mean for that image
       - Build an instruction-tuning sample:
           User:  image + CoT-style prompt
           Assistant: lines like
             Attr1: a1
             Attr2: a2
             ...
             Overall: s_i

  3. Fine-tune a user-specific LoRA-augmented VLM on these support examples.
       - Loss is applied ONLY on assistant tokens (attributes + Overall).

  4. For each image j in the test set:
       - Prompt LoRA model with image + CoT-style prompt (no assistant target).
       - Let model generate attributes + Overall.
       - Parse "Overall: Y" as PIAA prediction.

Output CSV (PIAA format + raw_output):

  user_id, image_path, model_id, support_set, method, giaa, piaa_pred, user_score, raw_output

  - method:
      * target_score=piaa   : "cot_attr_lora_per_user_<support_set>"
      * target_score=giaa_gt: "cot_attr_lora_per_user_giaa_gt_<support_set>"
  - giaa      : NaN (GIAA is unused here)
  - piaa_pred : parsed Overall score
  - user_score: always personalized ground truth (PIAA)
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

from utils.aadb import AESTHETIC_ATTRIBUTES as AADB_ATTRS
from utils.para import get_personalized_para_dataset, get_para_dataset
from utils.lapis import get_personalized_lapis_dataset, get_lapis_dataset
from utils.mm_embed import load_mm_model, build_inputs, extract_all_pools


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


def make_prompt_for_mm_embed(prompt_mode: str) -> str:
    if prompt_mode == "base":
        return "Assess the aesthetics of this image."
    elif prompt_mode == "format":
        return (
            "Assess the overall aesthetic quality of this image. "
            "Please rate it on a scale from 1 to 5. "
            "Output only the numeric score, and do not output any other text."
        )
    elif prompt_mode == "attributes":
        return "Describe the aesthetic properties of this image."
    elif prompt_mode == "unrelated":
        return "Describe the weather today in one sentence."
    else:
        return "Assess the aesthetics of this image."


# ---------- Helpers ----------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_overall_from_text(text: str) -> float:
    """
    Parse the value `Y` from `Overall: Y` in the output text.
    If absent, fall back to the last number in the text.
    """
    m = re.search(r"Overall\s*[:=]\s*([-+]?\d+(\.\d+)?)", text, flags=re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return math.nan
    nums = list(re.finditer(r"[-+]?\d+(\.\d+)?", text))
    if nums:
        try:
            return float(nums[-1].group(0))
        except Exception:
            return math.nan
    return math.nan


def extract_feature_vector(pools, source: str, layer_idx: int) -> np.ndarray:
    if source == "llm_text":
        vec = pools.llm_text[layer_idx]
    elif source == "llm_visual":
        vec = pools.llm_visual[layer_idx]
    elif source == "llm_text_tail":
        vec = pools.llm_text_tail[layer_idx]
    elif source == "vision":
        if pools.vision_layers is None:
            raise RuntimeError("vision_layers is None; vision source not available.")
        vec = pools.vision_layers[layer_idx]
    elif source == "bridge_text":
        vec = pools.bridge_text[0]
    elif source == "bridge_visual":
        vec = pools.bridge_visual[0]
    else:
        raise ValueError(f"Unknown feature_source: {source}")
    return vec.astype(np.float32)


# ---------- Dataset for per-user instruction tuning ----------

class UserInstrDataset(Dataset):
    """
    Convert one user's support set into a Dataset for instruction tuning.

    items: List[{"image_path": str, "attrs": np.ndarray[K], "target_score": float}]
      - attrs: AADB-style attribute vector from the projection (roughly in the range 1..5)
      - target_score: Overall Y (either PIAA or GIAA_GT, depending on --target_score)
    """

    def __init__(self, items, processor, tokenizer, device, attr_names: List[str], prompt_text: str):
        self.items = items
        self.processor = processor
        self.tokenizer = tokenizer
        self.device = device
        self.attr_names = attr_names
        self.prompt_text = prompt_text

    def __len__(self):
        return len(self.items)

    def _build_messages(self, image: Image.Image, attrs: np.ndarray, target_score: float):
        lines = []
        for name, val in zip(self.attr_names, attrs):
            v = max(1.0, min(5.0, float(val)))
            lines.append(f"{name}: {v:.2f}")
        s = max(1.0, min(5.0, float(target_score)))
        lines.append(f"Overall: {s:.2f}")
        ans_text = "\n".join(lines)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompt_text},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": ans_text},
                ],
            },
        ]
        return messages

    def __getitem__(self, idx):
        item = self.items[idx]
        img = Image.open(item["image_path"]).convert("RGB")
        attrs = item["attrs"]
        target_score = item["target_score"]

        # full conversation
        messages_full = self._build_messages(img, attrs, target_score)
        out_full = self.processor.apply_chat_template(
            messages_full,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids_full = out_full["input_ids"][0]
        attention_mask_full = out_full["attention_mask"][0]

        # user only (to know where the assistant segment begins)
        messages_user_only = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": self.prompt_text},
                ],
            }
        ]
        out_user = self.processor.apply_chat_template(
            messages_user_only,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids_user = out_user["input_ids"][0]
        user_len = input_ids_user.size(0)

        labels = input_ids_full.clone()
        labels[:user_len] = -100

        return {
            "input_ids": input_ids_full,
            "attention_mask": attention_mask_full,
            "labels": labels,
        }


def collate_fn(batch):
    input_ids = [b["input_ids"] for b in batch]
    attention_mask = [b["attention_mask"] for b in batch]
    labels = [b["labels"] for b in batch]

    max_len = max(x.size(0) for x in input_ids)
    padded_ids, padded_mask, padded_labels = [], [], []

    for ids, mask, lab in zip(input_ids, attention_mask, labels):
        pad_len = max_len - ids.size(0)
        padded_ids.append(torch.cat([ids, torch.full((pad_len,), 0, dtype=ids.dtype)]))
        padded_mask.append(torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)]))
        padded_labels.append(torch.cat([lab, torch.full((pad_len,), -100, dtype=lab.dtype)]))

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
        help="Personalized dataset to use (para or lapis).",
    )
    ap.add_argument(
        "--proj_file",
        required=True,
        help="Projection .npz file trained on AADB (from train_attr_projection_aadb.py).",
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
        "--support_set",
        default="small",
        choices=["small", "large"],
        help="Which support set from personalized dataset to use for per-user LoRA training.",
    )
    ap.add_argument(
        "--target_score",
        default="piaa",
        choices=["piaa", "giaa_gt"],
        help=(
            "Which score to use as the 'Overall' target in instruction tuning:\n"
            "  piaa   : user-specific score (PIAA)\n"
            "  giaa_gt: dataset-level GIAA mean (from GIAA files)"
        ),
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
        help="Learning rate for per-user LoRA.",
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

    # 1) Load the projection (trained on AADB)
    proj = np.load(args.proj_file, allow_pickle=True)
    proj_model_id = proj["model_id"].item()
    feature_source = proj["feature_source"].item()
    feature_layer = int(proj["feature_layer"].item())
    prompt_mode = proj["prompt_mode"].item()
    attr_names = proj["attr_names"].tolist()
    coef = proj["coef"]              # [K, D]
    intercept = proj["intercept"]    # [K]
    scaler_mean = proj["scaler_mean"]
    scaler_scale = proj["scaler_scale"]

    print(f"[info] loaded projection from {args.proj_file}")
    print(f"        proj_model_id = {proj_model_id}")
    print(f"        feature_source= {feature_source}")
    print(f"        feature_layer = {feature_layer}")
    print(f"        prompt_mode   = {prompt_mode}")
    print(f"        attr_names    = {attr_names}")
    print(f"        coef shape    = {coef.shape}")

    # 2) Load personalized data + GIAA ground truth (if needed)
    if args.dataset == "para":
        print(f"[info] loading personalized PARA (seed={args.seed})...")
        personalized = get_personalized_para_dataset(seed=args.seed, dataset_dir=args.dataset_dir)

        image_to_giaa = {}
        if args.target_score == "giaa_gt":
            print("[info] loading PARA GIAA ground truth (PARA-Giaa* files)...")
            ga_items = get_para_dataset(None, dataset_dir=args.dataset_dir)
            for it in ga_items:
                image_to_giaa[it.image_path] = float(it.score)
            print(f"[info] GIAA GT entries (PARA) = {len(image_to_giaa)}")

    else:
        print(f"[info] loading personalized LAPIS (seed={args.seed})...")
        personalized = get_personalized_lapis_dataset(seed=args.seed, dataset_dir=args.dataset_dir)

        image_to_giaa = {}
        if args.target_score == "giaa_gt":
            print("[info] loading LAPIS GIAA ground truth (LAPIS_GIAA_* files)...")
            ga_items = get_lapis_dataset(None, dataset_dir=args.dataset_dir)
            for it in ga_items:
                image_to_giaa[it.image_path] = float(it.score)
            print(f"[info] GIAA GT entries (LAPIS) = {len(image_to_giaa)}")

    all_user_ids = sorted(personalized.keys())
    print(f"[info] num users in personalized {args.dataset.upper()} dataset: {len(all_user_ids)}")

    if args.quick_users is not None and args.quick_users < len(all_user_ids):
        user_ids = all_user_ids[: args.quick_users]
        print(f"[info] quick_users: using first {len(user_ids)} users out of {len(all_user_ids)}")
    else:
        user_ids = all_user_ids

    # 3) Load the VLM used by mm_embed (same model as the projection)
    print(f"[info] loading VLM for hidden features: {proj_model_id}")
    mm_model, mm_processor = load_mm_model(proj_model_id, dtype="auto", device_map="auto", attn_impl=None)
    mm_model.eval()
    mm_device = mm_model.device
    mm_prompt = make_prompt_for_mm_embed(prompt_mode)

    def hidden_to_attr_vec(h: np.ndarray) -> np.ndarray:
        xs = (h - scaler_mean) / (scaler_scale + 1e-12)  # [D]
        raw = xs @ coef.T + intercept                    # [K], expected range roughly [-1, 1]
        # Linear mapping from [-1, 1] to [1, 5]: f(x) = 2x + 3
        mapped = 2.0 * raw + 3.0
        return mapped  # [K], expected range [1, 5]

    # 4) Load the target model for LoRA (Gemma or Qwen)
    model_specs = []
    if args.gemma_model_id:
        model_specs.append(("gemma", args.gemma_model_id))
    if args.qwen_model_id:
        model_specs.append(("qwen", args.qwen_model_id))

    if len(model_specs) != 1:
        raise ValueError("Please specify exactly one of --gemma_model_id or --qwen_model_id.")

    family, mid = model_specs[0]
    print(f"[info] base model for LoRA: {mid}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_kwargs = dict(
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    # CoT prompt built from AADB attribute names
    cot_prompt = build_prompt(attr_names)

    rows: List[dict] = []
    if args.target_score == "piaa":
        method_name = f"cot_attr_lora_per_user_{args.support_set}"
    else:
        method_name = f"cot_attr_lora_per_user_giaa_gt_{args.support_set}"
    support_set_value = args.support_set

    # 5) Run instruction tuning user by user
    for user_id in tqdm(user_ids, desc="Users(cot-attr-lora-per-user)"):
        pdata = personalized[user_id]
        if args.support_set == "small":
            support_items = pdata.support_small
        else:
            support_items = pdata.support_large
        test_items = pdata.test

        if len(support_items) < 1:
            continue

        # Support set: hidden→attr projection + target_score
        train_examples = []
        sup_items = support_items
        if args.quick_items_per_user is not None and args.quick_items_per_user < len(sup_items):
            sup_items = sup_items[: args.quick_items_per_user]

        for it in sup_items:
            path = it.image_path
            img = Image.open(path).convert("RGB")
            inputs = build_inputs(mm_processor, img, mm_prompt)
            pools = extract_all_pools(mm_model, inputs, processor=mm_processor)
            try:
                h = extract_feature_vector(pools, feature_source, feature_layer)
            except IndexError:
                raise IndexError(
                    f"feature_layer={feature_layer} is out of range for feature_source={feature_source}"
                )
            a_pred = hidden_to_attr_vec(h)  # [K]

            # Select target_score
            user_score = float(it.score)
            if args.target_score == "piaa":
                target = user_score
            else:
                if path not in image_to_giaa:
                    continue
                target = float(image_to_giaa[path])

            train_examples.append(
                {
                    "image_path": path,
                    "attrs": a_pred.astype(np.float32),
                    "target_score": target,
                }
            )

        if len(train_examples) < 1:
            continue

        # Test set
        eval_items = test_items
        if args.quick_items_per_user is not None and args.quick_items_per_user < len(eval_items):
            eval_items = eval_items[: args.quick_items_per_user]

        # --- per-user LoRA model ---
        if family == "qwen":
            base_model = Qwen3VLForConditionalGeneration.from_pretrained(mid, **base_kwargs)
        else:
            base_model = AutoModelForCausalLM.from_pretrained(mid, **base_kwargs)

        processor = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
        tokenizer = processor.tokenizer

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

        # Dataset & DataLoader
        train_dataset = UserInstrDataset(
            train_examples,
            processor=processor,
            tokenizer=tokenizer,
            device=device,
            attr_names=attr_names,
            prompt_text=cot_prompt,
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

        for epoch in range(args.epochs):
            model.train()
            epoch_losses = []
            for batch in tqdm(
                train_loader,
                desc=f"  LoRA train user={user_id} epoch {epoch+1}/{args.epochs}",
                leave=False,
            ):
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

        # --- Evaluation on test set ---
        model.eval()
        for it in tqdm(eval_items, desc=f"  eval user={user_id}", leave=False):
            path = it.image_path
            img = Image.open(path).convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": cot_prompt},
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
                    max_new_tokens=200,  # large enough to generate all attributes + Overall
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

            piaa_pred = parse_overall_from_text(text)
            user_score = float(it.score)

            rows.append(
                {
                    "user_id": user_id,
                    "image_path": path,
                    "model_id": mid,
                    "support_set": support_set_value,
                    "method": method_name,
                    "giaa": math.nan,
                    "piaa_pred": piaa_pred,
                    "user_score": user_score,
                    "raw_output": text,
                }
            )

        # Release the LoRA model for this user
        del model
        torch.cuda.empty_cache()

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