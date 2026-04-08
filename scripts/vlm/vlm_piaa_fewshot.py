#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Few-shot PIAA baseline using multimodal VLMs (Gemma3 / Qwen3-VL) on PARA / LAPIS.

For each user u:
  - We have support set S_u (support_small or support_large) and test set T_u
    from get_personalized_*_dataset.
  - Few-shot inference:
    - Prompt the VLM with:
        - Instructions: "You are an expert judge of image aesthetics ... "
        - For each support image: show the image and the user's rating (1–5).
        - Then show the test image and ask: "What is this user's rating?"
    - Parse the numeric score from the generated text.
  - Evaluate on T_u by comparing the predicted score to the user's ground truth.

Output (CSV):
  user_id, image_path, model_id, support_set, method, giaa, piaa_pred, user_score

  - method: "vlm_fewshot_<support_set>" (e.g. vlm_fewshot_small)
  - giaa : NaN (we don't use GIAA here)
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
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers import Qwen3VLForConditionalGeneration

from utils.para import get_personalized_para_dataset
from utils.lapis import get_personalized_lapis_dataset


# ---------- Few-shot Prompt Template ----------

def build_fewshot_messages(
    support_items,
    test_image: Image.Image,
    dataset_name: str,
) -> List[Dict]:
    """
    support_items: List of PersonalizedPARAItem / PersonalizedLAPISItem
      Each item has .image_path and .score (1–5)
    test_image: PIL.Image
    dataset_name: "PARA" / "LAPIS" (for wording, if needed)

    Returns:
      list of messages to pass to apply_chat_template
    """

    # Introductory instruction text
    intro_text = (
        "You are an expert judge of image aesthetics.\n"
        "I will show you some example images with this user's ratings on a 1 to 5 scale.\n"
        "From these examples, infer the user's personal preferences.\n"
        "Then I will show you a new image; please predict this user's rating for it.\n\n"
        "For the examples, each rating is this user's own rating, already mapped to a 1 to 5 scale.\n"
        "When you answer for the final image, respond with a single number from 1 to 5, and nothing else.\n"
    )

    content = []
    # Instruction text
    content.append({"type": "text", "text": intro_text})

    # Support images + scores
    for idx, item in enumerate(support_items):
        img = Image.open(item.image_path).convert("RGB")
        content.append({"type": "image", "image": img})
        content.append(
            {
                "type": "text",
                "text": f"Example {idx+1}: This user rated this image {item.score:.2f} out of 5.",
            }
        )

    # Test image
    content.append({"type": "image", "image": test_image})
    content.append(
        {
            "type": "text",
            "text": (
                "Now, based on the user's previous ratings, "
                "what is this user's rating for THIS image? "
                "Answer with a single number from 1 to 5, and do not output any other text."
            ),
        }
    )

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]
    return messages


# ---------- Common Helpers ----------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_inputs_from_messages(processor, messages, device: torch.device):
    """
    Tokenize `messages` with `processor.apply_chat_template` to build model inputs.
    """
    if not hasattr(processor, "apply_chat_template"):
        raise RuntimeError(
            "Processor does not support `apply_chat_template`. "
            "This script assumes a chat-style multimodal model (Gemma3/Qwen3-VL)."
        )

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    model_inputs = {k: v.to(device) for k, v in inputs.items()}
    model_inputs.pop("token_type_ids", None)
    return model_inputs


def parse_float_from_text(text: str) -> float:
    """
    Parse the first float-like number from the output text.
    """
    m = re.search(r"[-+]?\d+(\.\d+)?", text)
    if not m:
        return math.nan
    try:
        return float(m.group(0))
    except Exception:
        return math.nan


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
        help="Which support set from personalized dataset to use.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for personalized split and quick mode.",
    )
    ap.add_argument(
        "--quick",
        type=int,
        default=None,
        help="If set, limit to at most N users (for debugging). Use 1 for a single-user check.",
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

    # Load personalized data
    print(f"[info] loading personalized {args.dataset.upper()} dataset...")
    if args.dataset == "para":
        personalized = get_personalized_para_dataset(seed=args.seed, dataset_dir=args.dataset_dir)
    else:
        personalized = get_personalized_lapis_dataset(seed=args.seed, dataset_dir=args.dataset_dir)

    all_user_ids = sorted(personalized.keys())
    print(f"[info] num users in personalized dataset: {len(all_user_ids)}")

    # quick: limit the number of users
    if args.quick is not None and args.quick < len(all_user_ids):
        user_ids = all_user_ids[:args.quick]
        print(f"[info] quick mode: using first {len(user_ids)} users out of {len(all_user_ids)}")
    else:
        user_ids = all_user_ids

    # Load the requested model set (Gemma, Qwen, or both)
    model_specs = []
    if args.gemma_model_id:
        model_specs.append(("gemma", args.gemma_model_id))
    if args.qwen_model_id:
        model_specs.append(("qwen", args.qwen_model_id))

    if not model_specs:
        raise ValueError("Please specify at least one of --gemma_model_id or --qwen_model_id")

    all_rows: List[dict] = []

    for family, mid in model_specs:
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

        method_name = f"vlm_fewshot_{args.support_set}"

        # Run few-shot inference for each user
        for user_id in tqdm(user_ids, desc=f"Users[{mid}]"):
            pdata = personalized[user_id]
            if args.support_set == "small":
                support_items = pdata.support_small
            else:
                support_items = pdata.support_large
            test_items = pdata.test

            # Skip users with insufficient support examples
            if len(support_items) < 1:
                continue

            # Predict one test image at a time
            for item in tqdm(
                test_items,
                desc=f"  Test[{user_id}]",
                leave=False,
            ):
                test_img = Image.open(item.image_path).convert("RGB")
                messages = build_fewshot_messages(
                    support_items=support_items,
                    test_image=test_img,
                    dataset_name=args.dataset.upper(),
                )
                inputs = build_inputs_from_messages(processor, messages, device)

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

                score_pred = parse_float_from_text(text)
                user_score = float(item.score)

                all_rows.append(
                    {
                        "user_id": user_id,
                        "image_path": item.image_path,
                        "model_id": mid,
                        "support_set": args.support_set,
                        "method": method_name,
                        "giaa": math.nan,          # GIAA is unused in few-shot mode
                        "piaa_pred": score_pred,
                        "user_score": user_score,
                    }
                )

    # Save CSV
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
        for r in all_rows:
            w.writerow(r)

    print(f"[done] wrote {len(all_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()