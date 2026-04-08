#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build per-user LOGO splits using LAPIS 2_styles tag.

For each user:
  - Use Support (large) + Test from personalized dataset
  - Split by 2_styles (e.g., FIGURATIVE / ABSTRACT)
  - Leave-one-style-out
  - Generate train/test split

Output CSV:
  user_id, image_id, image_path, user_score, style_2, holdout_tag, logo_split

Also reports users missing one of the styles.
"""

import os
import argparse
import pandas as pd
from tqdm import tqdm

from utils.lapis import get_personalized_lapis_dataset


def load_2styles_mapping(dataset_dir):
    """
    Load mapping:
        image_filename -> 2_styles
    """
    giaa_train = os.path.join(dataset_dir, "annotation", "LAPIS_GIAA_Trainsplit.csv")
    giaa_val   = os.path.join(dataset_dir, "annotation", "LAPIS_GIAA_Valsplit.csv")
    giaa_test  = os.path.join(dataset_dir, "annotation", "LAPIS_GIAA_Testsplit.csv")

    dfs = []
    for p in [giaa_train, giaa_val, giaa_test]:
        df = pd.read_csv(p)
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # image_filename → 2_styles
    mapping = dict(zip(df_all["image_filename"], df_all["2_styles"]))
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default="datasets/LAPIS")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    print("[info] loading personalized LAPIS dataset...")
    personalized = get_personalized_lapis_dataset(
        seed=args.seed,
        dataset_dir=args.dataset_dir
    )

    print("[info] loading 2_styles mapping...")
    style_map = load_2styles_mapping(args.dataset_dir)

    rows = []
    missing_style_users = []
    missing_style_items = 0

    for user_id, pdata in tqdm(personalized.items(), desc="Building LOGO 2_styles"):
        combined = pdata.support_large + pdata.test

        user_records = []
        for item in combined:
            filename = os.path.basename(item.image_path)
            style = style_map.get(filename)
            if style is None:
                missing_style_items += 1
                continue

            user_records.append({
                "user_id": int(user_id),
                "image_id": int(item.image_id),
                "image_path": item.image_path,
                "user_score": float(item.score),
                "style_2": style,
            })

        if not user_records:
            continue

        df_user = pd.DataFrame(user_records)
        unique_styles = df_user["style_2"].unique()

        if len(unique_styles) < 2:
            missing_style_users.append(int(user_id))
            continue

        # Leave-one-style-out
        for holdout in unique_styles:
            tmp = df_user.copy()
            tmp["holdout_tag"] = holdout
            tmp["logo_split"] = tmp["style_2"].apply(lambda s: "test" if s == holdout else "train")
            rows.extend(tmp.to_dict(orient="records"))

    df_out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df_out.to_csv(args.out_csv, index=False)

    print("\n[done]")
    print("Total LOGO rows:", len(df_out))
    print("Users missing one of the styles:", len(missing_style_users))
    print("Items skipped due to missing 2_styles mapping:", missing_style_items)

    if missing_style_users:
        print("Example missing-style users:", missing_style_users[:10])


if __name__ == "__main__":
    main()