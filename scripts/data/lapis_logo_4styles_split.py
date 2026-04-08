#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from utils.lapis import get_personalized_lapis_dataset


def load_imageid_to_4styles(dataset_dir: str) -> Tuple[Dict[int, str], List[str]]:
    """
    Build mapping image_id -> 4_styles across all GIAA splits.
    Also return global unique 4_styles list.
    """
    split_map = {"train": "Trainsplit", "val": "Valsplit", "test": "Testsplit"}
    mapping: Dict[int, str] = {}
    styles = set()

    for split, name in split_map.items():
        p = os.path.join(dataset_dir, "annotation", f"LAPIS_GIAA_{name}.csv")
        df = pd.read_csv(p)

        for _, r in df.iterrows():
            try:
                iid = int(r["image_id"])
            except Exception:
                continue

            tag = str(r["4_styles"]) if pd.notna(r["4_styles"]) else "UNKNOWN"
            styles.add(tag)

            if iid not in mapping:
                mapping[iid] = tag

    return mapping, sorted(styles)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", default="datasets/LAPIS")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--min_tags", type=int, default=2)
    args = ap.parse_args()

    # 1️⃣ Global style vocabulary
    imageid_to_tag, global_styles = load_imageid_to_4styles(args.dataset_dir)
    print(f"[info] global 4_styles: {global_styles}")

    # 2️⃣ Personalized users
    personalized = get_personalized_lapis_dataset(seed=args.seed, dataset_dir=args.dataset_dir)
    user_ids = sorted(list(personalized.keys()))
    print(f"[info] personalized users: {len(user_ids)}")

    rows = []
    missing_styles_per_user = {}

    for uid in tqdm(user_ids, desc="Users"):
        pdata = personalized[uid]
        items = list(pdata.support_large) + list(pdata.test)

        item_rows = []
        user_styles = set()

        for it in items:
            iid = int(it.image_id)
            tag = imageid_to_tag.get(iid)
            if tag is None:
                continue

            user_styles.add(tag)
            item_rows.append({
                "user_id": int(uid),
                "image_id": iid,
                "image_path": it.image_path,
                "user_score": float(it.score),
                "tag_4styles": tag,
            })

        if not item_rows:
            continue

        # 🔎 Missing style detection
        missing = set(global_styles) - user_styles
        if missing:
            missing_styles_per_user[int(uid)] = sorted(list(missing))

        if len(user_styles) < args.min_tags:
            continue

        # Leave-one-group-out
        for holdout in user_styles:
            for r in item_rows:
                split = "test" if r["tag_4styles"] == holdout else "train"
                rows.append({
                    **r,
                    "holdout_tag": holdout,
                    "logo_split": split,
                })

    df_out = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df_out.to_csv(args.out_csv, index=False)

    print(f"\n[save] {args.out_csv}")
    print(f"[info] total rows: {len(df_out)}")

    # 📊 Missing style report
    print("\n========== Missing 4_styles Report ==========")
    print(f"Total users with missing styles: {len(missing_styles_per_user)} / {len(user_ids)}")

    if missing_styles_per_user:
        example = list(missing_styles_per_user.items())[:10]
        print("Examples (first 10 users):")
        for uid, styles in example:
            print(f"  user {uid}: missing {styles}")

        # style frequency statistics
        style_missing_count = {s: 0 for s in global_styles}
        for styles in missing_styles_per_user.values():
            for s in styles:
                style_missing_count[s] += 1

        print("\nMissing count per style:")
        for s, c in style_missing_count.items():
            print(f"  {s}: {c} users")

    print("=============================================\n")


if __name__ == "__main__":
    main()