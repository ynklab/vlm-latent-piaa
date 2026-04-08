#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build ICI splits under the policy:

  Phase 1,2: PARA -> Phase 3 PARA  -> Eval PARA
  Phase 1,2: PARA -> Phase 3 LAPIS -> Eval LAPIS

So:
  - PARA: build BOTH train split (Phase1/2) and eval split (Phase3/Eval)
  - LAPIS: build ONLY eval split (Phase3/Eval). No LAPIS train split.

Outputs:
  out_root/
    para/
      train_users.json
      train_images.json
      train_interactions.csv
      eval_users.json
      eval_support_small.csv
      eval_support_large.csv
      eval_test.csv
      user_attrs.csv
      image_attrs.csv
      image_dis.csv

    lapis/
      eval_users.json
      eval_support_small.csv
      eval_support_large.csv
      eval_test.csv
      # optionally:
      # user_attrs.csv (from LAPIS_PIAA.csv demographics)
      # image_attrs.csv / image_dis.csv  (NOT needed for Phase1/2 under this policy)
"""

import os
import json
import argparse
from pathlib import Path

import pandas as pd

from utils.para import get_personalized_para_dataset
from utils.lapis import get_personalized_lapis_dataset

# PARA attrs/dis/user
from utils.para_ici_attrs import (
    load_para_image_attrs,
    load_para_image_distribution,
    load_para_user_attrs,
)

# (Optional) LAPIS user demographics (if you still want it for reporting)
# If you don't have utils.lapis_ici_attrs or don't want it, you can remove these imports.
try:
    from utils.lapis_ici_attrs import load_lapis_user_attrs_from_piaa
except Exception:
    load_lapis_user_attrs_from_piaa = None


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def build_para_splits(dataset_dir: str, seed: int, out_root: str):
    print("\n[PARA] Building ICI splits (train + eval)...")

    out_dir = Path(out_root) / "para"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- eval split (keep your existing 200-user split) ---
    personalized = get_personalized_para_dataset(seed=seed, dataset_dir=dataset_dir)
    eval_users = sorted(list(personalized.keys()))

    rows_s, rows_l, rows_t = [], [], []
    for uid, pdata in personalized.items():
        for it in pdata.support_small:
            rows_s.append({"user_id": uid, "image_path": it.image_path, "score": float(it.score)})
        for it in pdata.support_large:
            rows_l.append({"user_id": uid, "image_path": it.image_path, "score": float(it.score)})
        for it in pdata.test:
            rows_t.append({"user_id": uid, "image_path": it.image_path, "score": float(it.score)})

    df_s_small = pd.DataFrame(rows_s)
    df_s_large = pd.DataFrame(rows_l)
    df_test = pd.DataFrame(rows_t)

    df_s_small.to_csv(out_dir / "eval_support_small.csv", index=False)
    df_s_large.to_csv(out_dir / "eval_support_large.csv", index=False)
    df_test.to_csv(out_dir / "eval_test.csv", index=False)
    save_json(eval_users, out_dir / "eval_users.json")

    # --- build train split for Phase1/2 using remaining users/images (no overlap with eval images) ---
    df_raw = pd.read_csv(os.path.join(dataset_dir, "annotation", "PARA-Images.csv"))
    df_raw = df_raw.dropna(subset=["userId", "sessionId", "imageName", "aestheticScore"])

    base_img_path = os.path.abspath(os.path.join(dataset_dir, "imgs"))
    df_raw["image_path"] = base_img_path + os.sep + df_raw["sessionId"] + os.sep + df_raw["imageName"]
    df_raw = df_raw.rename(columns={"userId": "user_id", "aestheticScore": "score"})

    eval_user_set = set(eval_users)
    eval_images_all = set(pd.concat([
        df_s_small["image_path"],
        df_s_large["image_path"],
        df_test["image_path"],
    ]).astype(str).tolist())

    train_df = df_raw[~df_raw["user_id"].isin(eval_user_set)].copy()
    train_df = train_df[~train_df["image_path"].isin(eval_images_all)].copy()

    train_users = sorted(train_df["user_id"].astype(str).unique().tolist())
    train_images = sorted(train_df["image_path"].astype(str).unique().tolist())

    save_json(train_users, out_dir / "train_users.json")
    save_json(train_images, out_dir / "train_images.json")
    train_df[["user_id", "image_path", "score"]].to_csv(out_dir / "train_interactions.csv", index=False)

    print(f"[PARA] total users (raw) = {df_raw['user_id'].nunique()}")
    print(f"[PARA] eval users = {len(eval_users)}")
    print(f"[PARA] train users = {len(train_users)}")
    print(f"[PARA] eval images (union) = {len(eval_images_all)}")
    print(f"[PARA] train images = {len(train_images)}")
    print(f"[PARA] train interactions = {len(train_df)}")

    # --- attrs/dis/user meta for Phase1/2 ---
    df_user = load_para_user_attrs(dataset_dir).rename(columns={"userId": "user_id"})
    df_user.to_csv(out_dir / "user_attrs.csv", index=False)

    df_img_attr = load_para_image_attrs(dataset_dir)
    df_img_attr.to_csv(out_dir / "image_attrs.csv", index=False)

    df_img_dis = load_para_image_distribution(dataset_dir)
    df_img_dis.to_csv(out_dir / "image_dis.csv", index=False)

    print("[PARA] Done.")


def build_lapis_eval_only(dataset_dir: str, seed: int, out_root: str, write_user_attrs: bool = True):
    print("\n[LAPIS] Building ICI splits (eval-only)...")

    out_dir = Path(out_root) / "lapis"
    out_dir.mkdir(parents=True, exist_ok=True)

    personalized = get_personalized_lapis_dataset(seed=seed, dataset_dir=dataset_dir)
    eval_users = sorted([str(x) for x in personalized.keys()])

    rows_s, rows_l, rows_t = [], [], []
    for uid, pdata in personalized.items():
        uid_str = str(uid)
        for it in pdata.support_small:
            rows_s.append({"user_id": uid_str, "image_id": int(it.image_id), "image_path": it.image_path, "score": float(it.score)})
        for it in pdata.support_large:
            rows_l.append({"user_id": uid_str, "image_id": int(it.image_id), "image_path": it.image_path, "score": float(it.score)})
        for it in pdata.test:
            rows_t.append({"user_id": uid_str, "image_id": int(it.image_id), "image_path": it.image_path, "score": float(it.score)})

    df_s_small = pd.DataFrame(rows_s)
    df_s_large = pd.DataFrame(rows_l)
    df_test = pd.DataFrame(rows_t)

    df_s_small.to_csv(out_dir / "eval_support_small.csv", index=False)
    df_s_large.to_csv(out_dir / "eval_support_large.csv", index=False)
    df_test.to_csv(out_dir / "eval_test.csv", index=False)
    save_json(eval_users, out_dir / "eval_users.json")

    print(f"[LAPIS] eval users = {len(eval_users)}")
    print(f"[LAPIS] eval images (union) = {df_s_small['image_path'].nunique() + df_s_large['image_path'].nunique() + df_test['image_path'].nunique()} (note: may overlap across splits)")

    # Optional: still export user demographic attributes for analysis / later use
    if write_user_attrs and load_lapis_user_attrs_from_piaa is not None:
        df_u = load_lapis_user_attrs_from_piaa(dataset_dir)
        df_u.to_csv(out_dir / "user_attrs.csv", index=False)
        print("[LAPIS] wrote user_attrs.csv (demographics)")

    print("[LAPIS] Done.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_root", default="ici_splits")
    ap.add_argument("--para_dir", default="datasets/PARA")
    ap.add_argument("--lapis_dir", default="datasets/LAPIS")
    ap.add_argument("--lapis_write_user_attrs", action="store_true",
                    help="If set, export LAPIS user demographics (optional).")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)

    build_para_splits(args.para_dir, args.seed, args.out_root)
    build_lapis_eval_only(args.lapis_dir, args.seed, args.out_root, write_user_attrs=args.lapis_write_user_attrs)

    print("\n[done] ICI split artifacts generated under:", args.out_root)


if __name__ == "__main__":
    main()