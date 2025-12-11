# utils/para_hard_images.py

import os
import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

# 既存の PARA ユーティリティから再利用
from utils.para import (
    PersonalizedPARA,
    PersonalizedPARAItem,
    PERSONALIZED_SCORE_ATTRIBUTES,
    _load_personalized_data,
)


def get_personalized_para_hard_dataset(
    seed: int,
    dataset_dir: str = "datasets/PARA",
    top_fraction: float = 0.5,
    min_raters: int = 3,
    num_support_small: int = 10,
    num_support_large: int = 100,
    num_test: int = 50,
    max_users: int = 200,
) -> Dict[str, PersonalizedPARA]:
    """
    Personalized PARA データセット（PIAAの個人嗜好データ）から、
    「aestheticScore の分散が高い（= 意見の割れている）画像」だけを使った
    Personalized データセットを構築する。

    手順:
      1. utils.para._load_personalized_data() で全アノテーションを読み込む
      2. image_path ごとに aestheticScore の分散と評価者数を計算
      3. 評価者数 >= min_raters の画像のみを対象にし、分散が上位 top_fraction に入る画像を「Hard image」とする
      4. Hard image のみから、元の get_personalized_para_dataset と同様に
         ユーザーごとの (support_small, support_large, test) を構築する

    Args:
        seed:
            ランダムシャッフルとユーザーサンプリングのシード。
        dataset_dir:
            PARA データセットのルートディレクトリ。
        top_fraction:
            分散が高い上位何パーセントを Hard image とみなすか (0〜1]。
            0.5 なら「上位50%の分散を持つ画像」。
        min_raters:
            画像を Hard 候補にするために必要な最小評価者数。
        num_support_small:
            support_small の枚数（デフォルト: 10）。
        num_support_large:
            support_large の枚数（デフォルト: 100）。
        num_test:
            test の枚数（デフォルト: 50）。
        max_users:
            最大何ユーザー分の Personalized データを構築するか（多い場合はサンプリング）。

    Returns:
        user_id → PersonalizedPARA の dict。
    """
    if not (0.0 < top_fraction <= 1.0):
        raise ValueError(f"top_fraction must be in (0, 1], got {top_fraction}")

    # ---- 1. パーソナライズド生データをロード ----
    df = _load_personalized_data(dataset_dir)
    # 必要な列が揃っているか確認
    required_cols = {"userId", "image_path", "aestheticScore"}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"PARA personalized data missing columns: {missing}")

    # ---- 2. 画像ごとの aestheticScore 分散 & 評価者数 ----
    grp = df.groupby("image_path")["aestheticScore"]
    img_var = grp.var(ddof=0)        # population variance
    img_cnt = grp.size()

    img_stats = pd.DataFrame(
        {
            "image_path": img_var.index,
            "score_var": img_var.values,
            "num_raters": img_cnt.values,
        }
    )

    # 評価者数が少なすぎる画像を除外
    img_stats = img_stats[img_stats["num_raters"] >= min_raters]
    if img_stats.empty:
        raise RuntimeError(
            f"No images have at least min_raters={min_raters} ratings; "
            "cannot build hard-image dataset."
        )

    # ---- 3. 分散の上位 top_fraction を Hard image とする ----
    # 分散の閾値を quantile から決める
    # top_fraction=0.5 のとき、median 以上が Hard image
    threshold = img_stats["score_var"].quantile(1.0 - top_fraction)
    hard_stats = img_stats[img_stats["score_var"] >= threshold]
    hard_image_paths = set(hard_stats["image_path"].tolist())

    if not hard_image_paths:
        raise RuntimeError("No hard images selected; check top_fraction / min_raters settings.")

    print(
        f"[info] total unique images={len(img_stats)}, "
        f"hard images={len(hard_image_paths)} "
        f"(top_fraction={top_fraction}, threshold={threshold:.4f})"
    )

    # ---- 4. Hard image のみを使って Personalized データ構築 ----
    df_hard = df[df["image_path"].isin(hard_image_paths)].copy()
    if df_hard.empty:
        raise RuntimeError("Filtered personalized data is empty after applying hard-image mask.")

    # ユーザーごとの Hard image 枚数
    user_counts = df_hard["userId"].value_counts()

    total_required = num_support_small + num_support_large + num_test
    valid_users = user_counts[user_counts >= total_required].index.tolist()
    print(
        f"[info] users with >= {total_required} hard-image ratings: "
        f"{len(valid_users)}"
    )

    if not valid_users:
        raise RuntimeError(
            f"No users have at least {total_required} hard-image ratings; "
            "cannot construct PersonalizedPARA splits."
        )

    # ユーザーを max_users に制限しつつサンプル
    rng = random.Random(seed)
    if len(valid_users) > max_users:
        selected_user_ids = rng.sample(valid_users, max_users)
    else:
        selected_user_ids = list(valid_users)
        rng.shuffle(selected_user_ids)

    df_user = df_hard[df_hard["userId"].isin(selected_user_ids)].copy()

    personalized_dataset: Dict[str, PersonalizedPARA] = {}

    grouped = df_user.groupby("userId")

    for user_id, user_group_df in tqdm(grouped, desc="Processing hard-image users"):
        # ユーザーごとのデータをシャッフル
        shuffled_group = user_group_df.sample(frac=1.0, random_state=seed)

        # support_small / support_large / test に分割
        if len(shuffled_group) < total_required:
            # 念のためのガード（valid_users でフィルタしているので通常は起きない）
            continue

        support_small_df = shuffled_group.iloc[:num_support_small]
        support_large_df = shuffled_group.iloc[
            num_support_small : num_support_small + num_support_large
        ]
        test_df = shuffled_group.iloc[
            num_support_small + num_support_large : num_support_small
            + num_support_large
            + num_test
        ]

        # 足りない場合はスキップ
        if len(test_df) < num_test:
            continue

        def df_to_items(split_df: pd.DataFrame) -> List[PersonalizedPARAItem]:
            items: List[PersonalizedPARAItem] = []
            for _, row in split_df.iterrows():
                # 属性ベクトルを構築（aestheticScore, qualityScore, ...）
                attributes = {
                    new_name: row[original_name]
                    for original_name, new_name in PERSONALIZED_SCORE_ATTRIBUTES.items()
                    if original_name in row and pd.notna(row[original_name])
                }
                items.append(
                    PersonalizedPARAItem(
                        image_path=row["image_path"],
                        user_id=row["userId"],
                        score=float(row["aestheticScore"]),
                        attributes=attributes,
                    )
                )
            return items

        personalized_dataset[user_id] = PersonalizedPARA(
            support_small=df_to_items(support_small_df),
            support_large=df_to_items(support_large_df),
            test=df_to_items(test_df),
        )

    print(
        f"[info] built hard-image PersonalizedPARA for {len(personalized_dataset)} users "
        f"(max_users={max_users})"
    )

    return personalized_dataset


if __name__ == "__main__":
    # 簡易テスト用
    try:
        ds = get_personalized_para_hard_dataset(seed=42)
        print(f"Constructed hard-image personalized dataset for {len(ds)} users.")
        if ds:
            first_user = list(ds.keys())[0]
            udata = ds[first_user]
            print(f"First user={first_user}")
            print(f"  support_small: {len(udata.support_small)}")
            print(f"  support_large: {len(udata.support_large)}")
            print(f"  test         : {len(udata.test)}")
    except FileNotFoundError:
        print("PARA dataset not found. Please check dataset_dir.")