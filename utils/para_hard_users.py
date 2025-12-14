# utils/para_hard_users.py

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr

from utils.para import (
    PersonalizedPARA,
    PersonalizedPARAItem,
    PERSONALIZED_SCORE_ATTRIBUTES,
    _load_personalized_data,
    get_para_dataset,
)


def compute_user_giaa_correlations(
    dataset_dir: str = "datasets/PARA",
    metric: str = "spearman",
    min_corr_items: int = 30,
) -> pd.DataFrame:
    """
    PARA の Personalized データと GIAA を用いて、
    各ユーザーごとの「自分のスコア vs GIAA score」の相関を計算する。

    Returns:
        pd.DataFrame with columns:
          - userId
          - corr       : 相関係数
          - n_items    : 相関計算に使った画像数
    """
    if metric not in ("spearman", "pearson"):
        raise ValueError(f"metric must be 'spearman' or 'pearson', got {metric}")

    # --- Personalized data (PIAA) ---
    df_p = _load_personalized_data(dataset_dir)
    # 必要な列があるかチェック
    required_p = {"userId", "image_path", "aestheticScore"}
    missing_p = required_p - set(df_p.columns)
    if missing_p:
        raise RuntimeError(f"PARA personalized data missing columns: {missing_p}")

    # --- General data (GIAA) ---
    items_g = get_para_dataset(None, dataset_dir=dataset_dir)
    if not items_g:
        raise RuntimeError("get_para_dataset(None) returned no items; check dataset_dir.")
    rows_g = []
    for it in items_g:
        rows_g.append(
            {
                "image_path": it.image_path,
                "giaa_score": float(it.score),
            }
        )
    df_g = pd.DataFrame(rows_g)

    # --- join on image_path ---
    df = pd.merge(df_p, df_g, on="image_path", how="inner")
    if df.empty:
        raise RuntimeError(
            "Joined personalized + GIAA data is empty. "
            "Check that image_path formats match between PARA-Giaa* and PARA-Images."
        )

    # --- per-user correlation ---
    results = []
    grouped = df.groupby("userId")

    for user_id, g in tqdm(grouped, desc="Computing user–GIAA correlations"):
        g = g.dropna(subset=["aestheticScore", "giaa_score"])
        if len(g) < min_corr_items:
            continue
        x = g["aestheticScore"].to_numpy(dtype=float)
        y = g["giaa_score"].to_numpy(dtype=float)
        if np.all(x == x[0]) or np.all(y == y[0]):
            # 定数配列 → 相関定義できないのでスキップ
            continue

        if metric == "spearman":
            r = spearmanr(x, y).correlation
        else:
            r = pearsonr(x, y)[0]

        if np.isnan(r):
            continue

        results.append({"userId": user_id, "corr": float(r), "n_items": int(len(g))})

    df_corr = pd.DataFrame(results)
    df_corr = df_corr.sort_values("corr", ascending=True).reset_index(drop=True)
    return df_corr


def get_personalized_para_hard_users_dataset(
    seed: int,
    dataset_dir: str = "datasets/PARA",
    max_users: int = 50,
    metric: str = "spearman",
    min_corr_items: int = 30,
    num_support_small: int = 10,
    num_support_large: int = 100,
    num_test: int = 50,
) -> Tuple[Dict[str, PersonalizedPARA], pd.DataFrame]:
    """
    GIAA (general score) と個人スコアの相関が低い順にユーザーを選抜し、
    そのユーザーだけを使った PersonalizedPARA データセットを構築する。

    具体的には:
      1. compute_user_giaa_correlations() で userId ごとの corr を計算
      2. corr が低い順に最大 max_users ユーザーを選ぶ
         （ただし後の split 用に十分な枚数があるユーザーのみ）
      3. 各ユーザーについて、元の get_personalized_para_dataset と同じ形式
         (support_small, support_large, test) を構築する

    Args:
        seed:
            ランダムシャッフルとユーザーサンプリングのシード。
        dataset_dir:
            PARA データセットのルートディレクトリ。
        max_users:
            選抜する最大ユーザー数（corr が低い順に詰める）。
        metric:
            "spearman" or "pearson"。相関の種類。
        min_corr_items:
            相関を計算するために最低限必要な画像数。
        num_support_small:
            support_small の枚数。
        num_support_large:
            support_large の枚数。
        num_test:
            test の枚数。

    Returns:
        (personalized_dataset, df_corr_selected)

        personalized_dataset:
          userId -> PersonalizedPARA の dict
        df_corr_selected:
          選抜されたユーザーの (userId, corr, n_items) を含む DataFrame
    """
    # 1) 全ユーザーの corr を計算
    df_corr = compute_user_giaa_correlations(
        dataset_dir=dataset_dir,
        metric=metric,
        min_corr_items=min_corr_items,
    )
    if df_corr.empty:
        raise RuntimeError("No users with valid correlation; cannot build hard-users dataset.")

    print(f"[info] computed correlations for {len(df_corr)} users.")
    total_required = num_support_small + num_support_large + num_test

    # 2) Personalized 生データ全体をロード（split用）
    df_p = _load_personalized_data(dataset_dir)
    if df_p.empty:
        raise RuntimeError("Personalized PARA data is empty; check dataset_dir.")

    # ユーザーごとの総レーティング数（全画像）
    user_counts_total = df_p["userId"].value_counts()

    # corrが低い順に走査しつつ、split用に十分な枚数を持つユーザーを順にピックアップ
    selected_users: List[str] = []
    selected_rows = []
    for _, row in df_corr.iterrows():
        uid = row["userId"]
        if user_counts_total.get(uid, 0) >= total_required:
            selected_users.append(uid)
            selected_rows.append(row)
            if len(selected_users) >= max_users:
                break

    if not selected_users:
        raise RuntimeError(
            f"No users have at least {total_required} ratings while also having a valid correlation."
        )

    df_corr_selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    print(
        f"[info] selected {len(selected_users)} users with lowest {metric} corr "
        f"(max_users={max_users}, total_required={total_required})."
    )

    # 3) 選抜ユーザーだけを使って PersonalizedPARA を構築
    df_p_sel = df_p[df_p["userId"].isin(selected_users)].copy()

    personalized_dataset: Dict[str, PersonalizedPARA] = {}
    rng = random.Random(seed)

    grouped = df_p_sel.groupby("userId")

    for user_id, user_group_df in tqdm(grouped, desc="Building hard-user PersonalizedPARA"):
        # 念のためチェック（通常は満たしているはず）
        if len(user_group_df) < total_required:
            continue

        # ランダムシャッフル（seed 固定）
        shuffled_group = user_group_df.sample(frac=1.0, random_state=seed)

        support_small_df = shuffled_group.iloc[:num_support_small]
        support_large_df = shuffled_group.iloc[
            num_support_small : num_support_small + num_support_large
        ]
        test_df = shuffled_group.iloc[
            num_support_small + num_support_large : num_support_small
            + num_support_large
            + num_test
        ]

        if len(test_df) < num_test:
            # 足りなければスキップ
            continue

        def df_to_items(split_df: pd.DataFrame) -> List[PersonalizedPARAItem]:
            items: List[PersonalizedPARAItem] = []
            for _, r in split_df.iterrows():
                attributes = {
                    new_name: r[original_name]
                    for original_name, new_name in PERSONALIZED_SCORE_ATTRIBUTES.items()
                    if original_name in r and pd.notna(r[original_name])
                }
                items.append(
                    PersonalizedPARAItem(
                        image_path=r["image_path"],
                        user_id=r["userId"],
                        score=float(r["aestheticScore"]),
                        attributes=attributes,
                    )
                )
            return items

        personalized_dataset[user_id] = PersonalizedPARA(
            support_small=df_to_items(support_small_df),
            support_large=df_to_items(support_large_df),
            test=df_to_items(test_df),
        )

    print(f"[info] built hard-user PersonalizedPARA for {len(personalized_dataset)} users.")

    return personalized_dataset, df_corr_selected


if __name__ == "__main__":
    # 簡単な動作確認用
    try:
        ds, stats = get_personalized_para_hard_users_dataset(
            seed=42,
            dataset_dir="datasets/PARA",
            max_users=200,
            metric="spearman",
            min_corr_items=30,
        )
        print(f"Hard-user dataset users: {len(ds)}")
        print(stats.head())
        if ds:
            first_user = list(ds.keys())[0]
            udata = ds[first_user]
            print(f"First user={first_user}")
            print(f"  support_small: {len(udata.support_small)}")
            print(f"  support_large: {len(udata.support_large)}")
            print(f"  test         : {len(udata.test)}")
    except FileNotFoundError:
        print("PARA dataset not found. Please check dataset_dir.")