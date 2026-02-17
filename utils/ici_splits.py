# utils/ici_splits.py
import os
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional

import pandas as pd

from utils.para import get_personalized_para_dataset
from utils.lapis import get_personalized_lapis_dataset


@dataclass
class ICISplit:
    dataset: str
    seed: int

    # eval users (fixed 200)
    eval_users: List[str]

    # eval items (Phase3 only)
    eval_support_small: pd.DataFrame  # columns: user_id, image_path (and image_id if lapis), score
    eval_support_large: pd.DataFrame
    eval_test: pd.DataFrame

    # train interactions (Phase1/2 only)
    train_users: List[str]
    train_interactions: pd.DataFrame  # columns: user_id, image_path (and image_id if lapis), score, image_id/image_filename if available


def build_eval_frames_para(personalized: Dict[str, object]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows_s, rows_l, rows_t = [], [], []
    for uid, pdata in personalized.items():
        for it in pdata.support_small:
            rows_s.append({"user_id": uid, "image_path": it.image_path, "score": float(it.score)})
        for it in pdata.support_large:
            rows_l.append({"user_id": uid, "image_path": it.image_path, "score": float(it.score)})
        for it in pdata.test:
            rows_t.append({"user_id": uid, "image_path": it.image_path, "score": float(it.score)})
    return pd.DataFrame(rows_s), pd.DataFrame(rows_l), pd.DataFrame(rows_t)


def build_eval_frames_lapis(personalized: Dict[int, object]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows_s, rows_l, rows_t = [], [], []
    for uid, pdata in personalized.items():
        uid_str = str(uid)
        for it in pdata.support_small:
            rows_s.append({"user_id": uid_str, "image_id": int(it.image_id), "image_path": it.image_path, "score": float(it.score)})
        for it in pdata.support_large:
            rows_l.append({"user_id": uid_str, "image_id": int(it.image_id), "image_path": it.image_path, "score": float(it.score)})
        for it in pdata.test:
            rows_t.append({"user_id": uid_str, "image_id": int(it.image_id), "image_path": it.image_path, "score": float(it.score)})
    return pd.DataFrame(rows_s), pd.DataFrame(rows_l), pd.DataFrame(rows_t)


def build_ici_split(
    dataset: str,
    seed: int,
    dataset_dir: str,
    raw_personal_csv: Optional[str] = None,  # PARA-Images.csv / LAPIS_PIAA.csv
) -> ICISplit:
    """
    - Keeps existing 200-user personalized split for fairness (Phase3 only).
    - Builds Phase1/2 train interactions from users/images not appearing in that eval split.
    """
    if dataset == "para":
        personalized = get_personalized_para_dataset(seed=seed, dataset_dir=dataset_dir)
        eval_users = sorted(list(personalized.keys()))
        s_small, s_large, s_test = build_eval_frames_para(personalized)

        # load raw personalized interactions (all users)
        csv_path = raw_personal_csv or os.path.join(dataset_dir, "annotation", "PARA-Images.csv")
        df_raw = pd.read_csv(csv_path)
        # standardize columns
        df_raw = df_raw.dropna(subset=["userId", "sessionId", "imageName", "aestheticScore"])
        base_img_path = os.path.abspath(os.path.join(dataset_dir, "imgs"))
        df_raw["image_path"] = base_img_path + os.sep + df_raw["sessionId"] + os.sep + df_raw["imageName"]
        df_raw = df_raw.rename(columns={"userId": "user_id", "aestheticScore": "score"})

        eval_user_set = set(eval_users)
        eval_images_all = set(pd.concat([s_small["image_path"], s_large["image_path"], s_test["image_path"]]).astype(str).tolist())

        train_df = df_raw[~df_raw["user_id"].isin(eval_user_set)].copy()
        train_df = train_df[~train_df["image_path"].isin(eval_images_all)].copy()

        train_users = sorted(train_df["user_id"].astype(str).unique().tolist())

        return ICISplit(
            dataset="para",
            seed=seed,
            eval_users=eval_users,
            eval_support_small=s_small,
            eval_support_large=s_large,
            eval_test=s_test,
            train_users=train_users,
            train_interactions=train_df[["user_id", "image_path", "score"]].copy(),
        )

    elif dataset == "lapis":
        personalized = get_personalized_lapis_dataset(seed=seed, dataset_dir=dataset_dir)
        eval_users = sorted([str(x) for x in personalized.keys()])
        s_small, s_large, s_test = build_eval_frames_lapis(personalized)

        csv_path = raw_personal_csv or os.path.join(dataset_dir, "annotation", "LAPIS_PIAA.csv")
        df_raw = pd.read_csv(csv_path)
        df_raw = df_raw.dropna(subset=["participant_id", "image_id", "image_filename", "rating"])
        base_img_path = os.path.abspath(os.path.join(dataset_dir, "images"))
        df_raw["image_path"] = base_img_path + os.sep + df_raw["image_filename"]
        df_raw["user_id"] = df_raw["participant_id"].astype(str)
        df_raw["image_id"] = df_raw["image_id"].astype(int)
        df_raw["score"] = (df_raw["rating"].astype(float) / 100.0) * 4.0 + 1.0

        eval_user_set = set(eval_users)
        eval_images_all = set(pd.concat([s_small["image_path"], s_large["image_path"], s_test["image_path"]]).astype(str).tolist())

        train_df = df_raw[~df_raw["user_id"].isin(eval_user_set)].copy()
        train_df = train_df[~train_df["image_path"].isin(eval_images_all)].copy()

        train_users = sorted(train_df["user_id"].astype(str).unique().tolist())

        return ICISplit(
            dataset="lapis",
            seed=seed,
            eval_users=eval_users,
            eval_support_small=s_small,
            eval_support_large=s_large,
            eval_test=s_test,
            train_users=train_users,
            train_interactions=train_df[["user_id", "image_id", "image_path", "score", "image_filename"]].copy(),
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset}")