# utils/lapis_ici_attrs.py
import os
import pandas as pd
import numpy as np


LAPIS_USER_COLS = [
    "participant_id",
    "age",
    "nationality",
    "demo_gender",
    "demo_edu",
    "demo_colorblind",
    "Art Interest VAIAK",
]


def load_lapis_user_attrs_from_piaa(dataset_dir: str) -> pd.DataFrame:
    p = os.path.join(dataset_dir, "annotation", "LAPIS_PIAA.csv")
    df = pd.read_csv(p)
    df = df.dropna(subset=["participant_id"])
    df["user_id"] = df["participant_id"].astype(str)

    keep = ["user_id"] + [c for c in LAPIS_USER_COLS if c in df.columns]
    df = df[keep].copy()
    df = df.groupby("user_id").first().reset_index()
    return df


def load_lapis_image_attrs(dataset_dir: str) -> pd.DataFrame:
    p = os.path.join(dataset_dir, "annotation", "ImgAttributes_LAPIS.csv")
    df = pd.read_csv(p)
    # key is image_filename
    return df


def _rating_to_score_1to5(rating_0_100: np.ndarray) -> np.ndarray:
    """
    Convert LAPIS rating in [0,100] to score in [1,5].
    Same as your other scripts:
      score = (rating/100)*4 + 1
    """
    return (rating_0_100.astype(float) / 100.0) * 4.0 + 1.0


def _score_to_bin_1to5(score_1to5: np.ndarray) -> np.ndarray:
    """
    Bin score into 1..5 using 0.5-width centered bins:
      [0.5,1.5)->1, [1.5,2.5)->2, ..., [4.5,5.5)->5
    Returns int in {1,2,3,4,5}.
    """
    b = np.floor(score_1to5 + 0.5).astype(int)  # 1.0->1, 1.49->1, 1.5->2, ...
    b = np.clip(b, 1, 5)
    return b


def load_lapis_image_distribution_from_piaa(
    dataset_dir: str,
    train_user_ids: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build per-image score distribution from LAPIS_PIAA.csv.

    Args:
      train_user_ids:
        If provided, only ratings by these users are used (recommended to avoid leakage).
        If None, use all users.

    Returns:
      DataFrame with columns:
        image_id, image_filename, n_raters, dis_1..dis_5, mean_score
      where dis_* sums to 1 per image.
    """
    p = os.path.join(dataset_dir, "annotation", "LAPIS_PIAA.csv")
    df = pd.read_csv(p)
    df = df.dropna(subset=["participant_id", "image_id", "image_filename", "rating"])

    df["user_id"] = df["participant_id"].astype(str)
    if train_user_ids is not None:
        train_user_set = set(map(str, train_user_ids))
        df = df[df["user_id"].isin(train_user_set)].copy()

    # score in 1..5
    df["score_1to5"] = _rating_to_score_1to5(df["rating"].to_numpy())
    df["bin"] = _score_to_bin_1to5(df["score_1to5"].to_numpy())

    # count bins per image
    grp = df.groupby(["image_id", "image_filename"])
    rows = []
    for (image_id, image_filename), g in grp:
        bins = g["bin"].to_numpy(dtype=int)
        n = len(bins)
        counts = np.array([(bins == k).sum() for k in range(1, 6)], dtype=float)
        dis = counts / (counts.sum() + 1e-12)
        mean_score = float(np.mean(g["score_1to5"].to_numpy(dtype=float)))

        rows.append(
            {
                "image_id": int(image_id),
                "image_filename": image_filename,
                "n_raters": int(n),
                "mean_score": mean_score,
                "dis_1": float(dis[0]),
                "dis_2": float(dis[1]),
                "dis_3": float(dis[2]),
                "dis_4": float(dis[3]),
                "dis_5": float(dis[4]),
            }
        )

    out = pd.DataFrame(rows)
    return out