import os
import pandas as pd


# ---------------------------------------------------------
# IMAGE ATTRIBUTE (Phase 1 input)
# ---------------------------------------------------------

def load_para_image_attrs(dataset_dir: str) -> pd.DataFrame:
    """
    Load PARA image-level aesthetic attributes (mean scores only).

    Returns:
        DataFrame with:
            image_path
            score, quality, composition, color, dof, light, content
    """

    train_csv = os.path.join(dataset_dir, "annotation", "PARA-GiaaTrain.csv")
    test_csv  = os.path.join(dataset_dir, "annotation", "PARA-GiaaTest.csv")

    df_train = pd.read_csv(train_csv)
    df_test  = pd.read_csv(test_csv)

    df = pd.concat([df_train, df_test], ignore_index=True)

    base_img_path = os.path.abspath(os.path.join(dataset_dir, "imgs"))

    df["image_path"] = (
        base_img_path
        + os.sep
        + df["sessionId"]
        + os.sep
        + df["imageName"]
    )

    cols = [
        "image_path",
        "aestheticScore_mean",
        "qualityScore_mean",
        "compositionScore_mean",
        "colorScore_mean",
        "dofScore_mean",
        "lightScore_mean",
        "contentScore_mean",
    ]

    df = df[cols].rename(
        columns={
            "aestheticScore_mean": "score",
            "qualityScore_mean": "quality",
            "compositionScore_mean": "composition",
            "colorScore_mean": "color",
            "dofScore_mean": "dof",
            "lightScore_mean": "light",
            "contentScore_mean": "content",
        }
    )

    return df


# ---------------------------------------------------------
# IMAGE SCORE DISTRIBUTION (Phase 1 label distribution)
# ---------------------------------------------------------

def load_para_image_distribution(dataset_dir: str) -> pd.DataFrame:
    """
    Load full aesthetic score histogram for each image.

    Returns:
        image_path
        aestheticScore_1.0 ... aestheticScore_5.0
    """

    train_csv = os.path.join(dataset_dir, "annotation", "PARA-GiaaTrain.csv")
    test_csv  = os.path.join(dataset_dir, "annotation", "PARA-GiaaTest.csv")

    df_train = pd.read_csv(train_csv)
    df_test  = pd.read_csv(test_csv)

    df = pd.concat([df_train, df_test], ignore_index=True)

    base_img_path = os.path.abspath(os.path.join(dataset_dir, "imgs"))

    df["image_path"] = (
        base_img_path
        + os.sep
        + df["sessionId"]
        + os.sep
        + df["imageName"]
    )

    score_cols = [c for c in df.columns if c.startswith("aestheticScore_") and c.endswith(tuple(["1.0","1.5","2.0","2.5","3.0","3.5","4.0","4.5","5.0"]))]

    return df[["image_path"] + score_cols]


# ---------------------------------------------------------
# USER DEMOGRAPHIC ATTRIBUTES
# ---------------------------------------------------------

def load_para_user_attrs(dataset_dir: str) -> pd.DataFrame:
    """
    Load user demographic and personality attributes.
    """

    user_csv = os.path.join(dataset_dir, "annotation", "PARA-UserInfo.csv")
    df = pd.read_csv(user_csv)

    return df