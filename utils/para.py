import os
from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
import random
from collections import defaultdict
from tqdm import tqdm

AESTHETIC_ATTRIBUTES = [
    "score",
    "quality",
    "composition",
    "color",
    "dof",
    "light",
    "content",
]

SCORE_ATTRIBUTES = {
    "aestheticScore_mean": "score",
    "qualityScore_mean": "quality",
    "compositionScore_mean": "composition",
    "colorScore_mean": "color",
    "dofScore_mean": "dof",
    "lightScore_mean": "light",
    "contentScore_mean": "content",
}


@dataclass
class PARAItem:
    image_path: str
    session_id: str
    score: float
    attributes: Dict[str, float]


@dataclass
class PersonalizedPARA:
    support_small: List["PersonalizedPARAItem"]
    support_large: List["PersonalizedPARAItem"]
    test: List["PersonalizedPARAItem"]


PERSONALIZED_SCORE_ATTRIBUTES = {
    "aestheticScore": "score",
    "qualityScore": "quality",
    "compositionScore": "composition",
    "colorScore": "color",
    "dofScore": "dof",
    "lightScore": "light",
    "contentScore": "content",
}


@dataclass
class PersonalizedPARAItem:
    image_path: str
    user_id: str
    score: float
    attributes: Dict[str, float]


def get_para_dataset(
    split: str | None,
    dataset_dir: str = "datasets/PARA",
) -> List[PARAItem]:
    """
    Loads the PARA dataset for a given split.

    Args:
        split: The data split to load ("train" or "test").
        dataset_dir: The path to the PARA dataset directory.

    Returns:
        A list of PARAItem objects, each containing the image path, overall score,
        and a dictionary of aesthetic attribute scores.
    """
    if split is None:
        train_data = _load_split_data("train", dataset_dir)
        test_data = _load_split_data("test", dataset_dir)
        return train_data + test_data
    elif split in ["train", "test"]:
        return _load_split_data(split, dataset_dir)
    else:
        raise ValueError(
            f"Invalid split: {split}. PARA dataset only supports 'train', 'test', or None."
        )


def _load_split_data(
    split: str,
    dataset_dir: str,
) -> List[PARAItem]:
    split_map = {
        "train": "Train",
        "test": "Test",
    }
    split_name = split_map[split]

    csv_file = f"PARA-Giaa{split_name}.csv"
    csv_path = os.path.join(dataset_dir, "annotation", csv_file)

    df = pd.read_csv(csv_path)

    dataset = []
    for _, row in df.iterrows():
        session_id = row["sessionId"]
        image_name = row["imageName"]
        image_path = os.path.abspath(
            os.path.join(dataset_dir, "imgs", session_id, image_name)
        )

        score = row["aestheticScore_mean"]

        attributes = {
            new_name: row[original_name]
            for original_name, new_name in SCORE_ATTRIBUTES.items()
        }

        dataset.append(
            PARAItem(
                image_path=image_path,
                session_id=session_id,
                score=score,
                attributes=attributes,
            )
        )

    return dataset


def get_personalized_para_dataset(
    seed: int,
    dataset_dir: str = "datasets/PARA",
) -> Dict[str, PersonalizedPARA]:
    """
    Loads the PARA dataset grouped by user for personalization tasks.
    Randomly extracts 200 user IDs and for each user, returns groups of images
    with their annotations (mean scores). This function is independent of
    get_para_dataset and uses PARA-Images.csv as its source.

    Three groups are returned for each user:
    1. support_small: 10 images
    2. support_large: 100 images
    3. test: 50 images

    Users with fewer than 160 images are skipped.

    Args:
        seed: The random seed for shuffling and selection.
        dataset_dir: The path to the PARA dataset directory.

    Returns:
        A dictionary where keys are user IDs and values are PersonalizedPARA objects,
        each containing support and test sets for that user.
    """
    df = _load_personalized_data(dataset_dir)

    # Get user counts and filter for users with enough images
    user_counts = df["userId"].value_counts()
    num_support_small = 10
    num_support_large = 100
    num_test = 50
    total_required = num_support_small + num_support_large + num_test
    valid_users = user_counts[user_counts >= total_required].index.tolist()

    # Sample 200 users
    rng = random.Random(seed)
    if len(valid_users) >= 200:
        selected_user_ids = rng.sample(valid_users, 200)
    else:
        selected_user_ids = valid_users
        rng.shuffle(selected_user_ids)

    # Filter the main DataFrame to only include selected users
    user_df = df[df["userId"].isin(selected_user_ids)]

    personalized_dataset = {}

    # Group by user and create the splits
    grouped = user_df.groupby("userId")

    for user_id, user_group_df in tqdm(grouped, desc="Processing users"):
        # Shuffle and split the user's DataFrame
        shuffled_group = user_group_df.sample(frac=1, random_state=seed)

        support_small_df = shuffled_group.iloc[:num_support_small]
        support_large_df = shuffled_group.iloc[
            num_support_small : num_support_small + num_support_large
        ]
        test_df = shuffled_group.iloc[
            num_support_small + num_support_large : num_support_small
            + num_support_large
            + num_test
        ]

        # Convert DataFrame rows to PersonalizedPARAItem objects for the splits
        def df_to_items(split_df: pd.DataFrame) -> List[PersonalizedPARAItem]:
            items = []
            for _, row in split_df.iterrows():
                attributes = {
                    new_name: row[original_name]
                    for original_name, new_name in PERSONALIZED_SCORE_ATTRIBUTES.items()
                }
                items.append(
                    PersonalizedPARAItem(
                        image_path=row["image_path"],
                        user_id=row["userId"],
                        score=row["aestheticScore"],
                        attributes=attributes,
                    )
                )
            return items

        personalized_dataset[user_id] = PersonalizedPARA(
            support_small=df_to_items(support_small_df),
            support_large=df_to_items(support_large_df),
            test=df_to_items(test_df),
        )

    return personalized_dataset


def _load_personalized_data(
    dataset_dir: str,
) -> pd.DataFrame:
    csv_file = "PARA-Images.csv"
    csv_path = os.path.join(dataset_dir, "annotation", csv_file)

    df = pd.read_csv(csv_path)

    # Skip rows with missing scores
    score_cols = list(PERSONALIZED_SCORE_ATTRIBUTES.keys())
    df.dropna(subset=score_cols, inplace=True)

    # Construct image_path using vectorized operations
    base_img_path = os.path.abspath(os.path.join(dataset_dir, "imgs"))
    df["image_path"] = (
        base_img_path + os.sep + df["sessionId"] + os.sep + df["imageName"]
    )

    return df


if __name__ == "__main__":
    # Example usage:
    for split_option in ["train", "test", None]:
        print(f"--- {split_option} ---")
        try:
            dataset = get_para_dataset(split_option)
            print(f"Loaded {len(dataset)} items.")

            # Print the first 5 entries
            for i in range(min(5, len(dataset))):
                print(dataset[i])
        except FileNotFoundError:
            print(
                f"Dataset files not found for split '{split_option}'. Please check the path."
            )

    # Example for personalized dataset
    print("\n--- Personalized Dataset ---")
    try:
        personalized_data = get_personalized_para_dataset(seed=42)
        print(f"Loaded personalized data for {len(personalized_data)} users.")
        if personalized_data:
            first_user = list(personalized_data.keys())[0]
            print(f"Data for first user ({first_user}):")
            print(
                f"  Support (small): {len(personalized_data[first_user].support_small)} items"
            )
            print(
                f"  Support (large): {len(personalized_data[first_user].support_large)} items"
            )
            print(f"  Test: {len(personalized_data[first_user].test)} items")
    except FileNotFoundError:
        print("Dataset files not found. Please check the path.")
