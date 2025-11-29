import os
import random
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
from tqdm import tqdm


@dataclass
class LAPISItem:
    image_path: str
    image_id: int
    score: float


@dataclass
class PersonalizedLAPIS:
    support_small: List["PersonalizedLAPISItem"]
    support_large: List["PersonalizedLAPISItem"]
    test: List["PersonalizedLAPISItem"]


@dataclass
class PersonalizedLAPISItem:
    image_path: str
    image_id: int
    user_id: int
    score: float


def get_lapis_dataset(
    split: str | None,
    dataset_dir: str = "datasets/LAPIS",
) -> List[LAPISItem]:
    """
    Loads the LAPIS dataset for a given split.

    Args:
        split: The data split to load ("train", "val", or "test"). If None,
               loads all splits.
        dataset_dir: The path to the LAPIS dataset directory.

    Returns:
        A list of LAPISItem objects, each containing the image path, image ID,
        and the mean aesthetic score.
    """
    if split is None:
        train_data = _load_split_data("train", dataset_dir)
        val_data = _load_split_data("val", dataset_dir)
        test_data = _load_split_data("test", dataset_dir)
        return train_data + val_data + test_data
    elif split in ["train", "val", "test"]:
        return _load_split_data(split, dataset_dir)
    else:
        raise ValueError(
            f"Invalid split: {split}. LAPIS dataset only supports 'train', 'val', 'test', or None."
        )


def _load_split_data(
    split: str,
    dataset_dir: str,
) -> List[LAPISItem]:
    split_map = {
        "train": "Trainsplit",
        "val": "Valsplit",
        "test": "Testsplit",
    }
    split_name = split_map[split]

    csv_file = f"LAPIS_GIAA_{split_name}.csv"
    csv_path = os.path.join(dataset_dir, "annotation", csv_file)

    df = pd.read_csv(csv_path)

    dataset = []
    for _, row in df.iterrows():
        image_id = row["image_id"]
        image_filename = row["image_filename"]
        image_path = os.path.abspath(
            os.path.join(dataset_dir, "images", image_filename)
        )
        score = row["mean_response"]

        dataset.append(
            LAPISItem(
                image_path=image_path,
                image_id=image_id,
                score=score,
            )
        )

    return dataset


def get_personalized_lapis_dataset(
    seed: int,
    dataset_dir: str = "datasets/LAPIS",
) -> Dict[int, PersonalizedLAPIS]:
    """
    Loads the LAPIS dataset grouped by user for personalization tasks.

    Three groups are returned for each user:
    1. support_small: 10 images
    2. support_large: 100 images
    3. test: 50 images

    Users with fewer than 160 images are skipped.

    Args:
        seed: The random seed for shuffling and selection.
        dataset_dir: The path to the LAPIS dataset directory.

    Returns:
        A dictionary where keys are user IDs and values are PersonalizedLAPIS objects,
        each containing support and test sets for that user.
    """
    df = _load_personalized_data(dataset_dir)

    # Get user counts and filter for users with enough images
    user_counts = df["participant_id"].value_counts()
    num_support_small = 10
    num_support_large = 100
    num_test = 50
    total_required = num_support_small + num_support_large + num_test
    valid_users = user_counts[user_counts >= total_required].index.tolist()

    # We don't sample users, just take all valid ones and shuffle
    rng = random.Random(seed)
    rng.shuffle(valid_users)

    # Filter the main DataFrame to only include selected users
    user_df = df[df["participant_id"].isin(valid_users)]

    personalized_dataset = {}

    # Group by user and create the splits
    grouped = user_df.groupby("participant_id")

    for user_id, user_group_df in tqdm(grouped, desc="Processing users"):
        # Shuffle and split the user's DataFrame
        shuffled_group = user_group_df.sample(frac=1, random_state=seed)

        support_small_df = shuffled_group.iloc[:num_support_small]
        support_large_df = shuffled_group.iloc[
            num_support_small : num_support_small + num_support_large
        ]
        test_df = shuffled_group.iloc[
            num_support_small
            + num_support_large : num_support_small
            + num_support_large
            + num_test
        ]

        # Convert DataFrame rows to PersonalizedLAPISItem objects for the splits
        def df_to_items(split_df: pd.DataFrame) -> List[PersonalizedLAPISItem]:
            items = []
            for _, row in split_df.iterrows():
                items.append(
                    PersonalizedLAPISItem(
                        image_path=row["image_path"],
                        image_id=row["image_id"],
                        user_id=row["participant_id"],
                        score=row["rating"],
                    )
                )
            return items

        personalized_dataset[user_id] = PersonalizedLAPIS(
            support_small=df_to_items(support_small_df),
            support_large=df_to_items(support_large_df),
            test=df_to_items(test_df),
        )

    return personalized_dataset


def _load_personalized_data(
    dataset_dir: str,
) -> pd.DataFrame:
    csv_file = "LAPIS_PIAA.csv"
    csv_path = os.path.join(dataset_dir, "annotation", csv_file)

    df = pd.read_csv(csv_path)

    df.dropna(subset=["rating"], inplace=True)

    base_img_path = os.path.abspath(os.path.join(dataset_dir, "images"))
    df["image_path"] = base_img_path + os.sep + df["image_filename"]

    return df


if __name__ == "__main__":
    # Example usage for get_lapis_dataset:
    for split_option in ["train", "val", "test", None]:
        print(f"--- LAPIS GIAA ({split_option}) ---")
        try:
            dataset = get_lapis_dataset(split_option)
            print(f"Loaded {len(dataset)} items.")
            if dataset:
                print("First 5 entries:")
                for i in range(min(5, len(dataset))):
                    print(dataset[i])
        except FileNotFoundError:
            print(
                f"Dataset files not found for split '{split_option}'. Please check the path."
            )
        print()

    # Example usage for get_personalized_lapis_dataset
    print("--- Personalized LAPIS (PIAA) ---")
    try:
        personalized_data = get_personalized_lapis_dataset(seed=42)
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

