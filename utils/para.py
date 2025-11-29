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
    1. support_small: 5 images
    2. support_large: 20 images
    3. test: 10 images

    Users with fewer than 35 images are skipped.

    Args:
        seed: The random seed for shuffling and selection.
        dataset_dir: The path to the PARA dataset directory.

    Returns:
        A dictionary where keys are user IDs and values are PersonalizedPARA objects,
        each containing support and test sets for that user.
    """
    all_data = _load_personalized_data(dataset_dir)

    user_data = defaultdict(list)
    for item in all_data:
        user_data[item.user_id].append(item)

    user_ids = list(user_data.keys())
    rng = random.Random(seed)
    rng.shuffle(user_ids)

    personalized_dataset = {}
    num_support_small = 10
    num_support_large = 100
    num_test = 50
    total_required = num_support_small + num_support_large + num_test

    selected_users = 0
    for user_id in user_ids:
        if selected_users >= 200:
            break

        images = user_data[user_id]
        if len(images) < total_required:
            continue

        rng.shuffle(images)

        support_small_set = images[:num_support_small]
        support_large_set = images[
            num_support_small : num_support_small + num_support_large
        ]
        test_set = images[
            num_support_small + num_support_large : num_support_small
            + num_support_large
            + num_test
        ]

        personalized_dataset[user_id] = PersonalizedPARA(
            support_small=support_small_set,
            support_large=support_large_set,
            test=test_set,
        )
        selected_users += 1

    return personalized_dataset


def _load_personalized_data(
    dataset_dir: str,
) -> List[PersonalizedPARAItem]:
    csv_file = "PARA-Images.csv"
    csv_path = os.path.join(dataset_dir, "annotation", csv_file)

    df = pd.read_csv(csv_path)

    dataset = []
    score_cols = list(PERSONALIZED_SCORE_ATTRIBUTES.keys())

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        # Skip rows with missing scores
        if row[score_cols].isnull().any():
            continue

        user_id = row["userId"]
        session_id = row["sessionId"]
        image_name = row["imageName"]
        image_path = os.path.abspath(
            os.path.join(dataset_dir, "imgs", session_id, image_name)
        )

        score = row["aestheticScore"]

        attributes = {
            new_name: row[original_name]
            for original_name, new_name in PERSONALIZED_SCORE_ATTRIBUTES.items()
        }

        dataset.append(
            PersonalizedPARAItem(
                image_path=image_path,
                user_id=user_id,
                score=score,
                attributes=attributes,
            )
        )

    return dataset


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
