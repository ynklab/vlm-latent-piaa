import os
from dataclasses import dataclass
from typing import Dict, List
import pandas as pd

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
            PARAItem(image_path=image_path, score=score, attributes=attributes)
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
