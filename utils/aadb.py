import os
from dataclasses import dataclass
from typing import Dict, List

AESTHETIC_ATTRIBUTES = [
    "BalacingElements",
    "ColorHarmony",
    "Content",
    "DoF",
    "Light",
    "MotionBlur",
    "Object",
    "Repetition",
    "RuleOfThirds",
    "Symmetry",
    "VividColor",
]

@dataclass
class AADBItem:
    image_path: str
    score: float
    attributes: Dict[str, float]

def get_aadb_dataset(
    split: str,
    dataset_dir: str = "datasets/aadb",
) -> List[AADBItem]:
    """
    Loads the AADB dataset for a given split.

    Args:
        split: The data split to load ("train", "validation", "test", or "test_new").
        dataset_dir: The path to the AADB dataset directory.

    Returns:
        A list of AADBItem objects, each containing the image path, overall score,
        and a dictionary of aesthetic attribute scores.
    """
    if split not in ["train", "validation", "test", "test_new"]:
        raise ValueError(f"Invalid split: {split}")

    split_map = {
        "train": "Train",
        "validation": "Validation",
        "test": "Test",
        "test_new": "TestNew",
    }
    split_name = split_map[split]
    
    # Read all attributes and the overall score
    all_attributes = ["score"] + AESTHETIC_ATTRIBUTES
    attribute_data = {}

    for attribute in all_attributes:
        file_name = f"imgList{split_name}Regression_{attribute}.txt"
        file_path = os.path.join(dataset_dir, "imgListFiles_label", file_name)
        
        with open(file_path, "r") as f:
            lines = [line.strip().split() for line in f]
        
        if attribute == "score":
            image_names = [line[0] for line in lines]
        
        attribute_data[attribute] = {line[0]: float(line[1]) for line in lines}

    dataset = []
    for image_name in image_names:
        image_path = os.path.abspath(os.path.join(dataset_dir, "datasetImages_originalSize", image_name))
        score = attribute_data["score"][image_name]
        attributes = {attr: attribute_data[attr][image_name] for attr in AESTHETIC_ATTRIBUTES}
        
        dataset.append(AADBItem(image_path=image_path, score=score, attributes=attributes))

    return dataset

if __name__ == "__main__":
    # Example usage:
    for split in ["train", "validation", "test"]:
        print(f"--- {split} ---")
        dataset = get_aadb_dataset(split)
        print(f"Loaded {len(dataset)} items.")

        # Print the first 5 entries
        for i in range(5):
            print(dataset[i])