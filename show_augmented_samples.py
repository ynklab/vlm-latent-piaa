#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AADB / PARA の画像からランダムに1枚サンプルし、
  - orig (RGB)
  - gray (グレースケール→3ch)
  - tps (ThinPlateSpline augmentation)
の3種類を並べて比較するスクリプト。

前提:
  - utils/aadb.py に get_aadb_dataset, AESTHETIC_ATTRIBUTES がある
  - utils/para.py に get_para_dataset, AESTHETIC_ATTRIBUTES がある
  - albumentations, opencv-python(-headless) がインストール済み

例:

  # AADB から1枚サンプル
  python show_augmented_samples.py \
    --dataset aadb \
    --dataset_dir datasets/aadb \
    --out_path viz_aug_samples/aadb_sample.png

  # PARA から1枚サンプル
  python show_augmented_samples.py \
    --dataset para \
    --dataset_dir datasets/PARA \
    --out_path viz_aug_samples/para_sample.png
"""

import os
import argparse
import numpy as np
from typing import List

from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A

from utils.aadb import get_aadb_dataset
from utils.para import get_para_dataset


# ----- Albumentations transform -----

def build_tps_transform():
    # 必要に応じてパラメータは調整可能
    return A.Compose([
        A.ThinPlateSpline(p=1.0)
    ])


def apply_image_mode(img: Image.Image, mode: str, tps_transform: A.BasicTransform) -> Image.Image:
    """
    img : PIL.Image (RGB前提)
    mode: "orig", "gray", "tps"
    """
    if mode == "orig":
        return img

    elif mode == "gray":
        # グレースケール → 3ch に戻す（Visionモデルが3chを期待するため）
        g = img.convert("L")
        return g.convert("RGB")

    elif mode == "tps":
        img_np = np.array(img)
        aug = tps_transform(image=img_np)["image"]
        return Image.fromarray(aug)

    else:
        raise ValueError(f"Unknown image mode: {mode}")


# ----- データセットからランダムに1枚取ってくる -----

def sample_image_path(dataset: str, dataset_dir: str, split: str, seed: int) -> str:
    """
    dataset: "aadb" or "para"
    split  : "train"/"validation"/"test" (AADB) or "train"/"test" (PARA)
    """
    if dataset == "aadb":
        items = get_aadb_dataset(split, dataset_dir=dataset_dir)
    elif dataset == "para":
        items = get_para_dataset(split, dataset_dir=dataset_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if not items:
        raise RuntimeError(f"No items found for dataset={dataset}, split={split}, dir={dataset_dir}")

    rng = np.random.RandomState(seed)
    idx = int(rng.randint(0, len(items)))
    image_path = items[idx].image_path

    print(f"[info] sampled index={idx}, path={image_path}")
    return image_path


# ----- 可視化 -----

def show_and_save_triplet(image_path: str, dataset: str, out_path: str, seed: int):
    img = Image.open(image_path).convert("RGB")
    tps_transform = build_tps_transform()

    modes: List[str] = ["orig", "gray", "tps"]
    titles = {
        "orig": "Original (orig)",
        "gray": "Grayscale (gray)",
        "tps":  "ThinPlateSpline (tps)",
    }

    imgs = [apply_image_mode(img, m, tps_transform) for m in modes]

    # プロット
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, m, im in zip(axes, modes, imgs):
        ax.imshow(im)
        ax.set_title(titles[m], fontsize=27)
        ax.axis("off")

    # fig.suptitle(f"{dataset.upper()} sample\n{os.path.basename(image_path)}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        print(f"[save] {out_path}")

    plt.show()
    plt.close(fig)


# ----- main -----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="aadb", choices=["aadb", "para"],
                    help="Which dataset to sample from (aadb or para).")
    ap.add_argument("--dataset_dir", default=None,
                    help="Root dir of the dataset. If None, use datasets/aadb or datasets/PARA.")
    ap.add_argument("--split", default=None,
                    help="Split to sample from. "
                         "AADB: train/validation/test, PARA: train/test. "
                         "If None, defaults to 'train' for both.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed for image sampling.")
    ap.add_argument("--out_path", default=None,
                    help="Path to save the comparison image (PNG). If None, do not save.")
    args = ap.parse_args()

    if args.dataset_dir is None:
        args.dataset_dir = "datasets/aadb" if args.dataset == "aadb" else "datasets/PARA"

    if args.split is None:
        args.split = "train"

    print(f"[info] dataset={args.dataset}, dir={args.dataset_dir}, split={args.split}, seed={args.seed}")

    image_path = sample_image_path(args.dataset, args.dataset_dir, args.split, args.seed)
    show_and_save_triplet(image_path, args.dataset, args.out_path, args.seed)


if __name__ == "__main__":
    main()