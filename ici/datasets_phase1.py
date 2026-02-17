# ici/datasets_phase1.py
import os
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


@dataclass
class Phase1Meta:
    attr_cols: List[str]
    dis_cols: List[str]


def _load_json_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        xs = json.load(f)
    return [str(x) for x in xs]


def _normalize_distribution(df: pd.DataFrame, dis_cols: List[str]) -> pd.DataFrame:
    if not dis_cols:
        raise ValueError("dis_cols is empty (no distribution columns found).")
    dis = df[dis_cols].to_numpy(dtype=np.float32)
    s = dis.sum(axis=1, keepdims=True) + 1e-12
    dis = dis / s
    df.loc[:, dis_cols] = dis
    return df


class PARAPhase1Dataset(Dataset):
    """
    Phase1 dataset for PARA:
      - x: image (RGB)
      - y_attr: vector of aesthetic attribute means (float)
      - y_dis: aesthetic score distribution (float, sums to 1)

    The dataset reads:
      - train_images.json (list of image_path to use)
      - image_attrs.csv: image_path + attr mean columns
      - image_dis.csv: image_path + distribution columns (aestheticScore_*)
    """

    def __init__(
        self,
        split_dir: str,           # e.g. "ici_splits/para"
        split: str,               # "train" or "val"
        val_ratio: float = 0.1,
        seed: int = 42,
        transform=None,           # torchvision transform
        attr_cols: Optional[List[str]] = None,
        dis_cols: Optional[List[str]] = None,
    ):
        self.split_dir = split_dir
        self.split = split
        self.val_ratio = float(val_ratio)
        self.seed = int(seed)
        self.transform = transform

        # ---- load image list ----
        train_images_path = os.path.join(split_dir, "train_images.json")
        image_paths = _load_json_list(train_images_path)

        # ---- load labels ----
        df_attr = pd.read_csv(os.path.join(split_dir, "image_attrs.csv"))
        df_dis = pd.read_csv(os.path.join(split_dir, "image_dis.csv"))

        # ---- infer columns if not provided ----
        if attr_cols is None:
            # expect these names from your load_para_image_attrs()
            candidates = ["score", "quality", "composition", "color", "dof", "light", "content"]
            attr_cols = [c for c in candidates if c in df_attr.columns]
        if dis_cols is None:
            # PARA uses aestheticScore_1.0 ... 5.0 (includes 1.5 etc)
            dis_cols = [c for c in df_dis.columns if c.startswith("aestheticScore_")]
            dis_cols = sorted(dis_cols, key=lambda x: float(x.split("_")[1]))

        if not attr_cols:
            raise RuntimeError(f"No attr columns found in image_attrs.csv. Columns={df_attr.columns.tolist()}")
        if not dis_cols:
            raise RuntimeError(f"No distribution columns found in image_dis.csv. Columns={df_dis.columns.tolist()}")

        self.meta = Phase1Meta(attr_cols=attr_cols, dis_cols=dis_cols)

        # ---- join on image_path ----
        df = pd.merge(df_attr[["image_path"] + attr_cols], df_dis[["image_path"] + dis_cols],
                      on="image_path", how="inner")

        # ---- filter to split images ----
        df = df[df["image_path"].astype(str).isin(set(image_paths))].copy()

        # ---- normalize distribution ----
        df = _normalize_distribution(df, dis_cols)

        # ---- train/val split inside train_images ----
        rng = np.random.RandomState(self.seed)
        idx = np.arange(len(df))
        rng.shuffle(idx)

        n_val = int(round(self.val_ratio * len(df)))
        val_idx = set(idx[:n_val].tolist())
        if split == "val":
            df = df.iloc[list(val_idx)].copy()
        elif split == "train":
            df = df.iloc[[i for i in range(len(df)) if i not in val_idx]].copy()
        else:
            raise ValueError("split must be 'train' or 'val'")

        df = df.reset_index(drop=True)
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[i]
        p = row["image_path"]
        img = Image.open(p).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        y_attr = torch.tensor(row[self.meta.attr_cols].to_numpy(dtype=np.float32))
        y_dis = torch.tensor(row[self.meta.dis_cols].to_numpy(dtype=np.float32))
        return img, y_attr, y_dis