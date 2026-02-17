#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AADB / PARA データセット中の各 attribute のヒストグラムを描画するスクリプト。

前提:
  - utils/aadb.py に get_aadb_dataset, AESTHETIC_ATTRIBUTES, AADBItem が定義されている
  - utils/para.py に get_para_dataset, AESTHETIC_ATTRIBUTES, PARAItem が定義されている

実行例:
  # AADB だけ
  python plot_attr_histograms.py --dataset aadb --out_dir viz_hists

  # PARA だけ
  python plot_attr_histograms.py --dataset para --out_dir viz_hists

  # 両方
  python plot_attr_histograms.py --dataset both --out_dir viz_hists
"""

import os
import argparse
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from utils.aadb import get_aadb_dataset, AESTHETIC_ATTRIBUTES as AADB_ATTRS
from utils.para import get_para_dataset, AESTHETIC_ATTRIBUTES as PARA_ATTRS


# ------------- 共通ユーティリティ -------------

def _ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _collect_aadb_values(dataset_dir: str) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """
    AADB: 属性名リストと、属性→値配列 の辞書を返す。
    - overall score は "score" として扱う。
    - attributes dict に含まれる属性は AADB_ATTRS を使う。
    """
    items = get_aadb_dataset("train", dataset_dir=dataset_dir) + \
            get_aadb_dataset("validation", dataset_dir=dataset_dir) + \
            get_aadb_dataset("test", dataset_dir=dataset_dir)

    # overall score + 属性（AADB_ATTRS）
    attr_names: List[str] =  list(AADB_ATTRS)

    values: Dict[str, List[float]] = {a: [] for a in attr_names}

    for it in items:
        # 各 attribute
        for a in AADB_ATTRS:
            if a in it.attributes:
                values[a].append(float(it.attributes[a]))

    arr_values = {k: np.array(v, dtype=np.float32) for k, v in values.items()}
    return attr_names, arr_values


def _collect_para_values(dataset_dir: str) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """
    PARA: 属性名リストと、属性→値配列 の辞書を返す。
    - PARA_ATTRS は "score" を含むので、重複しないように注意する。
    - overall score は "score" として it.score を使う。
    """
    items = get_para_dataset("train", dataset_dir=dataset_dir) + \
            get_para_dataset("test", dataset_dir=dataset_dir)

    # attr_names は PARA_ATTRS をそのまま使う (score, quality, composition, ...)
    attr_names: List[str] = list(PARA_ATTRS)

    values: Dict[str, List[float]] = {a: [] for a in attr_names}

    for it in items:
        # score は it.score を優先
        if "score" in values:
            values["score"].append(float(it.score))
        # その他の attribute は attributes dict から
        for a in attr_names:
            if a == "score":
                continue
            if a in it.attributes:
                values[a].append(float(it.attributes[a]))

    arr_values = {k: np.array(v, dtype=np.float32) for k, v in values.items()}
    return attr_names, arr_values


def _plot_histograms(
    dataset_name: str,
    attr_names: List[str],
    values: Dict[str, np.ndarray],
    out_dir: str,
    bins: int = 30,
):
    """
    dataset_name: "aadb" / "para"
    attr_names  : プロット対象の順序つき属性名リスト
    values      : attr -> np.ndarray
    """
    _ensure_dir(out_dir)

    n_attr = len(attr_names)
    n_cols = 3
    n_rows = (n_attr + n_cols - 1) // n_cols

    plt.close("all")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)

    for idx, attr in enumerate(attr_names):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r, c]

        vals = values.get(attr, None)
        if vals is None or len(vals) == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            ax.set_title(attr)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # ヒストグラム
        ax.hist(vals, bins=bins)
        ax.set_title("Overall Score" if attr == 'score' else attr, fontsize=18)
        ax.grid(True, linestyle="--", alpha=0.3)

    # 余ったサブプロットを消す
    for idx in range(n_attr, n_rows * n_cols):
        r = idx // n_cols
        c = idx % n_cols
        axes[r, c].axis("off")

    # fig.suptitle(f"{dataset_name.upper()} attribute distributions", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(out_dir, f"{dataset_name}_attributes_hist.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out_path}")


# ------------- main -------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        default="both",
        choices=["aadb", "para", "both"],
        help="Which dataset(s) to plot: aadb / para / both.",
    )
    ap.add_argument(
        "--aadb_dir",
        default="datasets/aadb",
        help="Path to AADB dataset root (where imgListFiles_label, datasetImages_originalSize exist).",
    )
    ap.add_argument(
        "--para_dir",
        default="datasets/PARA",
        help="Path to PARA dataset root (where annotation, imgs exist).",
    )
    ap.add_argument(
        "--out_dir",
        default="viz_hists",
        help="Root output directory for histogram images.",
    )
    ap.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Number of histogram bins.",
    )
    args = ap.parse_args()

    if args.dataset in ("aadb", "both"):
        print("[info] Collecting AADB attributes...")
        aadb_attrs, aadb_vals = _collect_aadb_values(args.aadb_dir)
        out_dir_aadb = os.path.join(args.out_dir, "aadb")
        _plot_histograms("aadb", aadb_attrs, aadb_vals, out_dir_aadb, bins=args.bins)

    if args.dataset in ("para", "both"):
        print("[info] Collecting PARA attributes...")
        para_attrs, para_vals = _collect_para_values(args.para_dir)
        out_dir_para = os.path.join(args.out_dir, "para")
        _plot_histograms("para", para_attrs, para_vals, out_dir_para, bins=args.bins)


if __name__ == "__main__":
    main()