#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Export representative images for qualitative comparison across PIAA methods,
and create thumbnail-style grids (contact sheets) with matplotlib.

- ランダムにユーザーをサンプル
- 各ユーザーについて:
  - GT (user_score) に基づく gt_low / gt_mid / gt_high の代表画像を n_imgs 枚
  - 各 method について:
      * piaa_pred に基づく pred_low / pred_mid / pred_high の代表画像を n_imgs 枚
      * err = piaa_pred - user_score に基づく err_pos / err_zero / err_neg の代表画像を n_imgs 枚
  - さらに low / mid / high のサムネイルを縦に stack した1枚の画像も作成:
      * user_<id>/gt_stacked.png
      * user_<id>/method_<m>/pred_stacked.png
"""

import os
import re
import argparse
import shutil
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


# ---------- Utils ----------

def sanitize(s: str) -> str:
    if s is None:
        return "unknown"
    if not isinstance(s, str):
        s = str(s)
    return re.sub(r"[^0-9A-Za-z._\\-]+", "_", s)


def load_piaa_from_dir(input_dir: str) -> pd.DataFrame:
    required = {
        "user_id",
        "image_path",
        "model_id",
        "support_set",
        "method",
        "giaa",
        "piaa_pred",
        "user_score",
    }
    dfs: List[pd.DataFrame] = []

    if not os.path.isdir(input_dir):
        raise RuntimeError(f"input_dir is not a directory: {input_dir}")

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(".csv")]
    print(f"[info] found {len(files)} CSV files in {input_dir}")

    for name in files:
        path = os.path.join(input_dir, name)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"[warn] failed to read {path}: {e}, skip")
            continue

        if not required.issubset(df.columns):
            print(f"[info] skip {path} (missing PIAA columns)")
            continue

        print(f"[info] loaded PIAA CSV: {path} (rows={len(df)})")
        dfs.append(df)

    if not dfs:
        raise RuntimeError(f"No valid PIAA CSVs found in directory: {input_dir}")

    return pd.concat(dfs, ignore_index=True)


def sample_group(df: pd.DataFrame, col: str, n: int, group: str) -> pd.DataFrame:
    """
    df: ユーザー内の DataFrame (image_path, user_score, piaa_pred, ...)
    col: ソート基準 ('user_score', 'piaa_pred', 'err', 'abs_err')
    group: 'high' / 'mid' / 'low' など

    戻り値: subset DataFrame
    """
    if df.empty or n <= 0:
        return df.iloc[0:0]

    df_sorted = df.sort_values(by=col, ascending=True).reset_index(drop=True)
    m = len(df_sorted)
    if m <= n:
        return df_sorted

    if group == "low":
        return df_sorted.head(n)
    elif group == "high":
        return df_sorted.tail(n)
    elif group == "mid":
        center = m // 2
        half = n // 2
        start = max(0, center - half)
        end = min(m, start + n)
        return df_sorted.iloc[start:end]
    else:
        center = m // 2
        half = n // 2
        start = max(0, center - half)
        end = min(m, start + n)
        return df_sorted.iloc[start:end]


def copy_images(sub_df: pd.DataFrame, dst_dir: str, prefix: str):
    """
    sub_df: image_path, user_score, piaa_pred, err などの列を持つ DataFrame
    dst_dir: 出力ディレクトリ
    prefix: ファイル名に付けるprefix (e.g. 'gt_high', 'pred_low', 'err_pos')
    """
    os.makedirs(dst_dir, exist_ok=True)
    for row_idx, row in sub_df.iterrows():
        src = row["image_path"]
        if not isinstance(src, str) or not os.path.isfile(src):
            continue
        base = os.path.basename(src)
        us = row.get("user_score", np.nan)
        pp = row.get("piaa_pred", np.nan)
        err = row.get("err", np.nan)
        fname = f"{prefix}_idx{row_idx}_gt{us:.2f}_pred{pp:.2f}_err{err:.2f}_{base}"
        dst = os.path.join(dst_dir, sanitize(fname))
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            print(f"[warn] failed to copy {src} -> {dst}: {e}")


def create_thumbnail_grid(sub_df: pd.DataFrame, out_path: str, title: str, max_cols: int = 5):
    """
    sub_df: image_path, user_score, piaa_pred, err を含む DataFrame
    out_path: 保存先PNG
    title: グリッド全体のタイトル
    max_cols: グリッドの最大列数（画像数が多いときに折り返す）
    """
    if sub_df.empty:
        return

    paths = sub_df["image_path"].tolist()
    gt_scores = sub_df["user_score"].tolist()
    preds = sub_df.get("piaa_pred", pd.Series([np.nan] * len(sub_df))).tolist()
    errs = sub_df.get("err", pd.Series([np.nan] * len(sub_df))).tolist()

    n = len(paths)
    cols = min(max_cols, n)
    rows = (n + cols - 1) // cols

    plt.close("all")
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    for idx, (p, gt, pr, er) in enumerate(zip(paths, gt_scores, preds, errs)):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            ax.axis("off")
            continue
        ax.imshow(img)
        ax.axis("off")
        t = f"gt={gt:.2f}"
        if not np.isnan(pr):
            t += f"\npred={pr:.2f}"
        if not np.isnan(er):
            t += f"\nerr={er:.2f}"
        ax.set_title(t, fontsize=8)

    for idx in range(n, rows * cols):
        r = idx // cols
        c = idx % cols
        axes[r, c].axis("off")

    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[thumb] saved {out_path}")



def create_stacked_thumbnail(groups, out_path, title, max_cols=5):
    total_images = sum(len(df) for df, _ in groups)
    if total_images == 0:
        return

    group_dfs = [df for (df, _) in groups]
    # group_labels = [lbl for (_, lbl) in groups]
    group_labels = ["High Score", "Low Score"]

    max_n = max(len(df) for df in group_dfs)
    if max_n == 0:
        return
    cols = min(max_cols, max_n)
    rows = len(group_dfs)

    plt.close("all")
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    # 行帯の背景色（図全体に敷く）
    row_colors = ["#f0f0f0", "#e0f7fa", "#fce4ec", "#f3e5f5"]

    # --- まず各行の帯を figure 座標で敷く（imshowに隠されない） ---
    # これをやるために、一旦 tight_layout は使わず subplots_adjust で余白を確保する
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.06, wspace=0.08, hspace=0.12)

    # 行ごとの y 範囲を axes の position から取得して帯を敷く
    for r in range(rows):
        pos_left = axes[r, 0].get_position()
        pos_right = axes[r, cols - 1].get_position()
        x0 = pos_left.x0
        x1 = pos_right.x1
        y0 = pos_left.y0
        y1 = pos_left.y1
        rect = Rectangle(
            (x0, y0),
            width=(x1 - x0),
            height=(y1 - y0),
            transform=fig.transFigure,
            facecolor=row_colors[r % len(row_colors)],
            edgecolor="none",
            zorder=0,  # 背面
        )
        fig.add_artist(rect)

    # --- 画像を描画 + 行ラベルを Axes 内に text で描く ---
    for r, df_group in enumerate(group_dfs):
        paths = df_group["image_path"].tolist()
        gt_scores = df_group["user_score"].tolist()
        preds = df_group.get("piaa_pred", pd.Series([np.nan] * len(df_group))).tolist()
        errs = df_group.get("err", pd.Series([np.nan] * len(df_group))).tolist()

        for c in range(cols):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)

        for idx, (p, gt, pr, er) in enumerate(zip(paths, gt_scores, preds, errs)):
            if idx >= cols:
                break
            ax = axes[r, idx]
            try:
                img = Image.open(p).convert("RGB")
                ax.imshow(img)
            except Exception:
                pass
            ax.set_axis_off()

        # 行ラベル（Axesの左上に重ね書き。bboxで読みやすく）
        ax0 = axes[r, 0]
        ax0.text(
            0.01, 0.02,
            group_labels[r],
            transform=ax0.transAxes,
            fontsize=25,
            fontweight="bold",
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"),
            zorder=5,
        )

    fig.suptitle(title, fontsize=30)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[thumb-stacked] saved {out_path}")

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing PIAA result CSVs.",
    )
    ap.add_argument(
        "--out_dir",
        required=True,
        help="Directory to export representative images and thumbnails.",
    )
    ap.add_argument(
        "--n_users",
        type=int,
        default=5,
        help="Number of users to sample at random.",
    )
    ap.add_argument(
        "--n_imgs",
        type=int,
        default=3,
        help="Number of images per group (high/mid/low etc.).",
    )
    ap.add_argument(
        "--min_items_per_user",
        type=int,
        default=10,
        help="Minimum number of samples per user to be considered.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for user sampling.",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) 全データ読み込み
    df_all = load_piaa_from_dir(args.input_dir)
    print(f"[info] total rows = {len(df_all)}")

    # 2) 有効ユーザーを抽出
    user_counts = df_all["user_id"].value_counts()
    valid_users = user_counts[user_counts >= args.min_items_per_user].index.to_list()
    print(f"[info] users with >= {args.min_items_per_user} items: {len(valid_users)}")

    if not valid_users:
        print("[warn] no users with enough samples, nothing to export.")
        return

    rng = np.random.RandomState(args.seed)
    if args.n_users < len(valid_users):
        sampled_users = sorted(rng.choice(valid_users, size=args.n_users, replace=False))
    else:
        sampled_users = sorted(valid_users)
    print(f"[info] sampled users: {sampled_users}")

    # 3) グローバルな (method, support_set) の組を把握（参考用）
    global_method_groups = (
        df_all[["method", "support_set"]]
        .drop_duplicates()
        .sort_values(["method", "support_set"])
        .to_records(index=False)
    )
    print("[info] method/support_set combos found:")
    for m, s in global_method_groups:
        print(f"  method={m}, support_set={s}")

    # 4) ユーザーごとに処理
    for user_id in sampled_users:
        df_user = df_all[df_all["user_id"] == user_id].copy()
        if df_user.empty:
            continue

        user_dir = os.path.join(args.out_dir, f"user_{sanitize(user_id)}")
        os.makedirs(user_dir, exist_ok=True)

        # --- GTベースの代表画像 (手法に依存しない) ---
        df_user_gt = df_user[["image_path", "user_score"]].drop_duplicates("image_path").copy()
        df_user_gt["err"] = np.nan
        df_user_gt["piaa_pred"] = np.nan

        gt_groups = [
            ("high", "gt_high"),
            # ("mid", "gt_mid"),
            ("low", "gt_low"),
        ]
        gt_subs = []

        for grp, label in gt_groups:
            sub = sample_group(df_user_gt, col="user_score", n=args.n_imgs, group=grp)
            gt_subs.append((sub, label))
            group_dir = os.path.join(user_dir, label)
            copy_images(sub, group_dir, prefix=label)
            grid_path = os.path.join(group_dir, "grid.png")
            create_thumbnail_grid(sub, grid_path, title=f"user={user_id} {label}")

        # low / mid / high を stack した1枚 (GT)
        stacked_gt_path = os.path.join(user_dir, "gt_stacked.png")
        create_stacked_thumbnail(
            gt_subs,
            stacked_gt_path,
            # title=f"user={user_id} GT (low/mid/high)",
            title="Ground Truth",
            max_cols=args.n_imgs,
        )

        # --- このユーザーに対する (method, support_set) の組を列挙 ---
        method_groups_user = (
            df_user[["method", "support_set"]]
            .drop_duplicates()
            .sort_values(["method", "support_set"])
            .to_records(index=False)
        )

        for method, sup in method_groups_user:
            df_um = df_user[(df_user["method"] == method) & (df_user["support_set"] == sup)].copy()
            if df_um.empty:
                continue

            # per-user rho
            y_true = df_um["user_score"].to_numpy(dtype=float)
            y_pred = df_um["piaa_pred"].to_numpy(dtype=float)
            mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            if len(y_true) < args.min_items_per_user:
                continue
            rho = spearmanr(y_true, y_pred).correlation
            if np.isnan(rho):
                rho = 0.0

            df_um["err"] = df_um["piaa_pred"] - df_um["user_score"]

            # ディレクトリ名に support_set も含める
            method_dir_name = f"method_{sanitize(method)}__sup_{sanitize(sup)}"
            method_dir = os.path.join(user_dir, method_dir_name)
            os.makedirs(method_dir, exist_ok=True)

            # summary
            summary_path = os.path.join(method_dir, "summary.txt")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(f"user_id={user_id}\n")
                f.write(f"method={method}\n")
                f.write(f"support_set={sup}\n")
                f.write(f"rho={rho:.4f}\n")
                f.write(f"n_items={len(df_um)}\n")

            # pred-based groups: high/mid/low
            pred_groups = [
                ("high", "pred_high"),
                # ("mid", "pred_mid"),
                ("low", "pred_low"),
            ]
            pred_subs = []

            for grp, label in pred_groups:
                sub = sample_group(df_um, col="piaa_pred", n=args.n_imgs, group=grp)
                pred_subs.append((sub, label))
                group_dir = os.path.join(method_dir, label)
                copy_images(sub, group_dir, prefix=label)
                grid_path = os.path.join(group_dir, "grid.png")
                create_thumbnail_grid(
                    sub,
                    grid_path,
                    title=f"user={user_id} method={method} sup={sup} {label}",
                )

            # low / mid / high の pred サムネイルを stack した1枚
            stacked_pred_path = os.path.join(method_dir, "pred_stacked.png")
            if method == 'direct_linear_llm_text_L15':
                title = "Linear-Hidden"
            elif method == 'direct_linear_giaa_gt_llm_text_L15':
                title = "Linear-Hidden (GIAA)"
            else:
                continue
                # title = method

            create_stacked_thumbnail(
                pred_subs,
                stacked_pred_path,
                # title=f"user={user_id} method={method} sup={sup} pred (low/mid/high)",
                title=title,
                max_cols=args.n_imgs,
            )

            # error-based groups: pos / zero / neg（画像コピーのみ）
            df_pos = df_um.sort_values(by="err", ascending=False).head(args.n_imgs)
            copy_images(df_pos, os.path.join(method_dir, "err_pos"), prefix="err_pos")

            df_zero = df_um.reindex(df_um["err"].abs().sort_values().index).head(args.n_imgs)
            copy_images(df_zero, os.path.join(method_dir, "err_zero"), prefix="err_zero")

            df_neg = df_um.sort_values(by="err", ascending=True).head(args.n_imgs)
            copy_images(df_neg, os.path.join(method_dir, "err_neg"), prefix="err_neg")

    print("[done] exported representative images and thumbnails for sampled users.")

if __name__ == "__main__":
    main()