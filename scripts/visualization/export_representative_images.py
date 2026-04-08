#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Export representative images for qualitative comparison across PIAA methods,
and create thumbnail-style grids (contact sheets) with matplotlib.

- Randomly sample users
- For each user:
  - export `n_imgs` representative images for gt_low / gt_mid / gt_high based on GT (`user_score`)
  - for each method:
      * export `n_imgs` representative images for pred_low / pred_mid / pred_high based on `piaa_pred`
      * export `n_imgs` representative images for err_pos / err_zero / err_neg based on `err = piaa_pred - user_score`
  - also create one vertically stacked image of low / mid / high thumbnails:
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
    df: per-user DataFrame (image_path, user_score, piaa_pred, ...)
    col: sorting key ('user_score', 'piaa_pred', 'err', 'abs_err')
    group: e.g. 'high' / 'mid' / 'low'

    Returns: subset DataFrame
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
    sub_df: DataFrame containing columns such as image_path, user_score, piaa_pred, and err
    dst_dir: output directory
    prefix: filename prefix (e.g. 'gt_high', 'pred_low', 'err_pos')
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
    sub_df: DataFrame containing image_path, user_score, piaa_pred, and err
    out_path: output PNG path
    title: title of the whole grid
    max_cols: maximum number of grid columns before wrapping
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



def create_stacked_thumbnail(groups, out_path, title, max_cols=5,
                            cell_border=2.0, label_w=0.14, top_h=0.10,
                            annotate_scores=False):
    """
    Table-style 2xK grid (High / Low rows) with explicit borders and row headers.

    - groups: expected format is [(df_high, "High Score"), (df_low, "Low Score")]
    - title: figure title (e.g. method name)
    - label_w: width of the row-label area on the left (fraction of figure width)
    - top_h  : height of the top title area (fraction of figure height)
    - annotate_scores: if True, add small gt/pred/err annotations below each cell
    """

    total_images = sum(len(df) for df, _ in groups)
    if total_images == 0:
        return

    # ---- Fixed assumption: 2 rows (High/Low). Generalize if needed. ----
    group_dfs = [df for (df, _) in groups]
    group_labels = ["High Score", "Low Score"]

    rows = len(group_dfs)
    max_n = max(len(df) for df in group_dfs)
    cols = min(max_cols, max_n)
    if rows == 0 or cols == 0:
        return

    # Figure size: keep each cell roughly square
    fig_w = 3.0 * cols / (1.0 - label_w)
    fig_h = 3.0 * rows / (1.0 - top_h)
    plt.close("all")
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))

    # axes shape normalize
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    # Layout: reserve space on the left for row labels and on top for the title
    fig.subplots_adjust(
        left=label_w, right=0.99,
        top=1.0 - top_h, bottom=0.04,
        wspace=0.02, hspace=0.02
    )

    # ----- Draw images in each cell -----
    for r, df_group in enumerate(group_dfs):
        paths = df_group["image_path"].tolist()
        gt_scores = df_group.get("user_score", pd.Series([np.nan]*len(df_group))).tolist()
        preds = df_group.get("piaa_pred", pd.Series([np.nan]*len(df_group))).tolist()
        errs = df_group.get("err", pd.Series([np.nan]*len(df_group))).tolist()

        for c in range(cols):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_frame_on(False)

        for c in range(min(cols, len(paths))):
            ax = axes[r, c]
            try:
                img = Image.open(paths[c]).convert("RGB")
                ax.imshow(img)
            except Exception:
                pass

            # optional small annotation under the image (inside the cell)
            if annotate_scores:
                gt = gt_scores[c] if c < len(gt_scores) else np.nan
                pr = preds[c] if c < len(preds) else np.nan
                er = errs[c] if c < len(errs) else np.nan
                txt = f"GT={gt:.2f}  Pred={pr:.2f}  Err={er:.2f}"
                ax.text(
                    0.02, 0.02, txt,
                    transform=ax.transAxes,
                    fontsize=8,
                    color="white",
                    va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.55, edgecolor="none"),
                    zorder=10,
                )

    # ----- Draw borders to create the table-like layout -----
    # Get each axes position in figure coordinates and overlay rectangles on cell boundaries
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            pos = ax.get_position()
            rect = Rectangle(
                (pos.x0, pos.y0),
                pos.width, pos.height,
                transform=fig.transFigure,
                fill=False,
                linewidth=cell_border,
                edgecolor="black",
                zorder=20,
            )
            fig.add_artist(rect)

    # ----- Row labels (left side of the table) -----
    for r, lbl in enumerate(group_labels):
        # Use the axes positions to compute the y-center of each row
        pos = axes[r, 0].get_position()
        y_center = (pos.y0 + pos.y1) / 2
        fig.text(
            0.02, y_center, lbl,
            ha="left", va="center",
            fontsize=21, fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                alpha=0.6,          # transparency (adjust within 0.5-0.8 if desired)
                edgecolor="none",
            ),
        )

    # # ----- Optional column labels: #1, #2, ... -----
    # for c in range(cols):
    #     pos = axes[0, c].get_position()
    #     x_center = (pos.x0 + pos.x1) / 2
    #     fig.text(
    #         x_center, 1.0 - top_h + 0.01,
    #         f"#{c+1}",
    #         ha="center", va="bottom",
    #         fontsize=14, fontweight="bold"
    #     )

    # ----- Title -----
    fig.suptitle(title, fontsize=24, y=0.995)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
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

    # 1) Load all data
    df_all = load_piaa_from_dir(args.input_dir)
    print(f"[info] total rows = {len(df_all)}")

    # 2) Extract valid users
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

    # 3) Gather global (method, support_set) pairs for reference
    global_method_groups = (
        df_all[["method", "support_set"]]
        .drop_duplicates()
        .sort_values(["method", "support_set"])
        .to_records(index=False)
    )
    print("[info] method/support_set combos found:")
    for m, s in global_method_groups:
        print(f"  method={m}, support_set={s}")

    # 4) Process each user
    for user_id in sampled_users:
        df_user = df_all[df_all["user_id"] == user_id].copy()
        if df_user.empty:
            continue

        user_dir = os.path.join(args.out_dir, f"user_{sanitize(user_id)}")
        os.makedirs(user_dir, exist_ok=True)

        # --- GT-based representative images (method-independent) ---
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

        # One GT image stacking low / mid / high thumbnails
        stacked_gt_path = os.path.join(user_dir, "gt_stacked.pdf")
        create_stacked_thumbnail(
            gt_subs,
            stacked_gt_path,
            # title=f"user={user_id} GT (low/mid/high)",
            title="Ground Truth",
            max_cols=args.n_imgs,
        )

        # --- Enumerate (method, support_set) pairs for this user ---
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

            # Include support_set in the directory name as well
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

            # One image stacking low / mid / high predicted thumbnails
            stacked_pred_path = os.path.join(method_dir, "pred_stacked.pdf")
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

            # Error-based groups: pos / zero / neg (copy images only)
            df_pos = df_um.sort_values(by="err", ascending=False).head(args.n_imgs)
            copy_images(df_pos, os.path.join(method_dir, "err_pos"), prefix="err_pos")

            df_zero = df_um.reindex(df_um["err"].abs().sort_values().index).head(args.n_imgs)
            copy_images(df_zero, os.path.join(method_dir, "err_zero"), prefix="err_zero")

            df_neg = df_um.sort_values(by="err", ascending=True).head(args.n_imgs)
            copy_images(df_neg, os.path.join(method_dir, "err_neg"), prefix="err_neg")

    print("[done] exported representative images and thumbnails for sampled users.")

if __name__ == "__main__":
    main()