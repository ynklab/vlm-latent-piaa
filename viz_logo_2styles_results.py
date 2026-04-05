#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sanitize(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(s))


def save_bar(out_path: str, labels, values, title: str, xlabel: str, ylabel: str):
    plt.close("all")
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_box(out_path: str, data_groups, labels, title: str, ylabel: str, showfliers: bool = False):
    plt.close("all")
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.boxplot(data_groups, labels=labels, showfliers=showfliers)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_csv", required=True, help="LOGO Ridge results CSV (2_styles).")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--skip_log_csv", default=None, help="Optional skip log CSV.")
    ap.add_argument("--no_fliers", action="store_true", help="Hide outliers in boxplots.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.results_csv)

    # basic checks
    required = {"user_id", "holdout_tag", "n_train", "n_test", "rho", "r2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"results_csv missing required columns: {missing}")

    df["rho"] = pd.to_numeric(df["rho"], errors="coerce")
    df["r2"] = pd.to_numeric(df["r2"], errors="coerce")
    df = df.dropna(subset=["rho", "r2"])

    # -------- summary table --------
    rows = []

    rows.append({
        "group": "overall",
        "n_rows": len(df),
        "mean_rho": float(df["rho"].mean()),
        "std_rho": float(df["rho"].std()),
        "mean_r2": float(df["r2"].mean()),
        "std_r2": float(df["r2"].std()),
    })

    for holdout, g in df.groupby("holdout_tag"):
        rows.append({
            "group": f"holdout={holdout}",
            "n_rows": len(g),
            "mean_rho": float(g["rho"].mean()),
            "std_rho": float(g["rho"].std()),
            "mean_r2": float(g["r2"].mean()),
            "std_r2": float(g["r2"].std()),
        })

    df_summary = pd.DataFrame(rows)
    out_summary = os.path.join(args.out_dir, "summary.csv")
    df_summary.to_csv(out_summary, index=False)
    print("[save]", out_summary)

    # -------- bar: mean by holdout --------
    holdouts = sorted(df["holdout_tag"].astype(str).unique().tolist())
    mean_rho = [float(df[df["holdout_tag"].astype(str) == h]["rho"].mean()) for h in holdouts]
    mean_r2  = [float(df[df["holdout_tag"].astype(str) == h]["r2"].mean()) for h in holdouts]

    save_bar(
        os.path.join(args.out_dir, "mean_rho_by_holdout.png"),
        holdouts, mean_rho,
        title="Mean Spearman (rho) by held-out 2_styles",
        xlabel="Held-out style (test)",
        ylabel="Mean rho",
    )
    save_bar(
        os.path.join(args.out_dir, "mean_r2_by_holdout.png"),
        holdouts, mean_r2,
        title="Mean R2 by held-out 2_styles",
        xlabel="Held-out style (test)",
        ylabel="Mean R2",
    )

    # -------- boxplots: distribution by holdout --------
    rho_groups = [df[df["holdout_tag"].astype(str) == h]["rho"].to_numpy() for h in holdouts]
    r2_groups  = [df[df["holdout_tag"].astype(str) == h]["r2"].to_numpy() for h in holdouts]

    save_box(
        os.path.join(args.out_dir, "rho_boxplot_by_holdout.png"),
        rho_groups, holdouts,
        title="Per-user rho distribution (LOGO) by held-out style",
        ylabel="Spearman rho",
        showfliers=not args.no_fliers,
    )
    save_box(
        os.path.join(args.out_dir, "r2_boxplot_by_holdout.png"),
        r2_groups, holdouts,
        title="Per-user R2 distribution (LOGO) by held-out style",
        ylabel="R2",
        showfliers=not args.no_fliers,
    )

    # -------- optional: skip log --------
    if args.skip_log_csv and os.path.exists(args.skip_log_csv):
        df_skip = pd.read_csv(args.skip_log_csv)
        # expecting columns: user_id, holdout_tag, reason, n_train, n_test (or similar)
        if "reason" in df_skip.columns:
            counts = df_skip["reason"].value_counts()
            save_bar(
                os.path.join(args.out_dir, "skip_counts_by_reason.png"),
                labels=list(counts.index),
                values=list(counts.values),
                title="Skipped LOGO folds (count by reason)",
                xlabel="reason",
                ylabel="count",
            )
        print("[info] skip log loaded:", args.skip_log_csv)

    print("[done]")


if __name__ == "__main__":
    main()