# viz_layers.py (colors fixed: many distinct colors)
import os
import re
import json
import argparse
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

SPLITS = ["train", "val", "test"]
METRICS = ["rho", "r2", "rmse"]
SOURCE_ORDER = ["vision", "bridge_text", "bridge_visual", "llm_visual", "llm_text", "llm_text_tail"]

def sanitize(s: str) -> str:
    if s is None or not isinstance(s, str) or s.strip() == "":
        return "unknown_model"
    return re.sub(r"[^0-9A-Za-z._\\-]+", "_", s)

def collect_sources(results: Dict) -> List[str]:
    sources = set()
    for _, entry in results.get("attrs", {}).items():
        for it in entry.get("per_layer", []):
            src = it.get("source")
            if src:
                sources.add(src)
    return [s for s in SOURCE_ORDER if s in sources] + sorted(s for s in sources if s not in SOURCE_ORDER)

def collect_series(results: Dict, source: str, split: str, metric: str) -> Dict[str, Tuple[List[int], List[float]]]:
    out = {}
    attrs = results.get("attrs", {})
    for attr_name, entry in attrs.items():
        items = [it for it in entry.get("per_layer", []) if it.get("source") == source]
        if not items:
            continue
        items = sorted(items, key=lambda x: x.get("layer", 0))
        xs, ys = [], []
        for it in items:
            xs.append(int(it.get("layer", 0)))
            split_dict = it.get(split, {})
            val = split_dict.get(metric, None)
            ys.append(np.nan if val is None else float(val))
        out[attr_name] = (xs, ys)
    return out

def pick_distinct_colors(n: int, palette: str = "auto") -> List:
    """
    Return n visually distinct RGBA colors.
    - 'auto' -> use tab20 + tab20b + tab20c (up to 60), then fall back to HSV sampling.
    - otherwise -> try the named matplotlib colormap and sample evenly.
    """
    colors = []
    if palette == "auto":
        pools = []
        for name in ["tab20", "tab20b", "tab20c"]:
            cmap = plt.get_cmap(name)
            if hasattr(cmap, "colors"):
                pools.extend(list(cmap.colors))
        if n <= len(pools):
            colors = pools[:n]
        else:
            # fallback: evenly sample from a continuous cmap (HSV)
            cmap = plt.get_cmap("hsv")
            colors = [cmap(i / max(1, n)) for i in range(n)]
    else:
        cmap = plt.get_cmap(palette)
        if hasattr(cmap, "colors") and len(cmap.colors) >= n:
            colors = list(cmap.colors)[:n]
        else:
            colors = [cmap(i / max(1, n)) for i in range(n)]
    return colors

def plot_source_split_metric(model_id: str, source: str, split: str, metric: str,
                             series: Dict[str, Tuple[List[int], List[float]]],
                             out_path: str, attr2color: Dict[str, tuple],
                             figsize=(10,8.5), dpi=160,
                             marker_size: float = 3.0, line_width: float = 1.0):
    if not series:
        return False
    plt.close("all")
    fig, ax = plt.subplots(figsize=figsize)

    # draw each attribute's line with its fixed color
    for attr, (xs, ys) in series.items():
        ax.plot(xs, ys, marker="o", markersize=marker_size, linewidth=line_width,
                label=attr, color=attr2color.get(attr))

    ax.set_title(f"{model_id} | {source} | {split} | {metric}", fontsize=12)
    ax.set_xlabel("Layer index")
    ax.set_ylabel(metric.upper())
    ax.grid(True, linestyle="--", alpha=0.4)

    all_layers = sorted({x for (xs, _) in series.values() for x in xs})
    ax.set_xticks(all_layers)

    all_vals = np.array([v for (_, ys) in series.values() for v in ys if v is not None and not np.isnan(v)], dtype=float)
    if all_vals.size > 0:
        ymin, ymax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
        if np.isfinite(ymin) and np.isfinite(ymax):
            if ymin == ymax:
                pad = 0.05 * (abs(ymin) + 1.0)
                ax.set_ylim(ymin - pad, ymax + pad)
            else:
                pad = 0.05 * (ymax - ymin)
                ax.set_ylim(ymin - pad, ymax + pad)

    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True

def load_result_files(inputs: List[str]) -> List[str]:
    files = []
    for p in inputs:
        if os.path.isdir(p):
            for name in os.listdir(p):
                if name.lower().endswith(".json"):
                    files.append(os.path.join(p, name))
        elif os.path.isfile(p) and p.lower().endswith(".json"):
            files.append(p)
    return sorted(files)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="Result JSON input (file or directory)")
    ap.add_argument("--out_dir", default="viz", help="Output root directory")
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--fig_w", type=float, default=10.0)
    ap.add_argument("--fig_h", type=float, default=8.5)
    ap.add_argument("--marker_size", type=float, default=3.0)
    ap.add_argument("--line_width", type=float, default=1.0)
    ap.add_argument("--palette", type=str, default="auto",
                    help="Color palette: auto / tab20 / tab20b / tab20c / hsv / turbo / etc.")
    args = ap.parse_args()

    files = load_result_files(args.inputs)
    if not files:
        print("[viz] No JSON files found.")
        return

    for fp in files:
        try:
            with open(fp, "r") as f:
                results = json.load(f)
        except Exception as e:
            print(f"[viz] Skip (load error): {fp} -> {e}")
            continue

        cfg = results.get("config", {})
        prompt_mode = cfg.get("prompt_mode", "unknown_prompt")
        dataset     = cfg.get("dataset", "unknown_dataset")
        model_id    = cfg.get("model_id") or os.path.splitext(os.path.basename(fp))[0]

        prompt_dir  = sanitize(prompt_mode)
        dataset_dir = sanitize(dataset)
        model_dir   = os.path.join(args.out_dir, dataset_dir, prompt_dir, sanitize(model_id))
        sources = collect_sources(results)
        if not sources:
            print(f"[viz] No sources found in: {fp}")
            continue

        # Map attribute names to colors (fixed within each model)
        attr_names = sorted(list(results.get("attrs", {}).keys()))
        colors = pick_distinct_colors(len(attr_names), palette=args.palette)
        attr2color = {a: c for a, c in zip(attr_names, colors)}

        for source in sources:
            for split in SPLITS:
                for metric in METRICS:
                    series = collect_series(results, source=source, split=split, metric=metric)
                    out_path = os.path.join(model_dir, f"{source}__{split}__{metric}.png")
                    ok = plot_source_split_metric(
                        model_id, source, split, metric, series, out_path, attr2color,
                        figsize=(args.fig_w, args.fig_h), dpi=args.dpi,
                        marker_size=args.marker_size, line_width=args.line_width
                    )
                    if ok:
                        print(f"[viz] saved: {out_path}")

if __name__ == "__main__":
    main()
