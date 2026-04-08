# VLM Latent PIAA

Code for the experiments in "What Do Vision-Language Models Encode for Personalized Image Aesthetics Assessment?"

This repository focuses on two themes:

- probing VLM hidden representations for aesthetic attributes
- using those representations for personalized image aesthetics assessment (PIAA)

Run commands from the repository root with `python -m ...`.

## Repository Layout

```text
datasets/                  dataset files (not versioned)
ici/                       PIAA-ICI baseline implementation
scripts/
  data/                    split construction and dataset filtering
  piaa/                    ridge / residual / hidden-feature PIAA experiments
  probing/                 feature extraction and linear probing
  reporting/               bootstrap tests and table generation
  visualization/           plots and figure scripts
  vlm/                     VLM text-output and LoRA baselines
utils/                     shared dataset loaders and multimodal feature helpers
outputs/, runs/, logs/     generated artifacts (ignored by git)
```

## Setup

With `uv`:

```bash
uv sync
```

With `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

`torch` is intentionally left hardware-agnostic in `pyproject.toml`. If you need a CUDA-specific build, install the appropriate `torch` and `torchvision` wheels for your machine before running the heavier VLM scripts.

```bash
cp requirements.local.example.txt requirements.local.txt
uv sync
uv pip install -r requirements.local.txt
```

This keeps local CUDA or `triton` paths out of the public dependency metadata.

## Expected Data Layout

The default dataset roots are:

- `datasets/aadb`
- `datasets/PARA`
- `datasets/LAPIS`

Key annotation files referenced by the loaders include:

- `datasets/aadb/imgListFiles_label/...`
- `datasets/PARA/annotation/PARA-GiaaTrain.csv`
- `datasets/PARA/annotation/PARA-Images.csv`
- `datasets/LAPIS/annotation/LAPIS_GIAA_Trainsplit.csv`
- `datasets/LAPIS/annotation/LAPIS_PIAA.csv`

If your data lives elsewhere, pass `--dataset_dir` explicitly.

## Dataset Sources

- `AADB`: derived from the AADB release associated with Kong et al., "Photo Aesthetics Ranking Network with Attributes and Content Adaptation" (ECCV 2016). The dataset and related files are distributed from the `deepImageAestheticsAnalysis` repository: <https://github.com/aimerykong/deepImageAestheticsAnalysis>
- `PARA`: based on Yang et al., "Personalized Image Aesthetics Assessment With Rich Attributes" (CVPR 2022). The paper points to the PARA project page hosted on the Institute of Computer Vision dataset site: <https://cv-datasets.institutecv.com/#/data-sets>
- `LAPIS`: based on Maerten et al., "LAPIS: a novel dataset for personalized image aesthetic assessment" (CVPR Workshops 2025). The official repository and access instructions are here: <https://github.com/Anne-SofieMaerten/LAPIS>

Please make sure your use of each dataset follows the original license, access conditions, and citation requirements from the respective authors.

## Main Entry Points

Representative commands:

```bash
python -m scripts.probing.probing --help
python -m scripts.probing.train_attr_projection_aadb --help
python -m scripts.piaa.hidden_attr_linear_piaa --help
python -m scripts.piaa.train_direct_linear_piaa --help
python -m scripts.vlm.vlm_giaa --help
python -m scripts.visualization.viz_probe_layers_attr --help
python -m scripts.reporting.bootstrap_piaa_significance --help
python -m ici.phase1_train_resnet --help
python -m ici.phase2_train_graph --help
python -m ici.phase3_personalize --help
```

## Notes For Public Release

- Run commands from the repository root so relative paths like `datasets/...` and `outputs/...` resolve consistently.
- Large datasets, checkpoints, and generated results are not tracked.
- Machine-local paths have been removed from tracked configuration. Prefer CLI arguments over source edits when changing paths.
