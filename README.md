# DyMoE-Pert

**Single-Cell Genetic Perturbation Effect Prediction via Dynamic Mixture-of-Experts**

DyMoE-Pert predicts how gene perturbations change single-cell gene expression, including interventions that were not measured during training. Experimental screens cannot cover every single-gene and multi-gene combination, so accurate prediction can help prioritize new experiments and study genetic interactions.

## Overview

The method combines two complementary sources of information about a perturbed gene: biological knowledge-graph embeddings and representations derived from the single-cell foundation model scGPT. Rather than applying one fixed fusion rule to every perturbation, DyMoE-Pert learns how much to use each source and their interaction for the current input.

Its dynamic residual mixture-of-experts module contains a knowledge-graph expert, an scGPT expert, and a cross-modal interaction expert. A soft router assigns input-dependent weights to their outputs, while a residual base branch carries shared multimodal information. The resulting representation is combined with gene-context features and passed to a Transformer to predict expression changes. For a two-gene perturbation, the model composes information from both perturbed genes.

The accompanying manuscript evaluates unseen single-gene and combinatorial perturbations across Norman, Dixit, Adamson, Replogle K562, and Replogle RPE1. Performance varies by dataset and metric; the clearest reported advantage is for the challenging combinations in which neither constituent gene was observed individually during training.

This repository provides the **core runnable implementation**: the model, data loader, training and prediction commands, and the fixed gene-embedding files needed at runtime. It does not include expression datasets, trained checkpoints, experiment logs, or scripts for reproducing every manuscript table.

## Installation

```bash
git clone https://github.com/zhyzhy12512/DyMoE-Pert.git
cd DyMoE-Pert
```

The code was validated with Python 3.8.20, PyTorch 2.3.0, and PyTorch Geometric 2.6.1 on Linux. Install a PyTorch 2.3.0 build suitable for your CUDA driver first (see the [official PyTorch installation guide](https://pytorch.org/get-started/previous-versions/)), then run:

```bash
python -m pip install -r requirements.txt
```

Run all commands below **from this directory**. The model currently reads two common embedding files by paths relative to the working directory.

## Data and embeddings

No expression dataset is included. The loader downloads and extracts the processed GEARS-format data automatically into `./data/` on the first run. You may also download the ZIPs yourself and extract them under `data/<dataset-name>/` (each directory must contain `perturb_processed.h5ad`).

| `--data-name` | Processed data ZIP |
| --- | --- |
| `norman` | https://dataverse.harvard.edu/api/access/datafile/6154020 |
| `adamson` | https://dataverse.harvard.edu/api/access/datafile/6154417 |
| `dixit` | https://dataverse.harvard.edu/api/access/datafile/6154416 |
| `replogle_k562_essential` | https://dataverse.harvard.edu/api/access/datafile/7458695 |
| `replogle_rpe1_essential` | https://dataverse.harvard.edu/api/access/datafile/7458694 |

The Replogle files are the filtered essential-gene subsets used by this code. The loader also retrieves the [essential perturbation gene list](https://dataverse.harvard.edu/api/access/datafile/6934320) and [GO similarity graph](https://dataverse.harvard.edu/api/access/datafile/6934319) when needed. These links are the exact URLs used in `models/ProcePertdata.py` and `models/utils.py`.

The `embeddings/` directory contains two common gene representations from the upstream scPert repository and five dataset-aligned 512-dimensional arrays generated for this project from an scGPT checkpoint. Genes absent from the scGPT checkpoint vocabulary received deterministic random fallback vectors. The fallback counts are 498/5045 (Norman), 660/5060 (Adamson), 619/5012 (Dixit), 187/5000 (Replogle K562), and 101/5000 (Replogle RPE1). These arrays are fixed model inputs, not expression datasets. The embedding-generation script and original scGPT checkpoint are outside this minimal runtime package; the arrays themselves are included so the training and prediction commands work without regeneration. See the [scGPT project](https://github.com/bowang-lab/scGPT) for the source model.

## Training

The defaults correspond to the main DyMoE-Pert configuration: simulation split seed 77, 25 epochs, batch size 64, hidden size 64, dynamic routing, dropout 0.1, and balance weight 0.0. Model initialization uses PyTorch seed 0 in `models/scpert.py`. For a memory-conscious single-GPU run on Norman, reduce the batch size:

```bash
python scripts/train.py --data-name norman --device cuda:0 --batch-size 4 --test-batch-size 4 --model-save-name norman_dymoe
```

To use four GPUs, as in the reported benchmark runs:

```bash
python scripts/train.py --data-name norman --multi-gpu --device-ids 0,1,2,3 --model-save-name norman_dymoe
```

For another dataset, change `--data-name` to one of the names in the table. The script chooses the corresponding dataset-aligned embedding automatically. `--embedding-dir` controls that dataset-aligned file; the two common embedding files must remain in `./embeddings/`. Training saves `config.pkl` and `model.pt` in the directory passed to `--model-save-name`.

## Prediction

Use a checkpoint produced by `scripts/train.py`. For a single-gene perturbation on Norman:

```bash
python scripts/infer.py --data-name norman --checkpoint norman_dymoe --perturbation CEBPA+ctrl --device cuda:0
```

A double-gene perturbation has the form `GENE1+GENE2`. Both gene names must occur in the selected dataset's perturbation vocabulary. The script writes `pred_dymoe_<perturbation>.npz` to the current directory. The file contains a mean predicted expression vector and individual predictions for 32 sampled control cells by default. Set `--num-cells 300` to use 300 cells, and adjust `--batch-size` to fit your GPU memory (default: 8).

If running on CPU, set `--device cpu`; model training and inference will be much slower. The first run also takes time and disk space to download/process the selected dataset and build graph caches.

## Repository structure

```text
models/          DyMoE-Pert architecture, graph/data processing, loss and metrics
scripts/train.py Training CLI
scripts/infer.py Prediction CLI
embeddings/      Required fixed model inputs
requirements.txt Python dependencies
LICENSE          Upstream code license
```

## Attribution and scope

This implementation builds on the [scPert source code](https://github.com/AI4VirtualCell/scPert), uses [scGPT](https://github.com/bowang-lab/scGPT)-derived representations, and loads processed perturbation datasets made available through [GEARS](https://github.com/snap-stanford/GEARS). The upstream scPert `LICENSE` is preserved verbatim, including its placeholder copyright field.

The repository focuses on training and prediction. Reproducing every manuscript table additionally requires the separate evaluation workflow and dataset-specific checkpoints. Only load dataset archives and model checkpoints from sources you trust: the loader uses Python pickle for processed graph caches and checkpoint configuration.
