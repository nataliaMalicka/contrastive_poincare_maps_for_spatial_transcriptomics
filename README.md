# Contrastive Poincaré Maps for Spatial Transcriptomics (CPM-ST)

> **Replication and benchmarking extension** of [Contrastive Poincaré Maps](https://github.com/NithyaBhasker/ContrastivePoincareMaps) (Bhasker et al., 2025).
> All credit for the original method, architecture, and training procedure belongs to the original authors. This repository reproduces their results and benchmarks CPM against alternative embedding methods on spatial transcriptomics datasets.

---

## Overview

Contrastive Poincaré Maps (CPM) learn low-distortion hyperbolic embeddings of single-cell transcriptomic data by combining contrastive self-supervised learning with Poincaré ball geometry. The method is designed to preserve hierarchical and lineage relationships that are naturally tree-like — a property that Euclidean embeddings (UMAP, t-SNE) tend to distort.

**This repository extends the original work by:**

- Reproducing the original CPM pipeline and verifying reported results
- Benchmarking CPM embeddings against Euclidean baselines (UMAP, t-SNE, PCA) and other hyperbolic methods on spatial transcriptomics data
- Evaluating performance on datasets not explored in the original paper (e.g., CosMx, MERFISH, Allen Brain Cell Atlas)

## Original Paper

**Uncovering Developmental Lineages from Single-cell Data with Contrastive Poincaré Maps**
Nithya Bhasker, Hyelim Chung, Léa Boucherie, Victor Kim, Sebastian Speidel, & Melanie Weber

bioRxiv (2025) — [https://doi.org/10.1101/2025.08.22.671789](https://doi.org/10.1101/2025.08.22.671789)

```bibtex
@article{bhasker2025uncovering,
  title   = {Uncovering Developmental Lineages from Single-cell Data with Contrastive Poincaré Maps},
  author  = {Bhasker, Nithya and Chung, Hyelim and Boucherie, L\'{e}a and Kim, Victor and Speidel, Sebastian and Weber, Melanie},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/2025.08.22.671789}
}
```

**Original code:** [github.com/NithyaBhasker/ContrastivePoincareMaps](https://github.com/NithyaBhasker/ContrastivePoincareMaps)

## What is Original vs. Replicated

| Component | Source |
|---|---|
| CPM model architecture | Replicated from Bhasker et al. (original repo) |
| Training loop and loss functions | Replicated from Bhasker et al. (original repo) |
| Benchmark baselines (UMAP, t-SNE, etc.) | Implemented by this repo |
| Evaluation metrics and comparisons | Implemented by this repo |
| Spatial transcriptomics preprocessing | Implemented by this repo |
| Application to new datasets | This repo |

## Datasets

| Dataset | Source | Description |
|---|---|---|
| *TBD — datasets under evaluation* | | |

## Installation

```bash
git clone https://github.com/<your-username>/CPM-ST.git
cd CPM-ST
pip install -r requirements.txt
```

## Usage

*Instructions will be added as the benchmarking pipeline is finalized.*

## Results

*Benchmarking results will be added here as experiments are completed.*

## Project Structure

```
CPM-ST/
├── README.md
├── requirements.txt
├── main.py
├── model/             # CPM model (from original repo)
├── helpers/           # Utilities (from original repo)
├── optimiser/         # Optimiser (from original repo)
├── benchmarks/        # Baseline methods and evaluation (this repo)
├── data/              # Dataset loading and preprocessing (this repo)
├── configs/           # Experiment config files
└── results/           # Outputs and figures
```

## Acknowledgements

This work builds entirely on the method and implementation by Bhasker et al. Their original repository and the following references were instrumental:

- [PoincareMaps](https://github.com/facebookresearch/PoincareMaps) (Facebook Research)
- [pytorch-scarf](https://github.com/clabrugere/pytorch-scarf) (Labrugere)
- [HGCN](https://github.com/HazyResearch/hgcn) (Chami et al.)

## License

This repository follows the license of the [original CPM repository](https://github.com/NithyaBhasker/ContrastivePoincareMaps). See [LICENSE](LICENSE) for details.

## Contact

For questions about the **original method**, contact the original authors:
[nithya.bhasker@nct-dresden.de](mailto:nithya.bhasker@nct-dresden.de) or [mweber@seas.harvard.edu](mailto:mweber@seas.harvard.edu)

For questions about **this replication/extension**, open an issue on this repository.
