# Contrastive Poincaré Maps Reproduction and Analysis

> **Replication and analysis** of [Contrastive Poincaré Maps](https://github.com/NithyaBhasker/ContrastivePoincareMaps) (Bhasker et al., 2025).
> All credit for the original method, architecture, and training procedure belongs to the original authors. This repository reproduces their results, applies the method to a dataset not used in the original paper, and explores an alternative loss function.

---

## Overview

Contrastive Poincaré Maps (CPM) learn low-distortion hyperbolic embeddings of single-cell transcriptomic data by combining contrastive self-supervised learning with Poincaré ball geometry. The method is designed to preserve hierarchical and lineage relationships that are naturally tree-like — a property that Euclidean embeddings (UMAP, t-SNE) tend to distort.

**This repository extends the original work by:**

- Reproducing the original CPM pipeline on the mouse hematopoiesis (Paul et al., 2015) and chicken cardiogenesis (Mantri et al., 2021) datasets
- Applying CPM to mouse pancreatic endocrinogenesis (Bastidas-Ponce et al., 2019), a dataset not used in the original paper
- Implementing a weighted k-nearest-neighbor loss as an alternative to the InfoNCE contrastive objective and comparing the resulting embeddings

## Original Paper

**Uncovering Developmental Lineages from Single-cell Data with Contrastive Poincaré Maps**
Nithya Bhasker, Hyelim Chung, Léa Boucherie, Victor Kim, Sebastian Speidel, & Melanie Weber

bioRxiv (2025) — [https://doi.org/10.1101/2025.08.22.671789](https://doi.org/10.1101/2025.08.22.671789)

```bibtex
