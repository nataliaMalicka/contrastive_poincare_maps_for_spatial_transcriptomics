import numpy as np
import pandas as pd
from ContrastivePoincareMaps.helpers.embedding_quality_score import get_quality_metrics

df = pd.read_csv("ContrastivePoincareMaps/Embeddings/pancreas.csv", sep=",")
X = df.iloc[:, :-1].values
poincare_emb = pd.read_csv("ContrastivePoincareMaps/Embeddings/pancreas_knn_emb.csv", header=None).values

# Euclidean
print(get_quality_metrics(
    coord_high=X,
    coord_low=poincare_emb,
    distance='euclidean',
    setting='manifold',
    k_neighbours=15,
    map_k=15,
    verbose=True
))

# for k in [5, 10, 15, 20, 30]:
print(get_quality_metrics(
    coord_high=X, # original gene expression matx
    coord_low=poincare_emb,
    distance='poincare',
    setting='manifold',
    k_neighbours=15,
    map_k=15,
    verbose=True
))

Qlocal, Qglobal, Kmax, distortion, map_score = get_quality_metrics(
    coord_high=X,
    coord_low=poincare_emb,
    distance='poincare',
    setting='manifold',
    k_neighbours=15,
    map_k=15,
    verbose=True
)

# Average distortion (absolute value, same as worst-case but averaged)
avg_distortion = np.mean(np.abs(distortion))
median_distortion = np.median(np.abs(distortion))
p95_distortion = np.percentile(np.abs(distortion), 95)
p99_distortion = np.percentile(np.abs(distortion), 99)

print(f"Distortion")
print(f"Worst-case: {np.abs(distortion).max():.4f}")
print(f"Average:    {avg_distortion:.4f}")
print(f"Median:     {median_distortion:.4f}")
print(f"95th pctl:  {p95_distortion:.4f}")
print(f"99th pctl:  {p99_distortion:.4f}")

# Find the worst pairs
inds = np.triu_indices(n=len(X), k=1)
n_worst = 5
worst_idx = np.argsort(np.abs(distortion))[-n_worst:][::-1]

labels = df.iloc[:, -1].values

print(f"Top {n_worst} most distorted pairs")
for idx in worst_idx:
    i = inds[0][idx]
    j = inds[1][idx]
    print(f"  Cell {i} ({labels[i]}) <-> Cell {j} ({labels[j]}): distortion = {np.abs(distortion[idx]):.4f}")
