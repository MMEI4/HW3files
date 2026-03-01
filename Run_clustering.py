import os
import sys
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from my_kmeans import my_kmeans
from my_spectralclustering import my_spectralclustering

DATASETS = [
    ('Aggregation',  7, 1.0),
    ('Bridge',       2, 0.5),
    ('Compound',     6, 1.5),
    ('Flame',        2, 1.5),
    ('Jain',         2, 3.0),
    ('Spiral',       3, 0.5),
    ('TwoDiamonds',  2, 1.0),
]

DATA_DIR = os.path.dirname(os.path.abspath(__file__)) 


def compute_accuracy(pred, gt):
    pred = np.array(pred)
    gt   = np.array(gt).ravel()
    gt_labels = np.unique(gt)
    gt_mapped = np.zeros_like(gt)
    for i, lbl in enumerate(gt_labels):
        gt_mapped[gt == lbl] = i
    K = len(gt_labels)
    C = np.zeros((K, K), dtype=int)
    for p, g in zip(pred, gt_mapped):
        if p < K and g < K:
            C[p, g] += 1
    row_ind, col_ind = linear_sum_assignment(-C)
    return C[row_ind, col_ind].sum() / len(pred)


results = {}
fig, axes = plt.subplots(len(DATASETS), 3, figsize=(15, 4 * len(DATASETS)))

print(f"{'Dataset':<14} {'K':>3}  {'Sigma':>6}  {'K-Means':>9}  {'Spectral':>9}")
print("-" * 50)

for row, (name, K, sigma) in enumerate(DATASETS):
    mat_path = os.path.join(DATA_DIR, f'data_{name}.mat')
    mat  = scipy.io.loadmat(mat_path)
    D    = mat['D']
    L_gt = mat['L'].ravel()

    km_label = my_kmeans(D, K)
    sc_label = my_spectralclustering(D, K, sigma)

    km_acc = compute_accuracy(km_label, L_gt)
    sc_acc = compute_accuracy(sc_label, L_gt)
    results[name] = dict(K=K, sigma=sigma, kmeans_acc=km_acc, spectral_acc=sc_acc)

    print(f"{name:<14} {K:>3}  {sigma:>6}  {km_acc:>8.3f}  {sc_acc:>8.3f}")

    for col, (labels, title) in enumerate([
        (L_gt,     f'{name} - Ground Truth (K={K})'),
        (km_label, f'K-Means  (acc={km_acc:.3f})'),
        (sc_label, f'Spectral sigma={sigma}  (acc={sc_acc:.3f})'),
    ]):
        ax = axes[row, col]
        ax.scatter(D[:, 0], D[:, 1], c=labels, cmap='tab10', s=10)
        ax.set_title(title, fontsize=9)
        ax.axis('equal')
        ax.set_xticks([]); ax.set_yticks([])

print("-" * 50)

out_path = os.path.join(DATA_DIR, 'clustering_results.png')
plt.tight_layout()
plt.savefig(out_path, dpi=100, bbox_inches='tight')
plt.close()
print(f"\nPlot saved -> {out_path}")
