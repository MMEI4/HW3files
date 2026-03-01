import numpy as np

def my_spectralclustering(data, K, sigma):
    from my_kmeans import my_kmeans
    
    N = data.shape[0]
    
    diffs = data[:, None, :] - data[None, :, :]  
    sq_dists = (diffs ** 2).sum(axis=2)         
    W = np.exp(-sq_dists / (2 * sigma ** 2))
    np.fill_diagonal(W, 0)
    
    D = np.diag(W.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(W.sum(axis=1)))
    
    L_sym = D_inv_sqrt @ W @ D_inv_sqrt

    eigenvalues, eigenvectors = np.linalg.eigh(L_sym)

    idx = np.argsort(eigenvalues)[::-1][:K]
    V = eigenvectors[:, idx]

    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms == 0] = 1
    V = V / norms
    
    label = my_kmeans(V, K)
    
    return label
