import numpy as np

def my_kmeans(data, K):
    N, d = data.shape
    rng = np.random.default_rng(42)
    indices = rng.choice(N, K, replace=False)
    centroids = data[indices].copy()
    
    label = np.zeros(N, dtype=int)
    
    for _ in range(1000):
        diffs = data[:, None, :] - centroids[None, :, :] 
        distances = np.sqrt((diffs ** 2).sum(axis=2))     
        new_label = np.argmin(distances, axis=1)
        if np.all(new_label == label):
            break
        label = new_label
        
        for k in range(K):
            members = data[label == k]
            if len(members) > 0:
                centroids[k] = members.mean(axis=0)
    
    return label
