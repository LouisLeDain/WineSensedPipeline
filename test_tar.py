from process import pairwise_distance_matrix, perform_nmds
from sklearn.manifold import MDS, TSNE
import numpy as np
from sklearn.model_selection import ParameterGrid
NAPPING_CSV = "data/napping.csv"



D,wine_ids = pairwise_distance_matrix(NAPPING_CSV)


# compute TAR
N= D.shape[0]
triplets = [] # The index are only on the Human kernel
for i in range(N):
    for j in range(N):
        for k in range(N):
            if D[i,j] > 0 and D[i,k] > 0 and D[i,j] < D[i,k]: # We put >0 because we don't want to have 0 distance (missing value)
                triplets.append((i,j,k))
                
# We will now do T-SNE on the distance matrix
param_grid = {
    "n_components": [2],
    "metric": ["precomputed"],
    "n_jobs": [-1],
    "perplexity": [5, 10, 20, 30, 50],
    "max_iter": [300, 500, 700, 1000, 2000],
    "init" : ["random"],
}
    
    
results = []
for params in ParameterGrid(param_grid):    
    tsne = TSNE(**params)
    emb = tsne.fit_transform(D)
    
    # compute TAR
    agree = 0
    for i, j, k in triplets:
        dij = np.linalg.norm(emb[i] - emb[j])
        dik = np.linalg.norm(emb[i] - emb[k])
        if dij < dik:
            agree += 1
    tar = agree / len(triplets)
    
    results.append((tar, params))
    print(f"params={params} → TAR={tar:.4f}")
print(f"max tar={max(results, key=lambda x: x[0])}")
    
# # 3) Define your grid of hyperparameters
# param_grid = {
#     "max_iter":      [300, 500,700, 1000,2000],
#     "eps":           [1e-1, 1e-2, 1e-3,1e-4, 1e-5, 1e-6,1e-7],
#     "n_init":        [4, 10,20, 50,100,200,300,500,1000],
#     "metric":        [ False],
# }

# # 4) Loop over grid, fit MDS, compute TAR
# results = []
# for params in ParameterGrid(param_grid):    
#     mds = MDS(n_components=2,
#         dissimilarity="precomputed",
#         n_jobs=-1,
#         **params
#     )
#     emb = mds.fit_transform(D)
    
#     # compute TAR
#     agree = 0
#     for i, j, k in triplets:
#         dij = np.linalg.norm(emb[i] - emb[j])
#         dik = np.linalg.norm(emb[i] - emb[k])
#         if dij < dik:
#             agree += 1
#     tar = agree / len(triplets)
    
#     results.append((tar, params))
#     print(f"params={params} → TAR={tar:.4f}")
#     print(f"stress={mds.stress_:.4f}")
    
# # 5) Find best
# best_tar, best_params = max(results, key=lambda x: x[0])
# print("\nBest hyperparameters:")
# print(f"  TAR = {best_tar:.4f}")
# print(f"  params = {best_params}")