import numpy as np
from qutip import *
import torch  # For SVD/PQ
import matplotlib.pyplot as plt  # Heatmap amp

# Gates (unchanged)
def rx(theta): return Qobj([[np.cos(theta/2), -1j*np.sin(theta/2)], [-1j*np.sin(theta/2), np.cos(theta/2)]])
def ry(theta): return Qobj([[np.cos(theta/2), -np.sin(theta/2)], [np.sin(theta/2), np.cos(theta/2)]])
def cz(): return Qobj([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]], dims=[[2,2],[2,2]])
def cnot(): return Qobj([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dims=[[2,2],[2,2]])
def hadamard(): return (1/np.sqrt(2)) * Qobj([[1,1],[1,-1]])

# Entropy (unchanged)
def von_neumann_entropy(rho):
    evals = rho.eigenenergies()
    evals = evals[evals > 1e-10]
    return -np.sum(evals * np.log2(evals)) if len(evals) > 0 else 0.0

print("BrQin v1.7: Brain-Quantum Imprint Simulator - PQ/Heatmap Amp")

# Sections 1-7 (full expanded, unchanged from v1.3)
# (insert as before—qubit, density, entanglement, QNN, evo, branch, rewrite)

# Section 8: 2D Pixel Grid (unchanged from v1.3)

# Section 9-10: Water Hexamer EZ (unchanged from v1.3)

# Section 11: 4D Hypergrid with W (dims=[2,2,2,2] + PQ 3-Bit Scrub + Heatmap)
print("\n11. 4D Hypergrid Brain with W (Dims: [2,2,2,2]=16 Qubits) - PQ/Heatmap Amp")
dims = [2, 2, 2, 2]
N_hyper = np.prod(dims)
Jx, Jy, Jz, Jw = 1.0, 0.8, 0.3, 0.5
hidden_dim = 64

# H_hyper build (same as v1.5)

# PQ/OPQ pro for 3-bit embeddings (scrub H_matrix)
def pq_quantize(matrix, M=4, K=8, bits=3):
    dim = matrix.shape[1]
    sub_dim = dim // M
    codebooks = np.zeros((M, K, sub_dim))
    codes = np.zeros((matrix.shape[0], M), dtype=np.uint8)
    for m in range(M):
        sub_mat = matrix[:, m*sub_dim:(m+1)*sub_dim]
        centroids = np.random.randn(K, sub_dim)  # K-means sim
        for _ in range(5):  # Toy iter
            dists = np.linalg.norm(sub_mat[:, None] - centroids, axis=2)
            codes[:, m] = np.argmin(dists, axis=1)
            for k in range(K):
                mask = codes[:, m] == k
                if mask.sum() > 0:
                    centroids[k] = sub_mat[mask].mean(axis=0)
        codebooks[m] = centroids
    # Reconstruct approx
    recon = np.zeros_like(matrix)
    for m in range(M):
        recon[:, m*sub_dim:(m+1)*sub_dim] = codebooks[m, codes[:, m]]
    return recon, codebooks.nbytes + codes.nbytes

H_matrix = H_hyper.full().astype(np.float32)
H_recon, pq_size = pq_quantize(H_matrix)
print("PQ 3-bit scrubbed size (bytes):", pq_size)
H_approx = Qobj(H_recon, dims=H_hyper.dims)

# Low-rank + reverse stack + evo (same as v1.6)

# Heatmap amp for <sigma_z> slice
final_z_4d = np.array([expect_hyper[i][-1] for i in range(N_hyper)]).reshape(dims)
slice_2d = final_z_4d[:,:,0,0]  # z=0, w=0
plt.imshow(slice_2d, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.title("<σ_z> 4D Slice Heatmap")
plt.savefig('sigma_z_heatmap.png')
print("Heatmap saved as sigma_z_heatmap.png")

print("\nBrQin v1.7 complete - PQ scrubbed, heatmap amped!")