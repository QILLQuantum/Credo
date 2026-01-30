import numpy as np
from qutip import *
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import psutil
import os
import networkx as nx
import matplotlib.pyplot as plt

print("=== BrQin v4.9 - 10×10 Stable Living Cone Sinus Filter ===\n")

def print_mem(label=""):
    mb = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
    print(f"[{label}] Memory: {mb:.2f} MB")
    return mb

baseline = print_mem("Baseline")

# ====================== ENERGY NODE ======================
class EntanglementEnergyNode:
    def __init__(self, name):
        self.name = name
        self.energy = 0.0
    def harvest(self, local_entropy):
        harvested = max(0.0, local_entropy - 0.3)
        self.energy += harvested
        return harvested
    def can_mutate(self, cost=0.15):
        return self.energy >= cost
    def spend(self, cost=0.15):
        if self.can_mutate(cost):
            self.energy -= cost
            return True
        return False

# ====================== THIN SPARSE LOCAL H ======================
def build_sparse_local_ising(Lx, Ly, rank, world_size, J=1.0, Bx=1.0, disorder=0.5, device='cpu'):
    rows_per_gpu = Lx // world_size
    start_row = rank * rows_per_gpu + min(rank, Lx % world_size)
    local_rows = rows_per_gpu + (1 if rank < Lx % world_size else 0)
    local_N = local_rows * Ly

    indices = []
    values = []

    for r in range(local_rows):
        for c in range(Ly):
            i = r * Ly + c
            if c + 1 < Ly:
                j = r * Ly + (c + 1)
                indices.append([i, j]); indices.append([j, i])
                values.append(J); values.append(J)
            if r + 1 < local_rows:
                j = (r + 1) * Ly + c
                indices.append([i, j]); indices.append([j, i])
                values.append(J); values.append(J)

    for i in range(local_N):
        indices.append([i, i]); values.append(Bx)
        rand_h = disorder * (2 * np.random.rand() - 1)
        indices.append([i, i]); values.append(rand_h)

    indices = torch.tensor(indices, dtype=torch.long, device=device).t()
    values = torch.tensor(values, dtype=torch.float16, device=device)

    H_sparse = torch.sparse_coo_tensor(indices, values, size=(local_N, local_N), device=device)
    return H_sparse.coalesce()

# ====================== LIVING MPS SWEEP WITH DYNAMIC CONE SINUS FILTER ======================
def living_mps_sweep(psi_vec, bond_map, threshold=1e-8):
    d = 2
    n = psi_vec.shape[0]
    state = psi_vec.clone()
    left = torch.eye(1, dtype=torch.complex128, device=psi_vec.device).unsqueeze(0)

    # Find core (max entropy site)
    entropies = torch.abs(state).mean(dim=0) if state.dim() > 1 else torch.abs(state)
    core_idx = torch.argmax(entropies).item()

    for i in range(n):
        # Radial distance from core (0 = core, 1 = farthest)
        distance = abs(i - core_idx) / (n - 1)
        # Dynamic cone sinus taper
        scale = np.sin(np.pi / 2 * (1 - distance))
        current_bond = int(24 * scale)
        current_bond = max(4, min(24, current_bond))

        # Reshape to matrix
        if i == 0:
            mat = state[i].reshape(d, -1)
        elif i == n-1:
            mat = state[i].reshape(-1, d)
        else:
            mat = state[i].reshape(left.shape[-1], d)

        # SVD with fallback
        try:
            U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
        except:
            current_bond = max(4, current_bond // 2)
            U, S, Vh = torch.linalg.svd(mat.to(torch.float32), full_matrices=False)

        cum_s2 = torch.cumsum(S**2, dim=0) / torch.sum(S**2 + 1e-12)
        keep = int(torch.sum(cum_s2 < 1 - threshold)) + 1
        keep = min(keep, current_bond)

        U = U[:, :keep]
        S = S[:keep]
        Vh = Vh[:keep, :]

        # Update left environment
        if i == 0:
            left = U.unsqueeze(0)
        elif i == n-1:
            left = torch.einsum('...a,a->...', left, S[:,None] * Vh)
        else:
            left = torch.einsum('...a,ab->...b', left, U * S[None,:])

        # Living update with cone cap
        local_entropy = torch.abs(state[i]).mean().item()
        if local_entropy > 0.5:
            bond_map[i] = min(24, bond_map.get(i, 12) + 4)
        else:
            bond_map[i] = max(4, bond_map.get(i, 12) - 1)

    final_vec = left.flatten()[:d]
    return torch.cat([final_vec, torch.zeros(d * (n - 1), device=psi_vec.device, dtype=torch.complex128)]

# ====================== MAIN WORKER ======================
def main_worker(rank, world_size):
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')

    Lx, Ly = 10, 10
    N = Lx * Ly
    rows_per_gpu = Lx // world_size
    local_rows = rows_per_gpu + (1 if rank < Lx % world_size else 0)
    local_N = local_rows * Ly

    H_local = build_sparse_local_ising(Lx, Ly, rank, world_size, device=device)
    current_state = torch.randn(local_N * 2, dtype=torch.complex64, device=device, requires_grad=True)

    G = nx.grid_2d_graph(Lx, Ly)
    nodes = [EntanglementEnergyNode(f"GPU{rank}-Node{i}") for i in range(4)]
    bond_map = {i: 12 for i in range(N)}

    optimizer = torch.optim.Adam([current_state], lr=0.01)
    energy_history = [[] for _ in nodes]

    for step in range(12):
        optimizer.zero_grad()
        psi_norm = torch.norm(current_state) + 1e-12
        psi = current_state / psi_norm
        energy = torch.real(psi.conj() @ H_local @ psi)
        energy.backward()
        if current_state.grad is not None:
            dist.all_reduce(current_state.grad, op=dist.ReduceOp.SUM)
            current_state.grad /= world_size
        optimizer.step()
        with torch.no_grad():
            current_state /= torch.norm(current_state) + 1e-12

        local_entropy = torch.rand(local_N, device=device) * 0.9 + 0.2
        for i, node in enumerate(nodes):
            harvested = node.harvest(local_entropy.mean().item())
            energy_history[i].append(node.energy)

        G = mutate_topology(G, local_entropy.cpu().numpy())

        current_state = living_mps_sweep(current_state, bond_map)

        current_state = exchange_boundary_tensors(current_state, rank, world_size)

        if rank == 0 and step % 4 == 0:
            avg_bond = sum(bond_map.values()) / len(bond_map)
            print_mem(f"Step {step} | Edges: {G.number_of_edges()} | Avg bond_dim: {avg_bond:.1f}")

    if rank == 0:
        plt.figure(figsize=(10, 6))
        for i, hist in enumerate(energy_history):
            plt.plot(hist, marker='o', label=f"Node {i}")
        plt.title("Energy Accumulation Over Steps (10×10 Grid)")
        plt.xlabel("Step")
        plt.ylabel("Energy (mJ)")
        plt.legend()
        plt.grid(True)
        plt.savefig('energy_accumulation_10x10.png')
        plt.close()
        print("Energy plot saved: energy_accumulation_10x10.png")

        torch.save(current_state, 'brqin_10x10_final.pt')
        nx.write_graphml(G, 'brqin_10x10_topology.graphml')
        print("10×10 final state and topology saved.")

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count() or 1
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)
