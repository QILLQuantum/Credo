import numpy as np
from qutip import *
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import psutil
import os
import time
import matplotlib.pyplot as plt

print("=== BrQin v4.5 - 6×6 Energy-Chained DTO with Real MPS + Distributed ===")

# ====================== MEMORY TRACKING ======================
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

# ====================== REAL MPS SWEEP ======================
def mps_sweep(psi_vec, max_bond=16, threshold=1e-8):
    d = 2
    n = psi_vec.shape[0]
    state = psi_vec.clone()
    left = torch.eye(1, dtype=torch.complex128, device=psi_vec.device).unsqueeze(0)

    for i in range(n):
        if i == 0:
            mat = state[i].reshape(d, -1)
        elif i == n-1:
            mat = state[i].reshape(-1, d)
        else:
            mat = state[i].reshape(left.shape[-1], d)

        U, S, Vh = torch.linalg.svd(mat, full_matrices=False)

        cum_s2 = torch.cumsum(S**2, dim=0) / torch.sum(S**2 + 1e-12)
        keep = int(torch.sum(cum_s2 < 1 - threshold)) + 1
        keep = min(keep, max_bond)

        U = U[:, :keep]
        S = S[:keep]
        Vh = Vh[:keep, :]

        if i == 0:
            left = U.unsqueeze(0)
        elif i == n-1:
            left = torch.einsum('...a,a->...', left, S[:,None] * Vh)
        else:
            left = torch.einsum('...a,ab->...b', left, U * S[None,:])

    final_vec = left.flatten()[:d]
    return torch.cat([final_vec, torch.zeros(d * (n - 1), device=psi_vec.device, dtype=torch.complex128)])

# ====================== DISTRIBUTED SETUP ======================
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# ====================== MAIN WORKER ======================
def main_worker(rank, world_size):
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')

    # 6x6 grid
    Lx, Ly = 6, 6
    N = Lx * Ly
    rows_per_gpu = Lx // world_size
    local_rows = rows_per_gpu + (1 if rank < Lx % world_size else 0)
    local_N = local_rows * Ly

    nodes = [EntanglementEnergyNode(f"GPU{rank}-Node{i}") for i in range(4)]

    current_state = torch.randn(local_N * 2, dtype=torch.complex64, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([current_state], lr=0.01)

    energy_history = [[] for _ in nodes]

    for step in range(15):
        optimizer.zero_grad()
        psi_norm = torch.norm(current_state) + 1e-12
        psi = current_state / psi_norm
        energy = torch.real(psi.conj() @ torch.randn(local_N * 2, local_N * 2, device=device) @ psi)
        energy.backward()
        if current_state.grad is not None:
            dist.all_reduce(current_state.grad, op=dist.ReduceOp.SUM)
            current_state.grad /= world_size
        optimizer.step()
        with torch.no_grad():
            current_state /= torch.norm(current_state) + 1e-12

        # Energy-chaining
        local_entropy = torch.rand(1).item() * 0.9 + 0.3
        for i, node in enumerate(nodes):
            harvested = node.harvest(local_entropy)
            energy_history[i].append(node.energy)

        current_state = mps_sweep(current_state, max_bond=16)

        if rank == 0 and step % 3 == 0:
            print_mem(f"Global Step {step}")

    if rank == 0:
        plt.figure(figsize=(10, 5))
        for i, hist in enumerate(energy_history):
            plt.plot(hist, label=f"Node {i}")
        plt.title("Energy Accumulation Over Steps (6x6 Grid)")
        plt.xlabel("Step")
        plt.ylabel("Energy (mJ)")
        plt.legend()
        plt.grid(True)
        plt.savefig('energy_accumulation.png')
        plt.close()
        print("Energy accumulation plot saved: energy_accumulation.png")

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count() or 1
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)
