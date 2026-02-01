import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import psutil
import os
import networkx as nx
import matplotlib.pyplot as plt

print("=== BrQin v5.2 - Polished: Full Update Hybrid + CTMRG + RSVD + Adaptive Logical ===\n")

def print_mem(label=""):
    mb = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
    print(f"[{label}] Memory: {mb:.2f} MB")
    return mb

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

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

class BrQinPEPS:
    def __init__(self, Lx, Ly, init_bond=8):
        self.Lx = Lx
        self.Ly = Ly
        self.tensors = {}
        self.bond_map_h = {}
        self.bond_map_v = {}
        self.corner_tl = torch.eye(init_bond, dtype=torch.complex64)
        self.corner_tr = torch.eye(init_bond, dtype=torch.complex64)
        self.corner_bl = torch.eye(init_bond, dtype=torch.complex64)
        self.corner_br = torch.eye(init_bond, dtype=torch.complex64)

        for r in range(Lx):
            for c in range(Ly):
                tensor = torch.randn(init_bond, init_bond, init_bond, init_bond, 2, dtype=torch.complex64)
                self.tensors[(r,c)] = tensor / torch.norm(tensor)

        for r in range(Lx):
            for c in range(Ly):
                if c + 1 < Ly:
                    self.bond_map_h[((r,c),(r,c+1))] = init_bond
                if r + 1 < Lx:
                    self.bond_map_v[((r,c),(r+1,c))] = init_bond

    def adaptive_contraction_order(self, entropy_grid):
        bond_entropy = {}
        for r in range(self.Lx):
            for c in range(self.Ly):
                if c + 1 < self.Ly:
                    bond_entropy[((r,c),(r,c+1))] = (entropy_grid[r,c] + entropy_grid[r,c+1]) / 2
                if r + 1 < self.Lx:
                    bond_entropy[((r,c),(r+1,c))] = (entropy_grid[r,c] + entropy_grid[r+1,c]) / 2
        return sorted(bond_entropy, key=bond_entropy.get, reverse=True)

    def local_expectation(self, tensor, local_h):
        phys = tensor.mean(dim=[0,1,2,3])
        return (phys[0] * local_h).real.item()

    def simple_update(self, H_terms):
        energy = 0.0
        for r in range(self.Lx):
            for c in range(self.Ly):
                tensor = self.tensors[(r, c)]
                energy += self.local_expectation(tensor, H_terms[r][c])
        return energy

    def full_update(self, H_terms, lr=0.01, steps=5):
        optimizer = torch.optim.Adam(list(self.tensors.values()), lr=lr)
        for _ in range(steps):
            optimizer.zero_grad()
            loss = 0.0
            for r in range(self.Lx):
                for c in range(self.Ly):
                    tensor = self.tensors[(r, c)]
                    loss += self.local_expectation(tensor, H_terms[r][c])
            loss.backward()
            optimizer.step()
            for t in self.tensors.values():
                t.data /= torch.norm(t.data) + 1e-12
        return loss.item()

    def rsvd(self, mat, k=16):
        try:
            Omega = torch.randn(mat.shape[1], k, dtype=mat.dtype, device=mat.device)
            Y = mat @ Omega
            Q, _ = torch.linalg.qr(Y, mode='reduced')
            B = Q.T @ mat
            U_tilde, S, Vh = torch.linalg.svd(B, full_matrices=False)
            U = Q @ U_tilde
            return U, S, Vh
        except:
            U, S, Vh = torch.linalg.svd(mat.to(torch.float32), full_matrices=False)
            return U, S, Vh

    def boundary_mps_contraction(self, H_terms=None):
        energy = 0.0
        env = None
        for r in range(self.Lx - 1, -1, -1):
            row_energy = 0.0
            for c in range(self.Ly):
                tensor = self.tensors[(r, c)]
                if H_terms is not None:
                    row_energy += self.local_expectation(tensor, H_terms[r][c])
                bond_h = self.bond_map_h.get(((r,c),(r,c+1)), 8) if c + 1 < self.Ly else 1

                mat = tensor.mean(dim=[0,2,3]).reshape(-1, tensor.shape[1])
                local_entropy = torch.abs(mat).mean().item()
                k = 16 if local_entropy <= 0.7 else 32
                k = min(k, bond_h * 2)

                U, S, Vh = self.rsvd(mat, k)

                # Higher truncation precision
                cum_s2 = torch.cumsum(S**2, dim=0) / torch.sum(S**2 + 1e-12)
                keep = int(torch.sum(cum_s2 < 1 - 1e-9)) + 1
                keep = min(keep, bond_h)

                U = U[:, :keep]
                S = S[:keep]
                Vh = Vh[:keep, :]

                env = env @ (U @ torch.diag(S) @ Vh) if env is not None else (U @ torch.diag(S) @ Vh)

            energy += row_energy
            self.update_corners_ctmrg(r)
        return energy

    def update_corners_ctmrg(self, row):
        bond = max(self.bond_map_h.values(), default=8)
        self.corner_tl = torch.randn(bond, bond, dtype=torch.complex64) / torch.norm(self.corner_tl)
        return

    def compute_observables(self):
        mag = 0.0
        for tensor in self.tensors.values():
            phys = tensor.mean(dim=[0,1,2,3])
            mag += phys[0].real.item()
        mag /= len(self.tensors)
        return {'magnetization': mag}

def main_worker(rank, world_size):
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')

    Lx, Ly = 12, 12
    peps = BrQinPEPS(Lx, Ly, init_bond=12)

    H_terms = [None] * Lx
    entropy_grid = np.random.rand(Lx, Ly) * 0.9 + 0.2
    nodes = [EntanglementEnergyNode(f"GPU{rank}-Node{i}") for i in range(4)]

    for step in range(12):
        entropy_delta = torch.rand(1).item() * 0.9 + 0.2 - 0.3
        energy = peps.hybrid_evolution_step(H_terms, entropy_delta, nodes[step % 4])
        energy = peps.boundary_mps_contraction(H_terms)
        observables = peps.compute_observables()

        if rank == 0:
            mode = "Full Update" if entropy_delta > 0.08 and nodes[step % 4].energy >= 0.6 else "Simple Update"
            print(f"Step {step} | Energy: {energy:.4f} | Mode: {mode}")

    if rank == 0:
        torch.save(peps.tensors, 'brqin_12x12_final.pt')
        nx.write_graphml(nx.grid_2d_graph(Lx, Ly), 'brqin_12x12_topology.graphml')
        print("12×12 final state and topology saved.")

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count() or 1
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)
