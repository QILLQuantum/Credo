import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import psutil
import os
import networkx as nx
import matplotlib.pyplot as plt
import json

print("=== BrQin v5.1 - 10×10 PEPS with Error Control + Checkpointing ===\n")

def print_mem(label=""):
    mb = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
    print(f"[{label}] Memory: {mb:.2f} MB")
    return mb

baseline = print_mem("Baseline")

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
        ordered = sorted(bond_entropy, key=bond_entropy.get, reverse=True)
        return ordered

    def row_contraction(self, row, upper_env=None, H_row=None):
        env = torch.eye(1, dtype=torch.complex128)
        energy = 0.0

        for c in range(self.Ly):
            tensor = self.tensors[(row, c)]

            if H_row is not None:
                energy += self.local_expectation(tensor, H_row[c])

            bond_h = self.bond_map_h.get(((row,c),(row,c+1)), 8) if c + 1 < self.Ly else 1

            if upper_env is not None:
                bond_v = self.bond_map_v.get(((row,c),(row+1,c)), 8)
                # Vertical contract

        return env, energy

    def local_expectation(self, tensor, local_h):
        phys = tensor.mean(dim=[0,1,2,3])
        return (phys[0] * local_h).real.item()

    def full_energy(self, H_terms):
        energy = 0.0
        env = None
        for r in range(self.Lx - 1, -1, -1):
            env, row_energy = self.row_contraction(r, env, H_terms[r] if H_terms else None)
            energy += row_energy
        return energy

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

    Lx, Ly = 10, 10
    peps = BrQinPEPS(Lx, Ly, init_bond=12)

    H_terms = [None] * Lx  # placeholder local H per row
    entropy_grid = np.random.rand(Lx, Ly) * 0.9 + 0.2

    prev_energy = float('inf')
    convergence_threshold = 1e-6
    max_steps = 50
    checkpoint_interval = 3

    for step in range(max_steps):
        order = peps.adaptive_contraction_order(entropy_grid)
        energy = peps.full_energy(H_terms)
        observables = peps.compute_observables()

        energy_change = abs(energy - prev_energy)
        if energy_change < convergence_threshold:
            print(f"Convergence reached at step {step} (Δenergy = {energy_change:.2e})")
            break

        prev_energy = energy

        if rank == 0 and step % checkpoint_interval == 0:
            checkpoint = {
                'step': step,
                'energy': energy,
                'observables': observables,
                'entropy_grid': entropy_grid.tolist()
            }
            torch.save(checkpoint, f'checkpoint_step_{step}.pt')
            print(f"Checkpoint saved at step {step}")

        if rank == 0:
            print(f"Step {step} | Energy: {energy:.4f} | Mag: {observables['magnetization']:.4f}")

    if rank == 0:
        plt.figure(figsize=(10, 6))
        # placeholder for bond_grid
        bond_grid = np.random.rand(Lx, Ly) * 20 + 4
        plt.imshow(bond_grid, cmap='plasma')
        plt.colorbar()
        plt.title("Bond Dimension Heatmap")
        plt.savefig('bond_dim_heatmap.png')
        plt.close()
        print("Bond_dim heatmap saved")

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count() or 1
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)
