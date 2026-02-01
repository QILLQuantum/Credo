import torch
import numpy as np
from typing import List, Dict

class EntanglementEnergyNode:
    def __init__(self, name: str):
        self.name = name
        self.energy = 0.0

    def harvest(self, local_entropy: float):
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
    def __init__(self, Lx: int, Ly: int, init_bond: int = 12, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.Lx = Lx
        self.Ly = Ly
        self.device = device
        self.tensors = {}
        self.bond_map_h = {}
        self.bond_map_v = {}

        for r in range(Lx):
            for c in range(Ly):
                tensor = torch.randn(init_bond, init_bond, init_bond, init_bond, 2, dtype=torch.complex64, device=device)
                self.tensors[(r, c)] = tensor / torch.norm(tensor)

        for r in range(Lx):
            for c in range(Ly):
                if c + 1 < Ly:
                    self.bond_map_h[((r,c),(r,c+1))] = init_bond
                if r + 1 < Lx:
                    self.bond_map_v[((r,c),(r+1,c))] = init_bond

    def simple_update(self, H_terms):
        energy = 0.0
        for r in range(self.Lx):
            for c in range(self.Ly):
                tensor = self.tensors[(r, c)]
                energy += self.local_expectation(tensor, H_terms[r][c])
        return energy

    def full_update(self, H_terms, lr=0.1, steps=20):
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
            # Soft normalization
            for t in self.tensors.values():
                norm = torch.norm(t.data)
                if norm > 1e-8:
                    t.data /= norm ** 0.7
        return loss.item()

    def local_expectation(self, tensor, local_h):
        phys = tensor.mean(dim=[0,1,2,3])
        expectation = (phys[0] * local_h[0] + phys[1] * local_h[1]).real.item()
        return expectation

    def boundary_mps_contraction(self, H_terms=None):
        energy = 0.0
        for r in range(self.Lx):
            for c in range(self.Ly):
                tensor = self.tensors[(r, c)]
                if H_terms is not None:
                    energy += self.local_expectation(tensor, H_terms[r][c])
        return energy

    def adaptive_bond_growth(self, energy_node):
        for r in range(self.Lx):
            for c in range(self.Ly):
                tensor = self.tensors[(r, c)]
                mat = tensor.mean(dim=[0,2,3]).reshape(-1, tensor.shape[1])
                local_entropy = torch.abs(mat).mean().item()

                if local_entropy > 0.7 and energy_node.can_mutate(cost=0.2):
                    if c + 1 < self.Ly:
                        key = ((r,c),(r,c+1))
                        self.bond_map_h[key] = min(self.bond_map_h.get(key, 12) + 2, 24)
                    if r + 1 < self.Lx:
                        key = ((r,c),(r+1,c))
                        self.bond_map_v[key] = min(self.bond_map_v.get(key, 12) + 2, 24)
                    energy_node.spend(cost=0.2)

    def hybrid_evolution_step(self, H_terms, entropy_delta: float, energy_node):
        if H_terms is None:
            H_terms = [[torch.tensor([1.0, 0.0], device=self.device) for _ in range(self.Ly)] for _ in range(self.Lx)]
        threshold = 0.08
        use_full = (entropy_delta > threshold) and energy_node.can_mutate(cost=0.15)
        if use_full:
            energy = self.full_update(H_terms, lr=0.1, steps=20)
            energy_node.spend(cost=0.15)
            mode = "Full Update"
        else:
            energy = self.simple_update(H_terms)
            mode = "Simple Update"

        self.adaptive_bond_growth(energy_node)
        return energy, mode

    def compute_observables(self):
        mag = 0.0
        for tensor in self.tensors.values():
            phys = tensor.mean(dim=[0,1,2,3])
            mag += phys[0].real.item()
        mag /= len(self.tensors)
        return {'magnetization': mag}
