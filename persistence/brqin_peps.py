import torch
import numpy as np
from typing import List

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
        self.corner_tl = torch.eye(init_bond, dtype=torch.complex64, device=device)
        self.corner_tr = torch.eye(init_bond, dtype=torch.complex64, device=device)
        self.corner_bl = torch.eye(init_bond, dtype=torch.complex64, device=device)
        self.corner_br = torch.eye(init_bond, dtype=torch.complex64, device=device)

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

    @classmethod
    def from_checkpoint(cls, checkpoint: Dict):
        peps = cls(checkpoint['Lx'], checkpoint['Ly'], init_bond=12, device=torch.device('cpu'))
        peps.tensors = {eval(k) if isinstance(k, str) else k: v.to(peps.device) for k, v in checkpoint['tensors'].items()}
        peps.bond_map_h = checkpoint['bond_map_h']
        peps.bond_map_v = checkpoint['bond_map_v']
        return peps

    def hybrid_evolution_step(self, H_terms, entropy_delta: float, energy_node):
        if H_terms is None:
            H_terms = [[torch.tensor([0.0, -1.0], device=self.device) for _ in range(self.Ly)] for _ in range(self.Lx)]
        threshold = 0.08
        use_full = (entropy_delta > threshold) and energy_node.can_mutate(cost=0.15)
        if use_full:
            energy = self.full_update(H_terms)
            energy_node.spend(cost=0.15)
            mode = "Full Update"
        else:
            energy = self.simple_update(H_terms)
            mode = "Simple Update"
        return energy, mode

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

    def local_expectation(self, tensor, local_h):
        phys = tensor.mean(dim=[0,1,2,3])
        return (phys[0] * local_h).real.item()

    def boundary_mps_contraction(self, H_terms=None):
        energy = 0.0
        env = None
        entropy_grid = np.random.rand(self.Lx, self.Ly) * 0.9 + 0.2
        for r in range(self.Lx - 1, -1, -1):
            row_energy = 0.0
            for c in range(self.Ly):
                tensor = self.tensors[(r, c)]
                if H_terms is not None:
                    row_energy += self.local_expectation(tensor, H_terms[r][c])
                bond_h = self.bond_map_h.get(((r,c),(r,c+1)), 8) if c + 1 < self.Ly else 1
                mat = tensor.mean(dim=[0,2,3]).reshape(-1, tensor.shape[1])
                local_entropy = torch.abs(mat).mean().item()
                k = min(32, bond_h * 2) if local_entropy > 0.7 else 16
                U, S, Vh = self.rsvd(mat, k)
                cum_s2 = torch.cumsum(S**2, dim=0) / torch.sum(S**2 + 1e-12)
                keep = min(int(torch.sum(cum_s2 < 1 - 1e-9)) + 1, bond_h)
                U = U[:, :keep]
                S = S[:keep]
                Vh = Vh[:keep, :]
                env = env @ (U @ torch.diag(S) @ Vh) if env is not None else (U @ torch.diag(S) @ Vh)
            energy += row_energy
            self.update_corners_ctmrg(r)
        return energy

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

    def update_corners_ctmrg(self, row):
        bond = max(self.bond_map_h.values(), default=8)
        self.corner_tl = torch.randn(bond, bond, dtype=torch.complex64, device=self.device) / torch.norm(self.corner_tl)
        return

    def compute_observables(self):
        mag = 0.0
        for tensor in self.tensors.values():
            phys = tensor.mean(dim=[0,1,2,3])
            mag += phys[0].real.item()
        mag /= len(self.tensors)
        return {'magnetization': mag}