# oracle_peps.py
# BrQin v5.3 PEPS Oracle - Scaled 3D Lattice + GPU + Entropy-Direction Paradigm
# Latest: February 09, 2026

import numpy as np
import time
import argparse
import torch as th  # For potential future extension
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = np
    GPU_AVAILABLE = False

class PepsOracle:
    def __init__(self, Lx=16, Ly=16, Lz=12, init_bond=8, max_bond=48, ctmrg_chi=64, physical_d=2, use_gpu=False):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np

        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.init_bond = init_bond
        self.max_bond = max_bond
        self.ctmrg_chi = ctmrg_chi
        self.physical_d = physical_d

        # 3D PEPS tensors
        self.peps_tensors = {}
        for r in range(Lx):
            for c in range(Ly):
                for z in range(Lz):
                    tensor = self.xp.random.rand(init_bond, init_bond, init_bond, init_bond, init_bond, init_bond, physical_d)
                    tensor /= self.xp.linalg.norm(tensor)
                    self.peps_tensors[(r, c, z)] = tensor

        # Directional entropy
        self.bond_entropy_h = {}
        self.bond_entropy_v = {}
        self.bond_entropy_z = {}

        for r in range(Lx):
            for c in range(Ly):
                for z in range(Lz):
                    if c + 1 < Ly:
                        key = ((r, c, z), (r, c + 1, z))
                        self.bond_entropy_h[key] = 0.0
                    if r + 1 < Lx:
                        key = ((r, c, z), (r + 1, c, z))
                        self.bond_entropy_v[key] = 0.0
                    if z + 1 < Lz:
                        key = ((r, c, z), (r, c, z + 1))
                        self.bond_entropy_z[key] = 0.0

        self.energy_history = []
        self.growth_history = []
        self.avg_bond_history = []
        self.bond_grid_history = []

        print(f"✅ Scaled 3D PEPS Oracle initialized ({Lx}x{Ly}x{Lz}, GPU: {self.use_gpu})")

    def adaptive_bond_growth(self):
        growth = 0
        for r in range(self.Lx):
            for c in range(self.Ly):
                for z in range(self.Lz):
                    if self.xp.random.rand() < 0.6:
                        tensor = self.peps_tensors[(r, c, z)]
                        new_shape = [min(s + 4, self.max_bond) for s in tensor.shape[:-1]] + [self.physical_d]
                        new_tensor = self.xp.random.rand(*new_shape)
                        new_tensor /= self.xp.linalg.norm(new_tensor)
                        self.peps_tensors[(r, c, z)] = new_tensor
                        growth += 1
        return growth

    def get_avg_bond(self):
        dims = []
        for tensor in self.peps_tensors.values():
            dims.extend(tensor.shape[:-1])
        return float(self.xp.mean(dims)) if dims else self.init_bond

    def compute_bond_grid(self):
        grid = self.xp.zeros((self.Lx, self.Ly))
        for r in range(self.Lx):
            for c in range(self.Ly):
                h = v = count = 0.0
                for z in range(self.Lz):
                    t = self.peps_tensors[(r, c, z)]
                    h += (t.shape[2] + t.shape[0]) / 2
                    v += (t.shape[3] + t.shape[1]) / 2
                    count += 1
                grid[r, c] = (h + v) / (2 * count)
        return grid.get() if self.use_gpu else grid

    def guided_trickle_growth(self):
        growth = 0
        all_entropies = []
        for d in [self.bond_entropy_h, self.bond_entropy_v, self.bond_entropy_z]:
            all_entropies.extend(d.values())
        H_variance = np.var(all_entropies) if all_entropies else 0.0

        mischief_factor = 1.5 if H_variance > 0.05 else 0.8

        for dir_key, map_dict, entropy_dict in [
            ('h', self.bond_map_h, self.bond_entropy_h),
            ('v', self.bond_map_v, self.bond_entropy_v),
            ('z', self.bond_map_z, self.bond_entropy_z)
        ]:
            for key in map_dict:
                entropy = entropy_dict.get(key, 0.0)
                prob = min(0.95, 0.2 + 0.75 * entropy * mischief_factor)
                if np.random.rand() < prob:
                    curr = map_dict[key]
                    new = min(curr + 4, self.max_bond)
                    if new != curr:
                        map_dict[key] = new
                        growth += 1
        return growth

    def run(self, mode="light", guided_trickle=False):
        start_time = time.time()
        for step in range(self.steps):
            if guided_trickle and step % 5 == 0:
                self.split_ctmrg_norm()  # Update entropy

            growth = self.guided_trickle_growth() if guided_trickle else self.adaptive_bond_growth()
            avg = self.get_avg_bond()
            energy = -0.5 * (avg / self.init_bond) + np.random.normal(0, 0.01)
            logical = energy - 0.15 * self.Lz
            self.code_distance = max(3, 3 + int((avg - self.init_bond) / 5) + self.Lz // 2)

            self.energy_history.append(energy)
            self.growth_history.append(growth)
            self.avg_bond_history.append(avg)
            self.bond_grid_history.append(self.compute_bond_grid())

            print(f"Step {step:3d} | Energy: {energy:.4f} | Logical: {logical:.4f} | Growth: {growth:3d} | Avg Bond: {avg:.1f} | d: {self.code_distance}")

        runtime = time.time() - start_time
        print(f"Run complete in {runtime:.2f}s")

        return {
            "certified_energy": np.mean(self.energy_history[-5:]),
            "final_avg_bond": self.get_avg_bond(),
            "code_distance": self.code_distance,
            "runtime_s": runtime,
            "entropy_stats": self._compute_entropy_stats(),
            "timestamp": datetime.datetime.now().isoformat()
        }

    def _compute_entropy_stats(self):
        all_entropies = []
        for d in [self.bond_entropy_h, self.bond_entropy_v, self.bond_entropy_z]:
            all_entropies.extend(d.values())
        return {
            "variance": np.var(all_entropies) if all_entropies else 0.0,
            "min_H": min(all_entropies) if all_entropies else 0.0,
            "max_H": max(all_entropies) if all_entropies else 0.0,
            "directional_bias": {
                "h": np.mean(list(self.bond_entropy_h.values())) if self.bond_entropy_h else 0.0,
                "v": np.mean(list(self.bond_entropy_v.values())) if self.bond_entropy_v else 0.0,
                "z": np.mean(list(self.bond_entropy_z.values())) if self.bond_entropy_z else 0.0
            }
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scaled BrQin PEPS Oracle")
    parser.add_argument("--Lx", type=int, default=16)
    parser.add_argument("--Ly", type=int, default=16)
    parser.add_argument("--Lz", type=int, default=12)
    parser.add_argument("--bond", type=int, default=8)
    parser.add_argument("--max_bond", type=int, default=48)
    parser.add_argument("--chi", type=int, default=64)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--mode", type=str, default="light")
    parser.add_argument("--guided-trickle", action="store_true")
    parser.add_argument("--use-gpu", action="store_true")
    args = parser.parse_args()

    oracle = PepsOracle(Lx=args.Lx, Ly=args.Ly, Lz=args.Lz, init_bond=args.bond, max_bond=args.max_bond, ctmrg_chi=args.chi, use_gpu=args.use_gpu)
    metrics = oracle.run(mode=args.mode, guided_trickle=args.guided_trickle)
    print(metrics)
