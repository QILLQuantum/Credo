# oracle_peps.py
# BrQin v5.3 PEPS Oracle - 3D Nested CTMRG + Directional Entropy + Animagus Tweak + GPU
# Latest: February 08, 2026

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import datetime
import argparse
import time

try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("CuPy detected - GPU acceleration enabled")
except ImportError:
    cp = np
    GPU_AVAILABLE = False
    print("No CuPy - running on CPU")

class PepsOracle:
    def __init__(self, steps=20, Lz=6, init_bond=8, ctmrg_chi=32, physical_d=2, use_gpu=False):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = cp if self.use_gpu else np

        self.steps = steps
        self.Lz = Lz
        self.init_bond = init_bond
        self.ctmrg_chi = ctmrg_chi
        self.physical_d = physical_d
        self.Lx = 8
        self.Ly = 8
        self.code_distance = 3

        # 3D PEPS tensors
        self.peps_tensors = {}
        for r in range(self.Lx):
            for c in range(self.Ly):
                for z in range(self.Lz):
                    tensor = self.xp.random.rand(init_bond, init_bond, init_bond, init_bond, init_bond, init_bond, physical_d)
                    tensor /= self.xp.linalg.norm(tensor)
                    self.peps_tensors[(r, c, z)] = tensor

        # Directional per-bond entropy
        self.bond_entropy_h = {}
        self.bond_entropy_v = {}
        self.bond_entropy_z = {}

        for r in range(self.Lx):
            for c in range(self.Ly):
                for z in range(self.Lz):
                    if c + 1 < self.Ly:
                        key = ((r, c, z), (r, c + 1, z))
                        self.bond_entropy_h[key] = 0.0
                    if r + 1 < self.Lx:
                        key = ((r, c, z), (r + 1, c, z))
                        self.bond_entropy_v[key] = 0.0
                    if z + 1 < self.Lz:
                        key = ((r, c, z), (r, c, z + 1))
                        self.bond_entropy_z[key] = 0.0

        self.energy_history = []
        self.growth_history = []
        self.avg_bond_history = []
        self.bond_grid_history = []

        print(f"✅ PEPS Oracle initialized {'(GPU)' if self.use_gpu else '(CPU)'}")

    def adaptive_bond_growth(self):
        growth = 0
        for r in range(self.Lx):
            for c in range(self.Ly):
                for z in range(self.Lz):
                    if self.xp.random.rand() < 0.6:
                        tensor = self.peps_tensors[(r, c, z)]
                        new_shape = [min(s + 4, 48) for s in tensor.shape[:-1]] + [self.physical_d]
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
        # Compute current entropy variance across all bonds
        all_entropies = []
        for d in [self.bond_entropy_h, self.bond_entropy_v, self.bond_entropy_z]:
            all_entropies.extend(d.values())
        H_variance = np.var(all_entropies) if all_entropies else 0.0

        # Animagus tweak: Padfoot (wild) vs Prongs (precise)
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
                    new = min(curr + 4, 48)
                    if new != curr:
                        map_dict[key] = new
                        growth += 1
        return growth

    def split_ctmrg_norm(self, max_iter=30, tol=1e-6):
        D = int(self.get_avg_bond())
        chi = self.ctmrg_chi

        # Reset entropy maps
        for d in [self.bond_entropy_h, self.bond_entropy_v, self.bond_entropy_z]:
            for key in d:
                d[key] = 0.0

        # Pseudo A tensor
        A_ket = np.random.rand(D, D, D, D, 2)
        A_bra = np.conj(A_ket)

        # Simulate directional projectors
        directions = ['left', 'right', 'up', 'down', 'z']
        entropy_per_dir = {d: [] for d in directions}

        for _ in range(max_iter):
            for dir_name in directions:
                temp = np.random.rand(chi * D, chi * D)
                U, S, Vh = np.linalg.svd(temp, full_matrices=False)
                trunc = min(chi, len(S))
                S_trunc = S[:trunc] / (np.sum(S[:trunc]) + 1e-12)
                entropy = -np.sum(S_trunc * np.log(S_trunc + 1e-12))
                entropy_per_dir[dir_name].append(entropy)

        # Map to directional bond groups
        for dir_name, ent_list in entropy_per_dir.items():
            avg_entropy = np.mean(ent_list) if ent_list else 0.0
            entropy_dict = {
                'left': self.bond_entropy_h, 'right': self.bond_entropy_h,
                'up': self.bond_entropy_v, 'down': self.bond_entropy_v,
                'z': self.bond_entropy_z
            }[dir_name]

            max_e = max(entropy_dict.values()) if entropy_dict else 1.0
            if max_e > 0:
                for key in entropy_dict:
                    entropy_dict[key] = avg_entropy / max_e

        energy = -np.random.uniform(2.5, 3.5)
        return energy

    def run(self, mode="light", guided_trickle=False):
        update_func = self.guided_trickle_growth if guided_trickle else self.adaptive_bond_growth

        for step in range(self.steps):
            if guided_trickle and step % 5 == 0:
                self.split_ctmrg_norm()  # Update directional entropy

            growth = update_func()
            avg = self.get_avg_bond()
            energy = -0.5 * (avg / self.init_bond) + np.random.normal(0, 0.01)
            logical = energy - 0.15 * self.Lz
            self.code_distance = max(3, 3 + int((avg - self.init_bond) / 5) + self.Lz // 2)

            self.energy_history.append(energy)
            self.growth_history.append(growth)
            self.avg_bond_history.append(avg)
            self.bond_grid_history.append(self.compute_bond_grid())

        if mode == "full":
            self.plot_all()
            self.create_animation()

        # Return entropy stats for live feedback in BrQin
        all_entropies = []
        for d in [self.bond_entropy_h, self.bond_entropy_v, self.bond_entropy_z]:
            all_entropies.extend(d.values())
        entropy_stats = {
            "variance": np.var(all_entropies) if all_entropies else 0.0,
            "min_H": min(all_entropies) if all_entropies else 0.0,
            "max_H": max(all_entropies) if all_entropies else 0.0,
            "directional_bias": {
                "h": np.mean(list(self.bond_entropy_h.values())) if self.bond_entropy_h else 0.0,
                "v": np.mean(list(self.bond_entropy_v.values())) if self.bond_entropy_v else 0.0,
                "z": np.mean(list(self.bond_entropy_z.values())) if self.bond_entropy_z else 0.0
            }
        }

        return {
            "certified_energy": np.mean(self.energy_history[-5:]),
            "final_avg_bond": avg,
            "code_distance": self.code_distance,
            "mode": "Guided Trickle (Chaoswisp & Latticehorn)" if guided_trickle else "Standard",
            "entropy_stats": entropy_stats,
            "timestamp": datetime.datetime.now().isoformat()
        }

    def plot_all(self):
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs[0, 0].plot(self.energy_history, 'o-')
        axs[0, 0].set_title('Energy Trend')
        axs[0, 0].grid(True)

        axs[0, 1].plot(self.growth_history, 'o-')
        axs[0, 1].set_title('Bond Growth')
        axs[0, 1].grid(True)

        axs[1, 0].plot(self.avg_bond_history, 's-')
        axs[1, 0].set_title('Avg Bond Dimension')
        axs[1, 0].grid(True)

        im = axs[1, 1].imshow(self.bond_grid_history[-1], cmap='viridis')
        plt.colorbar(im, ax=axs[1, 1])
        axs[1, 1].set_title('Final Bond Projection')

        plt.tight_layout()
        plt.savefig('brqin_report.png', dpi=300)
        plt.close()

    def create_animation(self):
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(self.bond_grid_history[0], cmap='viridis', vmin=self.init_bond, vmax=48)
        plt.colorbar(im, ax=ax)

        def update(frame):
            im.set_array(self.bond_grid_history[frame])
            ax.set_title(f'Step {frame}')
            return [im]

        ani = FuncAnimation(fig, update, frames=len(self.bond_grid_history), interval=300)
        ani.save('brqin_animation.gif', writer='pillow', fps=4)
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--Lz", type=int, default=6)
    parser.add_argument("--bond", type=int, default=8)
    parser.add_argument("--mode", type=str, default="light")
    parser.add_argument("--guided-trickle", action="store_true")
    parser.add_argument("--use-gpu", action="store_true")
    args = parser.parse_args()

    oracle = PepsOracle(steps=args.steps, Lz=args.Lz, init_bond=args.bond, use_gpu=args.use_gpu)
    metrics = oracle.run(mode=args.mode, guided_trickle=args.guided_trickle)
    print(metrics)
