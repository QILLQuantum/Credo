# oracle_peps_sci.py
# EntroQin PEPS Oracle - BEDP for Physics Sims with Heisenberg Hamiltonian
# Latest: February 09, 2026

import numpy as np

class PepsOracleSci:
    def __init__(self, Lx=8, Ly=8, Lz=0, steps=50, init_bond=8, max_bond=48, J=1.0, use_gpu=False):
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz if Lz > 0 else 1  # Treat 2D as Lz=1
        self.steps = steps
        self.init_bond = init_bond
        self.max_bond = max_bond
        self.J = J

        # Simplified bond maps (horizontal, vertical, depth)
        self.horizontal_bonds = np.ones((Lx, Ly, self.Lz)) * init_bond
        self.vertical_bonds = np.ones((Lx, Ly, self.Lz)) * init_bond
        self.depth_bonds = np.ones((Lx, Ly, self.Lz - 1)) * init_bond if self.Lz > 1 else np.ones((Lx, Ly)) * init_bond

        self.entropies_h = np.random.normal(0.5, 0.2, self.horizontal_bonds.shape)
        self.entropies_v = np.random.normal(0.5, 0.2, self.vertical_bonds.shape)
        self.entropies_d = np.random.normal(0.5, 0.2, self.depth_bonds.shape) if self.Lz > 1 else np.random.normal(0.5, 0.2, (Lx, Ly))

        self.entropies_h = np.clip(self.entropies_h, 0, 1)
        self.entropies_v = np.clip(self.entropies_v, 0, 1)
        self.entropies_d = np.clip(self.entropies_d, 0, 1)

        print(f"EntroQin oracle initialized for {Lx}x{Ly}x{self.Lz} lattice with Heisenberg J={J}")

    def guided_trickle_growth(self):
        growth = 0
        all_entropies = np.concatenate([
            self.entropies_h.ravel(),
            self.entropies_v.ravel(),
            self.entropies_d.ravel()
        ])
        H_variance = np.var(all_entropies)

        mischief_factor = 1.5 if H_variance > 0.05 else 0.8  # CE/LP switch

        # Horizontal
        for x in range(self.Lx):
            for y in range(self.Ly):
                for z in range(self.Lz):
                    H_norm = self.entropies_h[x, y, z]
                    prob = min(0.95, 0.2 + 0.75 * H_norm * mischief_factor)
                    if np.random.rand() < prob:
                        curr = self.horizontal_bonds[x, y, z]
                        new = min(curr + 4, self.max_bond)
                        if new != curr:
                            self.horizontal_bonds[x, y, z] = new
                            growth += 1

        # Vertical
        for x in range(self.Lx):
            for y in range(self.Ly):
                for z in range(self.Lz):
                    H_norm = self.entropies_v[x, y, z]
                    prob = min(0.95, 0.2 + 0.75 * H_norm * mischief_factor)
                    if np.random.rand() < prob:
                        curr = self.vertical_bonds[x, y, z]
                        new = min(curr + 4, self.max_bond)
                        if new != curr:
                            self.vertical_bonds[x, y, z] = new
                            growth += 1

        # Depth (z-direction)
        for x in range(self.Lx):
            for y in range(self.Ly):
                for z in range(self.Lz - 1):
                    H_norm = self.entropies_d[x, y, z]
                    prob = min(0.95, 0.2 + 0.75 * H_norm * mischief_factor)
                    if np.random.rand() < prob:
                        curr = self.depth_bonds[x, y, z]
                        new = min(curr + 4, self.max_bond)
                        if new != curr:
                            self.depth_bonds[x, y, z] = new
                            growth += 1

        return growth

    def uniform_growth(self):
        growth = 0
        for x in range(self.Lx):
            for y in range(self.Ly):
                for z in range(self.Lz):
                    if np.random.rand() < 0.6:
                        curr = self.horizontal_bonds[x, y, z]
                        new = min(curr + 4, self.max_bond)
                        if new != curr:
                            self.horizontal_bonds[x, y, z] = new
                            growth += 1

                    if np.random.rand() < 0.6:
                        curr = self.vertical_bonds[x, y, z]
                        new = min(curr + 4, self.max_bond)
                        if new != curr:
                            self.vertical_bonds[x, y, z] = new
                            growth += 1

        if self.Lz > 1:
            for x in range(self.Lx):
                for y in range(self.Ly):
                    for z in range(self.Lz - 1):
                        if np.random.rand() < 0.6:
                            curr = self.depth_bonds[x, y, z]
                            new = min(curr + 4, self.max_bond)
                            if new != curr:
                                self.depth_bonds[x, y, z] = new
                                growth += 1
        return growth

    def compute_heisenberg_energy(self):
        """Approximate Heisenberg energy for 3D lattice."""
        # Average nearest-neighbor correlation proxy via bond dimensions
        avg_h = np.mean(self.horizontal_bonds)
        avg_v = np.mean(self.vertical_bonds)
        avg_d = np.mean(self.depth_bonds) if self.Lz > 1 else 0.0

        # Energy ~ -J * sum of neighbor correlations (simplified)
        total_neighbors = 2 * self.Lx * self.Ly * self.Lz + (self.Lz - 1) * self.Lx * self.Ly
        energy = -self.J * (avg_h * self.Lx * (self.Ly - 1) * self.Lz +
                            avg_v * (self.Lx - 1) * self.Ly * self.Lz +
                            avg_d * self.Lx * self.Ly * (self.Lz - 1)) / total_neighbors
        energy += np.random.normal(0, 0.01)  # Noise
        return energy

    def get_avg_bond_grid(self):
        """Average bond dimension grid for visualization (xy slice at z=0)."""
        grid = np.zeros((self.Lx, self.Ly))
        for x in range(self.Lx):
            for y in range(self.Ly):
                grid[x, y] = (self.horizontal_bonds[x, y, 0] + self.vertical_bonds[x, y, 0]) / 2
        return grid

    def run(self, guided=False):
        for step in range(self.steps):
            if guided:
                growth = self.guided_trickle_growth()
            else:
                growth = self.uniform_growth()
            energy = self.compute_heisenberg_energy()
            avg_bond = np.mean([np.mean(self.horizontal_bonds), np.mean(self.vertical_bonds), np.mean(self.depth_bonds) if self.Lz > 1 else 0])
            print(f"Step {step:3d} | Energy: {energy:.4f} | Avg Bond: {avg_bond:.2f} | Growth: {growth}")

        final_avg_bond = np.mean([np.mean(self.horizontal_bonds), np.mean(self.vertical_bonds), np.mean(self.depth_bonds) if self.Lz > 1 else 0])
        final_variance = np.var(np.concatenate([self.entropies_h.ravel(), self.entropies_v.ravel(), self.entropies_d.ravel()]))

        return {
            "energy": energy,
            "avg_bond": final_avg_bond,
            "entropy_variance": final_variance
        }

if __name__ == "__main__":
    oracle = PepsOracleSci()
    metrics = oracle.run(guided=True)
    print("\nFinal Metrics:")
    print(metrics)