import numpy as np
import matplotlib.pyplot as plt
import argparse

class BrQinDemo:
    def __init__(self, Lx=12, Ly=12, init_bond=12):
        self.Lx = Lx
        self.Ly = Ly
        self.init_bond = init_bond
        self.bond_map_h = {}
        self.bond_map_v = {}
        self.growth_history = []
        self.energy_history = []
        self.vqe_energy_history = []

        for r in range(Lx):
            for c in range(Ly):
                if c + 1 < Ly:
                    self.bond_map_h[((r,c),(r,c+1))] = init_bond
                if r + 1 < Lx:
                    self.bond_map_v[((r,c),(r+1,c))] = init_bond

    def adaptive_bond_growth(self):
        growth = 0
        for r in range(self.Lx):
            for c in range(self.Ly):
                if np.random.rand() < 0.3:
                    if c + 1 < self.Ly:
                        key = ((r,c),(r,c+1))
                        self.bond_map_h[key] = min(self.bond_map_h.get(key, self.init_bond) + 4, 32)
                        growth += 1
                    if r + 1 < self.Lx:
                        key = ((r,c),(r+1,c))
                        self.bond_map_v[key] = min(self.bond_map_v.get(key, self.init_bond) + 4, 32)
                        growth += 1
        self.growth_history.append(growth)
        return growth

    def simulate_energy(self):
        avg_bond = self.get_avg_bond()
        energy = -0.5 * (avg_bond / self.init_bond) + np.random.normal(0, 0.01)
        self.energy_history.append(energy)
        return energy

    def simulate_vqe_energy(self):
        avg_bond = self.get_avg_bond()
        vqe_energy = -0.8 * (avg_bond / self.init_bond) + np.random.normal(0, 0.02)
        self.vqe_energy_history.append(vqe_energy)
        return vqe_energy

    def get_avg_bond(self):
        total = sum(self.bond_map_h.values()) + sum(self.bond_map_v.values())
        count = len(self.bond_map_h) + len(self.bond_map_v)
        return total / count if count > 0 else self.init_bond

    def plot_all(self):
        plt.figure(figsize=(14, 10))
        plt.subplot(2, 3, 1)
        plt.plot(self.energy_history, marker='o', color='blue', linewidth=2)
        plt.title('Energy Trend')
        plt.xlabel('Step')
        plt.ylabel('Energy')
        plt.grid(True)

        plt.subplot(2, 3, 2)
        plt.plot(self.growth_history, marker='o', color='green', linewidth=2)
        plt.title('Bonds Grown per Step')
        plt.xlabel('Step')
        plt.ylabel('Growth Count')
        plt.grid(True)

        plt.subplot(2, 3, 3)
        bond_grid = np.zeros((self.Lx, self.Ly))
        for r in range(self.Lx):
            for c in range(self.Ly):
                h = self.bond_map_h.get(((r,c),(r,c+1)), self.init_bond) if c + 1 < self.Ly else self.init_bond
                v = self.bond_map_v.get(((r,c),(r+1,c)), self.init_bond) if r + 1 < self.Lx else self.init_bond
                bond_grid[r, c] = (h + v) / 2
        plt.imshow(bond_grid, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Avg Bond Dim')
        plt.title('Final Bond Dimension Heatmap')
        plt.xlabel('Column')
        plt.ylabel('Row')

        plt.subplot(2, 3, 4)
        plt.plot(self.vqe_energy_history, marker='d', color='red', linewidth=2)
        plt.title('VQE Energy')
        plt.xlabel('Step')
        plt.ylabel('VQE Energy')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('brqin_full_report.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Report saved as 'brqin_full_report.png'")

    def run(self, steps=12):
        print(f"Initial avg bond: {self.get_avg_bond():.1f}")

        for step in range(steps):
            growth = self.adaptive_bond_growth()
            energy = self.simulate_energy()
            vqe_energy = self.simulate_vqe_energy()
            avg_bond = self.get_avg_bond()
            print(f"Step {step:2d} | Energy: {energy:.6f} | VQE: {vqe_energy:.6f} | Growth: {growth:2d} | Avg Bond: {avg_bond:.1f}")

        self.plot_all()
        print("✅ Demo complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BrQin Demo")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--Lx", type=int, default=12)
    parser.add_argument("--Ly", type=int, default=12)
    parser.add_argument("--bond", type=int, default=12)
    args = parser.parse_args()

    demo = BrQinDemo(Lx=args.Lx, Ly=args.Ly, init_bond=args.bond)
    demo.run(steps=args.steps)