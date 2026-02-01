import numpy as np
import matplotlib.pyplot as plt
import sys
import traceback

class SelfHealingBrQin:
    def __init__(self, Lx=12, Ly=12, init_bond=12):
        self.Lx = Lx
        self.Ly = Ly
        self.init_bond = init_bond
        self.bond_map_h = {}
        self.bond_map_v = {}
        self.growth_history = []
        self.error_count = 0

        print("🚀 Self-Healing BrQin Demo Starting...")

        # Initialize bonds
        for r in range(Lx):
            for c in range(Ly):
                if c + 1 < Ly:
                    self.bond_map_h[((r,c),(r,c+1))] = init_bond
                if r + 1 < Lx:
                    self.bond_map_v[((r,c),(r+1,c))] = init_bond

    def adaptive_bond_growth(self):
        try:
            growth_count = 0
            for r in range(self.Lx):
                for c in range(self.Ly):
                    local_entropy = np.random.rand() * 0.9 + 0.2
                    if local_entropy > 0.7:
                        if c + 1 < self.Ly:
                            key = ((r,c),(r,c+1))
                            self.bond_map_h[key] = min(self.bond_map_h.get(key, self.init_bond) + 4, 32)
                            growth_count += 1
                        if r + 1 < self.Lx:
                            key = ((r,c),(r+1,c))
                            self.bond_map_v[key] = min(self.bond_map_v.get(key, self.init_bond) + 4, 32)
                            growth_count += 1
            self.growth_history.append(growth_count)
            return growth_count
        except Exception as e:
            self.error_count += 1
            print(f"⚠️ Growth error (self-healing): {e}")
            return 0

    def get_avg_bond(self):
        try:
            total = sum(self.bond_map_h.values()) + sum(self.bond_map_v.values())
            count = len(self.bond_map_h) + len(self.bond_map_v)
            return total / count if count > 0 else self.init_bond
        except:
            return self.init_bond

    def plot_heatmap(self):
        try:
            bond_grid = np.zeros((self.Lx, self.Ly))
            for r in range(self.Lx):
                for c in range(self.Ly):
                    h = self.bond_map_h.get(((r,c),(r,c+1)), self.init_bond) if c + 1 < self.Ly else self.init_bond
                    v = self.bond_map_v.get(((r,c),(r+1,c)), self.init_bond) if r + 1 < self.Lx else self.init_bond
                    bond_grid[r, c] = (h + v) / 2

            plt.figure(figsize=(10, 8))
            plt.imshow(bond_grid, cmap='viridis', interpolation='nearest')
            plt.colorbar(label='Average Bond Dimension')
            plt.title('Final Bond Dimension Heatmap')
            plt.xlabel('Column')
            plt.ylabel('Row')
            for i in range(self.Lx):
                for j in range(self.Ly):
                    plt.text(j, i, f'{bond_grid[i,j]:.0f}', ha='center', va='center', color='white', fontsize=9)
            plt.tight_layout()
            plt.savefig('bond_dimension_heatmap.png', dpi=300)
            plt.close()
            print("📊 Heatmap saved successfully")
        except Exception as e:
            print(f"⚠️ Plot error (self-healing): {e}")

    def run(self, steps=12):
        print(f"Initial avg bond: {self.get_avg_bond():.1f}")
        for step in range(steps):
            growth = self.adaptive_bond_growth()
            avg = self.get_avg_bond()
            print(f"Step {step:2d} | Growth: {growth:2d} bonds | Avg Bond: {avg:.1f}")

        self.plot_heatmap()
        print(f"✅ Self-Healing Demo complete (errors: {self.error_count})")
        print("📊 Files saved: bond_dimension_heatmap.png")

# Run the self-healing demo
if __name__ == "__main__":
    demo = SelfHealingBrQin(Lx=12, Ly=12, init_bond=12)
    demo.run(steps=12)