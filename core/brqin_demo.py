import torch
import numpy as np
import matplotlib.pyplot as plt

print("=== BrQin Demo: Energy-Triggered Adaptive Bond Growth ===")

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

class BrQinDemo:
    def __init__(self, Lx=12, Ly=12, init_bond=12):
        self.Lx = Lx
        self.Ly = Ly
        self.bond_map_h = {}
        self.bond_map_v = {}
        self.init_bond = init_bond
        self.growth_history = []

        for r in range(Lx):
            for c in range(Ly):
                if c + 1 < Ly:
                    self.bond_map_h[((r,c),(r,c+1))] = init_bond
                if r + 1 < Lx:
                    self.bond_map_v[((r,c),(r+1,c))] = init_bond

    def adaptive_bond_growth(self, energy_node):
        growth_count = 0
        for r in range(self.Lx):
            for c in range(self.Ly):
                # Simulate local entropy
                local_entropy = np.random.rand() * 0.9 + 0.2
                energy_node.harvest(local_entropy)

                if local_entropy > 0.7 and energy_node.can_mutate(cost=0.2):
                    if c + 1 < self.Ly:
                        key = ((r,c),(r,c+1))
                        self.bond_map_h[key] = min(self.bond_map_h[key] + 4, 32)
                        growth_count += 1
                    if r + 1 < self.Lx:
                        key = ((r,c),(r+1,c))
                        self.bond_map_v[key] = min(self.bond_map_v[key] + 4, 32)
                        growth_count += 1
                    energy_node.spend(cost=0.2)
        self.growth_history.append(growth_count)
        return growth_count

    def get_avg_bond(self):
        total = sum(self.bond_map_h.values()) + sum(self.bond_map_v.values())
        count = len(self.bond_map_h) + len(self.bond_map_v)
        return total / count if count > 0 else self.init_bond

    def plot_growth(self):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.growth_history, marker='o')
        plt.title('Bonds Grown per Step')
        plt.xlabel('Step')
        plt.ylabel('Growth Count')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        bond_grid = np.zeros((self.Lx, self.Ly))
        for r in range(self.Lx):
            for c in range(self.Ly):
                h = self.bond_map_h.get(((r,c),(r,c+1)), self.init_bond)
                v = self.bond_map_v.get(((r,c),(r+1,c)), self.init_bond)
                bond_grid[r, c] = (h + v) / 2
        plt.imshow(bond_grid, cmap='viridis')
        plt.colorbar(label='Avg Bond Dim')
        plt.title('Final Bond Dimension Heatmap')
        plt.tight_layout()
        plt.savefig('brqin_bond_growth.png', dpi=300)
        plt.close()
        print("📊 Growth plot saved as 'brqin_bond_growth.png'")

# Run
demo = BrQinDemo()
node = EntanglementEnergyNode("MainNode")

print(f"Initial avg bond: {demo.get_avg_bond():.1f}")

for step in range(12):
    growth = demo.adaptive_bond_growth(node)
    avg = demo.get_avg_bond()
    print(f"Step {step:2d} | Growth: {growth:2d} bonds | Avg Bond: {avg:.1f}")

demo.plot_growth()
print("✅ Demo complete")
