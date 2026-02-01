import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import psutil
import time

print("=== BrQin Extreme Bond Growth Stress Test (Very High Entropy Regimes) ===")

class BrQinExtremeGrowth:
    def __init__(self, Lx=12, Ly=12, init_bond=12):
        self.Lx = Lx
        self.Ly = Ly
        self.init_bond = init_bond
        self.max_bond = 128  # much higher limit for extreme growth
        self.bond_map_h = {}
        self.bond_map_v = {}
        self.growth_history = []
        self.energy_history = []
        self.vqe_energy_history = []
        self.entropy_history = []

        for r in range(Lx):
            for c in range(Ly):
                if c + 1 < Ly:
                    self.bond_map_h[((r,c),(r,c+1))] = init_bond
                if r + 1 < Lx:
                    self.bond_map_v[((r,c),(r+1,c))] = init_bond

    def compute_local_entropy(self):
        # Simulated high entropy regime
        entropy_h = np.log2(self.bond_map_h.values()) if self.bond_map_h else 0
        entropy_v = np.log2(self.bond_map_v.values()) if self.bond_map_v else 0
        avg_entropy = (np.mean(list(entropy_h)) + np.mean(list(entropy_v))) / 2 if entropy_h and entropy_v else 0
        self.entropy_history.append(avg_entropy)
        return avg_entropy

    def adaptive_bond_growth_extreme(self):
        growth = 0
        entropy = self.compute_local_entropy()

        # Force high growth in high entropy regime
        growth_prob = 0.8  # very high probability
        growth_amount = 8  # larger increments

        for r in range(self.Lx):
            for c in range(self.Ly):
                if np.random.rand() < growth_prob:
                    if c + 1 < self.Ly:
                        key = ((r,c),(r,c+1))
                        current = self.bond_map_h.get(key, self.init_bond)
                        new = min(current + growth_amount, self.max_bond)
                        if new != current:
                            self.bond_map_h[key] = new
                            growth += 1
                    if r + 1 < self.Lx:
                        key = ((r,c),(r+1,c))
                        current = self.bond_map_v.get(key, self.init_bond)
                        new = min(current + growth_amount, self.max_bond)
                        if new != current:
                            self.bond_map_v[key] = new
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

    def get_memory_mb(self):
        return psutil.Process().memory_info().rss / (1024**2)

    def run_extreme(self, steps=500):
        print(f"Starting extreme bond growth test ({steps} steps)...")
        start_time = time.time()

        for step in tqdm(range(steps), desc="Extreme growth"):
            growth = self.adaptive_bond_growth_extreme()
            energy = self.simulate_energy()
            vqe_energy = self.simulate_vqe_energy()
            avg_bond = self.get_avg_bond()
            mem = self.get_memory_mb()

            if step % 50 == 0:
                print(f"Step {step:3d} | Growth: {growth:3d} | Avg Bond: {avg_bond:.1f} | Energy: {energy:.4f} | Mem: {mem:.1f} MB")

        total_time = time.time() - start_time
        final_bond = self.get_avg_bond()
        final_mem = self.get_memory_mb()

        print("\n=== EXTREME GROWTH SUMMARY ===")
        print(f"Steps: {steps}")
        print(f"Final bond dim: {final_bond:.1f}")
        print(f"Avg growth rate: {np.mean(self.growth_history):.1f} bonds/step")
        print(f"Total runtime: {total_time:.1f} seconds")
        print(f"Final memory: {final_mem:.1f} MB")

        self.plot_extreme_growth()
        print("✅ Extreme growth test complete")

    def plot_extreme_growth(self):
        plt.figure(figsize=(14, 10))
        plt.subplot(2, 2, 1)
        plt.plot(self.growth_history, color='green')
        plt.title('Growth per Step (Extreme)')
        plt.xlabel('Step')
        plt.ylabel('Growth Count')
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(self.energy_history, color='blue')
        plt.title('Energy Trend')
        plt.xlabel('Step')
        plt.ylabel('Energy')
        plt.grid(True)

        plt.subplot(2, 2, 3)
        plt.plot(self.vqe_energy_history, color='red')
        plt.title('VQE Energy')
        plt.xlabel('Step')
        plt.ylabel('VQE Energy')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('extreme_growth_results.png', dpi=300)
        plt.close()
        print("📊 Extreme growth plots saved as 'extreme_growth_results.png'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BrQin Extreme Bond Growth Test")
    parser.add_argument("--steps", type=int, default=500, help="Number of steps")
    parser.add_argument("--Lx", type=int, default=12)
    parser.add_argument("--Ly", type=int, default=12)
    parser.add_argument("--bond", type=int, default=12)
    args = parser.parse_args()

    demo = BrQinExtremeGrowth(Lx=args.Lx, Ly=args.Ly, init_bond=args.bond)
    demo.run_extreme(steps=args.steps)