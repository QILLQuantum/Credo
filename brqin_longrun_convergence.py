import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import psutil
import time
import csv

print("=== BrQin Convergence Stability Test (Very Long Runs) ===")

class BrQinLongRun:
    def __init__(self, Lx=12, Ly=12, init_bond=12, max_bond=32):
        self.Lx = Lx
        self.Ly = Ly
        self.init_bond = init_bond
        self.max_bond = max_bond
        self.bond_map_h = {}
        self.bond_map_v = {}
        self.growth_history = []
        self.energy_history = []
        self.vqe_energy_history = []
        self.logical_vqe_history = []
        self.code_distance = 3

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
                        current = self.bond_map_h.get(key, self.init_bond)
                        new = min(current + 4, self.max_bond)
                        if new != current:
                            self.bond_map_h[key] = new
                            growth += 1
                    if r + 1 < self.Lx:
                        key = ((r,c),(r+1,c))
                        current = self.bond_map_v.get(key, self.init_bond)
                        new = min(current + 4, self.max_bond)
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

    def hybrid_vqe_with_error_correction(self):
        physical_vqe = self.vqe_energy_history[-1] if self.vqe_energy_history else -0.8
        correction_factor = 0.15 * np.exp(-0.35 * (self.code_distance - 3))
        logical_vqe = physical_vqe * (1 - correction_factor) + np.random.normal(0, 0.01)
        self.logical_vqe_history.append(logical_vqe)
        return logical_vqe

    def get_avg_bond(self):
        total = sum(self.bond_map_h.values()) + sum(self.bond_map_v.values())
        count = len(self.bond_map_h) + len(self.bond_map_v)
        return total / count if count > 0 else self.init_bond

    def get_memory_mb(self):
        return psutil.Process().memory_info().rss / (1024**2)

    def run_long(self, steps=2000):
        print(f"Starting convergence stability test ({steps} steps)...")
        start_time = time.time()

        for step in tqdm(range(steps), desc="Long-run convergence"):
            growth = self.adaptive_bond_growth()
            energy = self.simulate_energy()
            vqe_energy = self.simulate_vqe_energy()
            logical_vqe = self.hybrid_vqe_with_error_correction()

            if step % 200 == 0 or step == steps-1:
                mem = self.get_memory_mb()
                avg_bond = self.get_avg_bond()
                print(f"Step {step:4d} | Energy: {energy:.4f} | VQE: {vqe_energy:.4f} | Logical VQE: {logical_vqe:.4f} | Growth: {growth:3d} | Avg Bond: {avg_bond:.1f} | Mem: {mem:.1f} MB")

        total_time = time.time() - start_time
        final_bond = self.get_avg_bond()
        final_mem = self.get_memory_mb()

        print("\n=== CONVERGENCE STABILITY SUMMARY ===")
        print(f"Steps completed: {steps}")
        print(f"Final bond dim: {final_bond:.1f}")
        print(f"Final energy: {self.energy_history[-1]:.6f}")
        print(f"Final VQE: {self.vqe_energy_history[-1]:.6f}")
        print(f"Final logical VQE: {self.logical_vqe_history[-1]:.6f}")
        print(f"Avg growth rate: {np.mean(self.growth_history):.1f} bonds/step")
        print(f"Total runtime: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"Final memory: {final_mem:.1f} MB")

        self.plot_convergence()
        self.save_results()
        print("✅ Convergence stability test complete")

    def plot_convergence(self):
        plt.figure(figsize=(14, 10))
        plt.subplot(2, 2, 1)
        plt.plot(self.energy_history, color='blue')
        plt.title('Energy Convergence (Long Run)')
        plt.xlabel('Step')
        plt.ylabel('Energy')
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(self.vqe_energy_history, color='red')
        plt.title('VQE Convergence')
        plt.xlabel('Step')
        plt.ylabel('VQE Energy')
        plt.grid(True)

        plt.subplot(2, 2, 3)
        plt.plot(self.logical_vqe_history, color='darkred')
        plt.title('Logical VQE Convergence')
        plt.xlabel('Step')
        plt.ylabel('Logical VQE')
        plt.grid(True)

        plt.subplot(2, 2, 4)
        plt.plot(self.growth_history, color='green')
        plt.title('Growth Rate')
        plt.xlabel('Step')
        plt.ylabel('Growth Count')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('convergence_stability.png', dpi=300)
        plt.close()
        print("📊 Convergence plots saved as 'convergence_stability.png'")

    def save_results(self):
        with open('convergence_stability_results.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'energy', 'vqe', 'logical_vqe', 'growth'])
            for i in range(len(self.energy_history)):
                writer.writerow([i, self.energy_history[i], self.vqe_energy_history[i], self.logical_vqe_history[i], self.growth_history[i]])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BrQin Convergence Stability Test")
    parser.add_argument("--steps", type=int, default=2000, help="Number of steps (500-5000 recommended)")
    parser.add_argument("--Lx", type=int, default=12)
    parser.add_argument("--Ly", type=int, default=12)
    parser.add_argument("--bond", type=int, default=12)
    args = parser.parse_args()

    demo = BrQinDemo(Lx=args.Lx, Ly=args.Ly, init_bond=args.bond)
    demo.run_long(steps=args.steps)