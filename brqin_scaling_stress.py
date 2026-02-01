import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import psutil
from tqdm import tqdm

print("=== BrQin Scaling & Performance Stressing (8×8 → 32×32) ===")

class BrQinDemo:
    def __init__(self, Lx, Ly, init_bond=12):
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
                        current = self.bond_map_h.get(key, self.init_bond)
                        new = min(current + 4, 32)
                        if new != current:
                            self.bond_map_h[key] = new
                            growth += 1
                    if r + 1 < self.Lx:
                        key = ((r,c),(r+1,c))
                        current = self.bond_map_v.get(key, self.init_bond)
                        new = min(current + 4, 32)
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

    def run_benchmark(self, steps=20):
        start_time = time.time()
        for step in range(steps):
            self.adaptive_bond_growth()
            self.simulate_energy()
            self.simulate_vqe_energy()
        runtime = time.time() - start_time
        final_bond = self.get_avg_bond()
        memory_mb = psutil.Process().memory_info().rss / (1024**2)
        return runtime, final_bond, memory_mb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BrQin Scaling Stress Test")
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    grid_sizes = [8, 12, 16, 20, 24, 28, 32]
    results = []

    print("Running scaling stress test (8×8 → 32×32)...\n")

    for size in grid_sizes:
        print(f"Testing {size}×{size} grid...")
        start = time.time()
        demo = BrQinDemo(Lx=size, Ly=size, init_bond=12)
        runtime, final_bond, memory_mb = demo.run_benchmark(steps=args.steps)
        elapsed = time.time() - start

        results.append({
            "grid_size": size,
            "runtime": runtime,
            "final_bond": final_bond,
            "memory_mb": memory_mb
        })

        print(f"  Runtime: {runtime:.2f}s | Final bond: {final_bond:.1f} | Memory: {memory_mb:.1f} MB\n")

    # Plot scaling laws
    sizes = [r["grid_size"] for r in results]
    runtimes = [r["runtime"] for r in results]
    bond_dims = [r["final_bond"] for r in results]
    memory = [r["memory_mb"] for r in results]

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(sizes, runtimes, marker='o', color='blue')
    plt.title('Runtime vs Grid Size')
    plt.xlabel('Grid Size (N×N)')
    plt.ylabel('Runtime (seconds)')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(sizes, bond_dims, marker='o', color='green')
    plt.title('Final Bond Dimension vs Grid Size')
    plt.xlabel('Grid Size (N×N)')
    plt.ylabel('Avg Bond Dimension')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(sizes, memory, marker='o', color='purple')
    plt.title('Memory Usage vs Grid Size')
    plt.xlabel('Grid Size (N×N)')
    plt.ylabel('Memory (MB)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('scaling_stress_results.png', dpi=300)
    plt.close()

    # Save CSV
    with open('scaling_stress_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['grid_size', 'runtime_s', 'final_bond', 'memory_mb'])
        for r in results:
            writer.writerow([r['grid_size'], r['runtime'], r['final_bond'], r['memory_mb']])

    print("✅ Scaling stress test complete")
    print("📊 Results saved to 'scaling_stress_results.csv'")
    print("📊 Plots saved as 'scaling_stress_results.png'")