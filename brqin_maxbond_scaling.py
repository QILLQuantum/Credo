import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import psutil
from tqdm import tqdm

print("=== BrQin Scaling Stress Test: Runtime/Memory vs Max Bond Dimension ===")

class BrQinPatch:
    def __init__(self, Lx=12, Ly=12, init_bond=12, max_bond=32, patch_id=0):
        self.patch_id = patch_id
        self.Lx = Lx
        self.Ly = Ly
        self.init_bond = init_bond
        self.max_bond = max_bond
        self.bond_map_h = {}
        self.bond_map_v = {}
        self.growth_history = []

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

    def get_avg_bond(self):
        total = sum(self.bond_map_h.values()) + sum(self.bond_map_v.values())
        count = len(self.bond_map_h) + len(self.bond_map_v)
        return total / count if count > 0 else self.init_bond

class BrQinMultiPatch:
    def __init__(self, Lx=12, Ly=12, init_bond=12, max_bond=32, num_patches=4):
        self.patches = [BrQinPatch(Lx, Ly, init_bond, max_bond, patch_id=i) for i in range(num_patches)]

    def run(self, steps=20):
        start_time = time.time()
        for step in range(steps):
            for p in self.patches:
                p.adaptive_bond_growth()
        runtime = time.time() - start_time
        memory_mb = psutil.Process().memory_info().rss / (1024**2)
        final_bond = self.patches[0].get_avg_bond() if self.patches else 0
        return runtime, memory_mb, final_bond

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BrQin Max Bond Scaling Stress Test")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--Lx", type=int, default=12)
    parser.add_argument("--Ly", type=int, default=12)
    parser.add_argument("--bond", type=int, default=12)
    parser.add_argument("--patches", type=int, default=4)
    args = parser.parse_args()

    max_bond_values = [16, 24, 32, 48, 64]
    results = []

    print("Starting max_bond scaling test (16 → 64)...\n")

    for max_bond in max_bond_values:
        print(f"Testing max_bond = {max_bond}...")
        start = time.time()
        simulator = BrQinMultiPatch(
            Lx=args.Lx, 
            Ly=args.Ly, 
            init_bond=args.bond,
            max_bond=max_bond,
            num_patches=args.patches
        )
        runtime, memory_mb, final_bond = simulator.run(steps=args.steps)
        elapsed = time.time() - start

        results.append({
            "max_bond": max_bond,
            "runtime": runtime,
            "memory_mb": memory_mb,
            "final_bond": final_bond
        })

        print(f"  Runtime: {runtime:.2f}s | Memory: {memory_mb:.1f} MB | Final bond: {final_bond:.1f}\n")

    # Plot scaling laws
    max_bonds = [r["max_bond"] for r in results]
    runtimes = [r["runtime"] for r in results]
    memory = [r["memory_mb"] for r in results]
    bond_dims = [r["final_bond"] for r in results]

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(max_bonds, runtimes, marker='o', color='blue')
    plt.title('Runtime vs Max Bond Dimension')
    plt.xlabel('Max Bond Dimension')
    plt.ylabel('Runtime (seconds)')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(max_bonds, memory, marker='o', color='purple')
    plt.title('Memory Usage vs Max Bond Dimension')
    plt.xlabel('Max Bond Dimension')
    plt.ylabel('Memory (MB)')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(max_bonds, bond_dims, marker='o', color='green')
    plt.title('Final Bond Dimension vs Max Bond')
    plt.xlabel('Max Bond Dimension')
    plt.ylabel('Avg Bond Dimension')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('max_bond_scaling_results.png', dpi=300)
    plt.close()

    # Save CSV
    with open('max_bond_scaling_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['max_bond', 'runtime_s', 'memory_mb', 'final_bond'])
        for r in results:
            writer.writerow([r['max_bond'], r['runtime'], r['memory_mb'], r['final_bond']])

    print("✅ Max bond scaling stress test complete")
    print("📊 Results saved to 'max_bond_scaling_results.csv'")
    print("📊 Plots saved as 'max_bond_scaling_results.png'")