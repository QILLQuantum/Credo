import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import psutil
from tqdm import tqdm

print("=== BrQin Memory Exhaustion Test (Large Grids + Many Patches) ===")

class BrQinPatch:
    def __init__(self, Lx, Ly, init_bond=12, max_bond=32, patch_id=0):
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

class BrQinMemoryStress:
    def __init__(self):
        self.results = []

    def run_test(self, Lx, Ly, patches, init_bond=12, max_bond=32, steps=50):
        print(f"\nTesting: {Lx}×{Ly} grid, {patches} patches, max_bond={max_bond}")
        start_time = time.time()

        try:
            simulator = []
            for i in range(patches):
                patch = BrQinPatch(Lx, Ly, init_bond, max_bond, patch_id=i)
                simulator.append(patch)

            for step in range(steps):
                for p in simulator:
                    p.adaptive_bond_growth()

            runtime = time.time() - start_time
            memory_mb = psutil.Process().memory_info().rss / (1024**2)
            final_bond = simulator[0].get_avg_bond() if simulator else 0

            result = {
                "grid_size": f"{Lx}x{Ly}",
                "patches": patches,
                "max_bond": max_bond,
                "runtime": runtime,
                "memory_mb": memory_mb,
                "final_bond": final_bond,
                "status": "SUCCESS"
            }
            self.results.append(result)

            print(f"  Runtime: {runtime:.2f}s | Memory: {memory_mb:.1f} MB | Final bond: {final_bond:.1f}")
            return result

        except MemoryError:
            runtime = time.time() - start_time
            memory_mb = psutil.Process().memory_info().rss / (1024**2)
            result = {
                "grid_size": f"{Lx}x{Ly}",
                "patches": patches,
                "max_bond": max_bond,
                "runtime": runtime,
                "memory_mb": memory_mb,
                "final_bond": 0,
                "status": "MEMORY_ERROR"
            }
            self.results.append(result)
            print(f"  MEMORY EXHAUSTED after {runtime:.2f}s | Memory: {memory_mb:.1f} MB")
            return result

    def run_full_stress_test(self):
        configs = [
            {"Lx":8, "Ly":8, "patches":8, "max_bond":32},
            {"Lx":12, "Ly":12, "patches":16, "max_bond":32},
            {"Lx":16, "Ly":16, "patches":32, "max_bond":32},
            {"Lx":20, "Ly":20, "patches":48, "max_bond":32},
            {"Lx":24, "Ly":24, "patches":64, "max_bond":32},
            {"Lx":28, "Ly":28, "patches":32, "max_bond":64},
            {"Lx":32, "Ly":32, "patches":16, "max_bond":64},
        ]

        for config in configs:
            self.run_test(**config)

        self.plot_results()
        self.save_results()

    def plot_results(self):
        patches = [r["patches"] for r in self.results]
        runtimes = [r["runtime"] for r in self.results]
        memory = [r["memory_mb"] for r in self.results]
        grid_sizes = [r["grid_size"] for r in self.results]

        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(patches, runtimes, marker='o', color='blue')
        plt.title('Runtime vs Number of Patches')
        plt.xlabel('Number of Patches')
        plt.ylabel('Runtime (seconds)')
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(patches, memory, marker='o', color='purple')
        plt.title('Memory Usage vs Number of Patches')
        plt.xlabel('Number of Patches')
        plt.ylabel('Memory (MB)')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('memory_exhaustion_results.png', dpi=300)
        plt.close()
        print("📊 Memory exhaustion plots saved as 'memory_exhaustion_results.png'")

    def save_results(self):
        with open('memory_exhaustion_results.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['grid_size', 'patches', 'max_bond', 'runtime_s', 'memory_mb', 'status'])
            for r in self.results:
                writer.writerow([r['grid_size'], r['patches'], r['max_bond'], r['runtime'], r['memory_mb'], r['status']])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BrQin Memory Exhaustion Test")
    args = parser.parse_args()

    stress_test = BrQinMemoryStress()
    stress_test.run_full_stress_test()