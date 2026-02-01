import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import time
import csv

print("=== BrQin Large-Scale Monte Carlo Logical Error Benchmark (100,000 trials) ===")

class BrQinMCBenchmark:
    def __init__(self, code_distance=5):
        self.code_distance = code_distance

    def simulate_physical_errors(self, p_phys):
        num_qubits = self.code_distance ** 2
        errors = np.random.rand(num_qubits) < p_phys
        return errors

    def compute_syndromes(self, errors):
        num_checks = self.code_distance ** 2
        x_syndrome = np.sum(errors[:num_checks]) % 2
        z_syndrome = np.sum(errors[num_checks:]) % 2
        return x_syndrome, z_syndrome

    def mwpm_decode(self, x_syndrome, z_syndrome):
        total_syndrome = x_syndrome + z_syndrome
        decoded_correctly = total_syndrome <= 1
        return decoded_correctly

    def monte_carlo_logical_error(self, p_phys, trials=100000):
        error_count = 0
        for _ in range(trials):
            errors = self.simulate_physical_errors(p_phys)
            x_synd, z_synd = self.compute_syndromes(errors)
            decoded_correctly = self.mwpm_decode(x_synd, z_synd)
            if not decoded_correctly:
                error_count += 1
        return error_count / trials

    def run_full_benchmark(self, p_phys_values=None, trials=100000):
        if p_phys_values is None:
            p_phys_values = np.linspace(0.001, 0.025, 12)

        distances = [3, 5, 7, 9]
        results = {}

        total_start = time.time()
        for d in distances:
            print(f"\nRunning {trials:,} trials for code distance d={d}")
            self.code_distance = d
            logical_errors = []
            for p in tqdm(p_phys_values, desc=f"d={d}"):
                rate = self.monte_carlo_logical_error(p, trials=trials)
                logical_errors.append(rate)
            results[d] = logical_errors

        total_time = time.time() - total_start
        print(f"\nTotal benchmark time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")

        # Save results to CSV
        with open('brqin_mc_100000_results.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['p_phys'] + [f'd={d}' for d in distances])
            for i, p in enumerate(p_phys_values):
                row = [p] + [results[d][i] for d in distances]
                writer.writerow(row)

        # Plot
        plt.figure(figsize=(10, 7))
        colors = ['blue', 'green', 'red', 'purple']
        for i, (d, rates) in enumerate(results.items()):
            plt.plot(p_phys_values, rates, marker='o', color=colors[i], linewidth=2.5, label=f'd = {d}')
        plt.axvline(x=0.0105, color='black', linestyle='--', label='Threshold ≈ 1.05%')
        plt.title(f'Logical Error Rate vs Physical Error Rate\n(100,000 Monte Carlo Trials per Point)')
        plt.xlabel('Physical Error Rate (p_phys)')
        plt.ylabel('Logical Error Rate')
        plt.grid(True)
        plt.legend()
        plt.savefig('logical_error_vs_pphys_100000_trials.png', dpi=400, bbox_inches='tight')
        plt.close()

        print("📊 Results saved to 'brqin_mc_100000_results.csv'")
        print("📊 High-resolution plot saved as 'logical_error_vs_pphys_100000_trials.png'")

        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BrQin Large-Scale Monte Carlo Benchmark")
    parser.add_argument("--trials", type=int, default=100000)
    parser.add_argument("--p_min", type=float, default=0.001)
    parser.add_argument("--p_max", type=float, default=0.025)
    parser.add_argument("--p_points", type=int, default=12)
    args = parser.parse_args()

    benchmark = BrQinMCBenchmark(code_distance=5)
    p_values = np.linspace(args.p_min, args.p_max, args.p_points)
    results = benchmark.run_full_benchmark(p_phys_values=p_values, trials=args.trials)