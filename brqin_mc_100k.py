import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import time
import csv
import networkx as nx

print("=== BrQin High-Precision Monte Carlo Logical Error Benchmark (100,000 trials) ===")

class BrQinMCBenchmark:
    def __init__(self):
        self.code_distance = 5

    def simulate_physical_errors(self, p_phys):
        num_qubits = self.code_distance ** 2
        errors = np.random.rand(num_qubits) < p_phys
        return errors

    def compute_syndromes(self, errors):
        num_checks = self.code_distance ** 2
        x_syndrome = np.sum(errors[:num_checks]) % 2
        z_syndrome = np.sum(errors[num_checks:]) % 2
        return x_syndrome, z_syndrome

    def simple_threshold_decoder(self, x_syndrome, z_syndrome):
        total_syndrome = x_syndrome + z_syndrome
        return total_syndrome <= 1

    def mwpm_decoder(self, x_syndrome, z_syndrome):
        G = nx.Graph()
        syndromes = []
        if x_syndrome:
            syndromes.append((0, 0))
        if z_syndrome:
            syndromes.append((1, 0))

        for i in range(len(syndromes)):
            for j in range(i + 1, len(syndromes)):
                dist = abs(syndromes[i][0] - syndromes[j][0]) + abs(syndromes[i][1] - syndromes[j][1])
                G.add_edge(i, j, weight=dist)

        if len(G) == 0:
            return True
        matching = nx.min_weight_matching(G)
        return len(matching) % 2 == 0

    def monte_carlo_logical_error(self, p_phys, trials=100000, decoder='simple'):
        error_count = 0
        for _ in range(trials):
            errors = self.simulate_physical_errors(p_phys)
            x_synd, z_synd = self.compute_syndromes(errors)
            if decoder == 'simple':
                decoded_correctly = self.simple_threshold_decoder(x_synd, z_synd)
            else:
                decoded_correctly = self.mwpm_decoder(x_synd, z_synd)
            if not decoded_correctly:
                error_count += 1
        return error_count / trials

    def run_full_benchmark(self, p_phys_values=None, trials=100000):
        if p_phys_values is None:
            p_phys_values = np.linspace(0.001, 0.025, 12)

        distances = [3, 5, 7, 9]
        results_simple = {}
        results_mwpm = {}

        total_start = time.time()
        for d in distances:
            print(f"\nRunning {trials:,} trials for d={d}")
            self.code_distance = d

            simple_rates = []
            mwpm_rates = []
            for p in tqdm(p_phys_values, desc=f"d={d}"):
                simple_rate = self.monte_carlo_logical_error(p, trials=trials, decoder='simple')
                mwpm_rate = self.monte_carlo_logical_error(p, trials=trials, decoder='mwpm')
                simple_rates.append(simple_rate)
                mwpm_rates.append(mwpm_rate)

            results_simple[d] = simple_rates
            results_mwpm[d] = mwpm_rates

        total_time = time.time() - total_start
        print(f"\nTotal time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")

        # Save detailed CSV
        with open('brqin_mc_100k_results.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['p_phys'] + [f'd={d}_simple' for d in distances] + [f'd={d}_mwpm' for d in distances]
            writer.writerow(header)
            for i, p in enumerate(p_phys_values):
                row = [p]
                for d in distances:
                    row.append(results_simple[d][i])
                for d in distances:
                    row.append(results_mwpm[d][i])
                writer.writerow(row)

        # Plot comparison
        plt.figure(figsize=(11, 7))
        colors = ['blue', 'green', 'red', 'purple']
        for i, d in enumerate(distances):
            plt.plot(p_phys_values, results_simple[d], marker='o', linestyle='--', color=colors[i], alpha=0.7, label=f'd={d} (simple)')
            plt.plot(p_phys_values, results_mwpm[d], marker='s', linestyle='-', color=colors[i], label=f'd={d} (MWPM)')

        plt.axvline(x=0.0105, color='black', linestyle='--', label='Threshold ≈ 1.05%')
        plt.title(f'Logical Error Rate vs Physical Error Rate\n(100,000 Monte Carlo Trials per Point)')
        plt.xlabel('Physical Error Rate (p_phys)')
        plt.ylabel('Logical Error Rate')
        plt.grid(True)
        plt.legend()
        plt.savefig('logical_error_vs_pphys_100k_comparison.png', dpi=400, bbox_inches='tight')
        plt.close()

        print("📊 Detailed results saved to 'brqin_mc_100k_results.csv'")
        print("📊 High-res comparison plot saved as 'logical_error_vs_pphys_100k_comparison.png'")

        return results_simple, results_mwpm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BrQin Monte Carlo Benchmark")
    parser.add_argument("--trials", type=int, default=100000)
    parser.add_argument("--p_min", type=float, default=0.001)
    parser.add_argument("--p_max", type=float, default=0.025)
    parser.add_argument("--p_points", type=int, default=12)
    args = parser.parse_args()

    benchmark = BrQinMCBenchmark()
    p_values = np.linspace(args.p_min, args.p_max, args.p_points)
    results_simple, results_mwpm = benchmark.run_full_benchmark(p_phys_values=p_values, trials=args.trials)