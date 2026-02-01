import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import time
import csv

print("=== BrQin Edge-Case Stress Test: Near-Threshold Error Rates (p_phys = 0.009–0.012) ===")

class BrQinEdgeCaseBenchmark:
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
        total_syndrome = x_syndrome + z_syndrome
        return total_syndrome <= 1

    def monte_carlo_logical_error(self, p_phys, trials=200000, decoder='simple'):
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

    def run_edge_case_stress(self, trials=200000):
        # Dense sampling around threshold
        p_phys_values = np.linspace(0.009, 0.012, 13)  # 13 points in narrow range

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
        print(f"\nTotal time: {total_time:.1f} seconds")

        # Save CSV
        with open('brqin_edge_case_200k.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['p_phys']
            for d in distances:
                header.append(f'd={d}_simple')
            for d in distances:
                header.append(f'd={d}_mwpm')
            writer.writerow(header)
            for i, p in enumerate(p_phys_values):
                row = [p]
                for d in distances:
                    row.append(results_simple[d][i])
                for d in distances:
                    row.append(results_mwpm[d][i])
                writer.writerow(row)

        # Plot
        plt.figure(figsize=(11, 7))
        colors = ['blue', 'green', 'red', 'purple']
        for i, d in enumerate(distances):
            plt.plot(p_phys_values, results_simple[d], marker='o', linestyle='--', color=colors[i], alpha=0.7, label=f'd={d} (simple)')
            plt.plot(p_phys_values, results_mwpm[d], marker='s', linestyle='-', color=colors[i], label=f'd={d} (MWPM)')

        plt.axvline(x=0.0105, color='black', linestyle='--', label='Threshold ≈ 1.05%')
        plt.title(f'Edge-Case Stress: Near-Threshold Logical Error Rate\n(200,000 Trials per Point)')
        plt.xlabel('Physical Error Rate (p_phys)')
        plt.ylabel('Logical Error Rate')
        plt.grid(True)
        plt.legend()
        plt.savefig('edge_case_threshold_stress.png', dpi=400, bbox_inches='tight')
        plt.close()

        print("📊 Results saved to 'brqin_edge_case_200k.csv'")
        print("📊 High-res plot saved as 'edge_case_threshold_stress.png'")

        return results_simple, results_mwpm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BrQin Edge-Case Stress Test")
    parser.add_argument("--trials", type=int, default=200000)
    args = parser.parse_args()

    benchmark = BrQinEdgeCaseBenchmark()
    results_simple, results_mwpm = benchmark.run_edge_case_stress(trials=args.trials)