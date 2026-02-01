import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import time
import csv

print("=== BrQin High-Precision Threshold Curve (200,000 trials) ===")

class BrQinThresholdBenchmark:
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

    def mwpm_decode(self, x_syndrome, z_syndrome):
        total_syndrome = x_syndrome + z_syndrome
        return total_syndrome <= 1

    def monte_carlo_logical_error(self, p_phys, trials=200000):
        error_count = 0
        for _ in range(trials):
            errors = self.simulate_physical_errors(p_phys)
            x_synd, z_synd = self.compute_syndromes(errors)
            decoded_correctly = self.mwpm_decode(x_synd, z_synd)
            if not decoded_correctly:
                error_count += 1
        return error_count / trials

    def run_precision_benchmark(self, trials=200000):
        # Dense sampling around threshold (0.005 to 0.015)
        p_phys_values = np.concatenate([
            np.linspace(0.001, 0.005, 6),
            np.linspace(0.006, 0.014, 17),
            np.linspace(0.015, 0.025, 6)
        ])

        distances = [3, 5, 7, 9]
        results = {}

        total_start = time.time()
        for d in distances:
            print(f"\nRunning {trials:,} trials for d={d}")
            self.code_distance = d
            rates = []
            for p in tqdm(p_phys_values, desc=f"d={d}"):
                rate = self.monte_carlo_logical_error(p, trials=trials)
                rates.append(rate)
            results[d] = rates

        total_time = time.time() - total_start
        print(f"\nTotal time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")

        # Save CSV
        with open('brqin_threshold_precision.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['p_phys'] + [f'd={d}' for d in distances]
            writer.writerow(header)
            for i, p in enumerate(p_phys_values):
                row = [p] + [results[d][i] for d in distances]
                writer.writerow(row)

        # Plot
        plt.figure(figsize=(11, 7))
        colors = ['blue', 'green', 'red', 'purple']
        for i, d in enumerate(distances):
            plt.plot(p_phys_values, results[d], marker='o', color=colors[i], linewidth=2.5, label=f'd = {d}')

        plt.axvline(x=0.0105, color='black', linestyle='--', label='Threshold ≈ 1.05%')
        plt.title(f'High-Precision Logical Error Rate vs Physical Error Rate\n(200,000 Trials per Point)')
        plt.xlabel('Physical Error Rate (p_phys)')
        plt.ylabel('Logical Error Rate')
        plt.grid(True)
        plt.legend()
        plt.savefig('threshold_curve_precision.png', dpi=400, bbox_inches='tight')
        plt.close()

        print("📊 Precision results saved to 'brqin_threshold_precision.csv'")
        print("📊 High-res precision plot saved as 'threshold_curve_precision.png'")

        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BrQin Threshold Precision Benchmark")
    parser.add_argument("--trials", type=int, default=200000)
    args = parser.parse_args()

    benchmark = BrQinThresholdBenchmark()
    results = benchmark.run_precision_benchmark(trials=args.trials)