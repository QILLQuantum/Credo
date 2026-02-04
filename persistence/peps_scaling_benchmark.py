# peps_scaling_benchmark.py
# PEPS scaling benchmark for Credo DB
# Run: py -3 peps_scaling_benchmark.py

import datetime
import time
from BrQin v5.3 import BrQin  # adjust import if file name has spaces

def run_scaling_benchmark(num_loops=60):
    brqin = BrQin()
    print(f"Starting PEPS scaling benchmark: {num_loops} reflections")

    results = []
    for i in range(num_loops):
        # Scale Lz and bond progressively (every 10 loops increase)
        loop_group = i // 10
        Lz = 4 + loop_group * 2           # 4 → 6 → 8 → 10 → 12 → 14
        bond = 6 + loop_group * 3         # 6 → 9 → 12 → 15 → 18 → 21

        # Update oracle params dynamically
        brqin.oracle.Lz = Lz
        brqin.oracle.bond = bond

        ordeal = f"Scaling test {i+1}: Lz={Lz}, bond={bond}"
        initial_belief = f"Explore PEPS scaling at Lz={Lz}, bond={bond}"

        start_time = time.time()
        result = brqin.reflect(ordeal, initial_belief)
        runtime = time.time() - start_time

        if result:
            oracle_metrics = result["oracle_metrics"]
            contraction = oracle_metrics["contraction_info"]

            quality_score = (
                oracle_metrics["logical_advantage"] *
                (1 - oracle_metrics["uncertainty"]) *
                oracle_metrics["growth_rate"] / 1000
            )

            entry = {
                "reflection_id": result["reflection_id"],
                "timestamp": result["timestamp"],
                "Lz": Lz,
                "bond": bond,
                "opt_cost": contraction["opt_cost"],
                "naive_cost": contraction["naive_cost"],
                "speedup": contraction["speedup"],
                "runtime_seconds": runtime,
                "quality_score": quality_score,
                "certified_energy": oracle_metrics["certified_energy"],
                "logical_advantage": oracle_metrics["logical_advantage"]
            }

            results.append(entry)
            print(f"[{i+1}/{num_loops}] Lz={Lz}, bond={bond} | Quality: {quality_score:.2f} | Speedup: {contraction['speedup']:.1f}x | Runtime: {runtime:.2f}s")

    print("\nBenchmark complete.")
    print(f"Total reflections persisted: {len(results)}")

    # Quick DB query summary
    try:
        facade = brqin.persistence
        print(f"Current Merkle root: {facade.get_current_root()[:12] if facade.get_current_root() else 'None'}…")
        print(f"Total DB entries: {facade.count()}")
    except Exception as e:
        print(f"DB stats failed: {str(e)}")

    return results


if __name__ == "__main__":
    run_scaling_benchmark(num_loops=60)  # 60 reflections → ~6 groups of scaling