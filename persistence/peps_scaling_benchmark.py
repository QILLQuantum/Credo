# peps_scaling_benchmark.py - PEPS Scaling Benchmark for Credo
import datetime
import time
import json
import os
from BrQin_v5_3 import BrQin  # Rename file to BrQin_v5_3.py to avoid space issue

def run_scaling_benchmark(num_loops=60):
    brqin = BrQin()
    print(f"Starting PEPS scaling benchmark: {num_loops} reflections")
    results = []
    current_Lz = 4
    current_bond = 6
    
    for i in range(num_loops):
        # Scale every 10 loops
        if i % 10 == 0 and i > 0:
            current_Lz += 2
            current_bond += 3
            # Re-init oracle with new params (safer than mutation)
            brqin.oracle = PepsOracle(steps=12, Lz=current_Lz, bond=current_bond)
        
        ordeal = f"Scaling test {i+1}: Lz={current_Lz}, bond={current_bond}"
        initial_belief = f"Explore PEPS scaling at Lz={current_Lz}, bond={current_bond}"
        
        start_time = time.time()
        try:
            result = brqin.reflect(ordeal, initial_belief)
            runtime = time.time() - start_time
            
            if result and "oracle_metrics" in result:
                oracle_metrics = result["oracle_metrics"]
                contraction = oracle_metrics.get("contraction_info", {})
                quality_score = (
                    oracle_metrics.get("logical_advantage", 1.0) *
                    (1 - oracle_metrics.get("uncertainty", 0.01)) *
                    oracle_metrics.get("growth_rate", 100) / 1000
                )
                entry = {
                    "reflection_id": result.get("reflection_id"),
                    "timestamp": result.get("timestamp"),
                    "Lz": current_Lz,
                    "bond": current_bond,
                    "opt_cost": contraction.get("opt_cost"),
                    "naive_cost": contraction.get("naive_cost"),
                    "speedup": contraction.get("speedup"),
                    "runtime_seconds": round(runtime, 2),
                    "quality_score": round(quality_score, 4),
                    "certified_energy": oracle_metrics.get("certified_energy"),
                    "logical_advantage": oracle_metrics.get("logical_advantage")
                }
                results.append(entry)
                print(f"[{i+1}/{num_loops}] Lz={current_Lz}, bond={current_bond} | Quality: {quality_score:.4f} | Speedup: {contraction.get('speedup', 1):.1f}x | Runtime: {runtime:.2f}s")
            else:
                print(f"[{i+1}/{num_loops}] Reflection failed or no metrics")
        except Exception as e:
            print(f"[{i+1}/{num_loops}] Error: {str(e)}")
    
    print("\nBenchmark complete.")
    print(f"Total successful reflections: {len(results)}")
    try:
        print(f"Current Merkle root: {brqin.persistence.get_current_root()[:12] if brqin.persistence.get_current_root() else 'None'}…")
        print(f"Total DB entries: {brqin.persistence.count()}")
    except Exception as e:
        print(f"DB stats failed: {str(e)}")
    
    # Save results
    benchmark_file = f"benchmark_results_{datetime.date.today()}.json"
    with open(benchmark_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {benchmark_file}")
    
    return results

if __name__ == "__main__":
    run_scaling_benchmark(num_loops=60)
