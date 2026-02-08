# BrQin v5.3.py - Persistence Hook + Tensor Oracle + Self-Evolving Oracle
# Date: February 2026

import datetime
import traceback
import numpy as np
from credo_db_facade import CredoDBFacade
from oracle_peps import PepsOracle
from credo_logger import log_reflection, log_oracle_call

class BrQin:
    def __init__(self):
        self.version = "5.3"
        try:
            self.persistence = CredoDBFacade()
            self.oracle = PepsOracle(steps=12, Lz=6, bond=8)
            self.reflection_count = 0
            self.error_count = 0
            self.last_quality_score = 0.0
            self.quality_history = []  # for trend detection
            print(f"✅ BrQin v{self.version} initialized with self-evolving oracle")
        except Exception as e:
            print(f"CRITICAL: Init failed - {str(e)}")
            traceback.print_exc()
            raise

    def reflect(self, ordeal_context: str, initial_belief: str) -> dict:
        self.reflection_count += 1
        reflection_id = f"ref_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.reflection_count}"

        try:
            print("🔮 Calling Tensor Oracle...")
            oracle_metrics = self.oracle.run(mode="light")

            log_oracle_call(reflection_id, oracle_metrics)

            # Enriched belief (example – add your full metrics here)
            enriched_belief = f"{initial_belief}\n\n[Oracle v5.3]\n" \
                              f"Certified Energy: {oracle_metrics['certified_energy']:.4f} ± {oracle_metrics['uncertainty']:.4f}\n" \
                              f"Logical Advantage: {oracle_metrics['logical_advantage']:.3f} (d={oracle_metrics['code_distance']})\n" \
                              f"Final Avg Bond: {oracle_metrics['final_avg_bond']:.1f} | Growth Rate: {oracle_metrics['growth_rate']}"

            record = {
                "reflection_id": reflection_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "ordeal_context": ordeal_context,
                "initial_belief": initial_belief,
                "enriched_belief": enriched_belief,
                "oracle_metrics": oracle_metrics,
                "version": self.version,
                "status": "success"
            }

            saved_record = self.persistence.save_belief(record, "reflection")
            log_reflection(reflection_id, saved_record)

            # Compute quality score for self-evolution
            quality = oracle_metrics['logical_advantage'] * (1 - oracle_metrics['uncertainty']) * oracle_metrics['growth_rate'] / 1000
            self.quality_history.append(quality)
            if len(self.quality_history) > 20:
                self.quality_history.pop(0)

            # Periodic self-evolution (every 50 reflections)
            if self.reflection_count % 50 == 0 and len(self.quality_history) >= 20:
                self.self_evolve()

            print(f"💾 Reflection {reflection_id} persisted")
            return saved_record

        except Exception as e:
            self.error_count += 1
            error_msg = f"ERROR in reflection {reflection_id}: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)

            try:
                error_record = {
                    "reflection_id": reflection_id,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "ordeal_context": ordeal_context,
                    "initial_belief": initial_belief,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "status": "failed"
                }
                self.persistence.save_belief(error_record, "error")
                print("Error record persisted")
            except Exception as save_err:
                print(f"CRITICAL: Failed to save error - {str(save_err)}")

            return None

    def self_evolve(self):
        """Self-evolution: reflect on recent quality trend and adjust oracle params"""
        if len(self.quality_history) < 20:
            return

        recent_avg = np.mean(self.quality_history[-20:])
        prev_avg = np.mean(self.quality_history[:-20]) if len(self.quality_history) > 20 else recent_avg

        trend = recent_avg - prev_avg

        old_steps = self.oracle.steps
        old_lz = self.oracle.Lz
        old_bond = self.oracle.bond
        old_mc_trials = self.oracle.mc_trials

        reason = ""

        if trend > 0.05:  # improving — push harder
            self.oracle.steps += 2
            self.oracle.Lz += 1
            self.oracle.bond += 2
            self.oracle.mc_trials = max(20, self.oracle.mc_trials - 20)
            reason = "Trend improving → increase complexity, reduce MC trials"

        elif trend < -0.05:  # worsening — rollback / simplify
            self.oracle.steps = max(8, self.oracle.steps - 4)
            self.oracle.Lz = max(4, self.oracle.Lz - 2)
            self.oracle.bond = max(6, self.oracle.bond - 4)
            self.oracle.mc_trials += 50
            reason = "Trend worsening → simplify, increase MC trials"

        else:  # stable — small random perturbation
            if np.random.rand() < 0.3:
                delta = np.random.choice([-2, 0, 2])
                self.oracle.steps += delta
                reason = "Stable → small random perturbation"

        # Persist the evolution decision
        meta_record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "type": "meta_evolution",
            "old_params": {"steps": old_steps, "Lz": old_lz, "bond": old_bond, "mc_trials": old_mc_trials},
            "new_params": {
                "steps": self.oracle.steps,
                "Lz": self.oracle.Lz,
                "bond": self.oracle.bond,
                "mc_trials": self.oracle.mc_trials
            },
            "trend": float(trend),
            "reason": reason
        }

        self.persistence.save_belief(meta_record, "meta_evolution")
        print(f"Self-evolution: {reason}")
        print(f"New params: steps={self.oracle.steps}, Lz={self.oracle.Lz}, bond={self.oracle.bond}, mc_trials={self.oracle.mc_trials}")

    def run_reflection_loop(self, ordeals: list):
        for i, ordeal in enumerate(ordeals):
            print(f"\n=== Ordeal {i+1}/{len(ordeals)} ===")
            try:
                belief = input(f"Initial belief for '{ordeal}': ") if hasattr(__builtins__, 'input') else f"Belief for {ordeal}"
                result = self.reflect(ordeal, belief)
                if result:
                    print(f"✅ Completed: {result['reflection_id']}")
                else:
                    print(f"❌ Failed: {ordeal}")
            except KeyboardInterrupt:
                print("\nInterrupted. Stopping.")
                break
            except Exception as e:
                print(f"Loop error: {str(e)}")

        print(f"\n🎉 BrQin v{self.version} complete.")
        print(f"Reflections: {self.reflection_count} | Errors: {self.error_count}")

        try:
            print(f"Merkle root: {self.persistence.get_current_root()[:12] if self.persistence.get_current_root() else 'None'}…")
            print(f"Total entries: {self.persistence.count()}")
        except Exception as e:
            print(f"Stats failed: {str(e)}")

if __name__ == "__main__":
    brqin = BrQin()

    test_ordeals = [f"Ordeal {i+1}: Test self-evolving oracle" for i in range(20)]

    brqin.run_reflection_loop(test_ordeals)
