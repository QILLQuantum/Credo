# BrQin v5.3.py - Persistent PEPS Memory + Tensor Oracle Integration
# Date: February 2026

import datetime
import traceback
import numpy as np
from credo_db_facade import CredoDBFacade
from oracle_peps import PepsOracle

class BrQin:
    def __init__(self):
        self.version = "5.3"
        try:
            self.persistence = CredoDBFacade()
            self.oracle = PepsOracle(steps=12, Lz=6, bond=8)
            self.reflection_count = 0
            self.error_count = 0

            # Persistent PEPS state – starts empty, carries forward
            self.peps_state = None  # Will hold tensor network state dict
            self.state_entropy = 0.0
            print(f"✅ BrQin v{self.version} initialized with persistent PEPS memory")
        except Exception as e:
            print(f"CRITICAL: Init failed - {str(e)}")
            traceback.print_exc()
            raise

    def reflect(self, ordeal_context: str, initial_belief: str) -> dict:
        self.reflection_count += 1
        reflection_id = f"ref_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.reflection_count}"

        try:
            # 1. Load persistent PEPS state (if exists) or initialize
            if self.peps_state is None:
                print("Initializing fresh PEPS state")
                self.peps_state = self.oracle.initialize_state()  # Assume oracle has init method
            else:
                print("Loading persistent PEPS state from previous reflection")

            # 2. Run oracle with current state (incremental update)
            print("🔮 Calling Tensor Oracle (with memory)...")
            oracle_metrics = self.oracle.run_with_state(
                mode="light",
                previous_state=self.peps_state
            )

            # 3. Update persistent state
            self.peps_state = oracle_metrics["updated_state"]  # oracle returns new state
            self.state_entropy = oracle_metrics.get("state_entropy", self.state_entropy)

            # 4. Enriched belief with memory metrics
            enriched_belief = f"{initial_belief}\n\n[Oracle v5.3 + persistent memory]\n" \
                              f"Certified Energy: {oracle_metrics['certified_energy']:.4f} ± {oracle_metrics['uncertainty']:.4f}\n" \
                              f"Logical Advantage: {oracle_metrics['logical_advantage']:.3f} (d={oracle_metrics['code_distance']})\n" \
                              f"Final Avg Bond: {oracle_metrics['final_avg_bond']:.1f} | Growth Rate: {oracle_metrics['growth_rate']}\n" \
                              f"State Continuity: Entropy change {oracle_metrics.get('entropy_change', 0.0):.4f} | Retention: {oracle_metrics.get('retention_rate', 1.0):.2f}"

            # 5. Record with memory metrics
            record = {
                "reflection_id": reflection_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "ordeal_context": ordeal_context,
                "initial_belief": initial_belief,
                "enriched_belief": enriched_belief,
                "oracle_metrics": oracle_metrics,
                "memory_metrics": {
                    "state_entropy": self.state_entropy,
                    "entropy_change": oracle_metrics.get("entropy_change", 0.0),
                    "retention_rate": oracle_metrics.get("retention_rate", 1.0)
                },
                "version": self.version,
                "status": "success"
            }

            # 6. Persist
            saved_record = self.persistence.save_belief(record, "reflection")
            log_reflection(reflection_id, saved_record)

            print(f"💾 Reflection {reflection_id} persisted (with memory)")
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

    # Example loop – scale to 20–50 for dogfood
    test_ordeals = [f"Ordeal {i+1}: Reflect on persistent PEPS memory" for i in range(15)]

    brqin.run_reflection_loop(test_ordeals)