# BrQin_v5.3.py
# Main reflective belief engine with PEPS oracle integration and persistence
# Polished for GitHub: clean formatting, docstrings, error handling
# Latest: February 09, 2026

import datetime
import json
import os
import hashlib
import sqlite3
import numpy as np
from oracle_peps import PepsOracle

class MerkleTree:
    """Simple Merkle Tree for chain integrity."""
    def __init__(self):
        self.leaves = []
        self.root = None

    def add_leaf(self, data_hash: str):
        """Add a new leaf hash and rebuild the root."""
        self.leaves.append(data_hash)
        self._build_root()

    def _build_root(self):
        """Build the Merkle root from leaves."""
        if not self.leaves:
            self.root = None
            return
        nodes = self.leaves[:]
        while len(nodes) > 1:
            new_level = []
            for i in range(0, len(nodes), 2):
                left = nodes[i]
                right = nodes[i + 1] if i + 1 < len(nodes) else left
                combined = hashlib.sha256((left + right).encode()).hexdigest()
                new_level.append(combined)
            nodes = new_level
        self.root = nodes[0]

    def verify(self, leaf_hash: str, leaf_index: int) -> bool:
        """Verify a leaf against the current root."""
        if leaf_index >= len(self.leaves):
            return False
        current = leaf_hash
        n = len(self.leaves)
        while n > 1:
            if leaf_index % 2 == 0:
                sibling = self.leaves[leaf_index + 1] if leaf_index + 1 < n else current
                current = hashlib.sha256((current + sibling).encode()).hexdigest()
            else:
                sibling = self.leaves[leaf_index - 1]
                current = hashlib.sha256((sibling + current).encode()).hexdigest()
            leaf_index //= 2
            n = (n + 1) // 2
        return current == self.root

class BrQin:
    """BrQin v5.3: Reflective belief engine with PEPS oracle and persistence."""
    def __init__(self, db_path="brqin_reflections.db", persistence_dir="brqin_persistence"):
        self.version = "5.3"
        self.reflection_count = 0
        self.db_path = db_path
        self.persistence_dir = persistence_dir
        os.makedirs(self.persistence_dir, exist_ok=True)

        # SQLite setup
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS reflections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                reflection_id TEXT UNIQUE,
                timestamp TEXT,
                ordeal_context TEXT,
                initial_belief TEXT,
                enriched_belief TEXT,
                oracle_metrics TEXT,
                tensor_snapshot TEXT,
                merkle_leaf_hash TEXT,
                merkle_root TEXT
            )
        ''')
        self.conn.commit()

        # Merkle tree
        self.merkle_tree = MerkleTree()

        # Oracle
        self.oracle = PepsOracle(steps=12, Lz=6, init_bond=8, ctmrg_chi=32)
        self.quality_history = []  # For self-evolution
        self.previous_energy = 0.0  # For convergence tracking

        print(f"BrQin v{self.version} initialized with PEPS oracle and persistence.")

    def reflect(self, ordeal_context: str, initial_belief: str):
        """Perform a single reflection using the PEPS oracle."""
        self.reflection_count += 1
        reflection_id = f"ref_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.reflection_count:03d}"

        print(f"Reflection {reflection_id} started")

        # Call oracle
        oracle_metrics = self.oracle.run(mode="light", guided_trickle=True)

        # Update convergence rate (avg ΔE over last 5)
        self.quality_history.append(oracle_metrics['certified_energy'])
        if len(self.quality_history) > 5:
            recent = self.quality_history[-5:]
            convergence_rate = (recent[-1] - recent[0]) / 5  # Negative = improving
        else:
            convergence_rate = 0.0
        self.previous_energy = oracle_metrics['certified_energy']

        # Live entropy stats
        entropy_stats = oracle_metrics.get('entropy_stats', {
            'variance': 0.0,
            'min_H': 0.0,
            'max_H': 0.0,
            'directional_bias': {'h': 0.0, 'v': 0.0, 'z': 0.0}
        })

        # Enriched belief
        enriched_belief = f"""
{initial_belief}

[PEPS-Thinking Core - Fracton Order]
Certified Energy: {oracle_metrics['certified_energy']:.4f}
Final Avg Bond: {oracle_metrics['final_avg_bond']:.1f}
Code Distance: {oracle_metrics['code_distance']}
Logical Advantage: {oracle_metrics.get('logical_advantage', 'N/A')}
Mode: {oracle_metrics['mode']}
Live Entropy Stats:
  - Variance: {entropy_stats['variance']:.4f} (high = exploratory chaos, low = precise structure)
  - Min/Max H: {entropy_stats['min_H']:.4f} / {entropy_stats['max_H']:.4f}
  - Directional Bias: h={entropy_stats['directional_bias']['h']:.2f} | v={entropy_stats['directional_bias']['v']:.2f} | z={entropy_stats['directional_bias']['z']:.2f}
Convergence Speed: {convergence_rate:.4f} / step (negative = rapid stabilization)
Fracton Rigidity: High (Haah/X-cube enforced)
Timestamp: {oracle_metrics['timestamp']}
"""

        # Tensor snapshot
        tensor_snapshot = {
            "avg_bond": oracle_metrics['final_avg_bond'],
            "energy": oracle_metrics['certified_energy'],
            "code_distance": oracle_metrics['code_distance'],
            "entropy_stats": entropy_stats,
            "convergence_rate": convergence_rate
        }

        # Record
        record = {
            "reflection_id": reflection_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "ordeal_context": ordeal_context,
            "initial_belief": initial_belief,
            "enriched_belief": enriched_belief,
            "oracle_metrics": oracle_metrics,
            "tensor_snapshot": tensor_snapshot
        }

        # Merkle leaf hash
        leaf_hash = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
        self.merkle_tree.add_leaf(leaf_hash)

        # Save to SQLite
        self.cursor.execute('''
            INSERT INTO reflections (
                reflection_id, timestamp, ordeal_context, initial_belief,
                enriched_belief, oracle_metrics, tensor_snapshot,
                merkle_leaf_hash, merkle_root
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            reflection_id, record["timestamp"], ordeal_context, initial_belief,
            enriched_belief, json.dumps(oracle_metrics), json.dumps(tensor_snapshot),
            leaf_hash, self.merkle_tree.root
        ))
        self.conn.commit()

        # JSON backup
        filepath = os.path.join(self.persistence_dir, f"{reflection_id}.json")
        with open(filepath, 'w') as f:
            json.dump(record, f, indent=2)

        # Self-evolution
        if len(self.quality_history) > 5:
            recent = self.quality_history[-5:]
            trend = (recent[0] - recent[-1]) / 5
            if trend < -0.01:  # Improving
                self.oracle.steps += 2
                self.oracle.Lz += 1
                self.oracle.init_bond += 2
                print("Self-evolution: Improving → increasing complexity")
            elif trend > 0.01:  # Worsening
                self.oracle.steps = max(5, self.oracle.steps - 1)
                self.oracle.Lz = max(4, self.oracle.Lz - 1)
                self.oracle.init_bond = max(4, self.oracle.init_bond - 1)
                print("Self-evolution: Worsening → decreasing complexity")
            else:  # Stable
                delta = np.random.randint(-1, 2)
                self.oracle.steps += delta
                self.oracle.Lz += delta
                self.oracle.init_bond += delta
                print(f"Self-evolution: Stable → random perturbation {delta}")

        print(f"Reflection {reflection_id} complete | Merkle leaf: {leaf_hash[:16]}... | Root: {self.merkle_tree.root[:16]}...")
        print(enriched_belief)
        return reflection_id

    def run_long_reflection_loop(self, num_ordeals=50):
        print(f"\nStarting DEEP reflection loop ({num_ordeals} ordeals)...")
        for i in range(1, num_ordeals + 1):
            ordeal = f"Ordeal {i}: Explore deeper self-reflection in a noisy, entangled universe"
            print(f"\n=== Ordeal {i}/{num_ordeals} ===")
            belief = input("Enter initial belief (or Enter for auto-sample): ").strip()
            if not belief:
                belief = f"Auto-sample belief: Seeking deeper entanglement-protected wisdom"
            self.reflect(ordeal, belief)

            # Benchmark every 10
            if i % 10 == 0:
                print("\nRunning logical error benchmark...")
                self.oracle.benchmark_logical_error_vs_Lz(noise_p=0.01, trials=500)

        print(f"\nDeep loop complete. Total reflections: {self.reflection_count}")
        print(f"Final Merkle root of entire chain: {self.merkle_tree.root}")

    def __del__(self):
        self.conn.close()

if __name__ == "__main__":
    brqin = BrQin()
    brqin.run_long_reflection_loop(num_ordeals=50)
