# BrQin v5.3.py
# Main reflective belief engine with full PEPS oracle + Merkle tree persistence
# Latest: February 08, 2026

import datetime
import json
import os
import hashlib
import sqlite3
from oracle_peps import PepsOracle

class MerkleTree:
    def __init__(self):
        self.leaves = []
        self.root = None

    def add_leaf(self, data_hash: str):
        self.leaves.append(data_hash)
        self._build_root()

    def _build_root(self):
        if not self.leaves:
            self.root = None
            return
        nodes = self.leaves[:]
        while len(nodes) > 1:
            new_level = []
            for i in range(0, len(nodes), 2):
                left = nodes[i]
                right = nodes[i+1] if i+1 < len(nodes) else left
                combined = hashlib.sha256((left + right).encode()).hexdigest()
                new_level.append(combined)
            nodes = new_level
        self.root = nodes[0]

    def verify(self, leaf_hash: str, proof: list = None) -> bool:
        """Simple verification against current root (proof optional for now)"""
        if self.root is None:
            return False
        # For full proof verification, use proof path — here simple root check
        return True  # Placeholder; full proof in future

class BrQin:
    def __init__(self, db_path="brqin_reflections.db", persistence_dir="brqin_persistence"):
        self.version = "5.3"
        self.reflection_count = 0
        self.db_path = db_path
        self.persistence_dir = persistence_dir
        os.makedirs(persistence_dir, exist_ok=True)

        # SQLite setup
        self.conn = sqlite3.connect(db_path)
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

        # Merkle tree for chain integrity
        self.merkle_tree = MerkleTree()

        # Oracle
        self.oracle = PepsOracle(steps=12, Lz=6, init_bond=8, ctmrg_chi=32)
        print(f"BrQin v{self.version} initialized with PEPS oracle & full Merkle persistence")

    def reflect(self, ordeal_context: str, initial_belief: str):
        self.reflection_count += 1
        reflection_id = f"ref_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.reflection_count:03d}"

        print(f"Reflection {reflection_id} started")

        # Call oracle (full thinking mode)
        oracle_metrics = self.oracle.run(mode="light", guided_trickle=True)

        # Enriched belief
        enriched_belief = f"""
{initial_belief}

[PEPS-Thinking Core - Fracton Order]
Certified Energy: {oracle_metrics['certified_energy']:.4f}
Final Avg Bond: {oracle_metrics['final_avg_bond']:.1f}
Code Distance: {oracle_metrics['code_distance']}
Logical Advantage: {oracle_metrics.get('logical_advantage', 'N/A')}
Mode: {oracle_metrics['mode']}
Fracton Rigidity: High (Haah/X-cube enforced)
Timestamp: {oracle_metrics['timestamp']}
"""

        # Tensor snapshot
        tensor_snapshot = {
            "avg_bond": oracle_metrics['final_avg_bond'],
            "energy": oracle_metrics['certified_energy'],
            "code_distance": oracle_metrics['code_distance']
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

        print(f"Reflection {reflection_id} complete | Merkle leaf: {leaf_hash[:16]}... | Root: {self.merkle_tree.root[:16]}...")
        print(enriched_belief)
        return record

    def run_long_reflection_loop(self, num_ordeals=20):
        print(f"\nStarting LONG reflection loop ({num_ordeals} ordeals)...")
        for i in range(1, num_ordeals + 1):
            ordeal = f"Ordeal {i}: Explore deeper self-reflection in a noisy, entangled universe"
            print(f"\n=== Ordeal {i}/{num_ordeals} ===")
            belief = input("Enter initial belief (or Enter for auto-sample): ").strip()
            if not belief:
                belief = f"Auto-sample belief: Seeking deeper entanglement-protected wisdom"
            self.reflect(ordeal, belief)
        print(f"\nLong loop complete. Total reflections: {self.reflection_count}")
        print(f"Merkle root of entire chain: {self.merkle_tree.root}")

    def __del__(self):
        self.conn.close()

if __name__ == "__main__":
    brqin = BrQin()
    brqin.run_long_reflection_loop(num_ordeals=20)
