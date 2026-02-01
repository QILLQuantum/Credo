import sqlite3
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

class CredoSQLiteStore:
    def __init__(self, db_path: str = "brqin_history.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS simulation_steps (
                    step INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    energy REAL,
                    mode TEXT,
                    observables TEXT,
                    entropy_delta REAL,
                    syndromes TEXT
                )
            ''')

    def save_step(self, step: int, energy: float, mode: str, observables: Dict, entropy_delta: float, syndromes: Dict):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO simulation_steps 
                (step, timestamp, energy, mode, observables, entropy_delta, syndromes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                step,
                datetime.now().isoformat(),
                float(energy),
                mode,
                json.dumps(observables),
                float(entropy_delta),
                json.dumps(syndromes)
            ))

    def get_history(self, limit: int = 100):
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                'SELECT * FROM simulation_steps ORDER BY step DESC LIMIT ?', (limit,)
            ).fetchall()
            return [{
                "step": r[0],
                "timestamp": r[1],
                "energy": r[2],
                "mode": r[3],
                "observables": json.loads(r[4]),
                "entropy_delta": r[5],
                "syndromes": json.loads(r[6]) if r[6] else {}
            } for r in rows]

    def load_latest(self):
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                'SELECT * FROM simulation_steps ORDER BY step DESC LIMIT 1'
            ).fetchone()
            if row:
                return {
                    "step": row[0],
                    "timestamp": row[1],
                    "energy": row[2],
                    "mode": row[3],
                    "observables": json.loads(row[4]),
                    "entropy_delta": row[5],
                    "syndromes": json.loads(row[6]) if row[6] else {}
                }
        return {}

if __name__ == "__main__":
    store = CredoSQLiteStore()
    print("✅ CredoSQLiteStore initialized successfully")
