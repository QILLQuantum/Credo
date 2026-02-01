from credo_merkle_vault import CredoMerkleVault
from credo_sqlite_store import CredoSQLiteStore
from typing import Dict, Any, List, Optional, Tuple
import os

class CredoDBFacade:
    def __init__(self, encryption_key: Optional[bytes] = None, db_path: str = "brqin_history.db"):
        self.merkle = CredoMerkleVault(encryption_key)
        self.sqlite = CredoSQLiteStore(db_path)
        self.current_step = 0
        self.vault_dir = "brqin_vault"
        os.makedirs(self.vault_dir, exist_ok=True)

    def save_simulation_step(self, peps, energy_nodes: List, observables: Dict[str, Any], 
                             energy: float, mode: str, entropy_delta: float, 
                             syndromes: Optional[Dict] = None) -> bool:
        metadata = {
            "energy": float(energy),
            "mode": mode,
            "observables": observables,
            "entropy_delta": float(entropy_delta),
            "logical_error_estimate": syndromes.get("p_phys", 0.005) if syndromes else 0.005,
            "step": self.current_step
        }

        self.merkle.save_state(peps, energy_nodes, metadata, self.current_step)
        self.merkle.append_event("simulation_step", {
            "step": self.current_step,
            "energy": energy,
            "mode": mode,
            "entropy_delta": entropy_delta
        }, metadata)

        if syndromes:
            self.merkle.append_event("syndrome_measurement", syndromes, {"step": self.current_step})

        self.sqlite.save_step(self.current_step, energy, mode, observables, entropy_delta, syndromes or {})

        self.current_step += 1
        return True

    def load_latest_checkpoint(self):
        return self.merkle.load_state(-1)

    def load_checkpoint(self, step: int):
        return self.merkle.load_state(step)

    def verify_integrity(self) -> Tuple[bool, str]:
        return self.merkle.verify_chain()

    def get_history(self, event_type: Optional[str] = None):
        return self.merkle.get_event_history(event_type)

    def query_observables(self, limit: int = 50):
        return self.sqlite.get_history(limit=limit)

    def get_checkpoint_count(self) -> int:
        return len([f for f in os.listdir(self.vault_dir) if f.startswith("checkpoint_step_")])

if __name__ == "__main__":
    db = CredoDBFacade()
    print("✅ CredoDBFacade (hybrid Merkle Vault + SQLite) initialized")
