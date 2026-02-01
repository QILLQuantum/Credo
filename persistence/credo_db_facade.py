from persistence.persistence import MimisbrunnVault
from typing import Dict, Any, List, Optional
import torch

class CredoDBFacade:
    """High-level DB facade for Credo/BrQin using MimisbrunnVault"""
    def __init__(self, encryption_key: Optional[bytes] = None):
        self.vault = MimisbrunnVault(encryption_key)
        self.current_step = 0

    def save_simulation_step(self, peps, energy_nodes, observables: Dict, energy: float, mode: str, entropy_delta: float, syndromes: Optional[Dict] = None):
        metadata = {
            "energy": float(energy),
            "mode": mode,
            "observables": observables,
            "entropy_delta": float(entropy_delta),
            "logical_error_estimate": syndromes.get("p_phys", 0.005) if syndromes else 0.005
        }
        self.vault.save_state(peps, energy_nodes, metadata, self.current_step)
        self.vault.append_event("simulation_step", {
            "step": self.current_step,
            "energy": energy,
            "mode": mode,
            "entropy_delta": entropy_delta
        }, metadata)
        if syndromes:
            self.vault.append_event("syndrome_measurement", syndromes, {"step": self.current_step})
        self.current_step += 1
        return True

    def load_latest_checkpoint(self):
        return self.vault.load_state(-1)

    def load_checkpoint(self, step: int):
        return self.vault.load_state(step)

    def verify_integrity(self):
        return self.vault.verify_chain()

    def get_history(self, event_type: Optional[str] = None):
        return self.vault.get_event_history(event_type)

    def get_checkpoint_count(self) -> int:
        return len([f for f in os.listdir("brqin_vault") if f.startswith("checkpoint_step_")])

if __name__ == "__main__":
    db = CredoDBFacade()
    print("✅ CredoDBFacade initialized")
    print(f"Encryption: {db.vault.fernet is not None}")