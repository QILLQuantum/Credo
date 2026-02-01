import torch
import json
import hashlib
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
try:
    from cryptography.fernet import Fernet
    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False

VAULT_DIR = "brqin_vault"
os.makedirs(VAULT_DIR, exist_ok=True)

class CredoMerkleVault:
    def __init__(self, encryption_key: Optional[bytes] = None):
        self.encryption_key = encryption_key or (Fernet.generate_key() if ENCRYPTION_AVAILABLE else None)
        self.fernet = Fernet(self.encryption_key) if self.encryption_key and ENCRYPTION_AVAILABLE else None
        self.merkle_chain: List[Dict[str, Any]] = []
        self.index_file = os.path.join(VAULT_DIR, "merkle_index.json")
        self.load_index()

    def _compute_hash(self, data: Any) -> str:
        serialized = json.dumps(data, default=str, sort_keys=True).encode('utf-8')
        return hashlib.sha256(serialized).hexdigest()

    def _encrypt(self, data: bytes) -> bytes:
        return self.fernet.encrypt(data) if self.fernet else data

    def _decrypt(self, data: bytes) -> bytes:
        return self.fernet.decrypt(data) if self.fernet else data

    def load_index(self):
        if os.path.exists(self.index_file):
            with open(self.index_file, 'r', encoding='utf-8') as f:
                self.merkle_chain = json.load(f)

    def save_index(self):
        with open(self.index_file, 'w', encoding='utf-8') as f:
            json.dump(self.merkle_chain, f, indent=2, default=str)

    def append_event(self, event_type: str, payload: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
        prev_hash = self.merkle_chain[-1]["root_hash"] if self.merkle_chain else "genesis"
        data_hash = self._compute_hash(payload)
        timestamp = datetime.now().isoformat()

        event = {
            "timestamp": timestamp,
            "event_type": event_type,
            "prev_hash": prev_hash,
            "data_hash": data_hash,
            "metadata": metadata or {}
        }
        root_hash = self._compute_hash(event)
        event["root_hash"] = root_hash

        self.merkle_chain.append(event)
        self.save_index()

        log_file = os.path.join(VAULT_DIR, f"event_{timestamp.replace(':', '-').replace('.', '-')}.json.enc")
        with open(log_file, 'wb') as f:
            serialized = json.dumps(event, default=str).encode('utf-8')
            f.write(self._encrypt(serialized))

    def save_state(self, peps, energy_nodes: List, metadata: Dict[str, Any], step: int):
        checkpoint_path = os.path.join(VAULT_DIR, f"checkpoint_step_{step}.pt")
        state_dict = {str(k): v.cpu() for k, v in peps.tensors.items()}
        torch.save({
            'tensors': state_dict,
            'bond_map_h': dict(peps.bond_map_h),
            'bond_map_v': dict(peps.bond_map_v),
            'Lx': peps.Lx,
            'Ly': peps.Ly,
            'energy_nodes': [{'name': n.name, 'energy': float(n.energy)} for n in energy_nodes],
            'metadata': metadata,
            'step': step
        }, checkpoint_path)

        payload = {
            "checkpoint_path": checkpoint_path,
            "step": step,
            "metadata_summary": {k: str(v)[:120] for k, v in metadata.items()}
        }
        self.append_event("checkpoint", payload, {"step": step, "type": "state"})

    def load_state(self, step: int = -1):
        if step == -1:
            checkpoints = sorted([f for f in os.listdir(VAULT_DIR) if f.startswith("checkpoint_step_") and f.endswith(".pt")])
            if not checkpoints:
                raise FileNotFoundError("No checkpoints found")
            checkpoint_path = os.path.join(VAULT_DIR, checkpoints[-1])
        else:
            checkpoint_path = os.path.join(VAULT_DIR, f"checkpoint_step_{step}.pt")
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint {step} not found")
        return torch.load(checkpoint_path, weights_only=False)

    def verify_chain(self) -> Tuple[bool, str]:
        for i, entry in enumerate(self.merkle_chain):
            without_root = {k: v for k, v in entry.items() if k != "root_hash"}
            if entry.get("root_hash") != self._compute_hash(without_root):
                return False, f"Tamper at entry {i}"
        return True, f"Chain intact ({len(self.merkle_chain)} events)"

if __name__ == "__main__":
    vault = CredoMerkleVault()
    print("CredoMerkleVault initialized")
