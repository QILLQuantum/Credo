import hashlib
import json
import os

class CredoMimirVault:
    def __init__(self, log_path="credo_data/mimir_vault.log"):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_path = log_path
        self.current_root = None
        if os.path.exists(log_path):
            with open(log_path) as f:
                lines = f.readlines()
                if lines:
                    self.current_root = lines[-1].strip().split("|")[0]

    def append(self, payload: dict, entry_type: str) -> str:
        payload_str = json.dumps(payload)
        entry_hash = hashlib.sha256(payload_str.encode()).hexdigest()
        prev_root = self.current_root or "genesis"
        new_root = hashlib.sha256((prev_root + entry_hash).encode()).hexdigest()
        timestamp = datetime.datetime.now().isoformat()
        line = f"{new_root}|{entry_hash}|{entry_type}|{timestamp}\n"
        with open(self.log_path, "a") as f:
            f.write(line)
        self.current_root = new_root
        return entry_hash

    def get_last_root(self):
        return self.current_root
