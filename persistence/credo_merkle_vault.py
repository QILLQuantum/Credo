# persistence/credo_merkle_vault.py
"""
Mímisbrunnr – Encrypted Append-Only Merkle Log for Credo
Each entry is encrypted with Fernet (AES-128-CBC + HMAC-SHA256).
Merkle root is computed over plaintext hashes for tamper detection.
MIT License – QILLQuantum/Credo
"""

import hashlib
import json
import os
from base64 import urlsafe_b64encode, urlsafe_b64decode
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from cryptography.fernet import Fernet, InvalidToken

def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

class CredoMimirVault:
    def __init__(
        self,
        log_path: str = "./credo_data/mimir_vault.log",
        key: Optional[bytes] = None,
    ):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: List[Dict] = []           # decrypted in memory
        self.roots: List[str] = []              # historical Merkle roots

        if key is None:
            # Development / first run: generate new key (SAVE THIS SECURELY!)
            key = Fernet.generate_key()
            print("[WARN] New Fernet key generated. Save it securely!")
            print("[WARN] Example: set CREDO_MIMIR_KEY=" + key.decode())
        self.fernet = Fernet(key)

        self._load_log()

    def _load_log(self):
        if not self.log_path.exists():
            return
        with self.log_path.open("rb") as f:
            for line in f:
                if line.strip():
                    try:
                        token = line.strip()
                        plaintext = self.fernet.decrypt(token)
                        entry = json.loads(plaintext)
                        self.entries.append(entry)
                        self._update_root(entry["hash"])
                    except InvalidToken:
                        raise ValueError("Vault corruption: Invalid Fernet token")
                    except json.JSONDecodeError:
                        raise ValueError("Vault corruption: Invalid JSON after decryption")

    def _build_merkle_root(self, leaf_hashes: List[str]) -> Optional[str]:
        if not leaf_hashes:
            return None
        nodes = leaf_hashes[:]
        while len(nodes) > 1:
            next_level = []
            i = 0
            while i < len(nodes):
                left = nodes[i]
                right = nodes[i + 1] if i + 1 < len(nodes) else left
                parent = sha256_hex((left + right).encode())
                next_level.append(parent)
                i += 2
            nodes = next_level
        return nodes[0]

    def _update_root(self, new_leaf_hash: str):
        all_leaves = [e["hash"] for e in self.entries]
        root = self._build_merkle_root(all_leaves)
        if root:
            self.roots.append(root)

    def append(self, payload: Dict, entry_type: str = "generic") -> str:
        """Append entry → encrypted on disk, plaintext in memory, new root computed"""
        ts = datetime.now(timezone.utc).isoformat()
        entry = {
            "ts": ts,
            "type": entry_type,
            "payload": payload,
            "prev_root": self.roots[-1] if self.roots else None,
        }
        entry_str = json.dumps(entry, sort_keys=True, separators=(',', ':'))
        entry_hash = sha256_hex(entry_str.encode('utf-8'))
        entry["hash"] = entry_hash

        # Encrypt full entry
        token = self.fernet.encrypt(entry_str.encode('utf-8'))

        # Append atomically (binary mode)
        with self.log_path.open("ab") as f:
            f.write(token + b"\n")

        self.entries.append(entry)
        self._update_root(entry_hash)

        return entry_hash

    def get_last_root(self) -> Optional[str]:
        return self.roots[-1] if self.roots else None

    def verify_prefix(self, up_to_index: int) -> bool:
        """Verify that history up to index matches recorded root"""
        if up_to_index < 0 or up_to_index >= len(self.entries):
            return False
        prefix_hashes = [self.entries[i]["hash"] for i in range(up_to_index + 1)]
        computed = self._build_merkle_root(prefix_hashes)
        return computed == self.roots[up_to_index]

    def count(self) -> int:
        return len(self.entries)

    @classmethod
    def generate_key(cls) -> bytes:
        return Fernet.generate_key()


# Quick smoke test / demo
if __name__ == "__main__":
    vault = CredoMimirVault("./test_mimir_vault.log")

    vault.append({"thought": "The vault remembers everything"}, "reflection")
    vault.append({"fact": 42}, "fact")
    vault.append({"source": "BrQin v5.2"}, "meta")

    print(f"Entries in vault: {vault.count()}")
    print(f"Latest Merkle root: {vault.get_last_root()[:12]}…")
    print(f"Prefix up to 1 is valid: {vault.verify_prefix(1)}")

    # Simulate reload / corruption check
    vault2 = CredoMimirVault("./test_mimir_vault.log")
    print(f"Reloaded entries: {vault2.count()}")