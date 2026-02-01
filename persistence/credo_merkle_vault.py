# persistence/credo_merkle_vault.py
"""
Mímisbrunnr – Encrypted Append-Only Merkle Log for Credo
MIT License – QILLQuantum/Credo
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

try:
    from cryptography.fernet import Fernet, InvalidToken
except ImportError:
    print("ERROR: cryptography package missing. Run:")
    print("pip install cryptography")
    exit(1)

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
        self.entries: List[Dict] = []
        self.roots: List[str] = []

        if key is None:
            key = Fernet.generate_key()
            print("\n=== NEW FERNET KEY GENERATED ===")
            print("SAVE THIS KEY SECURELY! You will need it every time you want to read the log.")
            print("Key:", key.decode())
            print("To reuse it automatically, run in cmd:")
            print(f'set CREDO_MIMIR_KEY={key.decode()}')
            print("===================================\n")
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
                        print("WARNING: Invalid token skipped (wrong key or corrupted line)")
                    except json.JSONDecodeError:
                        print("WARNING: Invalid JSON skipped")

    def _build_merkle_root(self, leaf_hashes: List[str]) -> Optional[str]:
        if not leaf_hashes:
            return None
        nodes = leaf_hashes[:]
        while len(nodes) > 1:
            next_level = []
            for i in range(0, len(nodes), 2):
                left = nodes[i]
                right = nodes[i+1] if i+1 < len(nodes) else left
                parent = sha256_hex((left + right).encode())
                next_level.append(parent)
            nodes = next_level
        return nodes[0]

    def _update_root(self, new_leaf_hash: str):
        all_leaves = [e["hash"] for e in self.entries]
        root = self._build_merkle_root(all_leaves)
        if root:
            self.roots.append(root)

    def append(self, payload: Dict, entry_type: str = "generic") -> str:
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

        token = self.fernet.encrypt(entry_str.encode('utf-8'))

        with self.log_path.open("ab") as f:
            f.write(token + b"\n")

        self.entries.append(entry)
        self._update_root(entry_hash)

        return entry_hash

    def get_last_root(self) -> Optional[str]:
        return self.roots[-1] if self.roots else None

    def verify_prefix(self, up_to_index: int) -> bool:
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


# ────────────────────────────────────────────────
# FIXED TEST BLOCK – no crash, deletes log every time
# ────────────────────────────────────────────────

if __name__ == "__main__":
    test_log = "./test_mimir_vault.log"

    # Clean start every test run – no more InvalidToken hell
    if os.path.exists(test_log):
        os.remove(test_log)
        print("Test log deleted (clean start)")

    vault = CredoMimirVault(test_log)

    print("Adding test entries...")
    vault.append({"thought": "The vault remembers everything"}, "reflection")
    vault.append({"fact": 42}, "fact")
    vault.append({"source": "BrQin v5.2"}, "meta")

    print(f"\nEntries added: {vault.count()}")
    root = vault.get_last_root()
    print(f"Latest Merkle root: {root[:12]}… if root else 'None'")

    print(f"Prefix up to index 1 valid: {vault.verify_prefix(1)}")
    print("Test complete.")