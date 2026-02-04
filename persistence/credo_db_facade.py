# persistence/credo_db_facade.py  # or rename to credo_persistence.py

import os
from typing import Dict, List, Optional

from credo_sqlite_store import CredoSQLiteStore
from credo_merkle_vault import CredoMimirVault

class CredoPersistence:  # was CredoDBFacade
    def __init__(
        self,
        db_path: str = "./credo_data/credo.db",
        vault_log: str = "./credo_data/mimir_vault.log",
    ):
        self.sqlite = CredoSQLiteStore(db_path)
        key = os.getenv("CREDO_MIMIR_KEY")
        if key:
            key = key.encode()
        self.vault = CredoMimirVault(vault_log, key=key)

    def save(self, payload: Dict, entry_type: str = "belief") -> str:
        """Save to both layers – returns hash"""
        hash_sql = self.sqlite.persist(payload, entry_type)
        hash_vault = self.vault.append(payload, entry_type)
        if hash_sql != hash_vault:
            print("WARN: hash mismatch")
        return hash_sql

    def get_by_hash(self, entry_hash: str) -> Optional[Dict]:
        return self.sqlite.get_entry_by_hash(entry_hash)

    def recent(self, limit: int = 10) -> List[Dict]:
        return self.sqlite.list_recent_entries(limit=limit)

    def count(self) -> int:
        return self.vault.count()

    def last_root(self) -> Optional[str]:
        return self.vault.get_last_root()

    def close(self):
        self.sqlite.close()


if __name__ == "__main__":
    p = CredoPersistence()
    p.save({"thought": "Facade refined"}, "reflection")
    p.save({"fact": 42}, "fact")
    print(f"Root: {p.last_root()[:12] if p.last_root() else 'None'}…")
    print(f"Count: {p.count()}")
    print("Recent:", p.recent(3))
    p.close()