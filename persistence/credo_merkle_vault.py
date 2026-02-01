# persistence/credo_persistence.py
"""
Credo Persistence Facade – combines SQLite (fast queries) + Mímisbrunnr vault (immutable encrypted log)
MIT License – QILLQuantum/Credo
"""

import os
from .credo_sqlite_store import CredoSQLiteStore
from .credo_merkle_vault import CredoMimirVault

class CredoPersistence:
    def __init__(
        self,
        db_path: str = "./credo_data/credo.db",
        vault_log: str = "./credo_data/mimir_vault.log",
        vault_key: bytes | None = None,
    ):
        self.store = CredoSQLiteStore(db_path)
        # Use env var if set, else generate (warn)
        if vault_key is None and os.getenv("CREDO_MIMIR_KEY"):
            vault_key = os.getenv("CREDO_MIMIR_KEY").encode()
        self.vault = CredoMimirVault(vault_log, key=vault_key)

    def persist(self, payload: dict, entry_type: str = "belief") -> str:
        """Write to both SQLite (queryable) and vault (immutable) – returns hash"""
        h = self.store.persist(payload, entry_type)
        vault_h = self.vault.append(payload, entry_type)

        # Safety check – hashes should match
        if h != vault_h:
            print("WARNING: Hash mismatch between store and vault")

        return h

    def get_current_root(self) -> str | None:
        return self.vault.get_last_root()

    def recent(self, limit: int = 10) -> list[dict]:
        return self.store.recent(limit=limit)

    def close(self):
        self.store.close()
