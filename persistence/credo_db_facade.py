import os
from credo_sqlite_store import CredoSQLiteStore
from credo_merkle_vault import CredoMimirVault

class CredoPersistence:
    def __init__(self):
        self.sqlite = CredoSQLiteStore()
        self.vault = CredoMimirVault()

    def save(self, payload: dict, entry_type: str = "belief") -> str:
        hash_sql = self.sqlite.persist(payload, entry_type)
        hash_vault = self.vault.append(payload, entry_type)
        if hash_sql != hash_vault:
            print("WARN: hash mismatch")
        return hash_sql

    def recent(self, limit=10):
        return self.sqlite.list_recent_entries(limit)

    def last_root(self):
        return self.vault.get_last_root()

    def count(self):
        return self.sqlite.count()  # Add count to store if needed

    def close(self):
        self.sqlite.close()
