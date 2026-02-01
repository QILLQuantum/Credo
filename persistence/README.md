# Credo / persistence

Official persistence layer for Credo/BrQin.

## Backends
- **Mímisbrunn Merkle Vault** (`credo_merkle_vault.py`): Immutable, encrypted, append-only Merkle chain for audit trail, checkpoints, entropy/syndrome events.
- **SQLite Store** (`credo_sqlite_store.py`): Fast, queryable relational store for observables, history lookup, reporting.

## Main Interface
- `credo_db_facade.py` → `CredoDBFacade`: Hybrid facade that writes to both backends automatically.

## Directory Structure Recommendation