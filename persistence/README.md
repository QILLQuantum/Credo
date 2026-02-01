# persistence/

Official persistence layer for Credo/BrQin.

## Files
- `credo_merkle_vault.py` — Immutable encrypted append-only Merkle vault (Mímisbrunn)
- `credo_sqlite_store.py` — Fast SQLite relational store for queries
- `credo_db_facade.py` — Hybrid facade (writes to both backends)

## Usage
```python
from credo_db_facade import CredoDBFacade

db = CredoDBFacade()
db.save_simulation_step(peps, nodes, observables, energy, mode, entropy_delta, syndromes)
checkpoint = db.load_latest_checkpoint()
ok, msg = db.verify_integrity()
