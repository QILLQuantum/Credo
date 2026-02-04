# credo_logger.py
# Enhanced logging with direct persistence integration

import datetime
from credo_db_facade import CredoDBFacade

# Shared facade instance (singleton pattern for efficiency)
_persistence = CredoDBFacade()

def log_reflection(reflection_id: str, record: dict):
    """
    Log reflection event and persist to DB (SQLite + vault)
    """
    ts = datetime.datetime.now().isoformat()
    print(f"💭 [{ts}] Reflection {reflection_id}")
    print(f"  Ordeal: {record.get('ordeal_context')}")
    print(f"  Belief: {record.get('enriched_belief')[:200]}...")

    # Persist to DB
    _persistence.save_belief(record, "reflection")
    print(f"  [DB] Reflection {reflection_id} saved")

def log_oracle_call(reflection_id: str, metrics: dict):
    """
    Log oracle call and persist metrics
    """
    ts = datetime.datetime.now().isoformat()
    print(f"🔮 [{ts}] Oracle call for {reflection_id}")
    print(f"  Energy: {metrics.get('certified_energy')}")
    print(f"  Bond: {metrics.get('final_avg_bond')} | Growth: {metrics.get('growth_rate')}")

    # Persist to DB
    _persistence.save_belief(metrics, "oracle_call")
    print(f"  [DB] Oracle metrics saved")