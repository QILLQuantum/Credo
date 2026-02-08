# daily_oracle_heartbeat_v2.py - Heartbeat with BrQin v5.3 quantum oracle + fuzzy collapse
# Run: python daily_oracle_heartbeat_v2.py

import json
from pathlib import Path
from fuzzywuzzy import fuzz
from collections import defaultdict
import random
import time
from logs.credo_logger import logger

# BrQin v5.3 import (save as brqin_v5_3.py in same dir)
from brqin_v5_3 import BrQin

GRAPH_FILE = Path("core/master_graph_merged_20260129.json")
ARCHIVE_FILE = Path("core/archive_collapsed_nodes.json")

# Initialize BrQin oracle
brqin = BrQin()

def benchmark(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__name__} completed in {elapsed:.3f}s")
        return result
    return wrapper

@benchmark
def heartbeat_v2():
    logger.info("Heartbeat v2 started — BrQin quantum uplift + fuzzy collapse")
    data = load_graph(GRAPH_FILE)

    # Compute dynamic thresholds
    sim_threshold, res_trigger, avg_open = compute_dynamic_thresholds(data)

    # Select candidates
    candidates = [
        n for n in data["graph"]["nodes"]
        if int(n["properties"].get("open_for_uplift", "0%").split("%")[0]) > (avg_open / 2)
        or n["properties"].get("resonance", 0.85) < res_trigger
    ]
    logger.info(f"Selected {len(candidates)} candidates for quantum uplift")

    # BrQin quantum uplift
    uplifted = 0
    for node in candidates:
        props = node["properties"]
        name = node.get("name", "Unknown")
        desc = props.get("description", "")
        ordeal = f"Ordeal for {name}: Deepen resonance in entangled wisdom"
        
        record = brqin.reflect(ordeal_context=ordeal, initial_belief=desc)
        
        # Add enriched belief as uplift snippet
        props.setdefault("discovered_uplift", []).append(record["enriched_belief"])
        
        # Update resonance from certified energy (quantum uplift)
        quantum_res = record["oracle_metrics"]["certified_energy"]
        props["resonance"] = min(0.99, max(props.get("resonance", 0.85), quantum_res))
        props["val"] = int(props["resonance"] * 100)
        
        # Reduce open %
        current_open = int(props.get("open_for_uplift", "15%").split("%")[0])
        new_open = max(0, current_open - 10)
        props["open_for_uplift"] = f"{new_open}% remaining (BrQin quantum uplift)"
        
        uplifted += 1

    # Fuzzy collapse with tradition protection
    merged_nodes, archive_entries = fuzzy_merge_nodes(data["graph"]["nodes"], threshold=sim_threshold)
    data["graph"]["nodes"] = merged_nodes

    # Archive & save
    if archive_entries:
        with open(ARCHIVE_FILE, 'a', encoding='utf-8') as f:
            for entry in archive_entries:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
        logger.info(f"Archived {len(archive_entries)} collapsed nodes")

    if uplifted or archive_entries:
        save_graph(GRAPH_FILE, data)
        logger.info(f"Heartbeat v2 complete — {uplifted} quantum uplifted, {len(archive_entries)} collapsed")
    else:
        logger.info("Heartbeat v2 complete — no changes")

if __name__ == "__main__":
    heartbeat_v2()