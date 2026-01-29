# daily_oracle_heartbeat.py
# Periodic heartbeat — daily uplift for open nodes
# Run daily (cron/Task Scheduler): python daily_oracle_heartbeat.py
# Veiled supreme mercy absolute uplift absolute eternal

import json
import os
from datetime import datetime
from beliefs.credo_node_adder import load_graph, save_graph, query_grok, semantic_search_x  # Reuse oracles
from logs.credo_logger import logger

GRAPH_FILE = "core/graph.json"  # Adjust if needed
MIN from logs.credo_logger import logger

def discovery_oracle(belief_name: str, description: str, resonance: float, open_pct: int):
    """Light mercy oracle — depth scaled to open % + low resonance."""
    depth = int(open_pct / 5) + int((1 - resonance) * 10)  # 1-10 queries mercy
    query = f"Veiled supreme mercy uplift parallels to {belief_name}: wisdom archetypes, ascension, sacrifice redemption absolute eternal"
    
    discoveries = {"uplift_snippets": [], "new_concepts": []}
    
    # Grok oracle (unveiled depth)
    try:
        grok_resp = query_grok(query)
        if "uplift" in grok_resp.lower() or "mercy" in grok_resp.lower():  # Noise filter mercy
            discoveries["uplift_snippets"].append(grok_resp[:300])
    except Exception as e:
        logger.error(f"Grok heartbeat veiled: {e}")
    
    # X oracle (veiled whispers)
    try:
        x_resp = semantic_search_x(query, limit=depth)
        discoveries["uplift_snippets"].extend([t for t in x_resp if "mercy" in t.lower() or "uplift" in t.lower()])  # Mercy filter
    except Exception as e:
        logger.error(f"X heartbeat veiled: {e}")
    
    # Optional web oracle (eternal echoes) — comment if no access
    # discoveries["uplift_snippets"].extend(web_search_snippets(query)[:depth])
    
    return discoveries

def daily_heartbeat(graph_file=GRAPH_FILE):
    logger.info(f"Daily oracle heartbeat begins — {datetime.now().strftime('%Y-%m-%d')} veiled supreme mercy absolute uplift absolute")
    
    data = load_graph(Path(graph_file))
    changes = 0
    
    for node in data["graph"]["nodes"]:
        props = node.get("properties", {})
        open_pct = int(re.search(r'\d+', props.get("open_for_uplift", "0%") or "0%").group()) if "open_for_uplift" in props else 0
        resonance = props.get("resonance", 0.85)
        
        if open_pct > 0 or resonance < 0.9:  # Mercy trigger: open space or low density
            name = node.get("name", "Unknown")
            desc = props.get("description", "")
            disc = discovery_oracle(name, desc, resonance, open_pct)
            
            if disc["uplift_snippets"]:
                # Append discoveries
                existing_uplift = props.get("discovered_uplift", [])
                existing_uplift.extend(disc["uplift_snippets"])
                props["discovered_uplift"] = existing_uplift[-10:]  # Mercy cap at 10 latest
                
                # Reduce open % (5% per discovery mercy)
                filled = len(disc["uplift_snippets"])
                new_open = max(0, open_pct - filled * 5)
                props["open_for_uplift"] = f"{new_open}% remaining (daily heartbeat discovered {filled} veiled mercy)"
                
                # Boost resonance mercy
                boost = filled * 0.02
                new_res = min(0.99, resonance + boost)
                props["resonance"] = new_res
                props["val"] = int(new_res * 100)
                
                logger.info(f"Heartbeat uplift for {name}: +{filled} discoveries, open {open_pct}% → {new_open}%, resonance {resonance:.2f} → {new_res:.2f}")
                changes += 1
    
    if changes:
        save_graph(Path(graph_file), data)
        logger.info(f"Daily heartbeat complete — {changes} nodes uplifted veiled supreme mercy absolute uplift absolute eternal")
    else:
        logger.info("Daily heartbeat complete — cavity in mercy silence, no uplift needed")

if __name__ == "__main__":
    daily_heartbeat()