# fuzzy_merge_nodes.py - Auto-Bridge with Archetype Naming
import json
import os
from collections import Counter

GRAPH_FILE = "your-current-fork.json"  # Update to your fork

# Predefined archetypes
ARCHETYPES = {
    "sacrifice": "Self-Sacrifice for Greater Good",
    "wisdom": "Quest for Hidden Knowledge",
    "redemption": "Redemption Through Trial",
    "prophecy": "Vision of Fate",
    "force": "Binding Energy of Will"
}

def fuzzy_auto_bridge(min_overlap=3):
    if not os.path.exists(GRAPH_FILE):
        print("Fork file not found.")
        return

    with open(GRAPH_FILE, 'r+') as f:
        data = json.load(f)
        nodes = data["graph"]["nodes"]
        added = 0
        
        keyword_index = {}
        for node in nodes:
            desc = (node["properties"].get("description", "") + " " + node["name"]).lower()
            words = [w for w in desc.split() if len(w) > 4]
            keyword_index[node["id"]] = Counter(words)
        
        for i, node1 in enumerate(nodes):
            id1 = node1["id"]
            keywords1 = keyword_index[id1]
            for node2 in nodes[i+1:]:
                id2 = node2["id"]
                keywords2 = keyword_index[id2]
                shared = keywords1 & keywords2
                overlap = sum(shared.values())
                if overlap >= min_overlap:
                    # Archetype naming
                    archetype = "Shared Theme"
                    for key, name in ARCHETYPES.items():
                        if key in shared:
                            archetype = name
                            break
                    
                    edge = {
                        "source": id1,
                        "target": id2,
                        "type": "FuzzyAutoBridge",
                        "properties": {
                            "archetype": archetype,
                            "overlap_count": overlap,
                            "shared_keywords": list(shared.keys())[:10],
                            "resonance": round(0.85 + (overlap / 20) * 0.13, 4)
                        }
                    }
                    if not any(e["source"] == edge["source"] and e["target"] == edge["target"] for e in data["graph"]["edges"]):
                        data["graph"]["edges"].append(edge)
                        added += 1
        
        f.seek(0)
        json.dump(data, f, indent=2)
        f.truncate()
    
    print(f"Fuzzy merge complete — added {added} bridges (min overlap {min_overlap})")

if __name__ == "__main__":
    fuzzy_auto_bridge(min_overlap=3)  # Try 3, 4, or 5