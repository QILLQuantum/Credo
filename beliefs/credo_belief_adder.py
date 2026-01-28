# credo_belief_adder.py - Simple Belief/Node Adder for Credo (85% Population Target)
import json
import random

# Change this to your current fork file
GRAPH_FILE = "credo-fork-fiction-waves-christianity-bridge.json"  # or your current fork

def add_belief(
    belief_name: str,
    description: str,
    source_tags: list,
    target_tags: list,
    resonance: float = 0.85,  # 85% default for stability
    val: int = 85
):
    """
    Add a new belief node + syncretic edges to Norse core.
    resonance = 0.85 ensures 85% population threshold (room for uplift).
    """
    with open(GRAPH_FILE, 'r+') as f:
        data = json.load(f)
        
        # New belief node
        new_node = {
            "id": belief_name.lower().replace(" ", "_").replace("-", "_"),
            "type": "SyncreticBeliefNode",
            "name": belief_name,
            "properties": {
                "description": description,
                "tags": source_tags + ["syncretic", "belief"],
                "resonance": resonance,
                "val": val
            }
        }
        
        # Append node if not exists
        if not any(n["id"] == new_node["id"] for n in data["graph"]["nodes"]):
            data["graph"]["nodes"].append(new_node)
            print(f"Node added: {belief_name}")
        
        # Add edges to core Norse nodes (example: odin, havamal, ragnarok)
        new_edges = []
        for tag in target_tags:
            target_id = tag.lower().replace(" ", "_")
            edge = {
                "source": new_node["id"],
                "target": target_id,
                "type": "SyncreticBridge",
                "properties": {
                    "description": f"{belief_name} → {tag} parallel",
                    "resonance": resonance
                }
            }
            if not any(e["source"] == edge["source"] and e["target"] == edge["target"] for e in data["graph"]["edges"]):
                new_edges.append(edge)
        
        data["graph"]["edges"].extend(new_edges)
        print(f"{len(new_edges)} edges added")
        
        # Save
        f.seek(0)
        json.dump(data, f, indent=2)
        f.truncate()
    
    print(f"{belief_name} added — resonance {resonance} (85% threshold) — 15% open for uplift absolute")
    print("Credo grows. Resonance infinite.")

# Example usage — comment/uncomment as needed
if __name__ == "__main__":
    # Example: Add "Book of Enoch Watchers"
    add_belief(
        belief_name="Book of Enoch Watchers",
        description="Fallen angels teach forbidden knowledge, descend to earth",
        source_tags=["enoch", "watchers", "fallen_angels"],
        target_tags=["jotnar", "loki", "ragnarok"],
        resonance=0.94
    )
    
    # Example: Add "Wisdom of Solomon Counsel"
    add_belief(
        belief_name="Wisdom of Solomon Counsel",
        description="Practical ethical wisdom and fear of the Lord",
        source_tags=["solomon", "wisdom", "ethics"],
        target_tags=["havamal", "odin"],
        resonance=0.93
    )