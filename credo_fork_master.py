# credo_fork_master.py - One-Click Fork Tool for Credo Core
import shutil
import datetime
import os
import sys

# Frozen core file (your immutable original)
FROZEN_CORE = "credo-core-norse-frozen-v1.json"

def create_fork(layer_name):
    """Create a new fork for a specific layer (e.g. fiction, quantum)."""
    if not os.path.exists(FROZEN_CORE):
        print(f"Error: Frozen core '{FROZEN_CORE}' not found.")
        return

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    fork_filename = f"credo-fork-{layer_name.lower().replace(' ', '-')}-{timestamp}.json"
    
    shutil.copy(FROZEN_CORE, fork_filename)
    
    # Add fork metadata
    with open(fork_filename, 'r+') as f:
        data = json.load(f)
        data["metadata"] = data.get("metadata", {})
        data["metadata"]["fork_from"] = "Credo-Core-Norse-Final-FROZEN-v1"
        data["metadata"]["fork_layer"] = layer_name
        data["metadata"]["fork_date"] = timestamp
        data["metadata"]["status"] = "Active development fork – core remains immutable"
        f.seek(0)
        json.dump(data, f, indent=2)
        f.truncate()
    
    print(f"Fork created successfully!")
    print(f"New file: {fork_filename}")
    print(f"Layer: {layer_name}")
    print(f"Use this file for all work on '{layer_name}' – never modify the original frozen core.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python credo_fork_master.py \"Layer Name\"")
        print("Examples:")
        print("  python credo_fork_master.py \"Fiction Waves\"")
        print("  python credo_fork_master.py \"Quantum Neuro Layer\"")
        print("  python credo_fork_master.py \"Celtic Tradition\"")
        sys.exit(1)
    
    layer_name = " ".join(sys.argv[1:])
    create_fork(layer_name)