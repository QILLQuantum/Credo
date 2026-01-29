# credo_multihop_brqin_key.py - Multi-Hop with BrQin Key Per Hop
import asyncio
import random
import json
import os
import time
import torch  # For BrQin sim

# CONFIG
DATA_TO_SEND = {"credo_chunk": "Multi-hop wisdom — BrQin keys chain eternal"}
HOPS_NEEDED = 3  # Simulate 3 hops (transmitter → intermediate1 → intermediate2 → receiver)
STORAGE_FOLDER = "multihop_brqin_received"
os.makedirs(STORAGE_FOLDER, exist_ok=True)

# Piezo energy sim (per node)
VIBRATION_POWER_MW = 0.5
EFFICIENCY = 0.15
TX_ENERGY_MJ = 0.05

# BrQin key per hop
def brqin_key_per_hop(hop_number, seed_base=42):
    seed = seed_base + hop_number  # Evolve seed per hop
    torch.manual_seed(seed)
    N = 12
    state = torch.randn(N, dtype=torch.complex64)
    state = state / state.norm()
    optimizer = torch.optim.Adam([state], lr=0.01)
    for _ in range(20 + hop_number):  # Slightly longer per hop for "evolution"
        energy = torch.real(torch.conj(state) @ state)
        optimizer.zero_grad()
        (-energy).backward()
        optimizer.step()
        state.data = state.data / state.data.norm()
    noise = state.real.numpy()
    key_bytes = b''.join(int(abs(n) * 1000 + hop_number * 10) % 256 .to_bytes(1, 'big') for n in noise[:32])
    from cryptography.fernet import Fernet
    base_key = Fernet.generate_key()
    # Mix with noise for flavor
    mixed = bytes(a ^ b for a, b in zip(base_key, key_bytes[:len(base_key)]))
    return Fernet(mixed)

class MeshNode:
    def __init__(self, name, is_receiver=False):
        self.name = name
        self.is_receiver = is_receiver
        self.energy_mj = 0.0

    def harvest(self, seconds=10):
        harvested_mw = VIBRATION_POWER_MW * EFFICIENCY
        harvested_mj = harvested_mw * seconds
        self.energy_mj += harvested_mj
        print(f"{self.name} harvested {harvested_mj:.3f} mJ — total {self.energy_mj:.3f} mJ")

    def can_transmit(self):
        return self.energy_mj >= TX_ENERGY_MJ

    def transmit(self):
        if self.can_transmit():
            self.energy_mj -= TX_ENERGY_MJ
            return True
        return False

async def simulate_chain():
    transmitter = MeshNode("Transmitter")
    intermediates = [MeshNode(f"Intermediate {i+1}") for i in range(HOPS_NEEDED-1)]
    receiver = MeshNode("Receiver", is_receiver=True)
    nodes = [transmitter] + intermediates + [receiver]

    current_data = DATA_TO_SEND.copy()
    current_hop = 0

    print("Multi-hop with BrQin key per hop starting...")

    for i, node in enumerate(nodes[:-1]):  # All but receiver
        current_hop = i + 1
        print(f"\nHop {current_hop} — {node.name} preparing...")
        
        # Harvest until ready
        while not node.can_transmit():
            node.harvest()
            await asyncio.sleep(1)
        
        # Generate BrQin key for this hop
        key = brqin_key_per_hencrypted = key.encrypt(json.dumps(current_data).encode())
        print(f"{node.name} generated BrQin key for hop {current_hop} — encrypting & transmitting...")
        
        if node.transmit():
            current_data = {"hop": current_hop, "encrypted_payload": encrypted.decode()}  # Simulate pass
            print(f"Hop {current_hop} complete — data forwarded")
        else:
            print(f"Hop {current_hop} failed — low energy")
            return

    # Receiver gets final payload
    print(f"\nReceiver got final payload — decrypting with last BrQin key...")
    final_key = brqin_key_per_hop(current_hop)  # Match last hop
    try:
        decrypted = final_key.decrypt(current_data["encrypted_payload"].encode())
        data = json.loads(decrypted)
        filename = f"{STORAGE_FOLDER}/multihop_brqin_received_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Multi-hop chain complete — data decrypted & stored: {filename}")
        print("Data:", data)
    except:
        print("Final decryption failed")

if __name__ == "__main__":
    asyncio.run(simulate_chain())