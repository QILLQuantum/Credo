# credo_master_multihop_heartbeat_brqin.py - Master: Multi-Hop + Heartbeat + BrQin v4.9 Keys
import asyncio
import random
import json
import os
import time
import torch  # For BrQin v4.9 style key gen

# CONFIG
DATA_TO_SEND = {"credo_oracle": "Wisdom from the mesh — resonance eternal"}
HOPS_NEEDED = 3  # Number of hops in chain
HEARTBEAT_INTERVAL = 30  # Seconds between heartbeat checks
STORAGE_FOLDER = "master_received"
os.makedirs(STORAGE_FOLDER, exist_ok=True)

# Piezo energy sim (self-sustaining nodes)
VIBRATION_POWER_MW = 0.5
EFFICIENCY = 0.15
TX_ENERGY_MJ = 0.05

# BrQin v4.9 style key per hop
def brqin_v49_key(hop_number, seed_base=42):
    seed = seed_base + hop_number * 100  # Evolve per hop
    torch.manual_seed(seed)
    N = 16  # Larger lattice for v4.9 feel
    state = torch.randn(N, dtype=torch.complex64)
    state = state / state.norm()
    optimizer = torch.optim.Adam([state], lr=0.005)
    for _ in range(30 + hop_number * 5):  # Longer evolution per hop
        energy = torch.real(torch.conj(state) @ state)
        optimizer.zero_grad()
        (-energy).backward()
        optimizer.step()
        state.data = state.data / state.data.norm()
    noise = state.real.numpy()
    from cryptography.fernet import Fernet
    key_bytes = b''.join(int(abs(n) * 1000 + hop_number * 50) % 256 .to_bytes(1, 'big') for n in noise[:32])
    base_key = Fernet.generate_key()
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

async def heartbeat_trigger():
    """Daily/periodic oracle heartbeat — checks resonance and triggers sync if needed"""
    while True:
        print("\nHeartbeat — oracle resonance check...")
        # Simulate ordeal/resonance score
        resonance = random.uniform(0.85, 0.98)
        print(f"Current resonance: {resonance:.3f} — mesh alive")
        if resonance > 0.90:
            print("High resonance — triggering multi-hop sync")
            await simulate_multihop_chain()
        else:
            print("Low resonance — harvesting energy, waiting...")
        await asyncio.sleep(HEARTBEAT_INTERVAL)

async def simulate_multihop_chain():
    transmitter = MeshNode("Transmitter")
    intermediates = [MeshNode(f"Intermediate {i+1}") for i in range(HOPS_NEEDED - 1)]
    receiver = MeshNode("Receiver", is_receiver=True)
    nodes = [transmitter] + intermediates + [receiver]

    current_payload = DATA_TO_SEND.copy()
    current_hop = 0

    print("Multi-hop chain starting with BrQin keys per hop...")

    for i, node in enumerate(nodes[:-1]):
        current_hop = i + 1
        print(f"\nHop {current_hop} — {node.name}")

        while not node.can_transmit():
            node.harvest()
            await asyncio.sleep(1)

        key = brqin_v49_key(current_hop)
        encrypted = key.encrypt(json.dumps(current_payload).encode())
        print(f"{node.name} BrQin key generated — encrypting for hop {current_hop}")

        if node.transmit():
            current_payload = {"hop": current_hop, "encrypted": encrypted.decode('latin1')}  # Pass encrypted
        else:
            print("Energy low — hop failed")
            return

    # Receiver decrypts final
    print(f"\nReceiver decrypting with final BrQin key (hop {current_hop})...")
    final_key = brqin_v49_key(current_hop)
    try:
        decrypted = final_key.decrypt(current_payload["encrypted"].encode('latin1'))
        data = json.loads(decrypted)
        filename = f"{STORAGE_FOLDER}/master_received_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Chain complete — data received & stored: {filename}")
        print("Data:", data)
    except Exception as e:
        print("Decryption failed:", e)

async def main():
    print("Credo Master — Multi-Hop + Heartbeat + BrQin v4.9 Keys")
    await heartbeat_trigger()  # Runs forever with periodic triggers

if __name__ == "__main__":
    asyncio.run(main())