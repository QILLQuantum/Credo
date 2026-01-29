# credo_multihop_energy_chain.py - Multi-Hop with Piezo Energy Chaining
import asyncio
import random
import json
import os
import time

# CONFIG
DATA_TO_SEND = {"credo_chunk": "Multi-hop wisdom — energy chains eternal"}
HOPS_NEEDED = 2  # Simulate 2 intermediate hops
STORAGE_FOLDER = "multihop_received"
os.makedirs(STORAGE_FOLDER, exist_ok=True)

# Piezo energy sim (per node)
VIBRATION_POWER_MW = 0.5  # Average harvest from motion
EFFICIENCY = 0.15
TX_ENERGY_MJ = 0.05  # Cost per hop transmit

class MeshNode:
    def __init__(self, name, is_receiver=False):
        self.name = name
        self.is_receiver = is_receiver
        self.energy_mj = 0.0  # Starting energy

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
    intermediate = MeshNode("Intermediate")
    receiver = MeshNode("Receiver", is_receiver=True)

    print("Multi-hop energy chaining simulation starting...")
    
    # Transmitter harvests until ready
    while not transmitter.can_transmit():
        transmitter.harvest()
        await asyncio.sleep(1)
    
    print("Transmitter ready — sending to intermediate...")
    if transmitter.transmit():
        # Intermediate receives, harvests, relays
        while not intermediate.can_transmit():
            intermediate.harvest()
            await asyncio.sleep(1)
        
        print("Intermediate ready — relaying to receiver...")
        if intermediate.transmit():
            # Receiver gets data
            filename = f"{STORAGE_FOLDER}/multihop_received_{int(time.time())}.json"
            with open(filename, 'w') as f:
                json.dump(DATA_TO_SEND, f, indent=2)
            print(f"Receiver got data — stored {filename}")
            print("Data:", DATA_TO_SEND)
            print("Multi-hop chain complete — energy chained across nodes")
        else:
            print("Intermediate low energy — hop failed")
    else:
        print("Transmitter low energy — chain failed")

if __name__ == "__main__":
    asyncio.run(simulate_chain())