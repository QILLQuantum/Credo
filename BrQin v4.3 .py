import numpy as np
import torch
import asyncio
import time
import psutil
import os

print("=== BrQin v4.4 - 5x5 Energy-Chained DTO Demo ===\n")

def print_mem(label=""):
    mb = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
    print(f"[{label}] Memory: {mb:.2f} MB")
    return mb

baseline = print_mem("Baseline")

class EntanglementEnergyNode:
    def __init__(self, name):
        self.name = name
        self.energy = 0.0

    def harvest(self, local_entropy):
        harvested = max(0.0, local_entropy - 0.3)
        self.energy += harvested
        return harvested

    def can_mutate(self, cost=0.15):
        return self.energy >= cost

    def spend(self, cost=0.15):
        if self.can_mutate(cost):
            self.energy -= cost
            return True
        return False

# 5x5 grid simulation
Lx, Ly = 5, 5
N = Lx * Ly
current_state = torch.randn(N * 2, dtype=torch.complex64)

# 4 nodes (simulating partitions)
nodes = [EntanglementEnergyNode(f"Node{i}") for i in range(4)]

async def simulate_boundary_relay(sender_node, receiver_node, data):
    if sender_node.can_mutate():
        sender_node.spend()
        print(f"[{sender_node.name}] Relay successful to {receiver_node.name}")
        await asyncio.sleep(0.1)
        receiver_node.harvest(0.4)
        return True
    else:
        print(f"[{sender_node.name}] Low energy - relay failed")
        return False

async def main():
    global current_state
    print_mem("Start of simulation")

    for step in range(8):
        local_entropy = torch.rand(1).item() * 0.9 + 0.2
        
        for node in nodes:
            harvested = node.harvest(local_entropy)
            print(f"[{node.name}] Harvested {harvested:.3f} mJ | Total: {node.energy:.3f} mJ")

        relay_success = await simulate_boundary_relay(nodes[0], nodes[1], "boundary_tensor")

        current_state = current_state * 0.98

        print_mem(f"Step {step}")

    print("\n=== 5x5 Energy-Chained DTO Demo Complete ===")
    print("Final node energies:", [f"{node.energy:.3f}" for node in nodes])

asyncio.run(main())
print_mem("Final")
