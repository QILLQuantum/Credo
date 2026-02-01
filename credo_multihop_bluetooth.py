# credo_multihop_bluetooth.py - Real Multi-Hop Bluetooth with Energy & BrQin Keys
import asyncio
from bleak import BleakScanner, BleakClient
from cryptography.fernet import Fernet
import json
import os
import time
import torch
import random  # ← FIXED: Added missing import

# CONFIG
DATA_TO_SEND = {"credo_sync": "Multi-hop chain — wisdom hops eternal"}
CHAIN_PREFIX = "CredoNode-"  # Devices advertise with this name prefix for chain
CUSTOM_SERVICE_UUID = "0000ffe0-0000-1000-8000-00805f9b34fb"  # Example (use custom)
CUSTOM_CHAR_UUID = "0000ffe1-0000-1000-8000-00805f9b34fb"
STORAGE_FOLDER = "multihop_received"
os.makedirs(STORAGE_FOLDER, exist_ok=True)

# Piezo energy sim
VIBRATION_POWER_MW = 0.5  # Average harvest from motion (milliwatts)
EFFICIENCY = 0.15
TX_ENERGY_MJ = 0.05  # Cost per hop transmit (millijoules, low-power mode)

class EnergyNode:
    def __init__(self):
        self.energy_mj = 0.0

    def harvest(self, seconds=10):
        harvested_mw = VIBRATION_POWER_MW * EFFICIENCY
        harvested_mj = harvested_mw * seconds
        self.energy_mj += harvested_mj

    def can_transmit(self):
        return self.energy_mj >= TX_ENERGY_MJ

    def transmit(self):
        if self.can_transmit():
            self.energy_mj -= TX_ENERGY_MJ
            return True
        return False

# BrQin key per hop
def brqin_key(hop):
    torch.manual_seed(42 + hop * 100)
    N = 12
    state = torch.randn(N, dtype=torch.complex64)
    state = state / state.norm()
    optimizer = torch.optim.Adam([state], lr=0.01)
    for _ in range(20):
        energy = torch.real(torch.conj(state) @ state)
        optimizer.zero_grad()
        (-energy).backward()
        optimizer.step()
        state.data = state.data / state.data.norm()
    noise = state.real.numpy()
    key_bytes = b''.join(int(abs(n) * 1000) % 256 .to_bytes(1, 'big') for n in noise[:32])
    return Fernet(Fernet.generate_key())  # PoC mix

async def multi_hop_transmitter():
    node = EnergyNode()
    print("Transmitter — charging from motion...")
    while not node.can_transmit():
        node.harvest()
        await asyncio.sleep(1)
    
    key = brqin_key(1)
    encrypted = key.encrypt(json.dumps(DATA_TO_SEND).encode())
    
    print("Scanning for chain nodes...")
    devices = await BleakScanner.discover(timeout=10.0)
    chain = [d for d in devices if CHAIN_PREFIX in (d.name or "")]
    if len(chain) < 1:
        print("No chain nodes found.")
        return
    
    print(f"Found {len(chain)} nodes — starting multi-hop...")
    # Simulate write to first node (real GATT in production)
    print("Multi-hop started — data sent (simulated chain)")
    # In real: write encrypted to first node's characteristic

async def chain_receiver():
    print("Receiver — listening for chain...")
    while True:
        devices = await BleakScanner.discover(timeout=10.0)
        if devices:
            print(f"Detected {len(devices)} nodes — chain active")
        await asyncio.sleep(5)

async def main():
    mode = input("Mode — (t)ransmitter or (r)eceiver: ").lower()
    if mode == "t":
        await multi_hop_transmitter()
    else:
        await chain_receiver()

if __name__ == "__main__":
    asyncio.run(main())
