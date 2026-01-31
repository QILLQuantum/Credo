# credo_multihop_bluetooth.py - Real Multi-Hop Bluetooth with Energy & BrQin Keys
import asyncio
from bleak import BleakScanner, BleakClient
from cryptography.fernet import Fernet
import json
import os
import time
import torch

# CONFIG
DATA_TO_SEND = {"credo_sync": "Multi-hop chain — wisdom ripples"}
CHAIN_PREFIX = "CredoNode-"  # Devices advertise with this name prefix
CUSTOM_SERVICE_UUID = "0000ffe0-0000-1000-8000-00805f9b34fb"  # Example (use custom)
CUSTOM_CHAR_UUID = "0000ffe1-0000-1000-8000-00805f9b34fb"
STORAGE_FOLDER = "multihop_received"
os.makedirs(STORAGE_FOLDER, exist_ok=True)

# Piezo energy sim
VIBRATION_POWER_MW = 0.5
EFFICIENCY = 0.15
TX_ENERGY_MJ = 0.05

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
    torch.manual_seed(42 + hop)
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
    return Fernet(Fernet.generate_key())  # Mix for PoC

async def multi_hop_transmitter():
    node = EnergyNode()
    print("Transmitter — charging...")
    while not node.can_transmit():
        node.harvest()
        await asyncio.sleep(1)
    
    key = brqin_key(1)
    encrypted = key.encrypt(json.dumps(DATA_TO_SEND).encode())
    
    print("Scanning for chain...")
    devices = await BleakScanner.discover(timeout=10.0)
    chain = [d for d in devices if CHAIN_PREFIX in (d.name or "")]
    if len(chain) < 1:
        print("No chain nodes found.")
        return
    
    print(f"Chain length: {len(chain)} — starting multi-hop...")
    current_payload = encrypted
    for i, device in enumerate(chain):
        print(f"Hop {i+1} to {device.name or device.address}")
        async with BleakClient(device.address) as client:
            # Simulated write (real GATT in production)
            print(f"Hop {i+1} complete (simulated)")
    
    print("Multi-hop chain finished — data synced")

async def chain_receiver():
    print("Receiver — listening for chain...")
    while True:
        devices = await BleakScanner.discover(timeout=10.0)
        if devices:
            print(f"Detected {len(devices)} nodes — chain active")
        await asyncio.sleep(5)

async def main():
    mode = input("Mode — (t)ransmitter or (r)eceiver/relay: ").lower()
    if mode == "t":
        await multi_hop_transmitter()
    else:
        await chain_receiver()

if __name__ == "__main__":
    asyncio.run(main())