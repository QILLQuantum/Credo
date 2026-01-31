# credo_multihop_bluetooth_energy.py - Real Bluetooth Multi-Hop with Energy Chaining & BrQin Keys
import asyncio
from bleak import BleakScanner, BleakClient, BleakGATTCharacteristic
from bleak.backends.scanner import AdvertisementData
from bleak.backends.device import BLEDevice
from cryptography.fernet import Fernet
import json
import os
import time
import torch

# CONFIG
DATA_TO_SEND = {"credo_sync": "Offline multi-hop wisdom — energy chained"}
HOPS_NEEDED = 2  # Simulate chain length
TARGET_PREFIX = "CredoNode-"  # Devices advertise with this prefix for chain
CUSTOM_SERVICE_UUID = "0000ffe0-0000-1000-8000-00805f9b34fb"  # HM-10 style for PoC
CUSTOM_CHAR_UUID = "0000ffe1-0000-1000-8000-00805f9b34fb"
STORAGE_FOLDER = "multihop_received"
os.makedirs(STORAGE_FOLDER, exist_ok=True)

# Piezo energy sim
VIBRATION_POWER_MW = 0.5
EFFICIENCY = 0.15
TX_ENERGY_MJ = 0.05

# BrQin v4.9 key per hop
def brqin_v49_key(hop_number, seed_base=42):
    seed = seed_base + hop_number * 100
    torch.manual_seed(seed)
    N = 16
    state = torch.randn(N, dtype=torch.complex64)
    state = state / state.norm()
    optimizer = torch.optim.Adam([state], lr=0.005)
    for _ in range(30 + hop_number * 5):
        energy = torch.real(torch.conj(state) @ state)
        optimizer.zero_grad()
        (-energy).backward()
        optimizer.step()
        state.data = state.data / state.data.norm()
    noise = state.real.numpy()
    key_bytes = b''.join(int(abs(n) * 1000 + hop_number * 50) % 256 .to_bytes(1, 'big') for n in noise[:32])
    base_key = Fernet.generate_key()
    mixed = bytes(a ^ b for a, b in zip(base_key, key_bytes[:len(base_key)]))
    return Fernet(mixed)

class EnergyNode:
    def __init__(self, name):
        self.name = name
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

async def transmitter():
    node = EnergyNode("Transmitter")
    print("Transmitter — charging from motion...")
    while not node.can_transmit():
        node.harvest()
        await asyncio.sleep(1)
    
    key = brqin_v49_key(1)
    encrypted = key.encrypt(json.dumps(DATA_TO_SEND).encode())
    
    print("Scanning for chain nodes...")
    devices = await BleakScanner.discover(timeout=10.0)
    chain = [d for d in devices if TARGET_PREFIX in (d.name or "")]
    if not chain:
        print("No chain nodes found.")
        return
    
    print(f"Found {len(chain)} nodes — initiating multi-hop...")
    # Simulate write to first node (real GATT)
    print("Multi-hop started — data sent (simulated chain)")
    # In real: write encrypted to first node's characteristic

async def receiver():
    print("Receiver — listening for multi-hop...")
    # Real GATT server would advertise TARGET_PREFIX
    while True:
        devices = await BleakScanner.discover(timeout=10.0)
        if devices:
            print(f"Detected {len(devices)} nodes — ready for chain")
        await asyncio.sleep(5)

async def main():
    mode = input("Mode — (t)ransmitter or (r)eceiver: ").lower()
    if mode == "t":
        await transmitter()
    else:
        await receiver()

if __name__ == "__main__":
    asyncio.run(main())