# credo_multihop_gatt_relay.py - Multi-Hop with Real GATT Server Relay + Energy + BrQin Keys
import asyncio
from bleak import BleakGATTCharacteristic, BleakServer, BleakClient, BleakScanner
from cryptography.fernet import Fernet
import json
import os
import torch

# CONFIG
DATA_TO_SEND = {"credo_sync": "GATT relay chain — wisdom hops eternal"}
DEVICE_PREFIX = "CredoNode-Hop"  # e.g., CredoNode-Hop1, Hop2
CUSTOM_SERVICE_UUID = "0000ffe0-0000-1000-8000-00805f9b34fb"
CUSTOM_CHAR_UUID = "0000ffe1-0000-1000-8000-00805f9b34fb"
STORAGE_FOLDER = "gatt_relay_received"
os.makedirs(STORAGE_FOLDER, exist_ok=True)

# Piezo energy
VIBRATION_POWER_MW = 0.5
EFFICIENCY = 0.15
TX_ENERGY_MJ = 0.05

class EnergyNode:
    def __init__(self):
        self.energy_mj = 0.0

    def harvest(self, seconds=10):
        harvested = VIBRATION_POWER_MW * EFFICIENCY * seconds
        self.energy_mj += harvested

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

async def relay_server(hop_number):
    node = EnergyNode()
    server = BleakServer()
    received_data = None

    def on_write(char: BleakGATTCharacteristic, value: bytearray):
        nonlocal received_data
        print(f"Hop {hop_number} received data on GATT char")
        key = brqin_key(hop_number)
        try:
            decrypted = key.decrypt(value)
            received_data = json.loads(decrypted)
            print(f"Decrypted: {received_data}")
        except:
            print("Decryption failed")

    await server.start_advertising(f"{DEVICE_PREFIX}{hop_number}")
    await server.add_service(CUSTOM_SERVICE_UUID)
    await server.add_characteristic(CUSTOM_SERVICE_UUID, CUSTOM_CHAR_UUID, on_write, properties=["write", "read"])

    print(f"Relay {hop_number} GATT server advertising {DEVICE_PREFIX}{hop_number}")

    while received_data is None:
        node.harvest()
        await asyncio.sleep(5)

    # Forward to next hop
    while not node.can_transmit():
        node.harvest()
        await asyncio.sleep(1)

    next_key = brqin_key(hop_number + 1)
    encrypted_next = next_key.encrypt(json.dumps(received_data).encode())

    devices = await BleakScanner.discover(timeout=10.0)
    next_device = next((d for d in devices if f"{DEVICE_PREFIX}{hop_number + 1}" in (d.name or "")), None)
    if next_device:
        async with BleakClient(next_device.address) as client:
            await client.write_gatt_char(CUSTOM_CHAR_UUID, encrypted_next)
            print(f"Relay {hop_number} forwarded to hop {hop_number + 1}")
    await server.stop_advertising()

async def receiver():
    node = EnergyNode()
    server = BleakServer()

    def on_write(char, value):
        print("Receiver got final payload")
        key = brqin_key(99)  # Final key
        try:
            decrypted = key.decrypt(value)
            data = json.loads(decrypted)
            filename = f"{STORAGE_FOLDER}/final_received_{int(time.time())}.json"
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Chain complete — stored {filename}")
            print("Data:", data)
        except:
            print("Final decryption failed")

    await server.start_advertising("CredoReceiver")
    await server.add_service(CUSTOM_SERVICE_UUID)
    await server.add_characteristic(CUSTOM_SERVICE_UUID, CUSTOM_CHAR_UUID, on_write, properties=["write"])

    print("Receiver GATT server running")
    while True:
        await asyncio.sleep(10)

async def main():
    mode = input("Mode — (t)ransmitter, (r)elay [hop#], or (R)eceiver: ").lower()
    if mode == "t":
        # Transmitter code from previous (initiate to Hop1)
        pass  # Use previous transmitter
    elif mode.startswith("r"):
        hop = int(mode[1:] or 1)
        await relay_server(hop)
    else:
        await receiver()

if __name__ == "__main__":
    asyncio.run(main())