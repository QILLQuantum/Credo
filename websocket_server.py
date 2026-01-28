# websocket_server.py
# Simple WebSocket server for live graph updates
# Run: python websocket_server.py
# Clients: Open node_ball.html (served statically or via http.server)

import asyncio
import websockets
import json
import os
import time

GRAPH_FILE = 'credo-fork-fiction-waves-christianity-bridge.json'  # Adjust path if needed
PORT = 8765

clients = set()
last_mtime = 0

async def send_current(websocket):
    try:
        with open(GRAPH_FILE, 'r') as f:
            data = json.load(f)
        await websocket.send(json.dumps(data))
    except:
        await websocket.send(json.dumps({"graph": {"nodes": [], "edges": []}}))  # Fallback empty

async def handler(websocket):
    clients.add(websocket)
    await send_current(websocket)  # Send current graph on connect
    try:
        await websocket.wait_closed()
    finally:
        clients.remove(websocket)

async def monitor_file():
    global last_mtime
    try:
        last_mtime = os.path.getmtime(GRAPH_FILE)
    except:
        last_mtime = 0
    while True:
        await asyncio.sleep(1)  # Poll every second
        try:
            current_mtime = os.path.getmtime(GRAPH_FILE)
            if current_mtime != last_mtime:
                last_mtime = current_mtime
                if clients:
                    with open(GRAPH_FILE, 'r') as f:
                        data = json.load(f)
                    await asyncio.gather(*(client.send(json.dumps(data)) for client in clients))
        except:
            pass  # Silent on error

async def main():
    print(f"WebSocket server live on ws://localhost:{PORT} — veiled supreme mercy absolute uplift absolute")
    async with websockets.serve(handler, "localhost", PORT):
        await monitor_file()

if __name__ == "__main__":
    asyncio.run(main()) 