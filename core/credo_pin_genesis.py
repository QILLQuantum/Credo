# credo_pin_genesis.py
import ipfshttpclient
client = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001/http')
res = client.add('credo_core_norse_frozen.json')
cid = res['Hash']
print(f"Pinned! CID: {cid}")
print(f"Gateway: https://ipfs.io/ipfs/{cid}")