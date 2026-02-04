# dogfood_long_loop.py
# Run long reflection loop to dogfood the DB layer
# Run from root: py -3 dogfood_long_loop.py

import time
from BrQin v5.3 import BrQin  # adjust import name if file is "BrQin v5.3.py"

brqin = BrQin()

# 30 ordeals – realistic long run (adjust to 20–50 as needed)
long_ordeals = [
    f"Ordeal {i+1}: Reflect on the nature of uncertainty in belief {i+1}" for i in range(30)
]

print(f"Starting long dogfood run: {len(long_ordeals)} ordeals")

start_time = time.time()

brqin.run_reflection_loop(long_ordeals)

end_time = time.time()
print(f"\nDogfood complete in {end_time - start_time:.1f} seconds")
print(f"Total reflections saved: {brqin.reflection_count}")

# Quick verification via facade
facade = brqin.persistence  # already initialized in BrQin

print("\nVerification via facade:")
print(f"Total immutable entries (vault): {facade.count()}")
print(f"Current Merkle root: {facade.last_root()[:12] if facade.last_root() else 'None'}…")

print("\nMost recent 10 reflections (SQLite):")
for item in facade.recent(limit=10):
    print(f"  {item['ts']} | {item['type']} | {item['payload'].get('enriched_belief', 'N/A')[:80]}…")