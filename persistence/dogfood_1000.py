# dogfood_1000.py
# High-volume dogfood: 1000 reflections to stress-test Credo DB layer
# Run: py -3 dogfood_1000.py

import time
from BrQin v5.3 import BrQin

NUM_ORDEALS = 1000

print(f"Starting 1000-reflection dogfood run...")
print("This will take ~10–30 minutes depending on oracle speed.")
print("Press Ctrl+C to stop early if needed.")

brqin = BrQin()

start_total = time.time()

try:
    brqin.run_reflection_loop(
        ordeals=[f"Ordeal {i+1}: Stress test persistence endurance" for i in range(NUM_ORDEALS)]
    )
except KeyboardInterrupt:
    print("\nInterrupted by user.")

end_total = time.time()

print(f"\nDogfood run complete.")
print(f"Total reflections attempted: {brqin.reflection_count}")
print(f"Successful: {brqin.reflection_count - brqin.error_count}")
print(f"Errors: {brqin.error_count}")
print(f"Total time: {end_total - start_total:.1f} seconds ({(end_total - start_total)/brqin.reflection_count:.3f} s/reflection)")

# Final DB verification
try:
    facade = brqin.persistence
    print(f"Current Merkle root: {facade.get_current_root()[:12] if facade.get_current_root() else 'None'}…")
    print(f"Total persisted entries: {facade.count()}")

    # Quick recent check
    recent = facade.recent(limit=5)
    print("\nMost recent 5 entries (SQLite):")
    for r in recent:
        print(f"  {r['ts']} | {r['type']} | {r['payload'].get('enriched_belief', 'N/A')[:100]}…")

    # Vault integrity check on last entry
    last_idx = facade.vault.count() - 1
    valid = facade.vault.verify_prefix(last_idx)
    print(f"\nVault prefix valid for last entry ({last_idx}): {valid}")

except Exception as e:
    print(f"Final DB verification failed: {str(e)}")