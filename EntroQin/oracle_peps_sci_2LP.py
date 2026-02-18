"""
oracle_peps_sci_2LP.py – Final Polished EntroQin Deep Bulk Hybrid (Feb 14, 2026)
================================================================================
Directed Birth Channel (DBC-e) with Flux Cycle tandem toggle.
Power-of-two + optional 3-6-9 parallel resonance.
"""

import argparse
import torch
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import psutil
import json
import random
from typing import Dict, Optional

# ... [MemoryTracker, SimpleTensorProxy, ParityTracker, ratchet, lindblad_step classes unchanged from previous] ...

class EntroQinHybrid:
    def __init__(self, Lz: int = 512, seed: Optional[int] = None, flux_cycle: bool = False):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
        self.Lz = Lz
        self.flux_cycle = flux_cycle
        self.proxy = SimpleTensorProxy(Lz)
        self.parity = ParityTracker()
        self.entropies = []
        self.peak_mem = MemoryTracker.report_peak()

    def check_resonance(self, w: float) -> bool:
        iw = int(w)
        hits = []
        if iw >= 2 and (iw & (iw - 1)) == 0:
            hits.append("power-of-two")
        if self.flux_cycle:
            if iw % 3 == 0 or iw % 6 == 0 or iw % 9 == 0:
                hits.append("flux 3-6-9")
        if hits:
            print(f"→ Resonance hit @ w={iw}: {', '.join(hits)} – extra clean-up")
            return True
        return False

    def run_macro(self, steps: int):
        for step in range(steps):
            # ... [existing macro logic] ...
            if self.check_resonance(step + 1):
                ent *= 0.85 if "power-of-two" in hits else 0.90
                if self.flux_cycle and "flux" in hits:
                    ent *= 0.88  # extra flux boost
            # ... [rest unchanged] ...

    # ... [run_open, plot_entropy, summary unchanged] ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EntroQin Hybrid – Flux Cycle Tandem")
    parser.add_argument("--lz", type=int, default=512)
    parser.add_argument("--macro-steps", type=int, default=15)
    parser.add_argument("--open-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flux-cycle", action="store_true", help="Enable 3-6-9 flux tandem")
    args = parser.parse_args()

    sim = EntroQinHybrid(Lz=args.lz, seed=args.seed, flux_cycle=args.flux_cycle)
    sim.run_macro(args.macro_steps)
    sim.run_open(args.open_steps)
    sim.plot_entropy(f"entropy_Lz{args.lz}_flux{args.flux_cycle}.png")
    sim.summary()
