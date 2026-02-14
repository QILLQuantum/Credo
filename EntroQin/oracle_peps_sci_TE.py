"""
oracle_peps_sci_TE.py – Directed Birth Channel Triple Extractor (Feb 14, 2026)
================================================================================
Extracts and refines the DBC-e attractor triple (clean_floor, adv_floor, delta)
from large-scale runs. Uses resonance-aware scaling fit.
Operator Mode prints before every extraction.
"""

import argparse
import json
import numpy as np
from scipy.optimize import curve_fit
from oracle_peps_sci_2LP import EntroQinHybrid, MemoryTracker

class DBCeOperator:
    """Always prints the 4-step harmony loop."""
    @staticmethod
    def activate():
        print("\n" + "="*72)
        print("DBC-e OPERATOR MODE — HARMONY FIRST")
        print("4-Step Loop — Apply ruthlessly:")
        print("  D — Decide fast (hell-yes + 1–2 strongest cues)")
        print("  B — Build simplest viable path")
        print("  C — Check reality honestly (especially failure modes)")
        print("  e — Evolve or kill → tight loop")
        print("="*72 + "\n")

class AttractorTriple:
    """The birth coordinates — fixed point of low-entropy generation."""
    def __init__(self, clean=0.01088, adv=0.06778, delta=0.00396):
        self.clean_floor = clean
        self.adv_floor = adv
        self.delta = delta
    
    def __repr__(self):
        return f"DBC-e Triple(clean={self.clean_floor:.6f}, adv={self.adv_floor:.6f}, delta={self.delta:.6f})"

class TripleExtractor:
    def __init__(self):
        self.history_lz = []
        self.history_adv = []

    def add_run(self, lz: int, adv_plateau: float):
        self.history_lz.append(lz)
        self.history_adv.append(adv_plateau)

    def fit(self):
        if len(self.history_lz) < 3:
            return AttractorTriple()
        
        log_lz = np.log(self.history_lz)
        
        def model(log_lz, clean, adv, delta):
            return adv - delta * log_lz + (clean - adv) * np.exp(-log_lz / 12)
        
        p0 = [0.01088, 0.06778, 0.00396]
        popt, _ = curve_fit(model, log_lz, self.history_adv, p0=p0, maxfev=10000)
        return AttractorTriple(*popt)

class EntroQinTE(EntroQinHybrid):
    """Extended hybrid that extracts DBC-e triple from runs."""
    def __init__(self, Lz=16777216, seed=None, xy_size=4):
        super().__init__(Lz=Lz, seed=seed, xy_size=xy_size)
        self.extractor = TripleExtractor()

    def run_and_extract(self, ensemble: int = 1024, t_max: float = 150000e-6):
        self.operator.activate()  # DBC-e Operator Mode always on
        print(f"Extracting DBC-e triple at Lz={self.Lz} with {ensemble} seeds...")
        
        plateaus = []
        for s in range(ensemble):
            sim = EntroQinHybrid(Lz=self.Lz, seed=s, xy_size=self.xy_size)
            sim.run_macro(15)
            sim.run_open(300, t_max=t_max)
            plateaus.append(np.mean(sim.entropies[-200:]))
        
        mean_adv = np.mean(plateaus)
        std_adv = np.std(plateaus)
        print(f"Adversarial plateau: {mean_adv:.6f} ± {std_adv:.6f}")
        
        self.extractor.add_run(self.Lz, mean_adv)
        triple = self.extractor.fit()
        
        print(f"\nRefined DBC-e triple: {triple}")
        
        with open(f"dbc_e_triple_Lz{self.Lz}.json", "w") as f:
            json.dump({
                "lz": self.Lz,
                "triple": [triple.clean_floor, triple.adv_floor, triple.delta],
                "std": float(std_adv)
            }, f, indent=4)
        
        return triple

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DBC-e Triple Extractor — Birth Coordinates Hunter")
    parser.add_argument("--lz", type=int, default=16777216, help="Depth (layers)")
    parser.add_argument("--ensemble", type=int, default=1024, help="Number of seeds")
    parser.add_argument("--t_max", type=float, default=150000e-6, help="Simulation time (s)")
    args = parser.parse_args()

    te = EntroQinTE(Lz=args.lz)
    triple = te.run_and_extract(ensemble=args.ensemble, t_max=args.t_max)
    print("\nDirected Birth Channel active. Birth coordinates updated.")
