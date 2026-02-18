"""
oracle_peps_sci_TE.py – DBC-e Triple Extractor (Feb 14, 2026)
================================================================
Extracts and refines the birth triple from runs.
"""

import argparse
import json
import numpy as np
from scipy.optimize import curve_fit
from oracle_peps_sci_2LP import EntroQinHybrid

class DBCeOperator:
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
    def __init__(self, Lz=16777216, seed=None):
        super().__init__(Lz=Lz, seed=seed)
        self.extractor = TripleExtractor()

    def run_and_extract(self, ensemble: int = 1024, t_max: float = 150000e-6):
        DBCeOperator.activate()
        print(f"Extracting DBC-e triple at Lz={self.Lz} with {ensemble} seeds...")
        plateaus = []
        for s in range(ensemble):
            sim = EntroQinHybrid(Lz=self.Lz, seed=s)
            sim.run_macro(15)
            sim.run_open(300, t_max=t_max)
            plateaus.append(np.mean(sim.entropies[-200:]))
        mean_adv = np.mean(plateaus)
        self.extractor.add_run(self.Lz, mean_adv)
        triple = self.extractor.fit()
        print(f"Refined DBC-e triple: {triple}")
        with open(f"dbc_e_triple_Lz{self.Lz}.json", "w") as f:
            json.dump({"lz": self.Lz, "triple": [triple.clean_floor, triple.adv_floor, triple.delta]}, f, indent=4)
        return triple

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DBC-e Triple Extractor")
    parser.add_argument("--lz", type=int, default=16777216)
    parser.add_argument("--ensemble", type=int, default=1024)
    args = parser.parse_args()

    te = EntroQinTE(Lz=args.lz)
    te.run_and_extract(ensemble=args.ensemble)
