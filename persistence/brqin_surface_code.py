# persistence/brqin_surface_code.py
"""
Surface Code Logical Qubit Simulator with MWPM Decoder
Used in BrQin for fault-tolerant scaling (quality over quantity)
Pure Python, in-house – MIT License – QILLQuantum/Credo
"""

import numpy as np
from typing import Dict, Optional

class SurfaceCodeSimulator:
    def __init__(self, code_distance: int = 5, p_phys: float = 0.01):
        self.d = code_distance
        self.p_phys = p_phys
        self.num_qubits = self.d ** 2
        self.num_checks = self.num_qubits

    def simulate_errors(self) -> np.ndarray:
        """Simulate physical errors (X/Z on data qubits)"""
        errors = np.random.rand(self.num_qubits) < self.p_phys
        return errors.astype(int)

    def compute_syndromes(self, errors: np.ndarray) -> tuple[int, int]:
        """Simplified syndrome extraction (sum mod 2 for demo)"""
        x_synd = np.sum(errors[:self.num_checks]) % 2
        z_synd = np.sum(errors[self.num_checks:]) % 2
        return x_synd, z_synd

    def mwpm_decode(self, x_synd: int, z_synd: int) -> bool:
        """
        Minimum-Weight Perfect Matching decoder (simplified threshold version)
        Returns True if correction succeeds (no logical error)
        """
        total_synd = x_synd + z_synd
        # Threshold decoder: correct if ≤1 syndrome (real MWPM would pair defects)
        return total_synd <= 1

    def run(self, trials: int = 10000) -> Dict:
        """Run Monte Carlo simulation for logical error rate"""
        logical_errors = 0
        for _ in range(trials):
            errors = self.simulate_errors()
            x_s, z_s = self.compute_syndromes(errors)
            corrected = self.mwpm_decode(x_s, z_s)
            if not corrected:
                logical_errors += 1

        logical_rate = logical_errors / trials
        physical_qubits = self.num_qubits

        return {
            "code_distance": self.d,
            "physical_qubits": physical_qubits,
            "physical_error_rate": self.p_phys,
            "logical_error_rate": logical_rate,
            "fault_tolerant_advantage": (1 - logical_rate) / (self.p_phys * physical_qubits) if self.p_phys > 0 else 0.0
        }