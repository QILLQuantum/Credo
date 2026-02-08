# persistence/quantum_circuit_sim.py
# Pure-Python NISQ quantum circuit simulator for BrQin v5.3
# In-house – MIT License – QILLQuantum/Credo

import numpy as np
from typing import Dict, List, Tuple

class QuantumCircuitSimulator:
    def __init__(self, num_qubits: int = 8, noise_level: float = 0.01):
        self.num_qubits = num_qubits
        self.noise_level = noise_level
        self.state = np.zeros(2**num_qubits, dtype=complex)
        self.state[0] = 1.0  # |00...0>

    def hadamard(self, qubit: int):
        """Apply H gate to qubit"""
        idx = 2**qubit
        mask = np.arange(2**self.num_qubits) ^ idx
        self.state = (self.state + self.state[mask]) / np.sqrt(2)

    def cnot(self, control: int, target: int):
        """Apply CNOT gate"""
        mask = 1 << control
        for i in range(2**self.num_qubits):
            if i & mask:
                self.state[i ^ (1 << target)] = self.state[i]

    def depolarizing_noise(self):
        """Apply simple depolarizing noise to state"""
        p = self.noise_level
        for i in range(2**self.num_qubits):
            self.state[i] *= (1 - p)
            self.state += p / 2**self.num_qubits * np.random.normal(0, 0.01, 2**self.num_qubits)

    def measure_all(self) -> Dict[int, float]:
        """Measure all qubits, return probabilities"""
        probs = np.abs(self.state)**2
        return {i: probs[i] for i in range(2**self.num_qubits) if probs[i] > 1e-8}

    def run_vqe_like(self, depth: int = 3) -> Dict:
        """Simple VQE-style circuit: variational ansatz + measurement"""
        for d in range(depth):
            # Layer of Hadamards + CNOTs (hardware-efficient ansatz)
            for q in range(self.num_qubits):
                self.hadamard(q)
            for q in range(0, self.num_qubits - 1, 2):
                self.cnot(q, q + 1)
            self.depolarizing_noise()

        probs = self.measure_all()
        # Simulated energy (placeholder – real VQE would minimize <H>)
        energy = -np.sum([p * (bin(i).count('1') % 2) for i, p in probs.items()])
        fidelity = np.max(probs)  # max probability as proxy

        return {
            "num_qubits": self.num_qubits,
            "depth": depth,
            "energy": energy,
            "fidelity": fidelity,
            "noise_level": self.noise_level,
            "measurement_probs": probs
        }