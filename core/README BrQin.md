
# BrQin - Brain Quantum / Breakthrough Quantum

A classical tensor network simulator (PEPS) with **living** adaptive features, surface code logical qubits, adaptive code distance, logical gates, VQE simulation, and immutable persistence (Mímisbrunn Merkle Vault + SQLite).

## Features
- Living PEPS with dynamic bond dimensions per bond
- Adaptive bond growth triggered by local entropy/energy harvesting
- Surface code patches with syndrome extraction
- Adaptive code distance based on syndrome error rates
- Logical gates (H, CNOT, T with distillation)
- Simple VQE simulation
- Immutable Merkle Vault (Mímisbrunn) for audit trail
- SQLite fast query store
- Energy trend, bond growth, code distance, and fidelity tracking
- Automatic plots and summary tables

## Installation
```bash
pip install torch numpy matplotlib networkx cryptography