# EntroQin: Entropy-Guided Adaptive Tensor Networks for Quantum Simulations

EntroQin is the science-focused branch of BrQin, emphasizing the BEDP (BrQin Entropy-Direction Paradigm): a paradigm shift from uniform bond dimension allocation to dynamic, entropy-directed growth in tensor networks. It excels in faster convergence, lower energy, and richer structure, especially at scale.

## Features
- Adaptive bond growth guided by von Neumann entropy from SVD.
- Chao Exploration (CE) & Lattice Protection (LP) modes for exploration–exploitation balance.
- 3D nested CTMRG for volumetric simulations.
- Fracton support (X-cube/Haah).
- GPU-ready (CuPy).

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run: `python entroqin.py --model ising --steps 50 --guided`

## Example Output
See benchmarks in docs/ for energy trends and comparisons.

License: MIT