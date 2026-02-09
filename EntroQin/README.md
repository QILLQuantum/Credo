# EntroQin: Entropy-Guided Adaptive Tensor Networks for Quantum Simulations

EntroQin is the science-focused branch of BrQin, built around the **BEDP (BrQin Entropy-Direction Paradigm)** — a paradigm shift from uniform bond dimension allocation to dynamic, entropy-directed growth in tensor networks.

BEDP uses von Neumann entropy from SVD decompositions to guide bond growth, with **Chao Exploration (CE)** for high-variance wild discovery and **Lattice Protection (LP)** for low-variance precise locking. It converges faster, reaches lower energy, and produces richer structure, **Especially at scale**

## Key Features
- Entropy-guided adaptive bond growth (BEDP)  
- CE/LP mode switching for exploration – exploitation balance  
- 3D nested CTMRG for volumetric simulations  
- Fracton support (X-cube & Haah)  
- GPU-ready (CuPy)  
- Live entropy stats + convergence rate tracking  

## Installation
```bash
git clone https://github.com/QILLQuantum/Credo.git
cd Credo/EntroQin
pip install -r requirements.txt
