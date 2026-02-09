# entroqin.py
# Main driver for EntroQin science simulations
# Latest: February 09, 2026

import argparse
import numpy as np
import matplotlib.pyplot as plt
from oracle_peps_sci import PepsOracleSci

def plot_bond_heatmap(bonds, title="Bond Dimension Heatmap"):
    """Visualize average bond dimensions."""
    plt.figure(figsize=(8, 6))
    plt.imshow(bonds, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Bond Dimension')
    plt.title(title)
    plt.xlabel('y')
    plt.ylabel('x')
    plt.savefig('bond_heatmap.png')
    plt.close()
    print("Bond heatmap saved as bond_heatmap.png")

def main():
    parser = argparse.ArgumentParser(description="EntroQin: Entropy-Guided Adaptive PEPS for Quantum Simulations")
    parser.add_argument("--model", type=str, default="heisenberg", choices=["ising", "heisenberg"], help="Hamiltonian model")
    parser.add_argument("--steps", type=int, default=50, help="Number of simulation steps")
    parser.add_argument("--Lx", type=int, default=8, help="Lattice size x")
    parser.add_argument("--Ly", type=int, default=8, help="Lattice size y")
    parser.add_argument("--Lz", type=int, default=0, help="Lattice size z (0 = 2D, >0 = 3D)")
    parser.add_argument("--guided", action="store_true", help="Use BEDP guided mode (entropy-directed)")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU if available (CuPy)")
    parser.add_argument("--plot", action="store_true", help="Generate bond heatmap at end")
    args = parser.parse_args()

    print(f"Running EntroQin sim: {args.model} model, {args.steps} steps, guided={args.guided}, size={args.Lx}x{args.Ly}x{args.Lz if args.Lz > 0 else '2D'}")

    oracle = PepsOracleSci(Lx=args.Lx, Ly=args.Ly, Lz=args.Lz, steps=args.steps, use_gpu=args.use_gpu)
    metrics = oracle.run(guided=args.guided)

    print("\nFinal Metrics:")
    print(f"Energy: {metrics['energy']:.4f}")
    print(f"Avg Bond: {metrics['avg_bond']:.1f}")
    print(f"Entropy Variance: {metrics['entropy_variance']:.4f}")

    if args.plot:
        plot_bond_heatmap(oracle.get_avg_bond_grid(), title=f"{args.model} Bond Map (Guided={args.guided})")

if __name__ == "__main__":
    main()