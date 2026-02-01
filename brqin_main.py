import torch
import numpy as np
import networkx as nx
import os
import matplotlib.pyplot as plt
from brqin_peps import BrQinPEPS, EntanglementEnergyNode
from credo_db_facade import CredoDBFacade

print("=== BrQin v5.2 - Single Process Mode with Ising Hamiltonian ===")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

Lx, Ly = 12, 12
peps = BrQinPEPS(Lx, Ly, init_bond=12, device=device)
nodes = [EntanglementEnergyNode(f"Node{i}") for i in range(4)]
db = CredoDBFacade()

# Simple Ising Hamiltonian
J = 1.0
h = 0.5
H_terms = []
for r in range(Lx):
    row = []
    for c in range(Ly):
        local_h = torch.tensor([h, 0.0], device=device)
        row.append(local_h)
    H_terms.append(row)

energies = []
mags = []

for step in range(12):
    entropy_delta = np.random.uniform(0.2, 0.9) - 0.3
    energy, mode = peps.hybrid_evolution_step(H_terms, entropy_delta, nodes[step % 4])
    boundary_energy = peps.boundary_mps_contraction(H_terms)
    observables = peps.compute_observables()
    mag = observables['magnetization']

    energies.append(energy)
    mags.append(mag)

    print(f"Step {step:2d} | Energy: {energy:.6f} | Mag: {mag:.6f} | Mode: {mode}")

    if step % 3 == 0:
        syndromes = {"p_phys": 0.005, "weight": np.random.randint(0, 25)}
        db.save_simulation_step(peps, nodes, observables, energy, mode, entropy_delta, syndromes)
        print(f"   [DB] Step {step} persisted | Logical err est: {syndromes['p_phys']:.3f}")

# Plot saved automatically
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(energies, marker='o', linewidth=2, label='Energy')
plt.title('Energy over Steps (Ising)')
plt.xlabel('Step')
plt.ylabel('Energy')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(mags, marker='o', color='orange', linewidth=2, label='Magnetization')
plt.title('Magnetization over Steps')
plt.xlabel('Step')
plt.ylabel('Magnetization')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('brqin_energy_mag_plot.png', dpi=300)
plt.close()

print("\n=== Final Summary ===")
print(f"Final magnetization: {mags[-1]:.6f}")
print(f"Average energy:      {np.mean(energies):.6f}")
print(f"Energy trend:        {energies[-1] - energies[0]:.6f}")
print(f"DB entries:          {db.get_checkpoint_count()}")
ok, msg = db.verify_integrity()
print(f"Vault integrity:     {ok} - {msg}")

torch.save(peps.tensors, 'final_state.pt')
nx.write_graphml(nx.grid_2d_graph(Lx, Ly), 'topology.graphml')
print("✅ Run complete")
print("📊 Plot saved as 'brqin_energy_mag_plot.png'")
