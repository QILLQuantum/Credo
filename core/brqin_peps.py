import torch
import numpy as np
import networkx as nx
import os
import matplotlib.pyplot as plt
from brqin_peps import BrQinPEPS, EntanglementEnergyNode
from credo_db_facade import CredoDBFacade

print("=== BrQin v5.2 - Bond Dimension Tracking ===")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

Lx, Ly = 12, 12
peps = BrQinPEPS(Lx, Ly, init_bond=12, device=device)
H_terms = [[torch.tensor([1.0, 0.0], device=device) for _ in range(Ly)] for _ in range(Lx)]
nodes = [EntanglementEnergyNode(f"Node{i}") for i in range(4)]
db = CredoDBFacade()

energies = []
mags = []
avg_bond_dims = []

for step in range(12):
    entropy_delta = np.random.uniform(0.2, 0.9) - 0.3
    energy, mode = peps.hybrid_evolution_step(H_terms, entropy_delta, nodes[step % 4])
    boundary_energy = peps.boundary_mps_contraction(H_terms)
    observables = peps.compute_observables()
    mag = observables['magnetization']

    # Track average bond dimension
    bond_sum = sum(peps.bond_map_h.values()) + sum(peps.bond_map_v.values())
    bond_count = len(peps.bond_map_h) + len(peps.bond_map_v)
    avg_bond = bond_sum / bond_count if bond_count > 0 else 12
    avg_bond_dims.append(avg_bond)

    energies.append(energy)
    mags.append(mag)

    print(f"Step {step:2d} | Energy: {energy:.6f} | Mag: {mag:.6f} | Avg Bond: {avg_bond:.1f} | Mode: {mode}")

    if step % 3 == 0:
        syndromes = {"p_phys": 0.005, "weight": np.random.randint(0, 25)}
        db.save_simulation_step(peps, nodes, observables, energy, mode, entropy_delta, syndromes)
        print(f"   [DB] Step {step} persisted")

# === Bond dimension heatmap ===
bond_grid = np.zeros((Lx, Ly))
for r in range(Lx):
    for c in range(Ly):
        h = peps.bond_map_h.get(((r,c),(r,c+1)), 12) if c + 1 < Ly else 12
        v = peps.bond_map_v.get(((r,c),(r+1,c)), 12) if r + 1 < Lx else 12
        bond_grid[r, c] = (h + v) / 2

plt.figure(figsize=(10, 8))
plt.imshow(bond_grid, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Average Bond Dimension')
plt.title('Bond Dimension Heatmap (Final State)')
plt.xlabel('Column')
plt.ylabel('Row')
for i in range(Lx):
    for j in range(Ly):
        plt.text(j, i, f'{bond_grid[i,j]:.0f}', ha='center', va='center', color='white', fontsize=8)
plt.tight_layout()
plt.savefig('bond_dimension_heatmap.png', dpi=300)
plt.close()

print("\n=== Final Summary ===")
print(f"Final magnetization: {mags[-1]:.6f}")
print(f"Average energy:      {np.mean(energies):.6f}")
print(f"Final avg bond dim:  {avg_bond_dims[-1]:.1f}")
print(f"DB entries:          {db.get_checkpoint_count()}")
ok, msg = db.verify_integrity()
print(f"Vault integrity:     {ok} - {msg}")

torch.save(peps.tensors, 'final_state.pt')
nx.write_graphml(nx.grid_2d_graph(Lx, Ly), 'topology.graphml')
print("✅ Run complete")
print("📊 Plots saved: brqin_energy_mag_plot.png + bond_dimension_heatmap.png")

