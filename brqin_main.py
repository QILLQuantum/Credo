import torch
import numpy as np
import networkx as nx
import os
from brqin_peps import BrQinPEPS, EntanglementEnergyNode
from credo_db_facade import CredoDBFacade

print("=== BrQin v5.2 - Single Process Mode ===")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

Lx, Ly = 12, 12
peps = BrQinPEPS(Lx, Ly, init_bond=12, device=device)
H_terms = [[torch.tensor([0.0, -1.0], device=device) for _ in range(Ly)] for _ in range(Lx)]
nodes = [EntanglementEnergyNode(f"Node{i}") for i in range(4)]
db = CredoDBFacade()

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

    if rank := 0:  # single process
        print(f"Step {step:2d} | Energy: {energy:.6f} | Mag: {mag:.6f} | Mode: {mode}")

    if step % 3 == 0:
        syndromes = {"p_phys": 0.005, "weight": np.random.randint(0, 25)}
        db.save_simulation_step(peps, nodes, observables, energy, mode, entropy_delta, syndromes)
        print(f"   [DB] Step {step} persisted | Logical err est: {syndromes['p_phys']:.3f}")

# Final report
print("\n=== Final Summary ===")
print(f"Final magnetization: {mags[-1]:.6f}")
print(f"Average energy:      {np.mean(energies):.6f}")
print(f"Energy trend:        {energies[-1] - energies[0]:.6f}")
print(f"DB entries:          {db.get_checkpoint_count()}")
ok, msg = db.verify_integrity()
print(f"Vault integrity:     {ok} - {msg}")

torch.save(peps.tensors, 'final_state.pt')
nx.write_graphml(nx.grid_2d_graph(Lx, Ly), 'topology.graphml')
print("Files saved: final_state.pt + topology.graphml")
print("✅ Run complete")
