import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import numpy as np
import networkx as nx
from core.brqin_peps import BrQinPEPS, EntanglementEnergyNode
from persistence.credo_db_facade import CredoDBFacade

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main_worker(rank, world_size):
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    Lx, Ly = 12, 12
    peps = BrQinPEPS(Lx, Ly, init_bond=12, device=device)
    H_terms = [[torch.tensor([0.0, -1.0], device=device) for _ in range(Ly)] for _ in range(Lx)]
    nodes = [EntanglementEnergyNode(f"Node{i}") for i in range(4)]
    db = CredoDBFacade()

    for step in range(12):
        entropy_delta = np.random.uniform(0.2, 0.9) - 0.3
        energy, mode = peps.hybrid_evolution_step(H_terms, entropy_delta, nodes[step % 4])
        boundary_energy = peps.boundary_mps_contraction(H_terms)
        observables = peps.compute_observables()

        if rank == 0:
            print(f"Step {step} | Energy: {energy:.4f} | Mode: {mode} | Boundary: {boundary_energy:.4f}")

        if step % 3 == 0 and rank == 0:
            db.save_simulation_step(peps, nodes, observables, energy, mode, entropy_delta, {"p_phys": 0.005, "weight": 12})
            print(f"[DB] Step {step} persisted")

    if rank == 0:
        torch.save(peps.tensors, 'final_state.pt')
        nx.write_graphml(nx.grid_2d_graph(Lx, Ly), 'topology.graphml')
        integrity_ok, msg = db.verify_integrity()
        print(f"DB Integrity: {integrity_ok} - {msg}")
        print(f"Total checkpoints: {db.get_checkpoint_count()}")

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count() or 1
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)
