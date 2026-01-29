import numpy as np
from qutip import *
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import networkx as nx

print("=== BrQin v4.3 - 144-Qubit Threshold Edition with Distributed Variational Training ===\n")

# ====================== DISTRIBUTED SETUP ======================
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# ====================== RL POLICY ======================
class RLTopologyPolicy(torch.nn.Module):
    def __init__(self, input_dim=50):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 3),
            torch.nn.Softmax(dim=-1)
        )
    def forward(self, entropy, bond):
        x = torch.cat([entropy, bond], dim=-1)
        return self.net(x)

# ====================== HELPER FUNCTIONS ======================
def compute_entropy(state):
    """Dummy entropy computation for RL input."""
    probs = torch.abs(state) ** 2
    probs = probs / (probs.sum() + 1e-12)
    return -torch.sum(probs * torch.log(probs + 1e-12)).unsqueeze(0)

def build_ising_hamiltonian(Lx, Ly, rank, world_size, J=1.0, h=0.5, device='cpu'):
    """Build local part of 2D Ising Hamiltonian using NetworkX grid."""
    G = nx.grid_2d_graph(Lx, Ly)
    N = Lx * Ly
    rows_per_gpu = Lx // world_size
    start_row = rank * rows_per_gpu + min(rank, Lx % world_size)
    local_rows = rows_per_gpu + (1 if rank < Lx % world_size else 0)
    local_qubits = list(range(start_row * Ly, (start_row + local_rows) * Ly))
    
    # Full H would be too big; approximate local H for demo (ignores boundaries)
    H_local = torch.zeros(2**len(local_qubits), 2**len(local_qubits), dtype=torch.complex64, device=device)
    
    # Map global to local indices
    global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(local_qubits)}
    
    # Add transverse field (X terms)
    for i in local_qubits:
        local_i = global_to_local[i]
        X = torch.tensor(sigmax().full(), dtype=torch.complex64, device=device)
        op = torch.eye(2**local_i, device=device) @ X @ torch.eye(2**(len(local_qubits) - local_i - 1), device=device)
        H_local -= h * op
    
    # Add interactions (ZZ terms) - only within local rows for simplicity
    for (i, j) in G.edges():
        if i in global_to_local and j in global_to_local:
            local_i = global_to_local[i]
            local_j = global_to_local[j]
            Z = torch.tensor(sigmaz().full(), dtype=torch.complex64, device=device)
            op_i = torch.eye(2**local_i, device=device) @ Z @ torch.eye(2**(len(local_qubits) - local_i - 1), device=device)
            op_j = torch.eye(2**local_j, device=device) @ Z @ torch.eye(2**(len(local_qubits) - local_j - 1), device=device)
            H_local -= J * (op_i @ op_j)
    
    return H_local

# ====================== MAIN WORKER ======================
def main_worker(rank, world_size):
    setup(rank, world_size)

    # For demo, reduce grid size to 2x2 (4 qubits) to make it runnable; scale up as needed
    # Original 12x12 is impossible without approximations
    Lx, Ly = 2, 2  # Change back to 12,12 for concept, but won't run
    N = Lx * Ly
    rows_per_gpu = Lx // world_size
    local_rows = rows_per_gpu + (1 if rank < Lx % world_size else 0)
    local_N = local_rows * Ly

    device = torch.device(f'cuda:{rank}')
    
    # Build local Hamiltonian using QuTiP and NetworkX
    H_local = build_ising_hamiltonian(Lx, Ly, rank, world_size, device=device)
    
    # State as complex vector (full space for small N)
    current_state = torch.randn(2**local_N, dtype=torch.complex64, device=device) / np.sqrt(2**local_N)
    current_state.requires_grad = True

    # RL Policy - initialize randomly since no checkpoint
    policy = RLTopologyPolicy(input_dim=2).to(device)  # Simplified input_dim for demo
    policy = DDP(policy, device_ids=[rank])
    policy.eval()

    optimizer = torch.optim.Adam([current_state], lr=0.01)

    for step in range(50):  # More steps for better convergence
        optimizer.zero_grad()
        
        # Normalize state
        psi_norm = torch.norm(current_state)
        psi_normalized = current_state / (psi_norm + 1e-12)
        
        # Compute local energy
        energy_local = torch.real(psi_normalized.conj() @ (H_local @ psi_normalized))
        
        # All-reduce energy for global view
        dist.all_reduce(energy_local, op=dist.ReduceOp.SUM)
        energy = energy_local / world_size  # Approximate average
        
        # Backward (minimize energy)
        energy.backward()
        
        # All-reduce gradients
        for param in [current_state]:
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= world_size
        
        optimizer.step()
        
        # Re-normalize
        with torch.no_grad():
            current_state /= torch.norm(current_state) + 1e-12
        
        # Use RL policy (demo: select 'topology' based on entropy/bond)
        if step % 10 == 0:
            entropy = compute_entropy(psi_normalized).to(device)
            bond = torch.tensor([1.0], device=device)  # Dummy bond dim
            action_probs = policy(entropy, bond)
            # In real use: reconfigure H based on argmax(action_probs)
            if rank == 0:
                print(f"Step {step} | Energy ≈ {energy.item():.4f} | Topology Probs: {action_probs.detach().cpu().numpy()}")

    # Gather full state on rank 0 (for small N)
    full_state = None
    if rank == 0:
        full_state = torch.zeros(2**N, dtype=torch.complex64, device=device)
    local_size = 2**local_N
    offsets = [0] * world_size
    for r in range(1, world_size):
        offsets[r] = offsets[r-1] + (rows_per_gpu + (1 if r-1 < Lx % world_size else 0)) * Ly
    offset = offsets[rank] * (2**Ly)  # Assuming row-major sharding
    dist.gather(current_state, gather_list=[full_state[offset:offset + local_size]] if rank == 0 else None, dst=0)

    if rank == 0:
        torch.save(full_state, 'brqin_final_state.pt')
        print("Final distributed variational state saved.")

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size == 0:
        print("No GPUs found; running on CPU with world_size=1")
        main_worker(0, 1)
    else:
        mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)
