import numpy as np
from qutip import *
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os

# Distributed setup
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

class RLTopologyPolicy(torch.nn.Module):
    def __init__(self, input_dim=50):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 3), torch.nn.Softmax(dim=-1)
        )
    def forward(self, entropy, bond):
        x = torch.cat([entropy, bond], dim=-1)
        return self.net(x)

def main_worker(rank, world_size):
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')

    policy = RLTopologyPolicy().to(device)
    # policy.load_state_dict(torch.load('rl_policy_offline_checkpoint.pt', map_location=device))
    policy = DDP(policy, device_ids=[rank])

    Lx, Ly = 12, 12  # Concept only - reduce to 2x2 for actual run
    N = Lx * Ly
    rows_per_gpu = Lx // world_size
    local_rows = rows_per_gpu + (1 if rank < Lx % world_size else 0)
    local_N = local_rows * Ly

    current_state = torch.randn(local_N * 2, device=device, requires_grad=True)
    H_local = torch.randn(local_N * 2, local_N * 2, device=device)  # placeholder

    optimizer = torch.optim.Adam([current_state], lr=0.01)

    for step in range(15):
        optimizer.zero_grad()
        psi_norm = torch.norm(current_state) + 1e-12
        psi = current_state / psi_norm
        energy = torch.real(psi.conj() @ H_local @ psi)
        energy.backward()
        if current_state.grad is not None:
            dist.all_reduce(current_state.grad, op=dist.ReduceOp.SUM)
            current_state.grad /= world_size
        optimizer.step()
        with torch.no_grad():
            current_state /= torch.norm(current_state) + 1e-12
        if rank == 0 and step % 5 == 0:
            print(f"Step {step} | Energy = {energy.item():.4f}")

    if rank == 0:
        torch.save(current_state, 'brqin_12x12_final.pt')

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count() or 1
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)
