# oracle_recursive_cotengra.py - Recursive Higher-D PEPS Oracle with Full Cotengra Contraction
import time
import multiprocessing as mp
import numpy as np
import cotengra as ctg
import torch

class RecursiveCotengraOracle:
    def __init__(
        self,
        levels=4,
        base_dims=(8, 8, 8, 8),
        max_bond=8,
        device='cpu',
        num_workers=None
    ):
        self.levels = levels
        self.base_dims = base_dims
        self.max_bond = max_bond
        self.device = device
        self.num_workers = num_workers or mp.cpu_count()

    def init_base_block(self):
        core = np.array([d//2 for d in self.base_dims])
        max_dist = np.linalg.norm(np.array(self.base_dims)/2)
        tensors = {}
        
        for idx in np.ndindex(self.base_dims):
            pos = np.array(idx)
            dist = np.linalg.norm(pos - core) / max_dist if max_dist > 0 else 0
            scale = np.sin(np.pi / 2 * (1 - dist))
            D = max(2, min(self.max_bond, int(self.max_bond * scale + 0.5)))
            
            shape = (D,) * (2 * len(self.base_dims)) + (2,)
            tensor = torch.randn(*shape, dtype=torch.complex64, device=self.device)
            tensor = tensor / tensor.norm() if tensor.norm() > 0 else tensor
            tensors[idx] = tensor
        
        return {"tensors": tensors, "dims": self.base_dims}

    def coarse_grain_block(self, lower_state, reduction_factor=2):
        lower_tensors = lower_state["tensors"]
        lower_dims = lower_state["dims"]
        
        new_dims = tuple(d // reduction_factor for d in lower_dims)
        if any(d < 2 for d in new_dims):
            return None
        
        flat_list = list(lower_tensors.values())
        flat = torch.stack(flat_list)
        flat = flat.view(-1, flat.shape[-1])
        U, S, Vh = torch.svd_lowrank(flat, q=self.max_bond // reduction_factor)
        
        new_D = min(self.max_bond // reduction_factor, S.shape[0])
        coarse = U[:, :new_D] @ torch.diag(S[:new_D]) @ Vh[:new_D, :]
        coarse = coarse.view(*new_dims, new_D, new_D, 2)
        
        parent_tensors = {(0,)*len(new_dims): coarse}
        
        return {"tensors": parent_tensors, "dims": new_dims}

    def build_recursive_hierarchy(self):
        current = self.init_base_block()
        hierarchy = [current]
        
        for level in range(1, self.levels + 1):
            print(f"Coarse-graining level {level-1} → level {level}")
            parent = self.coarse_grain_block(current)
            if parent is None:
                print("Minimum size reached — stopping")
                break
            current = parent
            hierarchy.append(current)
        
        print(f"Recursive hierarchy complete — {len(hierarchy)} levels")
        return hierarchy[-1]

    def contract_with_cotengra(self, state):
        tensors_list = list(state["tensors"].values())
        num_tensors = len(tensors_list)
        
        input_sets = []
        for i in range(num_tensors):
            start = i * (2 * len(self.base_dims))
            input_sets.append(set(range(start, start + 2 * len(self.base_dims))))
        
        output_set = set()
        eq = ctg.get_equation(input_sets, output_set)
        
        opt = ctg.HyperOptimizer(
            max_repeats=64,
            max_time=30,
            parallel=self.num_workers
        )
        path, info = opt.search(tensors_list, eq)
        
        result = ctg.contract(path, *tensors_list)
        norm = torch.abs(result).item()
        energy = -np.log(norm + 1e-12) if norm > 0 else 0.0
        
        return energy, norm, info

    def run_with_state(self, mode="light", previous_state=None):
        start_time = time.time()
        if previous_state is None:
            state = self.build_recursive_hierarchy()
        else:
            state = previous_state
        
        energy, norm, info = self.contract_with_cotengra(state)
        runtime = time.time() - start_time
        
        metrics = {
            "updated_state": state,
            "certified_energy": energy,
            "norm": norm,
            "contraction_info": {
                "opt_cost": info.opt_cost,
                "speedup": info.speedup,
                "path_length": len(info.path),
                "runtime": runtime
            },
            "levels": self.levels
        }
        return metrics

if __name__ == "__main__":
    oracle = RecursiveCotengraOracle(levels=4, base_dims=(8,8,8,8), max_bond=8)
    metrics = oracle.run_with_state()
    print("Recursive Cotengra metrics:", metrics)