# oracle_peps.py - PEPS Oracle for BrQin v5.3 with Recursive Coarse-Graining (Higher-D Scaling)
import time
import numpy as np
import torch

class PepsOracle:
    def __init__(
        self,
        steps=12,
        base_dims=(8, 8, 8, 8),  # Base 3+1D block
        levels=3,  # Recursive depth
        max_bond=8,
        device='cpu'
    ):
        self.steps = steps
        self.base_dims = base_dims
        self.levels = levels
        self.max_bond = max_bond
        self.device = device

    def init_base_block(self):
        """Initialize base 3+1D block with adaptive bonds"""
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
        """Coarse-grain lower block to parent via SVD truncation"""
        lower_tensors = lower_state["tensors"]
        lower_dims = lower_state["dims"]
        
        new_dims = tuple(d // reduction_factor for d in lower_dims)
        if any(d < 2 for d in new_dims):
            return None
        
        # Flatten and SVD
        flat_list = list(lower_tensors.values())
        flat = torch.stack(flat_list)
        flat = flat.view(-1, flat.shape[-1])
        U, S, Vh = torch.svd_lowrank(flat, q=self.max_bond // reduction_factor)
        
        new_D = min(self.max_bond // reduction_factor, S.shape[0])
        coarse = U[:, :new_D] @ torch.diag(S[:new_D]) @ Vh[:new_D, :]
        
        # Reshape to parent (single site for simplicity)
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
        return hierarchy[-1]  # Top-level state

    def approximate_contraction(self, state):
        tensors = state["tensors"]
        current = None
        dims = state["dims"]
        for slice_idx in range(dims[-1]):
            layer = []
            for coord in np.ndindex(dims[:-1]):
                full_coord = coord + (slice_idx,)
                layer.append(tensors[full_coord])
            layer_tensor = torch.stack(layer)
            layer_tensor = layer_tensor.view(-1, layer_tensor.shape[-1])
            if current is None:
                current = layer_tensor
            else:
                current = torch.tensordot(current, layer_tensor, dims=([0], [0]))
        
        norm = torch.abs(current).max().item()
        energy = -np.log(norm + 1e-12) if norm > 0 else 0.0
        return energy, norm

    def run_with_state(self, mode="light", previous_state=None):
        start_time = time.time()
        if previous_state is None:
            state = self.build_recursive_hierarchy()
        else:
            state = previous_state
        
        energy, norm = self.approximate_contraction(state)
        runtime = time.time() - start_time
        
        metrics = {
            "updated_state": state,
            "certified_energy": energy,
            "norm": norm,
            "runtime_seconds": runtime,
            "levels": self.levels
        }
        return metrics

if __name__ == "__main__":
    oracle = PepsOracle(levels=4, base_dims=(8,8,8,8), max_bond=8)
    metrics = oracle.run_with_state()
    print("Recursive higher-D metrics:", metrics)
