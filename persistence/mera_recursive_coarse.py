# mera_recursive_coarse.py - Recursive Coarse-Graining for Higher-D PEPS (3+1D Blocks as Units)
import torch
import numpy as np

class RecursiveCoarsePEPS:
    def __init__(self, levels=4, base_L=8, max_D=8, device='cpu'):
        self.levels = levels
        self.base_L = base_L
        self.max_D = max_D
        self.device = device

    def init_base_block(self, dims=(self.base_L, self.base_L, self.base_L, self.base_L)):
        """Initialize base 3+1D block with adaptive bonds"""
        core = np.array([d//2 for d in dims])
        max_dist = np.linalg.norm(np.array(dims)/2)
        tensors = {}
        
        for idx in np.ndindex(dims):
            pos = np.array(idx)
            dist = np.linalg.norm(pos - core) / max_dist if max_dist > 0 else 0
            scale = np.sin(np.pi / 2 * (1 - dist))
            D = max(2, min(self.max_D, int(self.max_D * scale + 0.5)))
            
            shape = (D,) * (2 * len(dims)) + (2,)
            tensor = torch.randn(*shape, dtype=torch.complex64, device=self.device)
            tensor = tensor / tensor.norm() if tensor.norm() > 0 else tensor
            tensors[idx] = tensor
        
        return {"tensors": tensors, "dims": dims}

    def coarse_grain_block(self, lower_state, reduction_factor=2):
        """Coarse-grain lower block to parent site via SVD truncation"""
        lower_tensors = lower_state["tensors"]
        lower_dims = lower_state["dims"]
        
        # Reduce dimensions
        new_dims = tuple(d // reduction_factor for d in lower_dims)
        if any(d < 2 for d in new_dims):
            return None  # Too small
        
        # Simplified coarse-grain: average + SVD on flattened
        flat = torch.stack(list(lower_tensors.values()))
        flat = flat.view(-1, flat.shape[-1])
        U, S, Vh = torch.svd_lowrank(flat, q=self.max_D // 2)
        
        # Truncate to new bond
        new_D = min(self.max_D // reduction_factor, S.shape[0])
        coarse_tensor = U[:, :new_D] @ torch.diag(S[:new_D]) @ Vh[:new_D, :]
        coarse_tensor = coarse_tensor.view(*new_dims, new_D, new_D, 2)  # Reshape approximate
        
        # Parent as single "site" tensor
        parent_tensors = { (0,)*len(new_dims): coarse_tensor }
        
        return {"tensors": parent_tensors, "dims": new_dims}

    def build_recursive_hierarchy(self):
        """Build full recursive hierarchy — lower blocks coarse-grained into parent"""
        current = self.init_base_block()
        hierarchy = [current]  # Level 0
        
        for level in range(1, self.levels + 1):
            print(f"Coarse-graining level {level-1} to level {level}")
            parent = self.coarse_grain_block(current)
            if parent is None:
                print("Reached minimum size — stopping")
                break
            current = parent
            hierarchy.append(current)
        
        print(f"Recursive hierarchy complete — {len(hierarchy)} levels")
        return hierarchy[-1]  # Top-level state

if __name__ == "__main__":
    rc = RecursiveCoarsePEPS(levels=5, base_L=8, max_D=8)
    top_state = rc.build_recursive_hierarchy()
    print("Top-level dims:", top_state["dims"])
    print("Indefinite scaling hierarchy ready")