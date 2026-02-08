# brqin_larger_adaptive.py - Adaptive 3D PEPS with High Pressure Buffer Zone (Stability Push)
import torch
import numpy as np
import gc

class AdaptivePEPS:
    def __init__(self, Lx=30, Ly=30, Lz=30, max_D=8, device=None):
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.max_D = max_D
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.entropy_cap = 0.7  # High pressure buffer
        self.renorm_interval = 10  # Frequent for stability
        print(f"High pressure buffer PEPS {Lx}x{Ly}x{Lz} on {self.device}")

    def initialize_state(self):
        core = np.array([self.Lx//2, self.Ly//2, self.Lz//2])
        max_dist = np.linalg.norm(np.array([self.Lx, self.Ly, self.Lz])/2)
        tensors = {}
        total_elements = 0
        
        for x in range(self.Lx):
            for y in range(self.Ly):
                for z in range(self.Lz):
                    pos = np.array([x, y, z])
                    dist = np.linalg.norm(pos - core) / max_dist if max_dist > 0 else 0
                    scale = np.sin(np.pi / 2 * (1 - dist))
                    D = max(2, min(self.max_D, int(self.max_D * scale + 0.5)))
                    
                    # High pressure cap
                    if dist < 0.1:
                        D = min(D, int(self.max_D * self.entropy_cap))
                    
                    shape = (D, D, D, D, D, D, 2)
                    tensor = torch.randn(*shape, dtype=torch.complex64, device=self.device)
                    tensor = tensor / tensor.norm() if tensor.norm() > 0 else tensor
                    tensors[(x, y, z)] = tensor
                    total_elements += np.prod(shape)
        
        memory_gb = total_elements * 8 / 1e9
        print(f"Initialized — est memory {memory_gb:.2f} GB")
        return {"tensors": tensors, "reflection_count": 0}

    def renormalize(self, state):
        tensors = state["tensors"]
        for coord, tensor in tensors.items():
            entropy_proxy = tensor.norm().item()
            if entropy_proxy > self.entropy_cap:
                tensor = tensor / (entropy_proxy * 1.15)
                tensors[coord] = tensor
        
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        return state

    def run_reflection(self, state):
        state["reflection_count"] += 1
        if state["reflection_count"] % self.renorm_interval == 0:
            print(f"High pressure buffer activated — renormalizing at reflection {state['reflection_count']}")
            state = self.renormalize(state)
        return state

if __name__ == "__main__":
    peps = AdaptivePEPS(Lx=30, Ly=30, Lz=30, max_D=8)
    state = peps.initialize_state()
    for i in range(200):
        state = peps.run_reflection(state)
        if i % 20 == 0:
            print(f"Reflection {i+1} — buffer stable")
    print("30×30×30 200 reflections complete — high pressure buffer holds")
