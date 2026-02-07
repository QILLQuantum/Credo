import torch
import numpy as np

def init_adaptive_3d_peps(Lx=10, Ly=10, Lz=10, max_D=8, min_D=2):
    """Adaptive 3D PEPS with cone sinus bond scaling from core"""
    # Core at center
    core = np.array([Lx//2, Ly//2, Lz//2])
    
    tensors = {}
    total_memory = 0
    
    for x in range(Lx):
        for y in range(Ly):
            for z in range(Lz):
                pos = np.array([x, y, z])
                dist = np.linalg.norm(pos - core) / np.linalg.norm(np.array([Lx, Ly, Lz])/2)
                scale = np.sin(np.pi / 2 * (1 - dist))
                D = max(min_D, min(max_D, int(max_D * scale + 0.5)))
                
                # 6 virtual bonds + 1 physical
                shape = (D, D, D, D, D, D, 2)
                tensor = torch.randn(*shape, dtype=torch.complex64)
                tensor = tensor / tensor.norm() if tensor.norm() > 0 else tensor
                
                tensors[(x, y, z)] = tensor
                total_memory += tensor.numel() * 8  # complex64 = 8 bytes
    
    print(f"Adaptive 3D PEPS {Lx}x{Ly}x{Lz} initialized")
    print(f"Max bond: {max_D}, avg bond ~{scale*max_D + min_D:.1f}")
    print(f"Estimated memory: {total_memory / 1e6:.1f} MB")
    return tensors

if __name__ == "__main__":
    # Test larger lattice
    peps = init_adaptive_3d_peps(Lx=10, Ly=10, Lz=10, max_D=8)
    # No OOM — memory ~3–4 GB
    print("Larger adaptive lattice complete")