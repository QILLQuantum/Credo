def boundary_mps_contraction(self, H_terms=None):
    """Boundary MPS + CTMRG corners with RSVD + adaptive k everywhere"""
    energy = 0.0
    env = None

    for r in range(self.Lx - 1, -1, -1):  # bottom to top
        row_energy = 0.0
        row_env = torch.eye(1, dtype=torch.complex128)

        for c in range(self.Ly):
            tensor = self.tensors[(r, c)]

            if H_terms is not None:
                row_energy += self.local_expectation(tensor, H_terms[r][c])

            # Horizontal contraction example (proxy mat for RSVD)
            mat = tensor.mean(dim=[0,2,3]).reshape(-1, tensor.shape[1])  # horizontal slice

            local_entropy = torch.abs(mat).mean().item()

            # Adaptive k
            k = 16 if local_entropy <= 0.7 else 32
            k = min(k, self.bond_map_h.get(((r,c),(r,c+1)), 8) * 2 if c+1 < self.Ly else 16)

            # Randomized SVD
            try:
                Omega = torch.randn(mat.shape[1], k, dtype=mat.dtype, device=mat.device)
                Y = mat @ Omega
                Q, _ = torch.linalg.qr(Y, mode='reduced')
                B = Q.T @ mat
                U_tilde, S, Vh = torch.linalg.svd(B, full_matrices=False)
                U = Q @ U_tilde
                S = S
                Vh = Vh
            except RuntimeError:
                U, S, Vh = torch.linalg.svd(mat.to(torch.float32), full_matrices=False)

            # Truncation to living bond_dim
            bond_h = self.bond_map_h.get(((r,c),(r,c+1)), 8) if c + 1 < self.Ly else 1
            cum_s2 = torch.cumsum(S**2, dim=0) / torch.sum(S**2 + 1e-12)
            keep = int(torch.sum(cum_s2 < 1 - 1e-6)) + 1
            keep = min(keep, bond_h)

            U = U[:, :keep]
            S = S[:keep]
            Vh = Vh[:keep, :]

            # Update row environment
            row_env = row_env @ (U @ torch.diag(S) @ Vh) if row_env is not None else (U @ torch.diag(S) @ Vh)

        energy += row_energy

        # CTMRG corner update (simplified – real impl would contract corners with row_env)
        self.update_corners_ctmrg(row, row_env)

    return energy