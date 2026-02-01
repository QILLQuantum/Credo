def living_mps_sweep(psi_vec, bond_map, threshold=1e-8):
    d = 2
    n = psi_vec.shape[0]
    state = psi_vec.clone()
    left = torch.eye(1, dtype=torch.complex128, device=psi_vec.device).unsqueeze(0)

    for i in range(n):
        local_entropy = torch.abs(state[i]).mean().item()
        current_bond = bond_map.get(i, 12)

        # Cone sinus taper + core cap
        if local_entropy > 0.6:
            current_bond = min(20, current_bond)
        elif local_entropy > 0.4:
            current_bond = min(12, current_bond)
        else:
            current_bond = min(6, current_bond)

        # Reshape to matrix
        if i == 0:
            mat = state[i].reshape(d, -1)
        elif i == n-1:
            mat = state[i].reshape(-1, d)
        else:
            mat = state[i].reshape(left.shape[-1], d)

        # Precondition
        mat = mat / (torch.norm(mat) + 1e-12)

        # Adaptive sketch size k
        k = 16 if local_entropy <= 0.7 else 32
        k = min(k, current_bond * 2)

        # Randomized SVD
        try:
            Omega = torch.randn(mat.shape[1], k, dtype=mat.dtype, device=mat.device)
            Y = mat @ Omega
            Q, _ = torch.linalg.qr(Y, mode='reduced')
            B = Q.T @ mat
            U_tilde, S, Vh = torch.linalg.svd(B, full_matrices=False)
            U = Q @ U_tilde[:, :current_bond]
            S = S[:current_bond]
            Vh = Vh[:current_bond, :]
        except:
            # Strong fallback
            U, S, Vh = torch.linalg.svd(mat.to(torch.float32), full_matrices=False)

        # Truncation
        cum_s2 = torch.cumsum(S**2, dim=0) / torch.sum(S**2 + 1e-12)
        keep = int(torch.sum(cum_s2 < 1 - threshold)) + 1
        keep = min(keep, current_bond)

        U = U[:, :keep]
        S = S[:keep]
        Vh = Vh[:keep, :]

        # Update left environment
        if i == 0:
            left = U.unsqueeze(0)
        elif i == n-1:
            left = torch.einsum('...a,a->...', left, S[:,None] * Vh)
        else:
            left = torch.einsum('...a,ab->...b', left, U * S[None,:])

        # Living update with cone cap
        if local_entropy > 0.5:
            bond_map[i] = min(24, bond_map.get(i, 12) + 4)
        else:
            bond_map[i] = max(4, bond_map.get(i, 12) - 1)

    final_vec = left.flatten()[:d]
    return torch.cat([final_vec, torch.zeros(d * (n - 1), device=psi_vec.device, dtype=torch.complex128)]