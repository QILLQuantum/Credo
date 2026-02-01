class BrQinPEPS:
    def __init__(self, Lx, Ly, init_bond=8):
        # ... existing init ...
        self.corner_tl = torch.eye(init_bond, dtype=torch.complex64)
        self.corner_tr = torch.eye(init_bond, dtype=torch.complex64)
        self.corner_bl = torch.eye(init_bond, dtype=torch.complex64)
        self.corner_br = torch.eye(init_bond, dtype=torch.complex64)

    def simple_update(self, H_terms):
        """Fast local gate application"""
        energy = 0.0
        for r in range(self.Lx):
            for c in range(self.Ly):
                tensor = self.tensors[(r, c)]
                energy += self.local_expectation(tensor, H_terms[r][c])
                # Apply simple local gate (placeholder)
                # tensor = apply_gate(tensor, H_terms[r][c])
        return energy

    def full_update(self, H_terms, lr=0.01, steps=5):
        """Variational Full Update — optimize each tensor locally"""
        optimizer = torch.optim.Adam([t for t in self.tensors.values()], lr=lr)
        for _ in range(steps):
            optimizer.zero_grad()
            loss = 0.0
            for r in range(self.Lx):
                for c in range(self.Ly):
                    tensor = self.tensors[(r, c)]
                    loss += self.local_energy(tensor, H_terms[r][c])
            loss.backward()
            optimizer.step()
            # Re-normalize tensors
            for t in self.tensors.values():
                t.data /= torch.norm(t.data) + 1e-12
        return loss.item()

    def local_energy(self, tensor, local_h):
        """Local energy term for variational optimization"""
        phys = tensor.mean(dim=[0,1,2,3])
        return (phys[0] * local_h).real

    def update_corners_ctmrg(self):
        """Simplified CTMRG-style corner update"""
        # In real impl: contract corner with row/column environments
        # Here: placeholder growth with living bond_dim
        bond = max(self.bond_map_h.values(), default=8)
        self.corner_tl = torch.randn(bond, bond, dtype=torch.complex64) / torch.norm(self.corner_tl)
        # ... update other corners similarly ...

    def boundary_mps_contraction(self, H_terms=None):
        """Boundary MPS + CTMRG corner tensors"""
        energy = 0.0
        env = None

        for r in range(self.Lx - 1, -1, -1):
            row_energy = 0.0
            for c in range(self.Ly):
                tensor = self.tensors[(r, c)]

                if H_terms is not None:
                    row_energy += self.local_expectation(tensor, H_terms[r][c])

                bond_h = self.bond_map_h.get(((r,c),(r,c+1)), 8) if c + 1 < self.Ly else 1

                # Horizontal contraction with living bond_dim
                # (RSVD can be wired here as before)

            energy += row_energy
            self.update_corners_ctmrg()  # CTMRG corner update

        return energy

    def hybrid_evolution_step(self, H_terms, entropy_delta, budget_node):
        if entropy_delta > 0.08 and budget_node.can_mutate(cost=0.6):
            budget_node.spend(0.6)
            return self.full_update(H_terms)
        else:
            return self.simple_update(H_terms)