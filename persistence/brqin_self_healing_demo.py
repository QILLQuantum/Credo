import numpy as np
import matplotlib.pyplot as plt

print("=== BrQin Self-Healing Demo: Surface Code + VQE Simulation ===")

class EntanglementEnergyNode:
    def __init__(self, name):
        self.name = name
        self.energy = 0.0

    def harvest(self, local_entropy):
        harvested = max(0.0, local_entropy - 0.3)
        self.energy += harvested
        return harvested

    def can_mutate(self, cost=0.15):
        return self.energy >= cost

    def spend(self, cost=0.15):
        if self.can_mutate(cost):
            self.energy -= cost
            return True
        return False

class BrQinDemo:
    def __init__(self, Lx=12, Ly=12, init_bond=12):
        self.Lx = Lx
        self.Ly = Ly
        self.init_bond = init_bond
        self.bond_map_h = {}
        self.bond_map_v = {}
        self.growth_history = []
        self.energy_history = []
        self.syndrome_history = []
        self.code_distance_history = []
        self.gate_fidelity_history = []
        self.vqe_energy_history = []
        self.code_distance = 3
        self.error_count = 0

        for r in range(Lx):
            for c in range(Ly):
                if c + 1 < Ly:
                    self.bond_map_h[((r,c),(r,c+1))] = init_bond
                if r + 1 < Lx:
                    self.bond_map_v[((r,c),(r+1,c))] = init_bond

    def adaptive_bond_growth(self, energy_node):
        growth = 0
        for r in range(self.Lx):
            for c in range(self.Ly):
                local_entropy = np.random.rand() * 0.9 + 0.2
                energy_node.harvest(local_entropy)
                if local_entropy > 0.7 and energy_node.can_mutate(cost=0.2):
                    if c + 1 < self.Ly:
                        key = ((r,c),(r,c+1))
                        self.bond_map_h[key] = min(self.bond_map_h.get(key, self.init_bond) + 4, 32)
                        growth += 1
                    if r + 1 < self.Lx:
                        key = ((r,c),(r+1,c))
                        self.bond_map_v[key] = min(self.bond_map_v.get(key, self.init_bond) + 4, 32)
                        growth += 1
                    energy_node.spend(cost=0.2)
        self.growth_history.append(growth)
        return growth

    def simulate_energy(self):
        avg_bond = self.get_avg_bond()
        energy = -0.5 * (avg_bond / self.init_bond) + np.random.normal(0, 0.01)
        self.energy_history.append(energy)
        return energy

    def simulate_syndromes(self):
        num_checks = self.code_distance * self.code_distance
        error_rate = 0.005 + np.random.rand() * 0.01
        syndrome_weight = int(np.random.poisson(error_rate * num_checks))
        self.syndrome_history.append(syndrome_weight)
        return syndrome_weight

    def adapt_code_distance(self, syndrome_weight):
        estimated_p = syndrome_weight / (self.code_distance * self.code_distance + 1e-8)
        old_d = self.code_distance
        if estimated_p > 0.012 and self.code_distance < 9:
            self.code_distance += 2
        elif estimated_p < 0.005 and self.code_distance > 3:
            self.code_distance -= 2
        if self.code_distance != old_d:
            print(f"   [Code Distance] {old_d} → {self.code_distance} (p_phys ≈ {estimated_p:.3f})")
        self.code_distance_history.append(self.code_distance)
        return self.code_distance

    def apply_logical_gate(self, gate_type="H"):
        if gate_type == "H":
            fidelity = 0.995 + np.random.normal(0, 0.001)
        elif gate_type == "CNOT":
            fidelity = 0.99 + np.random.normal(0, 0.002)
        elif gate_type == "T":
            distillation_success = np.random.rand() < 0.85
            fidelity = 0.98 if distillation_success else 0.85
            print(f"   [T Gate] Distillation success: {distillation_success}")
        else:
            fidelity = 0.99 + np.random.normal(0, 0.002)

        self.gate_fidelity_history.append(fidelity)
        print(f"   [Logical Gate] Applied {gate_type} | Fidelity: {fidelity:.4f}")
        return fidelity

    def simple_vqe_step(self):
        avg_bond = self.get_avg_bond()
        vqe_energy = -0.8 * (avg_bond / self.init_bond) + np.random.normal(0, 0.02)
        self.vqe_energy_history.append(vqe_energy)
        return vqe_energy

    def get_avg_bond(self):
        total = sum(self.bond_map_h.values()) + sum(self.bond_map_v.values())
        count = len(self.bond_map_h) + len(self.bond_map_v)
        return total / count if count > 0 else self.init_bond

    def plot_all(self):
        plt.figure(figsize=(14, 10))

        plt.subplot(2, 3, 1)
        plt.plot(self.energy_history, marker='o', color='blue', linewidth=2)
        plt.title('Energy Trend')
        plt.xlabel('Step')
        plt.ylabel('Energy')
        plt.grid(True)

        plt.subplot(2, 3, 2)
        plt.plot(self.growth_history, marker='o', color='green', linewidth=2)
        plt.title('Bonds Grown per Step')
        plt.xlabel('Step')
        plt.ylabel('Growth Count')
        plt.grid(True)

        plt.subplot(2, 3, 3)
        bond_grid = np.zeros((self.Lx, self.Ly))
        for r in range(self.Lx):
            for c in range(self.Ly):
                h = self.bond_map_h.get(((r,c),(r,c+1)), self.init_bond) if c + 1 < self.Ly else self.init_bond
                v = self.bond_map_v.get(((r,c),(r+1,c)), self.init_bond) if r + 1 < self.Lx else self.init_bond
                bond_grid[r, c] = (h + v) / 2
        plt.imshow(bond_grid, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Avg Bond Dim')
        plt.title('Final Bond Dimension Heatmap')
        plt.xlabel('Column')
        plt.ylabel('Row')

        plt.subplot(2, 3, 4)
        plt.plot(self.code_distance_history, marker='s', color='purple', linewidth=2)
        plt.title('Code Distance Over Time')
        plt.xlabel('Step')
        plt.ylabel('Code Distance')
        plt.grid(True)

        plt.subplot(2, 3, 5)
        plt.plot(self.gate_fidelity_history, marker='^', color='orange', linewidth=2)
        plt.title('Logical Gate Fidelity')
        plt.xlabel('Gate Application')
        plt.ylabel('Fidelity')
        plt.grid(True)
        plt.ylim(0.8, 1.0)

        plt.subplot(2, 3, 6)
        plt.plot(self.vqe_energy_history, marker='d', color='red', linewidth=2)
        plt.title('VQE Energy')
        plt.xlabel('Step')
        plt.ylabel('VQE Energy')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('brqin_full_report.png', dpi=300)
        plt.close()
        print("📊 Full report saved as 'brqin_full_report.png'")

    def run(self, steps=12):
        print(f"Initial avg bond: {self.get_avg_bond():.1f} | Code distance: {self.code_distance}")

        for step in range(steps):
            growth = self.adaptive_bond_growth(EntanglementEnergyNode(f"Node{step}"))
            energy = self.simulate_energy()
            avg_bond = self.get_avg_bond()
            syndrome_weight = self.simulate_syndromes()
            new_d = self.adapt_code_distance(syndrome_weight)

            if step % 4 == 0:
                gate = np.random.choice(['H', 'CNOT', 'T'])
                self.apply_logical_gate(gate)

            vqe_energy = self.simple_vqe_step()

            print(f"Step {step:2d} | Energy: {energy:.6f} | Growth: {growth:2d} | Avg Bond: {avg_bond:.1f} | Syndrome: {syndrome_weight} | Code Dist: {new_d} | VQE: {vqe_energy:.6f}")

        self.plot_all()
        print(f"✅ Demo complete (errors: {self.error_count})")

if __name__ == "__main__":
    demo = BrQinDemo(Lx=12, Ly=12, init_bond=12)
    demo.run(steps=12)
