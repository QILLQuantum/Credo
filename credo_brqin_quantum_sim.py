import numpy as np
from qutip import *

# Manual gates
def rx(theta):
    return Qobj([[np.cos(theta/2), -1j*np.sin(theta/2)],
                 [-1j*np.sin(theta/2), np.cos(theta/2)]])

def ry(theta):
    return Qobj([[np.cos(theta/2), -np.sin(theta/2)],
                 [np.sin(theta/2), np.cos(theta/2)]])

def cz():
    return Qobj([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]], dims=[[2,2],[2,2]])

def cnot():
    return Qobj([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dims=[[2,2],[2,2]])

def hadamard():
    return (1/np.sqrt(2)) * Qobj([[1,1],[1,-1]])

# Entropy func
def von_neumann_entropy(rho):
    evals = rho.eigenenergies()
    evals = evals[evals > 1e-10]
    return -np.sum(evals * np.log2(evals))

print("BrQin: Brain-Quantum Imprint Simulator v1.0")

# Section 1-7: Core quantum basics (abbreviated for brevity; expand as needed)
print("\nBasic Qubit & Density (Section 1-2)")
alpha = 1/np.sqrt(2)
beta = 1j/np.sqrt(2)
psi = basis(2, 0) * alpha + basis(2, 1) * beta
rho_pure = ket2dm(psi)
print("Pure state norm:", psi.norm())
print("Pure entropy:", von_neumann_entropy(rho_pure))

# Section 8: Brain Microtubule Chain (Orch-OR)
print("\nSection 8: Microtubule Chain (N=4)")
N_brain = 4
J_brain = 1.0
H_brain = 0
for i in range(N_brain-1):
    op_xx = tensor([sigmax() if j in [i, i+1] else qeye(2) for j in range(N_brain)])
    op_yy = tensor([sigmay() if j in [i, i+1] else qeye(2) for j in range(N_brain)])
    H_brain += J_brain * (op_xx + op_yy)

psi0_brain_list = [basis(2, 0) for _ in range(N_brain)]
psi0_brain_list[1] = (basis(2, 0) + basis(2, 1)).unit()
psi0_brain = tensor(psi0_brain_list)
times_brain = np.linspace(0, 20 / J_brain, 200)
result_clean_brain = mesolve(H_brain, psi0_brain, times_brain)
print("Brain chain purity (clean):", (ket2dm(result_clean_brain.states[-1])**2).tr().real)

# Section 9-10: Water Sims Evolution (Dimer to Hexamer EZ)
print("\nSection 9-10: Water Quantum Sims (Up to Hexamer 2D EZ)")

# Parameters
delta = 0.05
epsilon_base = 0.02
J = 0.1
drive_amp = 0.1
omega = 0.5
pulse_center = 50.0
pulse_width = 10.0
gamma_dephase = 0.01
gamma_relax_down = 0.005
gamma_relax_up = 0.001

N = 6  # Hexamer
# Initial
psi0_list = [(basis(2, 0) + 0.1 * basis(2, 1)).unit() for _ in range(N)]
psi0_list[0] = (basis(2, 0) + 0.3 * basis(2, 1)).unit()
psi0 = tensor(psi0_list)

times = np.linspace(0, 200, 400)

# H_base with ring coupling + gradient
H_base = 0
for i in range(N):
    epsilon_i = epsilon_base * (1 + 0.15 * (i % 2))
    op_x = tensor([sigmax() if j == i else qeye(2) for j in range(N)])
    op_z = tensor([sigmaz() if j == i else qeye(2) for j in range(N)])
    H_base += delta * op_x + epsilon_i * op_z

for i in range(N):
    j = (i + 1) % N
    op_xx = tensor([sigmax() if k in [i, j] else qeye(2) for k in range(N)])
    op_yy = tensor([sigmay() if k in [i, j] else qeye(2) for k in range(N)])
    H_base += J * (op_xx + op_yy)

# Drive
op_drive = sum(tensor([sigmaz() if j == i else qeye(2) for j in range(N)]) for i in range(N))
def pulse_coeff(t, args):
    return drive_amp * np.exp(-((t - pulse_center)/pulse_width)**2) * np.cos(omega * t)

H = [H_base, [op_drive, pulse_coeff]]

# c_ops thermal + dephase
c_ops = []
for i in range(N):
    sm = tensor([sigmam() if j == i else qeye(2) for j in range(N)])
    sp = tensor([sigmap() if j == i else qeye(2) for j in range(N)])
    c_ops.append(np.sqrt(gamma_relax_down) * sm)
    c_ops.append(np.sqrt(gamma_relax_up) * sp)
c_ops += [np.sqrt(gamma_dephase) * tensor([sigmaz() if j == i else qeye(2) for j in range(N)]) for i in range(N)]

# e_ops
e_ops = [tensor([sigmaz() if j == i else qeye(2) for j in range(N)]) for i in range(N)]

# Run
result = mesolve(H, psi0, times, c_ops=c_ops, e_ops=e_ops)
rho_final = ket2dm(result.states[-1])
print("Hexamer final purity:", (rho_final * rho_final).tr().real)
print("Final <σ_z> per site:", [e[-1] for e in result.expect])
post_idx = 200
print("Post-pulse <σ_z> (t≈100):", [e[post_idx] for e in result.expect])

print("\nBrQin complete. Run tweaks on N, gamma, J for experiments!")