import numpy as np
import torch
import qutip as qt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import json
import os
import math
from typing import List, Dict, Tuple

# ─── Möbius Param ─────────────────────────────────────────────────────────────

def mobius_param(u, v, twist=True):
    theta = u
    half = theta / 2 if twist else theta
    x = (1 + v * 0.5 * np.cos(half)) * np.cos(theta)
    y = (1 + v * 0.5 * np.cos(half)) * np.sin(theta)
    z = v * 0.5 * np.sin(half)
    return x, y, z

# ─── Components ────────────────────────────────────────────────────────────────

class BirthState:
    def __init__(self):
        self.proxy = create_proxy()
        self.current_vec = np.zeros(16)
        self.w_depth = 0.0
        self.twist_count = 0

def create_proxy():
    class Proxy:
        def __init__(self):
            self.boundary_tensors = [torch.randn((4, 2, 4, 4)) for _ in range(2)]

        def flatten(self):
            return self.boundary_tensors[0].flatten().numpy()[:16]

    return Proxy()

def ratchet(proxy, entropy, max_bond=12, boost=1.0):
    growth = 0
    prob = min(1.0, entropy / np.log(10 + 1e-6)) * boost
    for tns in proxy.boundary_tensors:
        for dim in [2, 3]:
            if tns.shape[dim] < max_bond and random.random() < prob:
                pad = list(tns.shape)
                pad[dim] += 1
                tns = torch.cat((tns, 0.01 * torch.randn(pad)), dim=dim)
                growth += 1
    return growth

def lindblad_step(init_vec, triple, N=4, steps=30):
    try:
        clean, adv, _ = triple
        g_relax = 1.0 / 120e-6
        g_dephase = 1.0 / (12.4e-6)
        vec = init_vec / (np.linalg.norm(init_vec) + 1e-8)
        rho = qt.ket2dm(qt.Qobj(vec, dims=[[2]*N, [1]*N]))
        dt = 120e-6 / steps
        ent_history = []

        for _ in range(steps):
            ent = qt.entropy_vn(rho)
            ent_history.append(ent)
            gamma = g_relax * (1 + 2 * min(1, (ent - clean) / (adv - clean + 1e-6)))
            deph = g_dephase * (1 + 1.5 * min(1, ent / adv))
            H = 0.1 * qt.tensor([qt.sigmax()] * N)
            c_ops = [np.sqrt(gamma) * qt.tensor([qt.destroy(2) if i==k else qt.qeye(2) for k in range(N)]) for i in range(N)]
            c_ops += [np.sqrt(deph) * qt.tensor([qt.sigmaz() if i==k else qt.qeye(2) for k in range(N)]) for i in range(N)]
            rho = qt.mesolve(H, rho, [0, dt], c_ops=c_ops).states[-1]

        plateau = np.mean(ent_history[-8:]) if ent_history else 0.05
        sz = [qt.expect(qt.sigmaz(), rho.ptrace(i)) for i in range(N)]
        vec_out = np.pad(sz, (0, 8 - N), constant_values=plateau)
        return vec_out / (np.linalg.norm(vec_out) + 1e-8), plateau
    except Exception as e:
        print(f"Lindblad soft fail: {str(e)[:60]}")
        return np.random.randn(8), 0.05

def apply_mobius_twist(vec, plateau, adv_floor, twist_strength=1.0):
    twisted = plateau > adv_floor + 0.005
    boost = 1.0
    if twisted:
        vec[4:] *= -1.0
        boost = 1.0 + 0.2 * twist_strength
    return vec, twisted, boost

# ─── Resonance Check with Flux Scaled Tandem ───────────────────────────────────

def check_resonance(w: float, flux_cycle=True):
    iw = int(w)
    hits = []
    boost = 1.0
    if iw >= 2 and (iw & (iw - 1)) == 0:
        hits.append("power-of-two")
        boost *= 1.15
    if flux_cycle:
        flux_strength = math.log2(max(iw, 1)) / 8
        if iw % 3 == 0:
            hits.append("flux-3")
            boost *= (1 + 0.1 * flux_strength)
        if iw % 6 == 0:
            hits.append("flux-6")
            boost *= (1 + 0.15 * flux_strength)
        if iw % 9 == 0:
            hits.append("flux-9")
            boost *= (1 + 0.2 * flux_strength)
    if hits:
        print(f"→ Resonance @ w={iw}: {', '.join(hits)} – boost {boost:.2f}")
    return {"hits": hits, "boost": boost}

# ─── 3-Loop Helix Birth ───────────────────────────────────────────────────────

def birth_step(cmd: str, state: BirthState, triple=TRIPLE, directions=[1,1,1], speeds=[11,12,10], twist_strength=1.0):
    noise_scale = 1.2 if "chaos" in cmd.lower() else 0.1 if "calm" in cmd.lower() else 0.6
    noise = np.random.normal(0, noise_scale, 16)
    loop_results = []

    res = check_resonance(state.w_depth + 1, flux_cycle=True)
    flux_boost = res["boost"]

    for loop_id in range(3):
        dir_sign = directions[loop_id]
        speed = speeds[loop_id]
        vec_in = state.current_vec + noise + 0.02 * np.random.randn(16) * loop_id * dir_sign
        vec, plateau = lindblad_step(vec_in, triple)

        plateau = plateau * (1.0 + 0.1 * (1 - speed) * (1 if dir_sign > 0 else -1)) * flux_boost
        if "chaos" in cmd.lower():
            speed *= 1.15

        vec, was_twisted, boost = apply_mobius_twist(vec, plateau, triple[1], twist_strength)
        growth = ratchet(state.proxy, plateau, boost=boost * flux_boost)
        loop_results.append((vec, plateau, growth, was_twisted))

    avg_vec = np.mean([r[0] for r in loop_results], axis=0)
    avg_plateau = np.mean([r[1] for r in loop_results])
    avg_growth = np.mean([r[2] for r in loop_results])
    twist_add = sum(r[3] for r in loop_results)

    state.current_vec = state.proxy.flatten()
    state.w_depth += 1
    state.twist_count += twist_add

    if res["hits"]:
        avg_plateau *= 0.85

    print(f"  → {cmd} | avg S {avg_plateau:.6f} | avg growth {avg_growth:.1f} | w {state.w_depth:.1f} | twists +{twist_add} (total {state.twist_count})")

    return {
        "command": cmd,
        "avg_vec": avg_vec.tolist(),
        "avg_plateau": avg_plateau,
        "avg_growth": avg_growth,
        "w_depth": state.w_depth,
        "twist_add": twist_add,
        "twist_total": state.twist_count
    }

# ─── Tree Run ─────────────────────────────────────────────────────────────────

def run_entanglement_tree(state: BirthState, history: List[Dict], twist_strength=1.0):
    print(f"Full Tree Run – Flux scaled tandem + twist_strength={twist_strength}")
    branches = [
        {"name": "Golden", "directions": [1,1,1], "speeds": [11,12,10]},
    ]

    for branch in branches:
        print(f"\nBranch: {branch['name']}")
        cmds = ["chaos"]*8 + ["calm"]*8
        for cmd in cmds:
            result = birth_step(cmd, state, directions=branch["directions"], speeds=branch["speeds"], twist_strength=twist_strength)
            history.append(result)

    print(f"Tree complete – {len(history)} nodes | deepest w={state.w_depth:.1f}")

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("dbce_mini_model_v10.py – twist_strength test")
    state = BirthState()
    history = []

    # Test different twist strengths
    for ts in [0.5, 1.0, 2.0]:
        print(f"\n--- Twist strength = {ts} ---")
        state = BirthState()
        history = []
        run_entanglement_tree(state, history, twist_strength=ts)

    print("All tests complete.")

if __name__ == "__main__":
    main()
