"""
operators_destroy.py
────────────────────
Seven destroy operators for ALNS. Each takes (state, rng) → state.

  D1  destroy_random_temporal    – unfix k random months
  D2  destroy_cost_based          – unfix highest-cost months
  D3  destroy_shortage_based      – unfix province/period with worst shortage
  D4  destroy_geographic          – unfix one regional cluster
  D5  destroy_policy_emergency    – clear all emergency flows
  D6  destroy_bottleneck_port     – unfix the most-utilised port
  D7  destroy_relatedness         – Shaw-like: unfix provinces related to a pivot

All operators read parameters via `state.problem.X`.
"""
from __future__ import annotations

import numpy as np

from state import SoybeanState


def _destruction_size(rng, rho_min: float = 0.15, rho_max: float = 0.35) -> float:
    """Random destruction rate ρ ∈ [rho_min, rho_max]."""
    return rng.uniform(rho_min, rho_max)


# ─── D1 – Random Temporal ───────────────────────────────────────────────────
def destroy_random_temporal(state: SoybeanState, rng) -> SoybeanState:
    """Unfix k random months by zeroing all flows in those months."""
    s = state.copy()
    p = state.problem
    rho = _destruction_size(rng)
    k = max(1, int(rho * p.N_PERIOD))
    months = rng.choice(p.N_PERIOD, size=k, replace=False)
    for t in months:
        s.x_loc[:, :, t]  = 0.0
        s.x_imp[:, :, t]  = 0.0
        s.x_dist[:, :, t] = 0.0
        s.x_trns[:, :, t] = 0.0
    s._last_destruction_scale = float(rho)
    s._invalidate()
    return s


# ─── D2 – Cost-Based ────────────────────────────────────────────────────────
def destroy_cost_based(state: SoybeanState, rng) -> SoybeanState:
    """Unfix the k highest-cost months (where re-optimisation likely helps most)."""
    s = state.copy()
    p = state.problem
    rho = _destruction_size(rng)
    k = max(1, int(rho * p.N_PERIOD))
    cost_per_month = np.array([
        ((p.C_PROD[:, None] + p.C_SHIP) * s.x_loc[:, :, t]).sum()
        + (p.C_PURCH[:, None] * s.x_imp[:, :, t]).sum()
        + (p.C_DIST[:, :] * s.x_dist[:, :, t]).sum()
        for t in range(p.N_PERIOD)
    ])
    months = np.argsort(-cost_per_month)[:k]
    for t in months:
        s.x_loc[:, :, t]  = 0.0
        s.x_imp[:, :, t]  = 0.0
        s.x_dist[:, :, t] = 0.0
        s.x_trns[:, :, t] = 0.0
    s._last_destruction_scale = float(rho)
    s._invalidate()
    return s


# ─── D3 – Shortage-Based ────────────────────────────────────────────────────
def destroy_shortage_based(state: SoybeanState, rng) -> SoybeanState:
    """Unfix the (province, period) with worst shortage and a 5-month neighbourhood."""
    s = state.copy()
    p = state.problem
    sh_flat = s.sh.flatten()
    if sh_flat.sum() < 1e-6:
        return destroy_random_temporal(state, rng)
    worst = np.unravel_index(np.argmax(sh_flat), s.sh.shape)
    i_star, t_star = worst
    t_low  = max(0, t_star - 2)
    t_high = min(p.N_PERIOD, t_star + 3)
    for t in range(t_low, t_high):
        s.x_loc[:, i_star, t]  = 0.0
        s.x_dist[:, i_star, t] = 0.0
        s.x_trns[i_star, :, t] = 0.0
        s.x_trns[:, i_star, t] = 0.0
    s._last_destruction_scale = 5.0 / (p.N_PROV * p.N_PERIOD)
    s._invalidate()
    return s


# ─── D4 – Geographic ────────────────────────────────────────────────────────
def destroy_geographic(state: SoybeanState, rng) -> SoybeanState:
    """Unfix one random regional cluster across k random months."""
    s = state.copy()
    p = state.problem
    chosen_cluster = int(rng.integers(0, 6))
    cluster_provs  = [i for i, c in p.CLUSTER.items() if c == chosen_cluster]
    rho = _destruction_size(rng)
    k = max(1, int(rho * p.N_PERIOD))
    months = rng.choice(p.N_PERIOD, size=k, replace=False)
    for t in months:
        for i in cluster_provs:
            s.x_loc[:, i, t]  = 0.0
            s.x_dist[:, i, t] = 0.0
            s.x_trns[i, :, t] = 0.0
            s.x_trns[:, i, t] = 0.0
    s._last_destruction_scale = float(rho)
    s._invalidate()
    return s


# ─── D5 – Policy / Emergency ────────────────────────────────────────────────
def destroy_policy_emergency(state: SoybeanState, rng) -> SoybeanState:
    """Clear all emergency import flows so they get re-evaluated by feasibility_repair."""
    s = state.copy()
    s.x_emg[:] = 0.0
    s._last_destruction_scale = 1.0
    s._invalidate()
    return s


# ─── D6 – Bottleneck Port ───────────────────────────────────────────────────
def destroy_bottleneck_port(state: SoybeanState, rng) -> SoybeanState:
    """Unfix all flows through the most-utilised port."""
    s = state.copy()
    p = state.problem
    utilisation = s.x_imp.sum(axis=(0, 2)) + s.x_emg.sum(axis=(0, 2))
    h_star = int(np.argmax(utilisation))
    s.x_imp[:, h_star, :]  = 0.0
    s.x_emg[:, h_star, :]  = 0.0
    s.x_dist[h_star, :, :] = 0.0
    s._last_destruction_scale = 1.0 / p.N_PORT
    s._invalidate()
    return s


# ─── D7 – Relatedness (Shaw-like) ───────────────────────────────────────────
def destroy_relatedness(state: SoybeanState, rng) -> SoybeanState:
    """Unfix provinces related to a random pivot (similar demand + same cluster)."""
    s = state.copy()
    p = state.problem
    rho_prov = _destruction_size(rng)
    k_prov = max(2, int(rho_prov * p.N_PROV))
    pivot  = int(rng.integers(0, p.N_PROV))

    demand_norm = p.DEMAND.mean(axis=1)
    demand_norm = demand_norm / (demand_norm.max() + 1e-9)
    pivot_demand = demand_norm[pivot]
    demand_sim   = 1.0 - np.abs(demand_norm - pivot_demand)
    cluster_sim  = np.array([1.0 if p.CLUSTER[i] == p.CLUSTER[pivot] else 0.0
                             for i in range(p.N_PROV)])
    relatedness  = 0.6 * demand_sim + 0.4 * cluster_sim
    related_provs = np.argsort(-relatedness)[:k_prov]

    k = max(1, int(_destruction_size(rng) * p.N_PERIOD))
    months = rng.choice(p.N_PERIOD, size=k, replace=False)
    for t in months:
        for i in related_provs:
            s.x_loc[:, i, t]  = 0.0
            s.x_dist[:, i, t] = 0.0
            s.x_trns[i, :, t] = 0.0
            s.x_trns[:, i, t] = 0.0
    s._last_destruction_scale = float(rho_prov)
    s._invalidate()
    return s
