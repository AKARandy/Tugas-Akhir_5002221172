"""
prism_refraction_search.py
--------------------------
Discrete Prism Refraction Search (PRS) local improvement for SoybeanState.

The original PRS paper defines a single-solution continuous optimizer whose
incident angle is updated through a prism-refraction equation. This module keeps
that single-solution, exploitative role, but maps the refracted angle to discrete
flow-improvement moves that are meaningful for the soybean supply-chain state.

Use inside ALNS as the Step-6 local search/intensification phase, replacing
tabu_local_search(best, rng).
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

import config
from state import SoybeanState


Move = Callable[[SoybeanState, np.random.Generator, float], bool]


def prism_refraction_local_search(
    state: SoybeanState,
    rng_ls: np.random.Generator | None = None,
) -> SoybeanState:
    """
    Run a short PRS-inspired local search around the incumbent solution.

    Each sub-iteration:
      1. Computes a "deviation" angle from current shortage/import/inventory
         pressure.
      2. Updates an incident-angle vector using the PRS refraction equation.
      3. Maps the refracted angle to one discrete local move and an intensity.
      4. Repairs feasibility and keeps the candidate only if it improves.

    This deliberately has no tabu list. Its memory is the current incident-angle
    vector and decaying prism angle, matching the single-solution PRS idea.
    """
    if rng_ls is None:
        rng_ls = np.random.default_rng()

    moves: list[Move] = [
        _move_import_to_local,
        _move_reroute_to_cheaper_port,
        _move_fill_worst_shortage,
        _move_transfer_surplus_to_deficit,
    ]

    best_local = state.copy()
    current = state.copy()

    n_moves = len(moves)
    incident = np.linspace(0.20, 0.85, n_moves) * (math.pi / 2.0)
    prism_angle = float(config.PRS_PRISM_ANGLE)
    max_steps = max(1, int(config.PRS_SUB_IT))

    for step in range(max_steps):
        deviation = _deviation_angle(current)
        incident, prism_angle = _prs_update(
            incident=incident,
            deviation=deviation,
            prism_angle=prism_angle,
            step=step,
            max_steps=max_steps,
            rng=rng_ls,
        )

        scores = np.maximum(incident, 1e-9)
        probs = scores / scores.sum()
        move_idx = int(rng_ls.choice(n_moves, p=probs))
        intensity = _angle_to_intensity(incident[move_idx])

        cand = current.copy()
        changed = moves[move_idx](cand, rng_ls, intensity)
        if not changed:
            continue

        cand.feasibility_repair()

        if cand.objective() < current.objective():
            current = cand.copy()
            if cand.objective() < best_local.objective():
                best_local = cand.copy()

    return best_local


def _prs_update(
    incident: np.ndarray,
    deviation: float,
    prism_angle: float,
    step: int,
    max_steps: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Vectorized adaptation of the PRS incident-angle update."""
    eps = 1e-6
    a = float(np.clip(prism_angle, 0.05, math.pi / 2.0 - eps))
    delta = float(np.clip(deviation, 0.01, math.pi / 2.0 - eps))

    emergent = delta - incident + a
    mu = math.sin((a + delta) / 2.0) / max(math.sin(a / 2.0), eps)

    r = rng.uniform(-1.0, 1.0, size=incident.shape)
    root = np.sqrt(np.maximum(mu * mu - np.sin(emergent) ** 2, eps))
    arg = -np.sin(emergent) * math.cos(a) + r * math.sin(a) * root
    next_incident = np.arcsin(np.clip(arg, -1.0 + eps, 1.0 - eps))
    next_incident = np.abs(next_incident)
    next_incident = np.clip(next_incident, 0.05, math.pi / 2.0 - eps)

    alpha = float(config.PRS_ALPHA)
    next_prism_angle = a * math.exp(-alpha * (step + 1) / max_steps)
    next_prism_angle = max(0.05, next_prism_angle)
    return next_incident, next_prism_angle


def _deviation_angle(s: SoybeanState) -> float:
    """Map solution pressure to a bounded PRS deviation angle."""
    p = s.problem
    demand = max(float(p.DEMAND.sum()), 1.0)
    shortage = float(s.sh.sum()) / demand
    inv_floor = float(np.maximum(0.0, config.INV_MIN_FRAC * p.DEMAND - s.inv).sum())
    inv_pressure = inv_floor / max(config.INV_MIN_FRAC * demand, 1.0)

    total_import = float((s.x_imp + s.x_emg).sum())
    total_local = float(s.x_loc.sum())
    import_dep = total_import / max(total_import + total_local, 1.0)
    import_pressure = max(0.0, import_dep - config.EPS_IMPORT_DEP)

    pressure = shortage + 0.5 * inv_pressure + import_pressure
    return float(np.clip(0.05 + pressure * (math.pi / 3.0), 0.05, math.pi / 3.0))


def _angle_to_intensity(angle: float) -> float:
    low = float(config.PRS_MOVE_MIN_FRAC)
    high = float(config.PRS_MOVE_MAX_FRAC)
    scaled = float(np.clip(angle / (math.pi / 2.0), 0.0, 1.0))
    return low + (high - low) * scaled


def _move_import_to_local(
    s: SoybeanState,
    rng: np.random.Generator,
    intensity: float,
) -> bool:
    """Substitute a fraction of imported supply with available local supply."""
    p = s.problem
    tc_local = p.C_PROD[:, None] + p.C_SHIP

    candidates = np.argwhere(s.x_dist.sum(axis=0) > 1.0)
    if candidates.size == 0:
        return False

    # Prefer province-months with high import volume and high shortage pressure.
    scores = []
    for i, t in candidates:
        imp_to_i = float(s.x_dist[:, i, t].sum())
        scores.append(imp_to_i + 2.0 * float(s.sh[i, t]))
    i_try, t_try = candidates[int(np.argmax(scores))]

    local_left = np.maximum(0.0, p.PROD_CAP[:, t_try].astype(float) - s.x_loc[:, :, t_try].sum(axis=1))
    if local_left.sum() <= 1.0:
        return False

    max_shift = float(s.x_dist[:, i_try, t_try].sum()) * intensity
    remaining = max_shift
    local_plan = []

    for k in np.argsort(tc_local[:, i_try]):
        alloc = min(remaining, local_left[k])
        if alloc <= 1.0:
            continue
        local_plan.append((int(k), float(alloc)))
        remaining -= alloc
        if remaining <= 1.0:
            break

    planned = sum(alloc for _, alloc in local_plan)
    if planned <= 1.0:
        return False

    reduced = _reduce_import_to_province(s, int(i_try), int(t_try), planned)
    if reduced <= 1.0:
        return False

    remaining = reduced
    for k, alloc in local_plan:
        add = min(remaining, alloc)
        if add <= 0.0:
            continue
        s.x_loc[k, i_try, t_try] += add
        remaining -= add
        if remaining <= 0.0:
            break
    return reduced - remaining > 1.0


def _move_reroute_to_cheaper_port(
    s: SoybeanState,
    rng: np.random.Generator,
    intensity: float,
) -> bool:
    """Move part of a province import flow from an expensive port to a cheaper one."""
    p = s.problem
    positive = np.argwhere(s.x_dist > 1.0)
    if positive.size == 0:
        return False

    best_gain = 0.0
    best_tuple = None
    for h_old, i, t in positive:
        h_new = int(np.argmin(p.C_DIST[:, i]))
        gain = float(p.C_DIST[h_old, i] - p.C_DIST[h_new, i])
        if h_new != h_old and gain > best_gain:
            best_gain = gain
            best_tuple = (int(h_old), int(h_new), int(i), int(t))

    if best_tuple is None:
        return False

    h_old, h_new, i, t = best_tuple
    amount = float(s.x_dist[h_old, i, t]) * intensity
    port_left = max(0.0, float(p.PORT_THRU_CAP[h_new, t] - (s.x_imp[:, h_new, t] + s.x_emg[:, h_new, t]).sum()))
    amount = min(amount, port_left)
    if amount <= 1.0:
        return False

    source_choice = _cheapest_source_with_capacity(s, t, amount)
    if source_choice is None:
        return False
    s_idx, source_left = source_choice
    amount = min(amount, source_left)
    if amount <= 1.0:
        return False

    reduced = _reduce_port_import(s, h_old, t, amount)
    if reduced <= 1.0:
        return False

    s.x_dist[h_old, i, t] = max(0.0, s.x_dist[h_old, i, t] - reduced)
    s.x_imp[s_idx, h_new, t] += reduced
    s.x_dist[h_new, i, t] += reduced
    return True


def _move_fill_worst_shortage(
    s: SoybeanState,
    rng: np.random.Generator,
    intensity: float,
) -> bool:
    """Add supply to the worst shortage province-month if capacity exists."""
    if float(s.sh.sum()) <= 1.0:
        return False

    p = s.problem
    i, t = np.unravel_index(np.argmax(s.sh), s.sh.shape)
    need = float(s.sh[i, t]) * max(0.25, intensity)
    if need <= 1.0:
        return False

    changed = _add_local_supply(s, int(i), int(t), need)
    if not changed:
        changed = _add_import_supply(s, int(i), int(t), need)
    return changed


def _move_transfer_surplus_to_deficit(
    s: SoybeanState,
    rng: np.random.Generator,
    intensity: float,
) -> bool:
    """Transfer surplus from an eligible province to a deficit province."""
    p = s.problem
    deficits = np.argwhere(s.sh > 1.0)
    if deficits.size == 0:
        return False

    best = None
    for j, t in deficits:
        eligible = [i for i in range(p.N_PROV) if i != j and s.w[i, t]]
        for i in eligible:
            supply_i = (
                s.x_loc[:, i, t].sum()
                + s.x_dist[:, i, t].sum()
                + s.x_trns[:, i, t].sum()
                - s.x_trns[i, :, t].sum()
            )
            surplus = max(0.0, float(supply_i - p.DEMAND[i, t] - 0.5 * p.SAFETY_STOCK[i]))
            if surplus <= 1.0:
                continue
            score = float(s.sh[j, t]) / max(p.C_TRANS[i, j], 1.0)
            if best is None or score > best[0]:
                best = (score, int(i), int(j), int(t), surplus)

    if best is None:
        return False

    _, i, j, t, surplus = best
    amount = min(float(s.sh[j, t]), surplus) * intensity
    if amount <= 1.0:
        return False
    s.x_trns[i, j, t] += amount
    return True


def _add_local_supply(s: SoybeanState, i: int, t: int, amount: float) -> bool:
    p = s.problem
    tc_local = p.C_PROD[:, None] + p.C_SHIP
    local_left = np.maximum(0.0, p.PROD_CAP[:, t].astype(float) - s.x_loc[:, :, t].sum(axis=1))
    remaining = amount
    changed = False
    for k in np.argsort(tc_local[:, i]):
        alloc = min(remaining, local_left[k])
        if alloc <= 1.0:
            continue
        s.x_loc[k, i, t] += alloc
        remaining -= alloc
        changed = True
        if remaining <= 1.0:
            break
    return changed


def _add_import_supply(s: SoybeanState, i: int, t: int, amount: float) -> bool:
    p = s.problem
    routes = []
    imp_left = np.maximum(0.0, p.IMP_CAP_NORMAL[:, t].astype(float) - s.x_imp[:, :, t].sum(axis=1))
    port_left = np.maximum(0.0, p.PORT_THRU_CAP[:, t].astype(float) - (s.x_imp[:, :, t] + s.x_emg[:, :, t]).sum(axis=0))
    for src in range(p.N_IMP):
        for h in range(p.N_PORT):
            if imp_left[src] > 1.0 and port_left[h] > 1.0:
                routes.append((p.C_PURCH[src] + p.C_DIST[h, i], src, h))
    if not routes:
        return False

    changed = False
    remaining = amount
    for _, src, h in sorted(routes):
        alloc = min(remaining, imp_left[src], port_left[h])
        if alloc <= 1.0:
            continue
        s.x_imp[src, h, t] += alloc
        s.x_dist[h, i, t] += alloc
        remaining -= alloc
        imp_left[src] -= alloc
        port_left[h] -= alloc
        changed = True
        if remaining <= 1.0:
            break
    return changed


def _reduce_import_to_province(s: SoybeanState, i: int, t: int, amount: float) -> float:
    p = s.problem
    remaining = amount
    reduced_total = 0.0
    # Reduce most expensive distribution arcs first.
    for h in np.argsort(-p.C_DIST[:, i]):
        if remaining <= 1.0:
            break
        red = min(remaining, float(s.x_dist[h, i, t]))
        if red <= 1.0:
            continue
        actual = _reduce_port_import(s, int(h), t, red)
        if actual <= 1.0:
            continue
        s.x_dist[h, i, t] = max(0.0, s.x_dist[h, i, t] - actual)
        remaining -= actual
        reduced_total += actual
    return reduced_total


def _reduce_port_import(s: SoybeanState, h: int, t: int, amount: float) -> float:
    remaining = amount
    reduced = 0.0
    for src in range(s.problem.N_IMP):
        red = min(remaining, float(s.x_imp[src, h, t]))
        if red <= 0.0:
            continue
        s.x_imp[src, h, t] -= red
        remaining -= red
        reduced += red
        if remaining <= 1.0:
            break
    return reduced


def _cheapest_source_with_capacity(s: SoybeanState, t: int, amount: float) -> tuple[int, float] | None:
    p = s.problem
    imp_left = np.maximum(0.0, p.IMP_CAP_NORMAL[:, t].astype(float) - s.x_imp[:, :, t].sum(axis=1))
    for src in np.argsort(p.C_PURCH):
        if imp_left[src] >= min(amount, 1.0):
            return int(src), float(imp_left[src])
    return None
