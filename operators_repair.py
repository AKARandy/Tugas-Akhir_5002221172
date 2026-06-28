"""
operators_repair.py
───────────────────
Repair operators + the initial-solution builder for ALNS.

  build_initial_solution(problem) → SoybeanState
                                  – two-phase greedy: cost allocation + feasibility repair

  R1  repair_greedy               – greedy cost order
  R2  repair_regret               – prioritise highest-regret provinces
  R3  repair_balanced             – cap each import source ≤ 60 % demand
  R4  repair_emergency_last       – defer emergency activation
  R5  repair_transfer_focused     – two-phase: seed donors + route surplus

All operators read parameters via `state.problem.X` and use `config.LOC_PREF`
as the local-preference bias on import sort keys.
"""
from __future__ import annotations

import numpy as np

import config
from state import SoybeanState
from problem import Problem


# ═══════════════════════════════════════════════════════════════════════════
#  Initial solution: two-phase greedy
# ═══════════════════════════════════════════════════════════════════════════

def build_initial_solution(problem: Problem) -> SoybeanState:
    """
    Phase 1: Forward greedy cost-based allocation
    Phase 2: Feasibility repair
    """
    p = problem
    s = SoybeanState(p)

    # Pre-compute total cost for each source→province pair
    TC_local = p.C_PROD[:, None] + p.C_SHIP   # (N_PROD, N_PROV)

    # Cheapest serving port for each (source, province) pair
    TC_imp = np.full((p.N_IMP, p.N_PROV), np.inf)
    best_port_for: dict = {}     # (s, i) → h
    for s_idx in range(p.N_IMP):
        for i in range(p.N_PROV):
            for h in range(p.N_PORT):
                cost = p.C_PURCH[s_idx] + p.C_DIST[h, i] + config.LOC_PREF
                if cost < TC_imp[s_idx, i]:
                    TC_imp[s_idx, i] = cost
                    best_port_for[(s_idx, i)] = h

    for t in range(p.N_PERIOD):
        prod_cap_left = p.PROD_CAP[:, t].copy().astype(float)
        imp_cap_left  = p.IMP_CAP_NORMAL[:, t].copy().astype(float)
        port_cap_left = p.PORT_THRU_CAP[:, t].copy().astype(float)

        for i in np.argsort(-p.DEMAND[:, t]):     # high-demand first
            deficit = float(p.DEMAND[i, t])

            sources = []
            for p_arr, _p_prov in enumerate(p.PROD_IDX):
                sources.append((TC_local[p_arr, i], "loc", p_arr))
            for s_idx in range(p.N_IMP):
                if TC_imp[s_idx, i] < np.inf:
                    sources.append((TC_imp[s_idx, i], "imp", s_idx))
            sources.sort()

            for cost, kind, idx in sources:
                if deficit <= 0:
                    break
                if kind == "loc":
                    alloc = min(deficit, prod_cap_left[idx])
                    if alloc > 0:
                        s.x_loc[idx, i, t] += alloc
                        prod_cap_left[idx]  -= alloc
                        deficit             -= alloc
                elif kind == "imp":
                    h = best_port_for.get((idx, i))
                    if h is None:
                        continue
                    alloc = min(deficit, imp_cap_left[idx], port_cap_left[h])
                    if alloc > 0:
                        s.x_imp[idx, h, t] += alloc
                        s.x_dist[h, i, t]  += alloc
                        imp_cap_left[idx]   -= alloc
                        port_cap_left[h]    -= alloc
                        deficit             -= alloc

    s.feasibility_repair()
    return s


def build_historical_initial(problem: Problem) -> SoybeanState:
    """
    Seed initial solution from BPS historical import data, then repair.

    Strategy:
      1. x_imp[s,h,t] ← HIST_IMPORT (disaggregated country×port×month data)
      2. x_loc[p,i,t] ← PROD_CAP (same as greedy — production is deterministic)
      3. x_dist[h,i,t] ← allocate port inflow to provinces using PORT_SERV
         shares, calibrated by HIST_PROV_IMPORT to match actual province totals
      4. feasibility_repair() handles all remaining constraints
    """
    p = problem
    s = SoybeanState(p)

    # ── 1. Seed x_imp from historical data ──────────────────────────────
    hist_imp = p.HIST_IMPORT
    if hist_imp is not None:
        for s_idx in range(p.N_IMP):
            for h in range(p.N_PORT):
                for t in range(p.N_PERIOD):
                    port_cap = p.PORT_THRU_CAP[h, t]
                    val = min(hist_imp[s_idx, h, t], port_cap * 0.95)
                    if val > 0:
                        s.x_imp[s_idx, h, t] = val

    # ── 2. Seed x_loc from production capacity ───────────────────────────
    for t in range(p.N_PERIOD):
        for k, i_prov in enumerate(p.PROD_IDX):
            cap = p.PROD_CAP[k, t]
            if cap > 0:
                local_demand_share = p.DEMAND[i_prov, t] * 0.5
                alloc = min(cap, local_demand_share)
                s.x_loc[k, i_prov, t] = alloc

    # ── 3. Allocate x_dist: free port selection, weighted by DEMAND ───────────
    # No PORT_SERV constraint. Each province chooses its cheapest port
    # (by C_DIST = matriks biaya distribusi pelabuhan→provinsi, mencerminkan
    # jarak + infrastruktur). Bobot alokasi = DEMAND[i,t] dari Proyeksi Neraca
    # Komoditas Kedelai (data nyata, tidak ada fill). Jika port termurah habis,
    # coba port termurah berikutnya secara otomatis.
    #
    # Shortage yang tersisa setelah Step 3 dibiarkan — tugas ALNS.

    # Pre-compute cheapest port index for each province (by C_DIST)
    cheapest_port = np.array([
        int(np.argmin(p.C_DIST[:, i]))
        for i in range(p.N_PROV)
    ])

    for t in range(p.N_PERIOD):
        total_inflow = s.x_imp[:, :, t].sum()
        if total_inflow <= 0.5:
            continue

        # Use DEMAND as allocation targets (real data from Neraca Kedelai)
        targets = np.maximum(0.0, p.DEMAND[:, t].copy())

        total_target = targets.sum()
        if total_target > total_inflow:
            # Scale down proportionally so total allocation <= actual port inflow
            targets *= (total_inflow / total_target)

        port_remaining = s.x_imp[:, :, t].sum(axis=0)

        # Allocate to each province: try cheapest port first, fallback to next cheapest
        for i in range(p.N_PROV):
            if targets[i] <= 0.5:
                continue
            for h in [cheapest_port[i]] + [hh for hh in range(p.N_PORT) if hh != cheapest_port[i]]:
                avail = min(targets[i], port_remaining[h])
                if avail > 0.5:
                    s.x_dist[h, i, t] += avail
                    port_remaining[h] -= avail
                    targets[i] -= avail
                if targets[i] <= 0.5:
                    break

    # Step 4 (inject impor tambahan untuk nutup shortage) DIHAPUS.
    # Shortage awal adalah kondisi nyata yang diselesaikan oleh ALNS.

    s.feasibility_repair()
    return s


# ═══════════════════════════════════════════════════════════════════════════
#  Internal helper used by R1 and R4
# ═══════════════════════════════════════════════════════════════════════════

def _greedy_fill(s: SoybeanState, t: int,
                 prod_cap: np.ndarray, imp_cap: np.ndarray,
                 port_cap: np.ndarray, emergency_last: bool = False) -> None:
    """Fill unmet demand for period t using greedy cost order."""
    p = s.problem
    TC_local = p.C_PROD[:, None] + p.C_SHIP   # (N_PROD, N_PROV)

    inflow_existing = (
        s.x_loc[:, :, t].sum(axis=0)
        + s.x_dist[:, :, t].sum(axis=0)
        + s.x_trns[:, :, t].sum(axis=0)
        - s.x_trns[:, :, t].sum(axis=1)
    )
    deficit = np.maximum(0.0, p.DEMAND[:, t] - inflow_existing)

    total_imp = float(s.x_imp.sum())
    total_loc = float(s.x_loc.sum())
    over_bound = total_imp / max(total_imp + total_loc, 1.0) >= config.EPS_IMPORT_DEP

    for i in np.argsort(-deficit / np.maximum(p.DEMAND[:, t], 1.0)):
        # Sorts by RELATIVE deficit ratio — Jakarta 50% shortage gets priority 
        # over Jatim 5% shortage, even if Jatim's absolute tonnage is larger.
        if deficit[i] <= 0:
            continue
        sources = []
        for p_arr in range(p.N_PROD):
            sources.append((TC_local[p_arr, i], "loc", p_arr))
        if not emergency_last:
            for s_idx in range(p.N_IMP):
                for h in range(p.N_PORT):
                    cost = p.C_PURCH[s_idx] + p.C_DIST[h, i] + config.LOC_PREF
                    if over_bound:
                        cost += 1e12
                    sources.append((cost, "imp", (s_idx, h)))
        sources.sort()

        for cost, kind, idx in sources:
            if deficit[i] <= 0:
                break
            if kind == "loc":
                alloc = min(deficit[i], prod_cap[idx])
                if alloc > 0:
                    s.x_loc[idx, i, t] += alloc
                    prod_cap[idx]       -= alloc
                    deficit[i]          -= alloc
            elif kind == "imp":
                s_idx, h = idx
                alloc = min(deficit[i], imp_cap[s_idx], port_cap[h])
                if alloc > 0:
                    s.x_imp[s_idx, h, t] += alloc
                    s.x_dist[h, i, t]    += alloc
                    imp_cap[s_idx]        -= alloc
                    port_cap[h]           -= alloc
                    deficit[i]            -= alloc


# ═══════════════════════════════════════════════════════════════════════════
#  R1 – Greedy Cost
# ═══════════════════════════════════════════════════════════════════════════

def repair_greedy(state: SoybeanState, rng) -> SoybeanState:
    s = state.copy()
    p = s.problem
    for t in range(p.N_PERIOD):
        prod_cap = np.maximum(0.0, p.PROD_CAP[:, t].astype(float)
                                   - s.x_loc[:, :, t].sum(axis=1))
        imp_cap  = p.IMP_CAP_NORMAL[:, t].copy().astype(float)
        port_cap = p.PORT_THRU_CAP[:, t].copy().astype(float)
        _greedy_fill(s, t, prod_cap, imp_cap, port_cap, emergency_last=False)
    s.feasibility_repair()
    return s


# ═══════════════════════════════════════════════════════════════════════════
#  R2 – Regret-Based
# ═══════════════════════════════════════════════════════════════════════════

def repair_regret(state: SoybeanState, rng) -> SoybeanState:
    """Prioritise provinces with highest regret (best vs 2nd-best source cost)."""
    s = state.copy()
    p = s.problem
    TC_local = p.C_PROD[:, None] + p.C_SHIP

    for t in range(p.N_PERIOD):
        prod_cap = np.maximum(0.0, p.PROD_CAP[:, t].astype(float)
                                   - s.x_loc[:, :, t].sum(axis=1))
        imp_cap  = p.IMP_CAP_NORMAL[:, t].copy().astype(float)
        port_cap = p.PORT_THRU_CAP[:, t].copy().astype(float)

        regret = np.zeros(p.N_PROV)
        for i in range(p.N_PROV):
            min_dist = [
                min((p.C_DIST[h, i] for h in range(p.N_PORT)),
                    default=np.inf)
                for _ in range(p.N_IMP)
            ]
            costs = sorted(
                [TC_local[p_arr, i] for p_arr in range(p.N_PROD)] +
                [p.C_PURCH[s_idx] + min_dist[s_idx] + config.LOC_PREF
                 for s_idx in range(p.N_IMP)]
            )
            if len(costs) >= 2:
                regret[i] = costs[1] - costs[0]

        for i in np.argsort(-regret):           # highest regret first
            inflow = (
                s.x_loc[:, i, t].sum()
                + s.x_dist[:, i, t].sum()
                + s.x_trns[:, i, t].sum()
                - s.x_trns[i, :, t].sum()
            )
            deficit = max(0.0, p.DEMAND[i, t] - inflow)
            if deficit <= 0:
                continue

            best_p       = int(np.argmin(TC_local[:, i]))
            local_cost   = TC_local[best_p, i]
            best_s       = int(np.argmin(p.C_PURCH + config.LOC_PREF))
            min_imp_cost = p.C_PURCH[best_s] + config.LOC_PREF + min(
                (p.C_DIST[h, i] for h in range(p.N_PORT)),
                default=np.inf)

            if local_cost <= min_imp_cost:
                alloc = min(deficit, prod_cap[best_p])
                if alloc > 0:
                    s.x_loc[best_p, i, t] += alloc
                    prod_cap[best_p]       -= alloc
                    deficit                -= alloc

            # Fill remaining via the cheapest CIF import
            if deficit > 0:
                best_s = int(np.argmin(p.C_PURCH))
                for h in range(p.N_PORT):
                    alloc = min(deficit, imp_cap[best_s], port_cap[h])
                    if alloc > 0:
                        s.x_imp[best_s, h, t] += alloc
                        s.x_dist[h, i, t]     += alloc
                        imp_cap[best_s]        -= alloc
                        port_cap[h]            -= alloc
                        deficit                -= alloc
                        break

    s.feasibility_repair()
    return s


# ═══════════════════════════════════════════════════════════════════════════
#  R3 – Balanced Multi-Source
# ═══════════════════════════════════════════════════════════════════════════

def repair_balanced(state: SoybeanState, rng) -> SoybeanState:
    """Diversify supply: cap each import source ≤ 60 % of provincial demand."""
    s = state.copy()
    p = s.problem
    TC_local = p.C_PROD[:, None] + p.C_SHIP
    CAP_FRAC = 0.6

    for t in range(p.N_PERIOD):
        prod_cap = np.maximum(0.0, p.PROD_CAP[:, t].astype(float)
                                   - s.x_loc[:, :, t].sum(axis=1))
        imp_cap  = p.IMP_CAP_NORMAL[:, t].copy().astype(float)
        port_cap = p.PORT_THRU_CAP[:, t].copy().astype(float)

        for i in np.argsort(-p.DEMAND[:, t]):
            inflow = (
                s.x_loc[:, i, t].sum()
                + s.x_dist[:, i, t].sum()
                + s.x_trns[:, i, t].sum()
                - s.x_trns[i, :, t].sum()
            )
            deficit = max(0.0, p.DEMAND[i, t] - inflow)
            if deficit <= 0:
                continue

            cap_per_src = CAP_FRAC * p.DEMAND[i, t]

            best_p          = int(np.argmin(TC_local[:, i]))
            local_cost_i    = TC_local[best_p, i]
            best_imp_cost_i = min(
                (p.C_PURCH[s_idx] + p.C_DIST[h, i] + config.LOC_PREF
                 for s_idx in range(p.N_IMP)
                 for h in range(p.N_PORT)),
                default=np.inf
            )
            prefer_local = (local_cost_i <= best_imp_cost_i)

            if prefer_local:
                alloc = min(deficit, prod_cap[best_p])
                if alloc > 0:
                    s.x_loc[best_p, i, t] += alloc
                    prod_cap[best_p]       -= alloc
                    deficit                -= alloc

            # Fill remaining via import round-robin (each source ≤ cap_per_src)
            if deficit > 0:
                for s_idx in range(p.N_IMP):
                    if deficit <= 0:
                        break
                    for h in range(p.N_PORT):
                        alloc = min(deficit, cap_per_src, imp_cap[s_idx], port_cap[h])
                        if alloc > 0:
                            s.x_imp[s_idx, h, t] += alloc
                            s.x_dist[h, i, t]    += alloc
                            imp_cap[s_idx]        -= alloc
                            port_cap[h]           -= alloc
                            deficit               -= alloc
                            break

            if not prefer_local and deficit > 0:
                # Import couldn't fully cover — fall back to local
                alloc = min(deficit, prod_cap[best_p])
                if alloc > 0:
                    s.x_loc[best_p, i, t] += alloc
                    prod_cap[best_p]       -= alloc

    s.feasibility_repair()
    return s


# ═══════════════════════════════════════════════════════════════════════════
#  R4 – Emergency-Last
# ═══════════════════════════════════════════════════════════════════════════

def repair_emergency_last(state: SoybeanState, rng) -> SoybeanState:
    """Maximise regular sources; emergency activates only if shortage > X_THRESHOLD."""
    s = state.copy()
    p = s.problem
    for t in range(p.N_PERIOD):
        prod_cap = np.maximum(0.0, p.PROD_CAP[:, t].astype(float)
                                   - s.x_loc[:, :, t].sum(axis=1))
        imp_cap  = p.IMP_CAP_NORMAL[:, t].copy().astype(float)
        port_cap = p.PORT_THRU_CAP[:, t].copy().astype(float)
        _greedy_fill(s, t, prod_cap, imp_cap, port_cap, emergency_last=True)
    s.feasibility_repair()    # will activate emergency if shortage > X_THRESHOLD
    return s


# ═══════════════════════════════════════════════════════════════════════════
#  R5 – Transfer-Focused (two-phase: seed donors + route surplus)
# ═══════════════════════════════════════════════════════════════════════════

def repair_transfer_focused(state: SoybeanState, rng) -> SoybeanState:
    """
    Two-phase transfer-focused repair.

      Phase 1 (months 0–2): Over-import into "donor" provinces so their
        inventory exceeds SAFETY_STOCK for 3 consecutive months → unlocks
        w[donor, t] = 1 for t ≥ 3.

      Phase 2 (months 3–11): Forward-simulate to find eligible donors (w=1)
        with surplus, route surplus to deficit provinces. Prefer same-cluster
        pairs (cheaper C_TRANS), only transfer if cheaper than fresh import.

    feasibility_repair validates: inadequate inventory history → transfer erased.
    """
    s = state.copy()
    p = s.problem

    # ── Donor scoring: cheap port access + high local-production ratio ──
    min_port_dist = np.array([
        min((p.C_DIST[h, i] for h in range(p.N_PORT)),
            default=np.inf)
        for i in range(p.N_PROV)
    ])
    local_annual = np.zeros(p.N_PROV)
    for k_arr, prov in enumerate(p.PROD_IDX):
        local_annual[prov] = p.PROD_CAP[k_arr].sum()

    max_dist = min_port_dist[min_port_dist < np.inf].max() + 1.0
    port_score  = 1.0 - min_port_dist / max_dist
    prod_score  = local_annual / max(local_annual.max(), 1.0)
    donor_score = 0.6 * port_score + 0.4 * prod_score
    donor_score[min_port_dist == np.inf] = -np.inf
    N_DONORS = min(10, p.N_PROV)
    donor_provinces = set(np.argsort(-donor_score)[:N_DONORS].tolist())

    # ── Phase 1: seed donor inventory in months 0, 1, 2 ────────────────
    for t in range(min(3, p.N_PERIOD)):
        imp_cap_left  = np.maximum(0.0,
            p.IMP_CAP_NORMAL[:, t].astype(float) - s.x_imp[:, :, t].sum(axis=1))
        port_cap_left = np.maximum(0.0,
            p.PORT_THRU_CAP[:, t].astype(float) - s.x_imp[:, :, t].sum(axis=0))

        for i in donor_provinces:
            current_inflow = (
                s.x_loc[:, i, t].sum()
                + s.x_dist[:, i, t].sum()
            )
            prev_inv = s.inv[i, t - 1] if t > 0 else 0.0
            needed = max(0.0,
                         p.DEMAND[i, t] + p.SAFETY_STOCK[i] - prev_inv - current_inflow)
            if needed < 1.0:
                continue

            # Cheapest available import route to province i
            best_s, best_h, best_cost = None, None, np.inf
            for s_idx in range(p.N_IMP):
                for h in range(p.N_PORT):
                    c = p.C_PURCH[s_idx] + p.C_DIST[h, i]
                    if c < best_cost:
                        best_cost, best_s, best_h = c, s_idx, h
            if best_s is None:
                continue

            alloc = min(needed, imp_cap_left[best_s], port_cap_left[best_h])
            if alloc > 0:
                s.x_imp[best_s, best_h, t] += alloc
                s.x_dist[best_h, i, t]     += alloc
                imp_cap_left[best_s]        -= alloc
                port_cap_left[best_h]       -= alloc

    # ── Phase 1.5: Route surplus to critical provinces in months 0-2 ──
    critical = set(p.CRITICAL_PROV)
    for t in range(min(3, p.N_PERIOD)):
        for j in critical:
            supply_j = (
                s.x_loc[:, j, t].sum()
                + s.x_dist[:, j, t].sum()
                + s.x_trns[:, j, t].sum()
                - s.x_trns[j, :, t].sum()
            )
            deficit_j = max(0.0, p.DEMAND[j, t] - supply_j)
            if deficit_j < 1.0:
                continue

            for i in range(p.N_PROV):
                if i == j or i in critical:
                    continue
                supply_i = (
                    s.x_loc[:, i, t].sum()
                    + s.x_dist[:, i, t].sum()
                    + s.x_trns[:, i, t].sum()
                    - s.x_trns[i, :, t].sum()
                )
                avail = max(0.0, supply_i - p.DEMAND[i, t] - p.SAFETY_STOCK[i] * 0.5)
                if avail < 1.0:
                    continue
                alloc = min(deficit_j, avail)
                s.x_trns[i, j, t] += alloc
                deficit_j          -= alloc
                if deficit_j < 1.0:
                    break

    # ── Run feasibility_repair to get REAL w after Phase 1.5 ──────
    s.feasibility_repair()

    # ── Phase 2: route surplus from eligible donors to deficit provinces ──
    for t in range(3, p.N_PERIOD):
        eligible = [i for i in range(p.N_PROV) if s.w[i, t]]
        if not eligible:
            continue

        # Available surplus per eligible province (above demand + half safety)
        surplus: dict = {}
        for i in eligible:
            supply_i = (
                s.x_loc[:, i, t].sum()
                + s.x_dist[:, i, t].sum()
                + s.x_trns[:, i, t].sum()
                - s.x_trns[i, :, t].sum()
            )
            avail = max(0.0, supply_i - p.DEMAND[i, t] - p.SAFETY_STOCK[i] * 0.5)
            if avail > 1.0:
                surplus[i] = avail

        if not surplus:
            continue

        for j in np.argsort(-p.DEMAND[:, t]):    # high-demand first
            supply_j = (
                s.x_loc[:, j, t].sum()
                + s.x_dist[:, j, t].sum()
                + s.x_trns[:, j, t].sum()
                - s.x_trns[j, :, t].sum()
            )
            deficit_j = max(0.0, p.DEMAND[j, t] - supply_j)
            if deficit_j < 1.0:
                continue

            same_cluster = [i for i in surplus
                            if surplus[i] > 0 and p.CLUSTER[i] == p.CLUSTER[j] and i != j]
            any_donor    = [i for i in surplus
                            if surplus[i] > 0 and i != j and i not in same_cluster]
            candidates   = (sorted(same_cluster, key=lambda i: p.C_TRANS[i, j]) +
                            sorted(any_donor,    key=lambda i: p.C_TRANS[i, j]))

            for i in candidates:
                if deficit_j < 1.0:
                    break

                alloc = min(deficit_j, surplus[i])
                s.x_trns[i, j, t] += alloc
                surplus[i]         -= alloc
                deficit_j          -= alloc

    s.feasibility_repair()
    return s
