"""
state.py
────────
The `SoybeanState` class — decision-variable container and ε-constraint objective.

A SoybeanState holds:
  - All 5 flow variable arrays (x_loc, x_imp, x_emg, x_dist, x_trns)
  - All derived variables (inv, sh, safe, w, y, z)
  - A reference to its `Problem` instance (immutable, shared)

The objective function (`_compute_objective`) implements the ε-constraint /
AUGMECON formulation: minimise Z_cost subject to penalty-enforced constraints
on shortage, import dependency, and local production floor.
"""
from __future__ import annotations

import numpy as np
from alns.State import State

import config
from problem import Problem


class SoybeanState(State):
    """
    Solusi ITP multi-periode untuk rantai pasok kedelai nasional.

    Following standard ITP notation (Sainathuni et al. 2014, De et al. 2020):
        x_{ij,t}  = flow from node i to node j in period t
        inv_{i,t} = inventory level at node i at end of period t

    Flow arc types:
        x_loc  [N_PROD, N_PROV, N_PERIOD]  – local: producer p → province i
        x_imp  [N_IMP,  N_PORT, N_PERIOD]  – normal import:    country s → port h
        x_emg  [N_IMP,  N_PORT, N_PERIOD]  – emergency import: country s → port h
        x_dist [N_PORT, N_PROV, N_PERIOD]  – distribution:     port h → province i
        x_trns [N_PROV, N_PROV, N_PERIOD]  – transfer:         province i → province j

    Policy binary variables:
        y[s,t]    = activation of normal import contract from source s in month t
        z[t]      = activation of emergency import mode in month t
        safe[i,t] = safety-stock indicator for province i in month t
        w[i,t]    = transfer eligibility (safe for 3 consecutive months)

    Ports are throughput pipes: all imports entering a port in month t
    are distributed in the same month (no port inventory carried over).
    """

    def __init__(self, problem: Problem):
        self.problem = problem
        p = problem

        self.x_loc  = np.zeros((p.N_PROD, p.N_PROV, p.N_PERIOD))
        self.x_imp  = np.zeros((p.N_IMP,  p.N_PORT, p.N_PERIOD))
        self.x_emg  = np.zeros((p.N_IMP,  p.N_PORT, p.N_PERIOD))
        self.x_dist = np.zeros((p.N_PORT, p.N_PROV, p.N_PERIOD))
        self.x_trns = np.zeros((p.N_PROV, p.N_PROV, p.N_PERIOD))

        # Derived (populated by feasibility_repair)
        self.inv    = np.zeros((p.N_PROV, p.N_PERIOD))
        self.sh     = np.zeros((p.N_PROV, p.N_PERIOD))
        self.safe   = np.zeros((p.N_PROV, p.N_PERIOD), dtype=int)
        self.w      = np.zeros((p.N_PROV, p.N_PERIOD), dtype=int)
        self.y      = np.zeros((p.N_IMP,  p.N_PERIOD), dtype=int)
        self.z      = np.zeros(p.N_PERIOD,             dtype=int)

        self._penalty   = 0.0
        self._obj_cache = None
        self._last_destruction_scale = 0.0

    # ── Deep-copy decision arrays; share the (immutable) problem reference ──
    def copy(self) -> "SoybeanState":
        s = SoybeanState.__new__(SoybeanState)
        s.problem = self.problem            # ← shared reference (frozen dataclass)
        s.x_loc   = self.x_loc.copy()
        s.x_imp   = self.x_imp.copy()
        s.x_emg   = self.x_emg.copy()
        s.x_dist  = self.x_dist.copy()
        s.x_trns  = self.x_trns.copy()
        s.inv     = self.inv.copy()
        s.sh      = self.sh.copy()
        s.safe    = self.safe.copy()
        s.w       = self.w.copy()
        s.y       = self.y.copy()
        s.z       = self.z.copy()
        s._penalty   = self._penalty
        s._obj_cache = self._obj_cache
        return s

    # ── ALNS library API ───────────────────────────────────────────────────
    def objective(self) -> float:
        if self._obj_cache is not None:
            return self._obj_cache
        self._obj_cache = self._compute_objective()
        return self._obj_cache

    def _invalidate(self) -> None:
        self._obj_cache = None

    # ── ε-constraint / AUGMECON objective ──────────────────────────────────
    def _compute_objective(self) -> float:
        p = self.problem

        # Z_cost: primary objective to minimise (Rp)
        loc_cost  = np.sum((p.C_PROD[:, None, None] + p.C_SHIP[:, :, None]) * self.x_loc)
        imp_cost  = np.sum(p.C_PURCH[:, None, None] * self.x_imp)
        emg_cost  = np.sum(p.C_EMG[:, None, None]   * self.x_emg)
        dist_cost = np.sum(p.C_DIST[:, :, None]      * self.x_dist)
        trns_cost = np.sum(p.C_TRANS[:, :, None]     * self.x_trns)
        hold_prov = np.sum(p.H_PROV[:, None] * self.inv)
        fix_act   = np.sum(p.F_ACT[:, None]  * self.y)
        fix_emg   = float(p.F_EMG) * float(self.z.sum())

        z_cost = (loc_cost + imp_cost + emg_cost + dist_cost +
                  trns_cost + hold_prov + fix_act + fix_emg)

        # ε₁: shortage ≤ EPS_SHORTAGE
        shortage_viol = max(0.0, float(self.sh.sum()) - config.EPS_SHORTAGE)

        # ε₂: imports / (imports + local) ≤ EPS_IMPORT_DEP
        # Formula standar BPS Neraca Komoditas: ketergantungan impor =
        # impor / (impor + produksi lokal). Bisa > 100% jika pakai demand.
        total_import = float((self.x_imp + self.x_emg).sum())
        total_local  = float(self.x_loc.sum())
        imp_dep      = total_import / max(total_import + total_local, 1.0)
        imp_dep_viol = max(0.0, imp_dep - config.EPS_IMPORT_DEP)
        local_viol  = max(0.0, config.EPS_LOCAL_MIN - total_local)

        # Priority mode: selectively relax penalty when constraints conflict
        if config.PRIORITY_MODE == "shortage":
            imp_dep_viol = 0.0      # prioritise zero shortage — ignore import bound

        # ε₄: inventory floor — provinces below INV_MIN_FRAC × demand
        inv_floor_viol = float(np.maximum(0.0,
            config.INV_MIN_FRAC * p.DEMAND - self.inv).sum())

        return (z_cost
                + config.M_SHORTAGE   * shortage_viol
                + config.M_IMPORT_DEP * imp_dep_viol
                + config.M_LOCAL      * local_viol
                + config.M_INV_FLOOR  * inv_floor_viol
                + config.BIG_M        * self._penalty)

    def cost_breakdown(self) -> dict:
        p = self.problem
        loc_cost  = float(np.sum((p.C_PROD[:, None, None] + p.C_SHIP[:, :, None]) * self.x_loc))
        imp_cost  = float(np.sum(p.C_PURCH[:, None, None] * self.x_imp))
        emg_cost  = float(np.sum(p.C_EMG[:, None, None]   * self.x_emg))
        dist_cost = float(np.sum(p.C_DIST[:, :, None]      * self.x_dist))
        trns_cost = float(np.sum(p.C_TRANS[:, :, None]     * self.x_trns))
        hold_prov = float(np.sum(p.H_PROV[:, None] * self.inv))
        fix_act   = float(np.sum(p.F_ACT[:, None]  * self.y))
        fix_emg   = float(p.F_EMG) * float(self.z.sum())
        z_cost = loc_cost + imp_cost + emg_cost + dist_cost + trns_cost + hold_prov + fix_act + fix_emg

        shortage_viol = max(0.0, float(self.sh.sum()) - config.EPS_SHORTAGE)
        total_import  = float((self.x_imp + self.x_emg).sum())
        total_local   = float(self.x_loc.sum())
        imp_dep       = total_import / max(total_import + total_local, 1.0)
        imp_dep_viol  = max(0.0, imp_dep - config.EPS_IMPORT_DEP)
        local_viol    = max(0.0, config.EPS_LOCAL_MIN - total_local)

        pen_shortage  = config.M_SHORTAGE  * shortage_viol
        pen_import    = config.M_IMPORT_DEP * imp_dep_viol
        pen_local     = config.M_LOCAL     * local_viol
        inv_floor_viol = float(np.maximum(0.0,
            config.INV_MIN_FRAC * p.DEMAND - self.inv).sum())
        pen_inv_floor = config.M_INV_FLOOR * inv_floor_viol
        pen_infeas    = config.BIG_M * self._penalty

        return {
            'loc_cost': loc_cost, 'imp_cost': imp_cost, 'emg_cost': emg_cost,
            'dist_cost': dist_cost, 'trns_cost': trns_cost,
            'hold_prov': hold_prov,
            'fix_act': fix_act, 'fix_emg': fix_emg,
            'z_cost': z_cost,
            'pen_shortage': pen_shortage, 'pen_import': pen_import,
            'pen_local': pen_local, 'pen_inv_floor': pen_inv_floor,
            'pen_infeas': pen_infeas,
            'total_penalty': pen_shortage + pen_import + pen_local + pen_inv_floor + pen_infeas,
            'objective': z_cost + pen_shortage + pen_import + pen_local + pen_inv_floor + pen_infeas,
        }

    # ── Feasibility repair: enforce balance + capacity, derive auxiliaries ──
    def feasibility_repair(self) -> None:
        """
        Recompute all derived variables from the current flow decisions and
        enforce hard capacity constraints via proportional reduction.
        Sets self._penalty for any infeasibility that survives the reduction.

        Ports are throughput pipes: all imports entering a port in month t
        must be distributed in the same month. Emergency imports also pass
        through ports but are routed directly to shortage provinces.
        """
        p = self.problem
        penalty = 0.0
        prev_inv_prov = np.zeros(p.N_PROV)

        for t in range(p.N_PERIOD):
            # ── 1. Port Throughput Capacity enforcement on x_imp ──────────
            port_in_normal = self.x_imp[:, :, t].sum(axis=0)

            self.x_emg[:, :, t] = 0.0
            self.z[t] = 0

            for h in range(p.N_PORT):
                if port_in_normal[h] > p.PORT_THRU_CAP[h, t]:
                    ratio = p.PORT_THRU_CAP[h, t] / port_in_normal[h]
                    self.x_imp[:, h, t] *= ratio
                    port_in_normal[h] = p.PORT_THRU_CAP[h, t]

            # ── 2. Pre-balance throughput enforcement on x_dist ──────────
            # Ensure x_dist == total port inflow (ports are throughput pipes).
            # Scale DOWN if x_dist > inflow; scale UP if x_dist < inflow;
            # if x_dist == 0 but inflow > 0, seed equally across PORT_SERV provinces.
            for h in range(p.N_PORT):
                total_inflow  = float(self.x_imp[:, h, t].sum() + self.x_emg[:, h, t].sum())
                total_outflow = float(self.x_dist[h, :, t].sum())
                if total_inflow <= 0:
                    self.x_dist[h, :, t] = 0.0
                elif total_outflow == 0:
                    # Seed evenly across served provinces so nothing disappears
                    served = p.PORT_SERV.get(h, list(range(p.N_PROV)))
                    share = total_inflow / len(served)
                    for i_prov in served:
                        self.x_dist[h, i_prov, t] = share
                elif total_outflow != total_inflow:
                    ratio = total_inflow / total_outflow
                    self.x_dist[h, :, t] *= ratio

            # ── 3. Province inventory balance (first pass) ────────────────
            inflow = (
                self.x_loc[:, :, t].sum(axis=0)
                + self.x_dist[:, :, t].sum(axis=0)
                + self.x_trns[:, :, t].sum(axis=0)
                - self.x_trns[:, :, t].sum(axis=1)
            )
            raw_inv = prev_inv_prov + inflow - p.DEMAND[:, t]
            self.sh[:, t]  = np.maximum(0.0, -raw_inv)
            self.inv[:, t] = np.maximum(0.0, raw_inv)

            # Capacity repair: proportional reduction on inflow
            overflow_mask = self.inv[:, t] > p.PROV_STOR_CAP
            for i in np.where(overflow_mask)[0]:
                excess = self.inv[i, t] - p.PROV_STOR_CAP[i]
                total_in = max(inflow[i], 1e-9)
                beta = 1.0 - excess / total_in
                beta = max(0.0, min(1.0, beta))
                self.x_loc[:, i, t]  *= beta
                self.x_dist[:, i, t] *= beta
                self.x_trns[:, i, t] *= beta
                new_inflow_i = (
                    self.x_loc[:, i, t].sum()
                    + self.x_dist[:, i, t].sum()
                    + self.x_trns[:, i, t].sum()
                    - self.x_trns[i, :, t].sum()
                )
                self.inv[i, t] = max(0.0, prev_inv_prov[i] + new_inflow_i - p.DEMAND[i, t])
                self.sh[i, t]  = max(0.0, -(prev_inv_prov[i] + new_inflow_i - p.DEMAND[i, t]))
                self.inv[i, t] = min(self.inv[i, t], p.PROV_STOR_CAP[i])

            # Safety stock indicator
            self.safe[:, t] = (self.inv[:, t] >= p.SAFETY_STOCK).astype(int)

            # ── 4. Emergency imports ─────────────────────────────────────
            if config.USE_EMERGENCY and self.sh[:, t].sum() > p.X_THRESHOLD:
                self.z[t] = 1
                for i in np.argsort(-self.sh[:, t]):
                    if self.sh[i, t] <= 0:
                        break
                    for s in range(p.N_IMP):
                        for h in range(p.N_PORT):
                            avail = max(0.0, p.IMP_CAP_EMERG[s, t] - self.x_emg[s, :, t].sum())
                            port_thru_used = port_in_normal[h] + self.x_emg[:, h, t].sum()
                            port_thru_avail = max(0.0, p.PORT_THRU_CAP[h, t] - port_thru_used)

                            alloc = min(self.sh[i, t], avail, port_thru_avail)
                            if alloc > 0:
                                self.x_emg[s, h, t] += alloc
                                self.x_dist[h, i, t] += alloc
                                self.sh[i, t]        -= alloc
                                self.inv[i, t]       += alloc
                                if self.sh[i, t] <= 0:
                                    break
                        if self.sh[i, t] <= 0:
                            break

            # ── 5. Post-emergency throughput enforcement + FINAL balance ──
            # Emergency may have added x_dist; recheck and recompute derived vars.
            for h in range(p.N_PORT):
                total_inflow  = float(self.x_imp[:, h, t].sum() + self.x_emg[:, h, t].sum())
                total_outflow = float(self.x_dist[h, :, t].sum())
                if total_inflow <= 0:
                    self.x_dist[h, :, t] = 0.0
                elif total_outflow == 0:
                    served = p.PORT_SERV.get(h, list(range(p.N_PROV)))
                    share = total_inflow / len(served)
                    for i_prov in served:
                        self.x_dist[h, i_prov, t] = share
                elif total_outflow != total_inflow:
                    ratio = total_inflow / total_outflow
                    self.x_dist[h, :, t] *= ratio

            # Final province balance — definitive sh, inv, safe for this period
            inflow = (
                self.x_loc[:, :, t].sum(axis=0)
                + self.x_dist[:, :, t].sum(axis=0)
                + self.x_trns[:, :, t].sum(axis=0)
                - self.x_trns[:, :, t].sum(axis=1)
            )
            raw_inv = prev_inv_prov + inflow - p.DEMAND[:, t]
            self.sh[:, t]  = np.maximum(0.0, -raw_inv)
            self.inv[:, t] = np.maximum(0.0, raw_inv)
            self.safe[:, t] = (self.inv[:, t] >= p.SAFETY_STOCK).astype(int)

            # ── Policy binaries ─────────────────────────────────────────
            self.y[:, t] = (self.x_imp[:, :, t].sum(axis=1) > 0).astype(int)

            prev_inv_prov = self.inv[:, t].copy()

        # ── Transfer eligibility: safe for 3 consecutive prior months ──
        self.w[:] = 0
        for t in range(3, p.N_PERIOD):
            self.w[:, t] = self.safe[:, t-1] & self.safe[:, t-2] & self.safe[:, t-3]

        # Policy violation: transfer from ineligible province → zero + penalty
        # Exception: transfers TO critical provinces in months 0-2 are allowed
        critical = set(p.CRITICAL_PROV)
        for t in range(p.N_PERIOD):
            for i in range(p.N_PROV):
                if self.w[i, t] == 0 and self.x_trns[i, :, t].sum() > 0:
                    if t < 3:
                        all_to_critical = all(
                            j in critical or self.x_trns[i, j, t] < 0.1
                            for j in range(p.N_PROV)
                        )
                        if all_to_critical:
                            continue
                    penalty += self.x_trns[i, :, t].sum()
                    self.x_trns[i, :, t] = 0.0

        self._penalty = penalty
        self._invalidate()
