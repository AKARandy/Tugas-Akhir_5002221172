"""
solver.py
─────────
The `SoybeanALNSSolver` class — orchestrator for the full optimisation run.

Implements a manual ALNS loop following Fathollahi-Fard et al. (2023):
  Destroy → Repair (pure) → SA → Score/Weight → Tabu (on S_best)

It owns:
  - The Problem instance
  - The list of destroy/repair operators
  - SA temperature schedule and operator weight tracking
  - Convergence tracking (obj_history, operator counts)

A typical run:
    problem = Problem.load(data_dir=".")
    solver  = SoybeanALNSSolver(problem)
    solver.run()
"""
from __future__ import annotations

import time

import numpy as np

import config
from problem import Problem
from state import SoybeanState
import reporting

import operators_destroy as ds
import operators_repair  as rp
from tabu_search import tabu_local_search
from prism_refraction_search import prism_refraction_local_search


class SoybeanALNSSolver:
    def __init__(self, problem: Problem,
                 *,
                 max_iter: int = None,
                 seed: int = None,
                 exclude_ops: set = None,
                 use_tabu: bool = True) -> None:
        self.problem  = problem
        self.max_iter = max_iter if max_iter is not None else config.MAX_ITER
        self.seed     = seed     if seed     is not None else config.SEED
        self.exclude_ops = exclude_ops or set()
        self.use_tabu = use_tabu

        # Tracking
        self.obj_history: list = []
        self.destroy_counts: dict = {}
        self.repair_counts: dict  = {}

    # ─────────────────────────────────────────────────────────────────────
    def build_initial(self) -> SoybeanState:
        """Build initial solution — historical data seed or two-phase greedy."""
        if config.USE_HISTORICAL_IC:
            return rp.build_historical_initial(self.problem)
        return rp.build_initial_solution(self.problem)

    # ─────────────────────────────────────────────────────────────────────
    def lock_in_eps_local_min(self, initial: SoybeanState, silent: bool = False) -> None:
        config.EPS_LOCAL_MIN = float(initial.x_loc.sum())
        if not silent:
            print(f"      e3 local prod >= {config.EPS_LOCAL_MIN:,.0f} ton  "
                  f"(= initial solution baseline)")

    # ─────────────────────────────────────────────────────────────────────
    def _calibrate_penalty_multipliers(self, initial: SoybeanState,
                                        silent: bool = False) -> None:
        """
        Auto-calibrate all ε-constraint penalty multipliers using normalized
        AUGMECON approach. Each multiplier is set so that a violation at 100%
        of its maximum possible range produces a penalty = PENALTY_SCALE × z0.

        This is fully agnostic — works for any data, any demand level, any cost.
        """
        p = self.problem
        z0 = max(initial.cost_breakdown()['z_cost'], 1.0)  # floor prevents div-by-zero
        S = config.PENALTY_SCALE

        # Max possible violation ranges (computed from data at runtime)
        r_shortage  = max(float(p.DEMAND.sum()), 1.0)       # worst: all demand unmet
        r_import    = 1.0                                    # ratio range [0, 1]
        r_local     = max(config.EPS_LOCAL_MIN, 1.0)         # worst: zero local
        r_inv_floor = max(config.INV_MIN_FRAC * float(p.DEMAND.sum()), 1.0)  # worst: all inv=0

        config.M_SHORTAGE   = S * z0 / r_shortage
        config.M_IMPORT_DEP = S * z0 / r_import
        config.M_LOCAL      = S * z0 / r_local
        config.M_INV_FLOOR  = S * z0 / r_inv_floor

        if not silent:
            print(f"      Penalty calibration (PENALTY_SCALE={S}, z0={z0:.2e}):")
            print(f"        M_SHORTAGE   = {config.M_SHORTAGE:.2e}  (per ton)")
            print(f"        M_IMPORT_DEP = {config.M_IMPORT_DEP:.2e}  (per unit ratio)")
            print(f"        M_LOCAL      = {config.M_LOCAL:.2e}  (per ton)")
            print(f"        M_INV_FLOOR  = {config.M_INV_FLOOR:.2e}  (per ton)")

    # ─────────────────────────────────────────────────────────────────────
    def _check_physical_feasibility(self, initial: SoybeanState,
                                     silent: bool = False) -> None:
        """Warn if e-constraints conflict with physical reality (demand, local cap)."""
        p = self.problem
        total_imp_init = float(initial.x_imp.sum() + initial.x_emg.sum())
        total_loc_init = float(initial.x_loc.sum())
        total_supply   = max(total_imp_init + total_loc_init, 1.0)
        local_share    = total_loc_init / total_supply
        min_feasible_import = 1.0 - local_share

        if not silent:
            print(f"      Physical limits:  max local = {local_share:.1%}  "
                  f"min import (for zero shortage) = {min_feasible_import:.1%}")

        if config.EPS_IMPORT_DEP < min_feasible_import and config.EPS_SHORTAGE <= 0.5:
            gap = min_feasible_import - config.EPS_IMPORT_DEP
            if not silent:
                print(f"      [WARNING] e2 import dep bound ({config.EPS_IMPORT_DEP:.0%}) "
                      f"< min possible import ({min_feasible_import:.0%})  "
                      f"gap = {gap:.1%}")
                print(f"      [WARNING] Zero shortage + e2 bound is PHYSICALLY IMPOSSIBLE "
                      f"(shortage {min_feasible_import * float(p.DEMAND.sum()):,.0f} ton unavoidable).")
            mode = config.PRIORITY_MODE
            if mode == "balanced" and not silent:
                print(f"      [INFO]    Priority: balanced -> solver finds best compromise.")
            elif mode == "shortage" and not silent:
                print(f"      [INFO]    Priority: shortage -> import penalty relaxed (e2 ignored).")
            elif mode == "import":
                demand_ton = float(p.DEMAND.sum())
                local_ton = float(initial.x_loc.sum())
                max_import_ton = config.EPS_IMPORT_DEP * demand_ton
                min_shortage = max(0.0, demand_ton - local_ton - max_import_ton)
                original_eps = config.EPS_SHORTAGE
                config.EPS_SHORTAGE = max(original_eps, min_shortage + 1.0)
                if not silent:
                    print(f"      [INFO]    Priority: import bound {config.EPS_IMPORT_DEP:.0%} "
                          f"-> shortage target relaxed ({original_eps:,.0f} -> "
                          f"{config.EPS_SHORTAGE:,.0f} ton, min unavoidable ~{min_shortage:,.0f}).")

    # ─────────────────────────────────────────────────────────────────────
    def _build_operator_lists(self) -> tuple[list, list]:
        """Build destroy/repair operator lists (repair operators are PURE —
        no tabu wrapper. Tabu is a separate step in the main loop)."""
        destroy_ops = [
            ds.destroy_random_temporal,
            ds.destroy_cost_based,
            ds.destroy_shortage_based,
            ds.destroy_geographic,
            ds.destroy_bottleneck_port,
            ds.destroy_relatedness,
        ]
        if config.USE_EMERGENCY:
            destroy_ops.insert(4, ds.destroy_policy_emergency)

        destroy_ops = [op for op in destroy_ops if op.__name__ not in self.exclude_ops]

        repair_ops = [rp.repair_greedy, rp.repair_regret,
                      rp.repair_balanced, rp.repair_transfer_focused]
        if config.USE_EMERGENCY:
            repair_ops.append(rp.repair_emergency_last)

        repair_ops = [op for op in repair_ops
                      if op.__name__ not in self.exclude_ops]
        return destroy_ops, repair_ops

    # ─────────────────────────────────────────────────────────────────────
    def configure_run(self, initial: SoybeanState, silent: bool = False) -> dict:
        """Setup SA parameters, operator weights, and tracking structures
        for the manual ALNS loop. Returns a config dict consumed by run()."""
        rng = np.random.default_rng(self.seed)

        destroy_ops, repair_ops = self._build_operator_lists()
        n_d = len(destroy_ops)
        n_r = len(repair_ops)

        # Initialise operator weights (all equal = 1.0, as in paper)
        d_weights = np.ones(n_d)
        r_weights = np.ones(n_r)

        # SA temperature setup
        f0 = initial.objective()
        if config.SA_COOLING == "geometric":
            # Auto-calibrate T0 proportional to initial objective value.
            # Static config.SA_PAPER_T0 is ignored — same pattern as EPS_LOCAL_MIN.
            T0 = 0.05 * abs(f0)
            T_end = T0 * 1e-4
            cooling_step = None
            redu = config.SA_REDU
        else:  # exponential (default)
            delta = 0.05 * abs(f0)
            T0 = -delta / np.log(0.5)
            T_end = T0 * 1e-4
            cooling_step = (T_end / T0) ** (1.0 / self.max_iter)
            redu = None

        # Initialise tracking
        self.destroy_counts = {op.__name__: 0 for op in destroy_ops}
        self.repair_counts  = {op.__name__: 0 for op in repair_ops}
        self.curr_history = [f0]
        self.best_history = [f0]
        bd_init = initial.cost_breakdown()
        self.z_cost_history = [bd_init['z_cost']]
        self.penalty_history = [bd_init['total_penalty']]
        self.op_history = []          # list of (d_name, r_name, outcome) per iteration
        self.destruction_scale_history = []  # per-iteration destruction scale

        tabu_mode = ("every iter" if config.TABU_EVERY_ITER else "on new best")
        prs_mode = ("every iter" if config.PRS_EVERY_ITER else "on new best")
        sa_mode   = ("paper (vs S_best)" if config.SA_PAPER else "standard (vs S_current)")

        if not silent:
            print(f"      Destroy ops : {n_d}")
            print(f"      Repair ops  : {n_r}  (pure — tabu is separate step)")
            print(f"      Max iters   : {self.max_iter}")
            print(f"      SA mode     : {sa_mode}")
            print(f"      Cooling     : {config.SA_COOLING}")
            print(f"      Tabu        : {tabu_mode if self.use_tabu else 'off'}")
            print(f"      PRS         : {prs_mode if config.USE_PRS else 'off'}")
            print(f"      T0 / T_end  : {T0:.2e} / {T_end:.2e}")

        return {
            'rng': rng,
            'destroy_ops': destroy_ops, 'repair_ops': repair_ops,
            'n_d': n_d, 'n_r': n_r,
            'd_weights': d_weights, 'r_weights': r_weights,
            'T0': T0, 'T_end': T_end,
            'cooling_step': cooling_step, 'redu': redu,
        }

    # ── Operator count formatting (actual counts from manual loop) ──────
    def _format_op_counts(self) -> None:
        """Actual counts tracked in the manual loop — no approximation needed."""
        pass  # counts are already populated directly in run() loop

    # ─────────────────────────────────────────────────────────────────────
    def run(self, silent: bool = False) -> dict:
        """Execute the full pipeline: build → configure → manual ALNS loop → report.

        If silent=True, suppress all console output, plots, and CSV exports.
        The return dict is always populated regardless of silent mode.
        """
        if not silent:
            self._print_banner()

        # 1. Build initial
        if not silent:
            print("\n[1/4] Building initial solution (two-phase greedy)…")
        t0 = time.perf_counter()
        initial = self.build_initial()
        if not silent:
            print(f"      Done in {time.perf_counter() - t0:.2f}s")
        self.lock_in_eps_local_min(initial, silent=silent)
        self._calibrate_penalty_multipliers(initial, silent=silent)
        self._check_physical_feasibility(initial, silent=silent)
        if not silent:
            reporting.print_summary(initial, "Initial Solution")

        # 2. Configure
        if not silent:
            print("[2/4] Configuring ALNS (manual loop — paper flow)…")
        cfg = self.configure_run(initial, silent=silent)
        rng = cfg['rng']
        destroy_ops = cfg['destroy_ops']
        repair_ops = cfg['repair_ops']
        n_d, n_r = cfg['n_d'], cfg['n_r']
        d_weights = cfg['d_weights']
        r_weights = cfg['r_weights']
        T = cfg['T0']
        T_end = cfg['T_end']

        # State initialization
        current = initial.copy()    # S_current — base for destroy in next iter
        best = initial.copy()       # S_best — target of local search (Step 6)

        # Score lookup
        SCORE_MAP = {
            'new_best':  config.ALNS_SCORE_GLOBAL_BEST,
            'better':   config.ALNS_SCORE_BETTER,
            'accepted': config.ALNS_SCORE_SA_ACCEPT,
            'rejected': config.ALNS_SCORE_REJECT,
        }

        # 3. Manual ALNS loop
        if not silent:
            print(f"\n[3/4] Running ALNS ({self.max_iter} iterations)…")
        t1 = time.perf_counter()

        for it in range(self.max_iter):
            # ── Step 1: Roulette wheel select operator pair ──
            d_idx = rng.choice(n_d, p=d_weights / d_weights.sum())
            r_idx = rng.choice(n_r, p=r_weights / r_weights.sum())
            d_op_name = destroy_ops[d_idx].__name__
            r_op_name = repair_ops[r_idx].__name__

            # ── Step 2: Destroy → Repair (pure, NO tabu) ──
            destroyed = destroy_ops[d_idx](current, rng)
            destruction_scale = getattr(destroyed, '_last_destruction_scale', 0.0)
            candidate = repair_ops[r_idx](destroyed, rng)

            cand_obj = candidate.objective()

            # ── Step 3: SA accept / reject ──
            outcome = None

            if config.SA_PAPER:
                # Paper mode (Eq. 32): compare vs S_best
                delta = abs(cand_obj - best.objective())
                if cand_obj < best.objective():
                    # New global best — always accept
                    best = candidate.copy()
                    current = candidate.copy()
                    outcome = 'new_best'
                elif rng.random() < np.exp(-delta / T):
                    # Worse than best, but SA accepts
                    current = candidate.copy()
                    outcome = 'accepted'
                else:
                    outcome = 'rejected'
            else:
                # Standard SA: compare vs S_current
                delta = cand_obj - current.objective()
                if delta <= 0:
                    # Better (or equal to) current
                    current = candidate.copy()
                    if cand_obj < best.objective():
                        best = candidate.copy()
                        outcome = 'new_best'
                    else:
                        outcome = 'better'
                elif rng.random() < np.exp(-delta / T):
                    # Worse but SA accepts
                    current = candidate.copy()
                    outcome = 'accepted'
                else:
                    outcome = 'rejected'

            # ── Steps 4 & 5: Score + Update weights ──
            score = SCORE_MAP[outcome]
            lam = config.ALNS_DECAY
            d_weights[d_idx] = lam * d_weights[d_idx] + (1 - lam) * score
            r_weights[r_idx] = lam * r_weights[r_idx] + (1 - lam) * score

            # Floor: prevent weight collapse (maintains operator diversity)
            d_weights = np.maximum(d_weights, config.ALNS_MIN_WEIGHT)
            r_weights = np.maximum(r_weights, config.ALNS_MIN_WEIGHT)

            # ── Step 6: local search on S_best ──
            if self.use_tabu:
                if config.TABU_EVERY_ITER or outcome == 'new_best':
                    improved = tabu_local_search(best, rng)
                    if improved.objective() < best.objective():
                        best = improved.copy()

            if config.USE_PRS:
                if config.PRS_EVERY_ITER or outcome == 'new_best':
                    improved = prism_refraction_local_search(best, rng)
                    if improved.objective() < best.objective():
                        best = improved.copy()

            # ── Temperature update ──
            if config.SA_COOLING == "geometric":
                T = max(T_end, T * config.SA_REDU)
            else:
                T *= cfg['cooling_step']

            # ── Tracking ──
            self.curr_history.append(current.objective())
            self.best_history.append(best.objective())
            bd_best = best.cost_breakdown()
            self.z_cost_history.append(bd_best['z_cost'])
            self.penalty_history.append(bd_best['total_penalty'])
            self.destroy_counts[d_op_name] += 1
            self.repair_counts[r_op_name] += 1
            self.op_history.append((d_op_name, r_op_name, outcome))
            self.destruction_scale_history.append(float(destruction_scale))

        elapsed = time.perf_counter() - t1
        if not silent:
            print(f"      Done in {elapsed:.2f}s  ({elapsed / self.max_iter * 1000:.1f}ms/iter)")

        best_state: SoybeanState = best
        if not silent:
            reporting.print_summary(best_state, "ALNS Best Solution")

        improvement = (initial.objective() - best_state.objective()) \
                       / abs(initial.objective()) * 100

        bd_i = initial.cost_breakdown()
        bd_b = best_state.cost_breakdown()

        if not silent:
            print(f"  Objective improvement: {improvement:+.2f}%")

            print(f"\n  -- Improvement Decomposition --")
            print(f"  {'Component':<22}  {'Initial':>16}  {'Best':>16}  {'d%':>8}")
            print(f"  {'-'*66}")
            for key, label in [('z_cost', 'Z_cost (murni)'),
                               ('total_penalty', 'Total Penalty'),
                               ('objective', 'Objective')]:
                vi, vb = bd_i[key], bd_b[key]
                pct = (vi - vb) / abs(vi) * 100 if abs(vi) > 0 else 0
                print(f"  {label:<22}  {vi:>16,.0f}  {vb:>16,.0f}  {pct:>+7.2f}%")

        # 4. Reports
        if not silent:
            print("[4/4] Generating plots and CSV exports…")
            try:
                reporting.plot_results(initial, best_state,
                                        self.best_history, self.z_cost_history,
                                        self.penalty_history,
                                        self.destroy_counts, self.repair_counts,
                                        out_dir=config.OUTPUT_DIR)
            except RuntimeError as exc:
                print(f"  [PLOT] Skipped: {exc}")
            reporting.export_csv_results(initial, best_state,
                                          out_dir=config.OUTPUT_DIR)

            import reporting_per_entity
            reporting_per_entity.export_all_per_entity(
                initial, best_state, base_dir=config.OUTPUT_DIR)

            self._print_final_table(best_state)
            self._print_eps_final(best_state)
            print(f"\n  Output files saved to: {config.OUTPUT_DIR}")
            print("\n  Optimization complete.\n")

        p_sh = best_state.sh
        worst_shortage_prov = int(np.argmax(p_sh.sum(axis=1)))
        worst_shortage_month = int(np.argmax(p_sh.sum(axis=0)))
        worst_shortage_val = float(p_sh.max())

        d_names = [op.__name__ for op in destroy_ops]
        r_names = [op.__name__ for op in repair_ops]

        result = {"initial": initial, "best": best_state,
                  "seed": self.seed,
                  "max_iter": self.max_iter,
                  "use_tabu": self.use_tabu,
                  "use_prs": config.USE_PRS,
                  "history_best": self.best_history,
                  "history_z_cost": self.z_cost_history,
                  "history_penalty": self.penalty_history,
                  "history_curr": self.curr_history,
                  "breakdown_initial": bd_i,
                  "breakdown_best": bd_b,
                  "destroy_counts": {k: v for k, v in self.destroy_counts.items()},
                  "repair_counts": {k: v for k, v in self.repair_counts.items()},
                  "d_weights_final": {d_names[i]: float(d_weights[i]) for i in range(n_d)},
                  "r_weights_final": {r_names[i]: float(r_weights[i]) for i in range(n_r)},
                  "op_history": list(self.op_history),
                  "destruction_scale_history": list(self.destruction_scale_history),
                  "worst_shortage_prov": worst_shortage_prov,
                  "worst_shortage_month": worst_shortage_month,
                  "worst_shortage_val": worst_shortage_val,
                  "improvement_pct": improvement,
                  "elapsed_seconds": elapsed}

        if not silent:
            from result_artifacts import export_optimization_result
            artifact_path = export_optimization_result(result, out_dir=config.OUTPUT_DIR)
            print(f"  JSON artifact saved to: {artifact_path}")

        return result

    # ─────────────────────────────────────────────────────────────────────
    def _print_banner(self) -> None:
        mode_str = ("WITH emergency imports" if config.USE_EMERGENCY
                    else "WITHOUT emergency imports (baseline)")
        print("\n" + "=" * 60)
        print("  Soybean Supply Chain Optimization  -  ALNS")
        print("  ITS Surabaya  |  Tugas Akhir  |  2026")
        print("=" * 60)
        print(f"  Mode       : {mode_str}")
        print(f"  Objective  : Minimize Z_cost  (e-constraint / AUGMECON)")
        print(f"  e1 shortage <= {config.EPS_SHORTAGE:,.0f} ton")
        print(f"  e2 import dep <= {config.EPS_IMPORT_DEP:.0%}  (impor / (impor + lokal))")
        print(f"  e3 local prod >= [dari solusi awal - dihitung setelah greedy]")
        print("=" * 60)

    def _print_final_table(self, best: SoybeanState) -> None:
        p = self.problem
        print("\n  All Provinces - Service Rate (ALNS best):")
        print(f"  {'Province':<28}  {'Demand':>9}  {'Shortage':>9}  {'Service':>8}")
        print(f"  {'-' * 58}")
        sh = best.sh.sum(axis=1)
        dm = p.DEMAND.sum(axis=1)
        for i in np.argsort(-sh):
            sr = 1.0 - sh[i] / max(dm[i], 1)
            flag = " !" if sh[i] > 0.5 else "  "
            print(f"  {p.PROV_NAMES[i]:<28}  {dm[i]:>9,.0f}  "
                  f"{sh[i]:>9,.0f}  {sr:>7.2%}{flag}")

        print("\n  Emergency import activations by month:")
        for t in range(p.N_PERIOD):
            flag = "ACTIVE" if best.z[t] else "  -"
            print(f"    Month {t + 1:>2}: {flag}  "
                  f"(national shortage = {best.sh[:, t].sum():>10,.0f} ton)")

    def _print_eps_final(self, best: SoybeanState) -> None:
        p = self.problem
        sh_fin       = best.sh.sum()
        total_imp    = best.x_imp.sum() + best.x_emg.sum()
        total_loc    = best.x_loc.sum()
        imp_dep_fin  = total_imp / max(total_imp + total_loc, 1.0)
        loc_fin      = best.x_loc.sum()

        print(f"\n  e-constraint final status:")
        print(f"    e1 shortage:   {sh_fin:>12,.1f} ton   (limit <= {config.EPS_SHORTAGE:,.0f})"
              + ("  OK" if sh_fin <= config.EPS_SHORTAGE + 0.5 else "  VIOLATED"))
        print(f"    e2 import dep: {imp_dep_fin:>11.2%}      (limit <= {config.EPS_IMPORT_DEP:.0%})"
              + ("  OK" if imp_dep_fin <= config.EPS_IMPORT_DEP + 1e-6 else "  VIOLATED"))
        print(f"    e3 local prod: {loc_fin:>12,.0f} ton   (limit >= {config.EPS_LOCAL_MIN:,.0f})"
              + ("  OK" if loc_fin >= config.EPS_LOCAL_MIN - 0.5 else "  VIOLATED"))
