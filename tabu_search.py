"""
tabu_search.py
──────────────
Short tabu local search applied as a post-iteration step on the incumbent
solution (S_best) — called from the main ALNS manual loop in solver.py.

This follows the paper flow: after SA acceptance + weight update, a tabu
sub-loop improves the current best-known solution (Step 6, Section 4.5).

Move definition: relocate a single (province, period) allocation by shifting
some volume from imports to local production (where local has spare capacity).
The tabu list (FIFO) holds recently visited (i, t) moves to prevent cycling.

NOTE: make_repair_with_tabu() has been REMOVED. Tabu is no longer embedded
inside repair operators. It is now a separate step in the ALNS main loop.
"""
from __future__ import annotations

import numpy as np

import config
from state import SoybeanState


def tabu_local_search(state: SoybeanState, rng_ls=None) -> SoybeanState:
    """
    Run a short tabu-list local search around `state`.

    Called from the main ALNS loop (solver.py) on S_best after each
    ALNS iteration (or only when S_best changes, depending on
    config.TABU_EVERY_ITER).

    For each sub-iteration:
      1. Pick a random (province i, period t) — skip if tabu.
      2. Move: shift up to 30 % of i's free local capacity from imports → local.
      3. Repair feasibility, accept if objective improves.
      4. Push (i, t) onto tabu list (FIFO with TABU_TENURE).

    Parameters in `config`: TABU_TENURE, SUB_IT.
    """
    if rng_ls is None:
        rng_ls = np.random.default_rng()

    p = state.problem
    best_local = state.copy()
    current    = state.copy()
    tabu: list = []                           # list of (i, t) tuples

    TC_local = p.C_PROD[:, None] + p.C_SHIP

    for _ in range(config.SUB_IT):
        i_try = int(rng_ls.integers(0, p.N_PROV))
        t_try = int(rng_ls.integers(0, p.N_PERIOD))
        move  = (i_try, t_try)

        if move in tabu:
            continue

        cand = current.copy()
        local_available = max(0.0, p.PROD_CAP[:, t_try].sum()
                                   - cand.x_loc[:, :, t_try].sum())
        shift = min(cand.x_dist[:, i_try, t_try].sum(), local_available * 0.3)

        if shift > 0:
            best_p = int(np.argmin(TC_local[:, i_try]))
            cand.x_loc[best_p, i_try, t_try] += shift

            # Reduce cheapest imports to keep flows balanced
            for h in range(p.N_PORT):
                red = min(shift, cand.x_dist[h, i_try, t_try])
                cand.x_dist[h, i_try, t_try] -= red
                for s_idx in range(p.N_IMP):
                    r2 = min(red, cand.x_imp[s_idx, h, t_try])
                    cand.x_imp[s_idx, h, t_try] -= r2
                    red -= r2
                    if red <= 0:
                        break
                if red <= 0:
                    break

        cand.feasibility_repair()

        if cand.objective() < best_local.objective():
            best_local = cand.copy()
        current = cand

        tabu.append(move)
        if len(tabu) > config.TABU_TENURE:
            tabu.pop(0)

    return best_local
