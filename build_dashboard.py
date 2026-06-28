"""
build_dashboard.py
──────────────────
Read hasil/optimization_result.json and export it as a JavaScript data
file for the local dashboard.

After running this script, open `dashboard/index.html` in your browser.
No server needed — data is embedded as a `const DATA = {...};` JS file.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np

import config
from result_artifacts import load_optimization_result


# ── Approximate province centroids (lat, lng) ──────────────────────────────
PROV_COORDS = [
    ( 4.70,  96.70),  # 0  Aceh
    ( 2.60,  98.70),  # 1  Sumatera Utara
    (-0.95, 100.40),  # 2  Sumatera Barat
    ( 0.50, 101.50),  # 3  Riau
    (-1.60, 102.50),  # 4  Jambi
    (-3.00, 104.00),  # 5  Sumatera Selatan
    (-3.60, 102.30),  # 6  Bengkulu
    (-4.60, 105.00),  # 7  Lampung
    (-2.70, 106.40),  # 8  Kep. Bangka Belitung
    ( 1.00, 104.50),  # 9  Kepulauan Riau
    (-6.20, 106.85),  # 10 DKI Jakarta
    (-7.00, 107.60),  # 11 Jawa Barat
    (-7.30, 110.00),  # 12 Jawa Tengah
    (-7.80, 110.40),  # 13 DI Yogyakarta
    (-7.50, 112.50),  # 14 Jawa Timur
    (-6.50, 106.20),  # 15 Banten
    (-8.40, 115.20),  # 16 Bali
    (-8.60, 117.40),  # 17 Nusa Tenggara Barat
    (-8.70, 121.00),  # 18 Nusa Tenggara Timur
    ( 0.00, 110.50),  # 19 Kalimantan Barat
    (-1.70, 113.40),  # 20 Kalimantan Tengah
    (-3.10, 115.30),  # 21 Kalimantan Selatan
    ( 0.50, 116.40),  # 22 Kalimantan Timur
    ( 3.50, 117.00),  # 23 Kalimantan Utara
    ( 1.50, 124.80),  # 24 Sulawesi Utara
    (-1.40, 121.40),  # 25 Sulawesi Tengah
    (-4.50, 119.80),  # 26 Sulawesi Selatan
    (-4.00, 122.50),  # 27 Sulawesi Tenggara
    ( 0.70, 122.50),  # 28 Gorontalo
    (-2.80, 119.00),  # 29 Sulawesi Barat
    (-3.20, 129.00),  # 30 Maluku
    ( 1.00, 127.80),  # 31 Maluku Utara
    (-1.00, 133.50),  # 32 Papua Barat
    (-1.50, 132.00),  # 33 Papua Barat Daya
    (-4.50, 138.50),  # 34 Papua
    (-7.50, 140.50),  # 35 Papua Selatan
    (-3.50, 136.50),  # 36 Papua Tengah
    (-4.00, 138.00),  # 37 Papua Pegunungan
]

# ── Port locations (lat, lng) ───────────────────────────────────────────────
PORT_COORDS = [
    ( 3.7876,  98.6957),  # 0  Belawan
    (-5.4759, 105.3198),  # 1  Panjang
    ( 1.1662, 104.0044),  # 2  Batu Ampar
    (-6.2088, 106.8456),  # 3  Tanjung Priok
    (-6.9932, 110.4203),  # 4  Tanjung Emas
    (-7.1539, 112.6561),  # 5  Gresik
    (-7.2575, 112.7521),  # 6  Tanjung Perak
    (-6.0123, 105.9570),  # 7  Cigading
    (-7.73844, 108.987578),  # 8  Cilacap / Tanjung Intan
    (-8.1166, 114.4000),  # 9  Banyuwangi / Tanjung Wangi
    (-0.01304626, 109.3354),  # 10 Pontianak
]

CLUSTER_NAMES = ["Sumatera", "Jawa", "Bali & Nusa Tenggara",
                 "Kalimantan", "Sulawesi", "Maluku & Papua"]

MONTHS = ["Januari", "Februari", "Maret", "April", "Mei", "Juni",
          "Juli", "Agustus", "September", "Oktober", "November", "Desember"]


# ════════════════════════════════════════════════════════════════════════════
def state_to_dict(state) -> dict:
    """Extract every per-month flow value from a solver state into JSON-friendly dicts."""
    p = state.problem

    provinces = []
    for i in range(p.N_PROV):
        monthly = []
        for t in range(p.N_PERIOD):
            # Where imports came from this month: {port_idx: vol}
            import_by_port = {
                str(h): float(state.x_dist[h, i, t])
                for h in range(p.N_PORT) if state.x_dist[h, i, t] > 0.1
            }
            # Where local supply came from: {producer_array_idx → vol}
            local_by_prod = {
                str(k): float(state.x_loc[k, i, t])
                for k in range(p.N_PROD) if state.x_loc[k, i, t] > 0.1
            }
            # Transfers IN this month: {from_prov: vol}
            transfer_in_from = {
                str(j): float(state.x_trns[j, i, t])
                for j in range(p.N_PROV) if state.x_trns[j, i, t] > 0.1
            }
            # Transfers OUT this month: {to_prov: vol}
            transfer_out_to = {
                str(j): float(state.x_trns[i, j, t])
                for j in range(p.N_PROV) if state.x_trns[i, j, t] > 0.1
            }
            monthly.append({
                "demand":       float(p.DEMAND[i, t]),
                "local":        float(state.x_loc[:, i, t].sum()),
                "import":       float(state.x_dist[:, i, t].sum()),
                "transfer_in":  float(state.x_trns[:, i, t].sum()),
                "transfer_out": float(state.x_trns[i, :, t].sum()),
                "inventory":    float(state.inv[i, t]),
                "shortage":     float(state.sh[i, t]),
                "safe":         int(state.safe[i, t]),
                "import_by_port":     import_by_port,
                "local_by_producer":  local_by_prod,
                "transfer_in_from":   transfer_in_from,
                "transfer_out_to":    transfer_out_to,
            })
        provinces.append({"monthly": monthly})

    ports = []
    for h in range(p.N_PORT):
        monthly = []
        for t in range(p.N_PERIOD):
            imports = {str(s): float(state.x_imp[s, h, t]) for s in range(p.N_IMP)}
            distribution = {
                str(i): float(state.x_dist[h, i, t])
                for i in range(p.N_PROV) if state.x_dist[h, i, t] > 0.1
            }
            monthly.append({
                "imports":      imports,
                "distribution": distribution,
                "total_in":     float(state.x_imp[:, h, t].sum()),
                "total_out":    float(state.x_dist[h, :, t].sum()),
                "thru_cap":     float(p.PORT_THRU_CAP[h, t]),
            })
        ports.append({"monthly": monthly})

    return {"provinces": provinces, "ports": ports}


def cost_breakdown(state) -> dict:
    """Decompose Z_cost into its 9 component terms (Rp)."""
    import numpy as np
    p = state.problem
    loc_cost  = float(np.sum((p.C_PROD[:, None, None] + p.C_SHIP[:, :, None]) * state.x_loc))
    imp_cost  = float(np.sum(p.C_PURCH[:, None, None] * state.x_imp))
    emg_cost  = float(np.sum(p.C_EMG[:, None, None]   * state.x_emg))
    dist_cost = float(np.sum(p.C_DIST[:, :, None]      * state.x_dist))
    trns_cost = float(np.sum(p.C_TRANS[:, :, None]     * state.x_trns))
    hold_prov = float(np.sum(p.H_PROV[:, None] * state.inv))
    fix_act   = float(np.sum(p.F_ACT[:, None]  * state.y))
    fix_emg   = float(p.F_EMG) * float(state.z.sum())

    total_imp = float((state.x_imp + state.x_emg).sum())
    total_local = float(state.x_loc.sum())
    imp_dep   = total_imp / max(total_imp + total_local, 1.0)

    # Monthly aggregates for time-series charts
    monthly_demand   = [float(p.DEMAND[:, t].sum()) for t in range(p.N_PERIOD)]
    monthly_local    = [float(state.x_loc[:, :, t].sum()) for t in range(p.N_PERIOD)]
    monthly_import   = [float(state.x_imp[:, :, t].sum() + state.x_emg[:, :, t].sum())
                        for t in range(p.N_PERIOD)]
    monthly_shortage = [float(state.sh[:, t].sum()) for t in range(p.N_PERIOD)]
    monthly_inv_prov = [float(state.inv[:, t].sum()) for t in range(p.N_PERIOD)]

    return {
        "terms": {
            "loc_cost":  loc_cost,
            "imp_cost":  imp_cost,
            "emg_cost":  emg_cost,
            "dist_cost": dist_cost,
            "trns_cost": trns_cost,
            "hold_prov": hold_prov,
            "fix_act":   fix_act,
            "fix_emg":   fix_emg,
        },
        "z_cost":    loc_cost + imp_cost + emg_cost + dist_cost + trns_cost
                     + hold_prov + fix_act + fix_emg,
        "objective": float(state.objective()),
        "total_shortage": float(state.sh.sum()),
        "total_local":    total_local,
        "total_import":   total_imp,
        "import_dep":     imp_dep,
        "monthly": {
            "demand":   monthly_demand,
            "local":    monthly_local,
            "import":   monthly_import,
            "shortage": monthly_shortage,
            "inv_prov": monthly_inv_prov,
        },
    }


def build_adjacency_edges() -> list:
    """Return list of {i, j, type} for W (land) and W_sea (ferry) edges."""
    from problem import _build_transport_costs, CLUSTER
    # Re-create adjacency defs (kept in sync with problem.py)
    ADJ_LAND = {
        0: {1}, 1: {0, 2, 3}, 2: {1, 3, 4, 6}, 3: {1, 2, 4}, 4: {2, 3, 5, 6},
        5: {4, 6, 7}, 6: {2, 4, 5, 7}, 7: {5, 6},
        8: set(), 9: set(),
        10: {11, 15}, 11: {10, 12, 15}, 12: {11, 13, 14}, 13: {12}, 14: {12},
        15: {10, 11}, 16: set(), 17: set(), 18: set(),
        19: {20}, 20: {19, 21, 22}, 21: {20, 22}, 22: {20, 21, 23}, 23: {22},
        24: {28}, 25: {26, 27, 28, 29}, 26: {25, 27, 29}, 27: {25, 26},
        28: {24, 25}, 29: {25, 26},
        30: set(), 31: set(),
        32: {33, 36}, 33: {32}, 34: {35, 36, 37}, 35: {34, 37},
        36: {32, 34, 37}, 37: {34, 35, 36},
    }
    SEA_PAIRS = [
        (3, 9), (5, 8), (7, 15), (14, 16),
        (16, 17), (17, 18), (24, 31), (30, 31),
    ]
    edges = []
    seen = set()
    for i, neighbors in ADJ_LAND.items():
        for j in neighbors:
            key = (min(i, j), max(i, j))
            if key not in seen:
                seen.add(key)
                edges.append({"i": key[0], "j": key[1], "type": "land"})
    for i, j in SEA_PAIRS:
        edges.append({"i": i, "j": j, "type": "sea"})
    return edges


def build_dashboard_data(initial, optimized,
                         obj_history: list = None) -> dict:
    p = initial.problem

    # Static info
    province_info = [
        {
            "idx":     i,
            "name":    p.PROV_NAMES[i],
            "coord":   PROV_COORDS[i],
            "cluster": p.CLUSTER[i],
            "is_producer": i in p.PROD_IDX,
            "annual_demand": float(p.DEMAND[i].sum()),
        }
        for i in range(p.N_PROV)
    ]
    port_info = [
        {
            "idx":      h,
            "name":     p.PORT_NAMES[h],
            "coord":    PORT_COORDS[h],
            "services": list(p.PORT_SERV[h]),
            "annual_capacity": float(p.PORT_THRU_CAP[h].sum()),
        }
        for h in range(p.N_PORT)
    ]
    cluster_info = [
        {
            "idx":  c,
            "name": CLUSTER_NAMES[c],
            "provinces": [i for i in range(p.N_PROV) if p.CLUSTER[i] == c],
        }
        for c in range(6)
    ]

    return {
        "meta": {
            "months":         MONTHS,
            "n_periods":      p.N_PERIOD,
            "imp_names":      list(p.IMP_NAMES),
            "prod_idx_to_prov": list(p.PROD_IDX),
            "eps_shortage":   config.EPS_SHORTAGE,
            "eps_import_dep": config.EPS_IMPORT_DEP,
            "eps_local_min":  config.EPS_LOCAL_MIN,
        },
        "provinces": province_info,
        "ports":     port_info,
        "clusters":  cluster_info,
        "adjacency": build_adjacency_edges(),
        "cost_initial":   cost_breakdown(initial),
        "cost_optimized": cost_breakdown(optimized),
        "obj_history":    obj_history or [],
        "cost_history":    [],
        "penalty_history": [],
        "transfers":       {"initial": [], "optimized": []},
        "initial":   state_to_dict(initial),
        "optimized": state_to_dict(optimized),
    }


def _int_key_dict(value: dict) -> dict:
    return {int(k): v for k, v in value.items()}


def _state_to_dict_from_artifact(state_payload: dict, problem_payload: dict) -> dict:
    """Extract dashboard-friendly flow values from a JSON state payload."""
    n_prov = int(problem_payload["N_PROV"])
    n_imp = int(problem_payload["N_IMP"])
    n_port = int(problem_payload["N_PORT"])
    n_period = int(problem_payload["N_PERIOD"])
    n_prod = int(problem_payload["N_PROD"])
    prod_idx = [int(i) for i in problem_payload["PROD_IDX"]]
    prod_pos_by_prov = {prov: k for k, prov in enumerate(prod_idx)}

    demand = np.asarray(problem_payload["DEMAND"], dtype=float)
    thru_cap = np.asarray(problem_payload["PORT_THRU_CAP"], dtype=float)
    x_loc = np.asarray(state_payload["x_loc"], dtype=float)
    x_imp = np.asarray(state_payload["x_imp"], dtype=float)
    x_dist = np.asarray(state_payload["x_dist"], dtype=float)
    x_trns = np.asarray(state_payload["x_trns"], dtype=float)
    inv = np.asarray(state_payload["inv"], dtype=float)
    sh = np.asarray(state_payload["sh"], dtype=float)
    safe = np.asarray(state_payload["safe"], dtype=float)

    provinces = []
    for i in range(n_prov):
        monthly = []
        for t in range(n_period):
            monthly.append({
                "demand":       float(demand[i, t]),
                "local":        float(x_loc[:, i, t].sum()),
                "import":       float(x_dist[:, i, t].sum()),
                "transfer_in":  float(x_trns[:, i, t].sum()),
                "transfer_out": float(x_trns[i, :, t].sum()),
                "inventory":    float(inv[i, t]),
                "shortage":     float(sh[i, t]),
                "safe":         int(safe[i, t]),
                "import_by_port": {
                    str(h): float(x_dist[h, i, t])
                    for h in range(n_port) if x_dist[h, i, t] > 0.1
                },
                "local_by_producer": {
                    str(k): float(x_loc[k, i, t])
                    for k in range(n_prod) if x_loc[k, i, t] > 0.1
                },
                "local_from_province": {
                    str(prod_idx[k]): float(x_loc[k, i, t])
                    for k in range(n_prod) if x_loc[k, i, t] > 0.1
                },
                "local_out_to": (
                    {
                        str(j): float(x_loc[prod_pos_by_prov[i], j, t])
                        for j in range(n_prov)
                        if x_loc[prod_pos_by_prov[i], j, t] > 0.1
                    }
                    if i in prod_pos_by_prov else {}
                ),
                "transfer_in_from": {
                    str(j): float(x_trns[j, i, t])
                    for j in range(n_prov) if x_trns[j, i, t] > 0.1
                },
                "transfer_out_to": {
                    str(j): float(x_trns[i, j, t])
                    for j in range(n_prov) if x_trns[i, j, t] > 0.1
                },
            })
        provinces.append({"monthly": monthly})

    ports = []
    for h in range(n_port):
        monthly = []
        for t in range(n_period):
            monthly.append({
                "imports": {
                    str(s): float(x_imp[s, h, t]) for s in range(n_imp)
                },
                "distribution": {
                    str(i): float(x_dist[h, i, t])
                    for i in range(n_prov) if x_dist[h, i, t] > 0.1
                },
                "total_in":  float(x_imp[:, h, t].sum()),
                "total_out": float(x_dist[h, :, t].sum()),
                "thru_cap":  float(thru_cap[h, t]),
            })
        ports.append({"monthly": monthly})

    return {"provinces": provinces, "ports": ports}


def _cost_breakdown_from_artifact(state_payload: dict, problem_payload: dict) -> dict:
    breakdown = state_payload.get("cost_breakdown", {})
    terms = {
        key: float(breakdown.get(key, 0.0))
        for key in ["loc_cost", "imp_cost", "emg_cost", "dist_cost",
                    "trns_cost", "hold_prov", "fix_act", "fix_emg"]
    }

    demand = np.asarray(problem_payload["DEMAND"], dtype=float)
    x_loc = np.asarray(state_payload["x_loc"], dtype=float)
    x_imp = np.asarray(state_payload["x_imp"], dtype=float)
    x_emg = np.asarray(state_payload["x_emg"], dtype=float)
    inv = np.asarray(state_payload["inv"], dtype=float)
    sh = np.asarray(state_payload["sh"], dtype=float)

    total_imp = float(x_imp.sum() + x_emg.sum())
    total_local = float(x_loc.sum())
    n_period = int(problem_payload["N_PERIOD"])

    return {
        "terms": terms,
        "z_cost": float(breakdown.get("z_cost", sum(terms.values()))),
        "objective": float(state_payload.get("objective", breakdown.get("objective", 0.0))),
        "total_shortage": float(sh.sum()),
        "total_local": total_local,
        "total_import": total_imp,
        "import_dep": total_imp / max(total_imp + total_local, 1.0),
        "monthly": {
            "demand": [float(demand[:, t].sum()) for t in range(n_period)],
            "local": [float(x_loc[:, :, t].sum()) for t in range(n_period)],
            "import": [float(x_imp[:, :, t].sum() + x_emg[:, :, t].sum())
                       for t in range(n_period)],
            "shortage": [float(sh[:, t].sum()) for t in range(n_period)],
            "inv_prov": [float(inv[:, t].sum()) for t in range(n_period)],
        },
    }


def _transfer_rows_from_artifact(state_payload: dict, problem_payload: dict,
                                 scenario: str) -> list[dict]:
    x_trns = np.asarray(state_payload["x_trns"], dtype=float)
    prov_names = problem_payload["PROV_NAMES"]
    rows = []
    for i, j, t in np.argwhere(x_trns > 0.1):
        volume = float(x_trns[i, j, t])
        rows.append({
            "scenario": scenario,
            "month_idx": int(t),
            "from_idx": int(i),
            "to_idx": int(j),
            "from": prov_names[int(i)],
            "to": prov_names[int(j)],
            "volume": volume,
        })
    rows.sort(key=lambda r: (-r["volume"], r["month_idx"], r["from"], r["to"]))
    return rows


def _province_plan_from_artifact(state_payload: dict, problem_payload: dict) -> dict:
    """Build all-month province plans for the decision cockpit."""
    n_prov = int(problem_payload["N_PROV"])
    n_period = int(problem_payload["N_PERIOD"])
    prod_idx = [int(i) for i in problem_payload["PROD_IDX"]]
    prod_pos_by_prov = {prov: k for k, prov in enumerate(prod_idx)}
    demand = np.asarray(problem_payload["DEMAND"], dtype=float)
    x_loc = np.asarray(state_payload["x_loc"], dtype=float)
    x_dist = np.asarray(state_payload["x_dist"], dtype=float)
    x_trns = np.asarray(state_payload["x_trns"], dtype=float)
    inv = np.asarray(state_payload["inv"], dtype=float)
    sh = np.asarray(state_payload["sh"], dtype=float)

    plans = {}
    for i in range(n_prov):
        monthly = []
        for t in range(n_period):
            demand_t = float(demand[i, t])
            shortage_t = float(sh[i, t])
            monthly.append({
                "month_idx": int(t),
                "month": MONTHS[t],
                "demand": demand_t,
                "local": float(x_loc[:, i, t].sum()),
                "import": float(x_dist[:, i, t].sum()),
                "transfer_in": float(x_trns[:, i, t].sum()),
                "transfer_out": float(x_trns[i, :, t].sum()),
                "inventory": float(inv[i, t]),
                "shortage": shortage_t,
                "service_rate": 1.0 - shortage_t / max(demand_t, 1.0),
                "local_from_province": {
                    str(prod_idx[k]): float(x_loc[k, i, t])
                    for k in range(len(prod_idx)) if x_loc[k, i, t] > 0.1
                },
                "local_out_to": (
                    {
                        str(j): float(x_loc[prod_pos_by_prov[i], j, t])
                        for j in range(n_prov)
                        if x_loc[prod_pos_by_prov[i], j, t] > 0.1
                    }
                    if i in prod_pos_by_prov else {}
                ),
            })
        totals = {
            "demand": float(demand[i, :].sum()),
            "local": float(x_loc[:, i, :].sum()),
            "import": float(x_dist[:, i, :].sum()),
            "transfer_in": float(x_trns[:, i, :].sum()),
            "transfer_out": float(x_trns[i, :, :].sum()),
            "inventory": float(inv[i, :].sum()),
            "shortage": float(sh[i, :].sum()),
        }
        totals["service_rate"] = 1.0 - totals["shortage"] / max(totals["demand"], 1.0)
        plans[str(i)] = {"monthly": monthly, "totals": totals}
    return plans


def _port_plan_from_artifact(state_payload: dict, problem_payload: dict) -> dict:
    """Build all-month port plans for the decision cockpit."""
    n_port = int(problem_payload["N_PORT"])
    n_imp = int(problem_payload["N_IMP"])
    n_prov = int(problem_payload["N_PROV"])
    n_period = int(problem_payload["N_PERIOD"])
    port_names = problem_payload["PORT_NAMES"]
    imp_names = problem_payload["IMP_NAMES"]
    prov_names = problem_payload["PROV_NAMES"]
    thru_cap = np.asarray(problem_payload["PORT_THRU_CAP"], dtype=float)
    x_imp = np.asarray(state_payload["x_imp"], dtype=float)
    x_dist = np.asarray(state_payload["x_dist"], dtype=float)

    plans = {}
    for h in range(n_port):
        monthly = []
        for t in range(n_period):
            imports = [
                {
                    "idx": s,
                    "name": imp_names[s],
                    "volume": float(x_imp[s, h, t]),
                }
                for s in range(n_imp)
                if x_imp[s, h, t] > 0.1
            ]
            destinations = [
                {
                    "idx": i,
                    "name": prov_names[i],
                    "volume": float(x_dist[h, i, t]),
                }
                for i in range(n_prov)
                if x_dist[h, i, t] > 0.1
            ]
            imports.sort(key=lambda r: -r["volume"])
            destinations.sort(key=lambda r: -r["volume"])
            total_in = float(x_imp[:, h, t].sum())
            total_out = float(x_dist[h, :, t].sum())
            capacity = float(thru_cap[h, t])
            monthly.append({
                "month_idx": int(t),
                "month": MONTHS[t],
                "total_in": total_in,
                "total_out": total_out,
                "capacity": capacity,
                "utilization": total_in / max(capacity, 1.0),
                "imports": imports,
                "destinations": destinations,
                "top_country": imports[0]["name"] if imports else "",
                "top_destination": destinations[0]["name"] if destinations else "",
            })
        totals = {
            "total_in": float(x_imp[:, h, :].sum()),
            "total_out": float(x_dist[h, :, :].sum()),
            "capacity": float(thru_cap[h, :].sum()),
        }
        totals["utilization"] = totals["total_in"] / max(totals["capacity"], 1.0)
        plans[str(h)] = {
            "name": port_names[h],
            "monthly": monthly,
            "totals": totals,
        }
    return plans


def _monthly_actions_from_artifact(state_payload: dict, problem_payload: dict) -> list[dict]:
    """Precompute decision-oriented top actions for each month."""
    n_prov = int(problem_payload["N_PROV"])
    n_port = int(problem_payload["N_PORT"])
    n_imp = int(problem_payload["N_IMP"])
    n_period = int(problem_payload["N_PERIOD"])
    prov_names = problem_payload["PROV_NAMES"]
    port_names = problem_payload["PORT_NAMES"]
    imp_names = problem_payload["IMP_NAMES"]
    demand = np.asarray(problem_payload["DEMAND"], dtype=float)
    thru_cap = np.asarray(problem_payload["PORT_THRU_CAP"], dtype=float)
    x_loc = np.asarray(state_payload["x_loc"], dtype=float)
    x_imp = np.asarray(state_payload["x_imp"], dtype=float)
    x_dist = np.asarray(state_payload["x_dist"], dtype=float)
    x_trns = np.asarray(state_payload["x_trns"], dtype=float)
    inv = np.asarray(state_payload["inv"], dtype=float)
    sh = np.asarray(state_payload["sh"], dtype=float)

    months = []
    for t in range(n_period):
        country = [
            {"idx": s, "name": imp_names[s], "volume": float(x_imp[s, :, t].sum())}
            for s in range(n_imp)
        ]
        ports = [
            {
                "idx": h,
                "name": port_names[h],
                "total_in": float(x_imp[:, h, t].sum()),
                "total_out": float(x_dist[h, :, t].sum()),
                "capacity": float(thru_cap[h, t]),
                "utilization": float(x_imp[:, h, t].sum()) / max(float(thru_cap[h, t]), 1.0),
            }
            for h in range(n_port)
        ]
        receivers = [
            {
                "idx": i,
                "name": prov_names[i],
                "demand": float(demand[i, t]),
                "local": float(x_loc[:, i, t].sum()),
                "import": float(x_dist[:, i, t].sum()),
                "transfer_in": float(x_trns[:, i, t].sum()),
                "shortage": float(sh[i, t]),
                "inventory": float(inv[i, t]),
            }
            for i in range(n_prov)
        ]
        transfers = []
        for i, j in np.argwhere(x_trns[:, :, t] > 0.1):
            transfers.append({
                "from_idx": int(i),
                "to_idx": int(j),
                "from": prov_names[int(i)],
                "to": prov_names[int(j)],
                "volume": float(x_trns[int(i), int(j), t]),
            })

        months.append({
            "month_idx": int(t),
            "month": MONTHS[t],
            "country": sorted(country, key=lambda r: -r["volume"]),
            "ports": sorted(ports, key=lambda r: -r["total_in"]),
            "receivers": sorted(receivers, key=lambda r: -r["import"]),
            "transfers": sorted(transfers, key=lambda r: -r["volume"]),
            "risks": {
                "shortage": sorted(
                    [r for r in receivers if r["shortage"] > 0.5],
                    key=lambda r: -r["shortage"],
                ),
                "port_utilization": sorted(
                    [r for r in ports if r["utilization"] >= 0.85],
                    key=lambda r: -r["utilization"],
                ),
                "low_inventory": sorted(
                    [r for r in receivers if r["inventory"] < 0.10 * max(r["demand"], 1.0)],
                    key=lambda r: r["inventory"] / max(r["demand"], 1.0),
                ),
            },
        })
    return months


def build_dashboard_data_from_artifact(artifact: dict) -> dict:
    p = artifact["problem"]
    cfg = artifact["config"]
    initial = artifact["states"]["initial"]
    optimized = artifact["states"]["best"]

    n_prov = int(p["N_PROV"])
    n_port = int(p["N_PORT"])
    prod_idx = [int(i) for i in p["PROD_IDX"]]
    clusters = _int_key_dict(p["CLUSTER"])
    port_serv = _int_key_dict(p["PORT_SERV"])
    demand = np.asarray(p["DEMAND"], dtype=float)
    thru_cap = np.asarray(p["PORT_THRU_CAP"], dtype=float)
    province_order = [int(i) for i in np.argsort(-demand.sum(axis=1))]
    hist_import = np.asarray(optimized["x_imp"], dtype=float)
    port_order = [int(h) for h in np.argsort(-hist_import.sum(axis=(0, 2)))]

    province_info = [
        {
            "idx": i,
            "name": p["PROV_NAMES"][i],
            "coord": PROV_COORDS[i],
            "cluster": int(clusters[i]),
            "is_producer": i in prod_idx,
            "annual_demand": float(demand[i].sum()),
        }
        for i in range(n_prov)
    ]
    port_info = [
        {
            "idx": h,
            "name": p["PORT_NAMES"][h],
            "coord": PORT_COORDS[h],
            "services": [int(i) for i in port_serv[h]],
            "annual_capacity": float(thru_cap[h].sum()),
        }
        for h in range(n_port)
    ]
    cluster_info = [
        {
            "idx": c,
            "name": CLUSTER_NAMES[c],
            "provinces": [i for i in range(n_prov) if int(clusters[i]) == c],
        }
        for c in range(6)
    ]

    return {
        "meta": {
            "months": MONTHS,
            "n_periods": int(p["N_PERIOD"]),
            "imp_names": list(p["IMP_NAMES"]),
            "prod_idx_to_prov": prod_idx,
            "eps_shortage": cfg["EPS_SHORTAGE"],
            "eps_import_dep": cfg["EPS_IMPORT_DEP"],
            "eps_local_min": cfg["EPS_LOCAL_MIN"],
            "schema_version": artifact["schema_version"],
            "created_at": artifact["created_at"],
        },
        "provinces": province_info,
        "ports": port_info,
        "clusters": cluster_info,
        "adjacency": build_adjacency_edges(),
        "cost_initial": _cost_breakdown_from_artifact(initial, p),
        "cost_optimized": _cost_breakdown_from_artifact(optimized, p),
        "obj_history": artifact["metrics"].get("history_best", []),
        "cost_history": artifact["metrics"].get("history_z_cost", []),
        "penalty_history": artifact["metrics"].get("history_penalty", []),
        "transfers": {
            "initial": _transfer_rows_from_artifact(initial, p, "initial"),
            "optimized": _transfer_rows_from_artifact(optimized, p, "optimized"),
        },
        "decision": {
            "default_province_idx": province_order[0],
            "default_port_idx": port_order[0],
            "province_order": province_order,
            "port_order": port_order,
            "province_plan": {
                "initial": _province_plan_from_artifact(initial, p),
                "optimized": _province_plan_from_artifact(optimized, p),
            },
            "port_plan": {
                "initial": _port_plan_from_artifact(initial, p),
                "optimized": _port_plan_from_artifact(optimized, p),
            },
            "monthly_actions": {
                "initial": _monthly_actions_from_artifact(initial, p),
                "optimized": _monthly_actions_from_artifact(optimized, p),
            },
        },
        "initial": _state_to_dict_from_artifact(initial, p),
        "optimized": _state_to_dict_from_artifact(optimized, p),
    }


def main() -> None:
    here = Path(__file__).parent
    dashboard_dir = here / "docs"
    dashboard_dir.mkdir(exist_ok=True)
    artifact_path = here / config.OUTPUT_DIR / "optimization_result.json"

    if not artifact_path.exists():
        raise SystemExit(
            f"Optimization JSON not found: {artifact_path}\n"
            "Run `python run_optimization.py` first, then run `python build_dashboard.py`."
        )

    print(f"Reading optimization artifact:\n  {artifact_path}")
    artifact = load_optimization_result(artifact_path)
    print("\nBuilding dashboard data file...")
    data = build_dashboard_data_from_artifact(artifact)

    out_path = dashboard_dir / "data.js"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("// Auto-generated by build_dashboard.py - do not edit\n")
        f.write("const DATA = ")
        json.dump(data, f, ensure_ascii=False, indent=1)
        f.write(";\n")
    print(f"  [JS]  {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")

    index_path = dashboard_dir / "index.html"
    print(f"\nDashboard ready:\n  {index_path}")


if __name__ == "__main__":
    main()
