"""
reporting.py
────────────
All output sinks for the solver:

  print_summary(state, label)    – console summary with cost breakdown + ε status
  export_csv_results(initial, optimized, out_dir)
                                  – writes 11 CSVs of per-province + flow detail
  plot_results(initial, optimized, obj_history, destroy_counts, repair_counts, out_dir)
                                  – writes 5 PNGs:
                                       1. Overview (convergence, monthly shortage,
                                          supply sources, operator usage)
                                       2. Province supply-demand balance (all 38)
                                       3. Province service rate (all 38)
                                       4. Import flow (port + province)
                                       5. What changed (Δ local / Δ import / Δ shortage)

All output paths default to `config.OUTPUT_DIR`.
"""
from __future__ import annotations

import csv
import os

import numpy as np
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
except ImportError:  # allow silent solver smoke tests in minimal runtimes
    plt = None
    Patch = None

import config
from state import SoybeanState

MONTHS_LBL = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
              "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]


def _short_prov_names(names):
    """Compact province labels for tight axes."""
    return [n.replace("Kep. ", "").replace("Kepulauan ", "Kep.")
             .replace("Nusa Tenggara", "NT").replace("Kalimantan", "Kal.")
             .replace("Sulawesi", "Sul.").replace("Sumatera", "Sum.")
            for n in names]


# ═══════════════════════════════════════════════════════════════════════════
#  CONSOLE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

def print_summary(state: SoybeanState, label: str = "") -> None:
    p = state.problem
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Objective (f):      {state.objective():>20,.0f}")

    # Cost breakdown
    loc_cost  = float(np.sum((p.C_PROD[:, None, None] + p.C_SHIP[:, :, None]) * state.x_loc))
    imp_cost  = float(np.sum(p.C_PURCH[:, None, None] * state.x_imp))
    emg_cost  = float(np.sum(p.C_EMG[:, None, None]   * state.x_emg))
    dist_cost = float(np.sum(p.C_DIST[:, :, None]      * state.x_dist))
    trns_cost = float(np.sum(p.C_TRANS[:, :, None]     * state.x_trns))
    hold_cost = float(np.sum(p.H_PROV[:, None] * state.inv))
    fix_cost  = float(np.sum(p.F_ACT[:, None] * state.y)) \
                + float(p.F_EMG) * float(state.z.sum())
    z_cost    = loc_cost + imp_cost + emg_cost + dist_cost + trns_cost + hold_cost + fix_cost

    print(f"  -- Cost (Z_cost) --")
    print(f"     Produksi lokal:  {loc_cost:>20,.0f} Rp")
    print(f"     Impor normal:    {imp_cost:>20,.0f} Rp")
    print(f"     Distribusi:      {dist_cost:>20,.0f} Rp")
    print(f"     Transfer:        {trns_cost:>20,.0f} Rp")
    print(f"     Holding:         {hold_cost:>20,.0f} Rp")
    print(f"     Fixed + Darurat: {fix_cost:>20,.0f} Rp")
    print(f"     TOTAL Z_cost:    {z_cost:>20,.0f} Rp")

    # ε-constraint status
    total_shortage = float(state.sh.sum())
    total_demand   = float(p.DEMAND.sum())
    total_import   = float((state.x_imp + state.x_emg).sum())
    total_local    = float(state.x_loc.sum())
    imp_dep        = total_import / max(total_import + total_local, 1.0)
    service_rate   = 1.0 - total_shortage / max(total_demand, 1.0)

    sh_ok = total_shortage <= config.EPS_SHORTAGE + 0.5
    id_ok = imp_dep        <= config.EPS_IMPORT_DEP + 1e-6
    lc_ok = total_local    >= config.EPS_LOCAL_MIN - 0.5

    sh_mark = "OK" if sh_ok else "!!"
    id_mark = "OK" if id_ok else "!!"
    lc_mark = "OK" if lc_ok else "!!"

    print(f"  -- e-constraint status --")
    print(f"  [{sh_mark}] Shortage <= {config.EPS_SHORTAGE:,.0f} ton:  {total_shortage:>12,.1f} ton"
          + (f"  (kelebihan {total_shortage - config.EPS_SHORTAGE:,.1f} ton)"
             if not sh_ok else "  OK"))
    print(f"  [{id_mark}] Import dep <= {config.EPS_IMPORT_DEP:.0%}:  {imp_dep:>11.2%}"
          + (f"  (kelebihan {(imp_dep - config.EPS_IMPORT_DEP)*100:.2f}%)"
             if not id_ok else "  OK"))
    print(f"  [{lc_mark}] Local prod >= {config.EPS_LOCAL_MIN:,.0f} ton:  {total_local:>12,.1f} ton"
          + (f"  (kurang {config.EPS_LOCAL_MIN - total_local:,.1f} ton)"
             if not lc_ok else "  OK"))
    print(f"  -- Supply & Demand --")
    print(f"  Total demand:       {total_demand:>20,.0f} ton")
    print(f"  Total local:        {total_local:>20,.1f} ton")
    print(f"  Total import:       {total_import:>20,.1f} ton")
    print(f"  Service level:      {service_rate:>19.2%}")
    print(f"  Infeasibility pen:  {state._penalty:>20,.1f}")
    print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════════════════════
#  CSV EXPORT
# ═══════════════════════════════════════════════════════════════════════════

def export_csv_results(initial: SoybeanState, optimized: SoybeanState,
                       out_dir: str = None) -> None:
    """
    Write 11 CSV files describing the initial and optimised solutions:

      province_summary_initial.csv / _optimized.csv      — annual per-province totals
      province_monthly_initial.csv / _optimized.csv      — monthly per-province detail
      import_by_port_initial.csv / _optimized.csv        — country → port flows
      import_to_province_initial.csv / _optimized.csv    — port → province distribution
      transfer_flows_initial.csv / _optimized.csv        — interprovincial transfers
      flow_changes.csv                                   — Δ initial→optimised per province
    """
    if out_dir is None:
        out_dir = config.OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    p = optimized.problem        # same Problem reference for both states

    def _write(fname, rows, header):
        path = os.path.join(out_dir, fname)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        print(f"  [CSV] {fname}")

    for tag, st in [("initial", initial), ("optimized", optimized)]:
        # ── 1. Province annual summary ────────────────────────────────
        rows = []
        for i in range(p.N_PROV):
            ann_dem   = float(p.DEMAND[i].sum())
            ann_loc   = float(st.x_loc[:, i, :].sum())
            ann_imp   = float(st.x_dist[:, i, :].sum())
            ann_trn_i = float(st.x_trns[:, i, :].sum())
            ann_trn_o = float(st.x_trns[i, :, :].sum())
            ann_sh    = float(st.sh[i].sum())
            svc       = 1.0 - ann_sh / max(ann_dem, 1.0)
            avg_inv   = float(st.inv[i].mean())
            safe_mo   = int(st.safe[i].sum())
            rows.append([p.PROV_NAMES[i], f"{ann_dem:.1f}", f"{ann_loc:.1f}",
                         f"{ann_imp:.1f}", f"{ann_trn_i:.1f}", f"{ann_trn_o:.1f}",
                         f"{ann_loc + ann_imp + ann_trn_i - ann_trn_o:.1f}",
                         f"{ann_sh:.1f}", f"{svc:.4f}", f"{avg_inv:.1f}", str(safe_mo)])
        _write(f"province_summary_{tag}.csv", rows,
               ["province", "demand_ton", "local_supply_ton", "import_received_ton",
                "transfer_in_ton", "transfer_out_ton", "total_supply_ton",
                "shortage_ton", "service_rate", "avg_inventory_ton", "months_safe_stock"])

        # ── 2. Province monthly detail ────────────────────────────────
        rows = []
        for i in range(p.N_PROV):
            for t in range(p.N_PERIOD):
                loc   = float(st.x_loc[:, i, t].sum())
                imp   = float(st.x_dist[:, i, t].sum())
                trn_i = float(st.x_trns[:, i, t].sum())
                trn_o = float(st.x_trns[i, :, t].sum())
                rows.append([p.PROV_NAMES[i], MONTHS_LBL[t],
                             f"{p.DEMAND[i, t]:.1f}", f"{loc:.1f}", f"{imp:.1f}",
                             f"{trn_i:.1f}", f"{trn_o:.1f}",
                             f"{st.inv[i, t]:.1f}", f"{st.sh[i, t]:.1f}",
                             str(int(st.safe[i, t]))])
        _write(f"province_monthly_{tag}.csv", rows,
               ["province", "month", "demand_ton", "local_supply_ton",
                "import_received_ton", "transfer_in_ton", "transfer_out_ton",
                "inventory_ton", "shortage_ton", "safe_stock"])

        # ── 3. Import by port ─────────────────────────────────────────
        rows = []
        for s in range(p.N_IMP):
            for h in range(p.N_PORT):
                for t in range(p.N_PERIOD):
                    vol = float(st.x_imp[s, h, t])
                    if vol > 0:
                        rows.append([p.IMP_NAMES[s], p.PORT_NAMES[h],
                                     MONTHS_LBL[t], f"{vol:.1f}"])
        _write(f"import_by_port_{tag}.csv", rows,
               ["country", "port", "month", "volume_ton"])

        # ── 4. Import distribution port → province ────────────────────
        rows = []
        for h in range(p.N_PORT):
            for i in range(p.N_PROV):
                ann = float(st.x_dist[h, i, :].sum())
                if ann > 0:
                    rows.append([p.PORT_NAMES[h], p.PROV_NAMES[i], f"{ann:.1f}"])
        _write(f"import_to_province_{tag}.csv", rows,
               ["port", "province", "annual_volume_ton"])

        # ── 5. Interprovincial transfers ──────────────────────────────
        rows = []
        for i in range(p.N_PROV):
            for j in range(p.N_PROV):
                ann = float(st.x_trns[i, j, :].sum())
                if ann > 0.1:
                    rows.append([p.PROV_NAMES[i], p.PROV_NAMES[j], f"{ann:.1f}"])
        _write(f"transfer_flows_{tag}.csv", rows,
               ["from_province", "to_province", "annual_volume_ton"])

    # ── 6. Flow changes (initial → optimized delta) ──────────────────
    rows = []
    metrics = [
        ("demand_ton",          lambda st, i: p.DEMAND[i].sum()),
        ("local_supply_ton",    lambda st, i: st.x_loc[:, i, :].sum()),
        ("import_received_ton", lambda st, i: st.x_dist[:, i, :].sum()),
        ("transfer_in_ton",     lambda st, i: st.x_trns[:, i, :].sum()),
        ("transfer_out_ton",    lambda st, i: st.x_trns[i, :, :].sum()),
        ("shortage_ton",        lambda st, i: st.sh[i].sum()),
        ("service_rate",        lambda st, i: 1.0 - st.sh[i].sum() / max(p.DEMAND[i].sum(), 1.0)),
    ]
    for i in range(p.N_PROV):
        for mname, mfunc in metrics:
            v_init = float(mfunc(initial, i))
            v_opt  = float(mfunc(optimized, i))
            delta  = v_opt - v_init
            pct    = (delta / abs(v_init) * 100) if abs(v_init) > 1e-9 else 0.0
            rows.append([p.PROV_NAMES[i], mname,
                         f"{v_init:.3f}", f"{v_opt:.3f}",
                         f"{delta:.3f}", f"{pct:.2f}"])
    _write("flow_changes.csv", rows,
           ["province", "metric", "initial", "optimized", "change", "change_pct"])
    print(f"  [CSV] All files saved to: {out_dir}")


# ═══════════════════════════════════════════════════════════════════════════
#  PLOTS — 5 figures
# ═══════════════════════════════════════════════════════════════════════════

def plot_results(initial: SoybeanState, optimized: SoybeanState,
                 obj_history: list, z_cost_history: list,
                 penalty_history: list,
                 destroy_counts: dict, repair_counts: dict,
                 out_dir: str = None) -> None:
    """Produce 5 PNG figures saved to out_dir (default config.OUTPUT_DIR)."""
    if plt is None:
        raise RuntimeError("matplotlib is required for plot_results()")
    if out_dir is None:
        out_dir = config.OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    p = optimized.problem
    PROV_SHORT = _short_prov_names(p.PROV_NAMES)

    _plot_overview(initial, optimized, obj_history, z_cost_history,
                   penalty_history, destroy_counts, repair_counts, out_dir)
    _plot_overview_individual(initial, optimized, obj_history, z_cost_history,
                              penalty_history, destroy_counts, repair_counts, out_dir)
    _plot_province_balance(initial, optimized, PROV_SHORT, out_dir)
    _plot_province_service(initial, optimized, PROV_SHORT, out_dir)
    _plot_import_flows(initial, optimized, PROV_SHORT, out_dir)
    _plot_flow_changes(initial, optimized, PROV_SHORT, out_dir)
    _plot_landed_cost_overview(initial, optimized, out_dir)
    _plot_landed_cost_cluster(initial, optimized, out_dir)
    _plot_landed_cost_heatmap(initial, optimized, out_dir)
    _plot_landed_cost_province(initial, optimized, out_dir)
    _plot_local_supply_matrix(initial, optimized, out_dir)
    _plot_local_supply_monthly(initial, optimized, out_dir)
    _plot_local_supply_by_producer(optimized, out_dir)


# ─── Fig 1: Overview ───────────────────────────────────────────────────────
def _plot_overview(initial, optimized, obj_history, z_cost_history,
                   penalty_history, destroy_counts, repair_counts, out_dir):
    p = optimized.problem
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.30)
    fig.suptitle("Optimasi Rantai Pasok Kedelai – ALNS Overview",
                 fontsize=13, fontweight="bold")

    # 1-1: convergence – Z_cost only (decomposed from penalties)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(z_cost_history, lw=1.2, color="#2ca02c", alpha=0.9)
    ax.set_title("Convergence – Z_cost (murni, tanpa penalti)", fontsize=10)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Z_cost (Rp)")
    ax.yaxis.get_major_formatter().set_scientific(True)
    ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
    ax.grid(True, alpha=0.3)

    # 1-2: convergence – Penalties only
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(penalty_history, lw=1.2, color="#d62728", alpha=0.9)
    ax.set_title("Convergence – Total Penalty", fontsize=10)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Penalty (Rp)")
    ax.yaxis.get_major_formatter().set_scientific(True)
    ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
    ax.grid(True, alpha=0.3)

    # 2-1: monthly national shortage
    ax = fig.add_subplot(gs[0, 1])
    months = np.arange(1, p.N_PERIOD + 1)
    sh_i = initial.sh.sum(axis=0) / 1e3;   sh_i[sh_i < 1e-6] = 0.0
    sh_b = optimized.sh.sum(axis=0) / 1e3; sh_b[sh_b < 1e-6] = 0.0
    ax.bar(months - 0.2, sh_i, width=0.4, label="Initial",   color="#ff7f0e", alpha=0.8)
    ax.bar(months + 0.2, sh_b, width=0.4, label="ALNS Best", color="#2ca02c", alpha=0.8)
    ax.set_title("National Shortage by Month", fontsize=10)
    ax.set_xlabel("Month"); ax.set_ylabel("Shortage (×1 000 ton)")
    ax.set_xticks(months); ax.set_xticklabels(MONTHS_LBL, fontsize=8)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")

    # 2-2: supply source composition
    ax = fig.add_subplot(gs[1, 1])
    imp_i = initial.x_imp.sum(axis=(1, 2)) / 1e3
    imp_b = optimized.x_imp.sum(axis=(1, 2)) / 1e3
    loc_i = initial.x_loc.sum() / 1e3
    loc_b = optimized.x_loc.sum() / 1e3
    x_pos = np.arange(p.N_IMP + 1)
    pal   = list(plt.cm.tab10(np.linspace(0, 0.8, p.N_IMP + 1)))
    bars_i = list(imp_i) + [loc_i]
    bars_b = list(imp_b) + [loc_b]
    ax.bar(x_pos - 0.2, bars_i, width=0.4, color=pal, alpha=0.6, label="Initial")
    ax.bar(x_pos + 0.2, bars_b, width=0.4, color=pal, alpha=0.95, label="ALNS Best",
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(p.IMP_NAMES + ["Prod. Lokal"], rotation=15, fontsize=9)
    ax.set_title("Supply Volume by Source (Initial vs Best)", fontsize=10)
    ax.set_ylabel("Volume (×1 000 ton)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    # 3: operator usage (spans bottom row)
    ax = fig.add_subplot(gs[2, :])
    d_names = list(destroy_counts.keys()); d_vals = list(destroy_counts.values())
    r_names = list(repair_counts.keys());  r_vals = list(repair_counts.values())
    labels_op = [n.replace("destroy_", "D:").replace("repair_", "R:").replace("_tabu", "★")
                 for n in d_names + r_names]
    vals_op   = d_vals + r_vals
    colors_op = ["#9467bd"] * len(d_names) + ["#8c564b"] * len(r_names)
    ax.barh(range(len(labels_op)), vals_op, color=colors_op, alpha=0.85)
    ax.set_yticks(range(len(labels_op))); ax.set_yticklabels(labels_op, fontsize=9)
    ax.set_title("Operator Usage Counts", fontsize=10); ax.set_xlabel("Times selected")
    ax.invert_yaxis(); ax.grid(True, alpha=0.3, axis="x")
    ax.legend(handles=[Patch(color="#9467bd", label="Destroy"),
                       Patch(color="#8c564b", label="Repair")], fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "alns_overview.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [PNG] alns_overview.png")


def _plot_overview_individual(initial, optimized, obj_history, z_cost_history,
                              penalty_history, destroy_counts, repair_counts, out_dir):
    p = optimized.problem
    iters = np.arange(len(z_cost_history))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(iters, z_cost_history, lw=1.2, color="#2ca02c", alpha=0.9)
    ax.set_title("Convergence – Z_cost (murni, tanpa penalti)", fontsize=11)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Z_cost (Rp)")
    ax.yaxis.get_major_formatter().set_scientific(True)
    ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "alns_convergence_zcost.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [PNG] alns_convergence_zcost.png")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(iters, penalty_history, lw=1.2, color="#d62728", alpha=0.9)
    ax.set_title("Convergence – Total Penalty", fontsize=11)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Penalty (Rp)")
    ax.yaxis.get_major_formatter().set_scientific(True)
    ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "alns_convergence_penalty.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [PNG] alns_convergence_penalty.png")

    total_history = [z + pen for z, pen in zip(z_cost_history, penalty_history)]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(iters, total_history, lw=1.2, color="#1f77b4", alpha=0.9)
    ax.set_title("Convergence – Total Objective (Z_cost + Penalty)", fontsize=11)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Objective (Rp)")
    ax.yaxis.get_major_formatter().set_scientific(True)
    ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "alns_convergence_total.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [PNG] alns_convergence_total.png")

    months = np.arange(1, p.N_PERIOD + 1)
    sh_i = initial.sh.sum(axis=0) / 1e3;   sh_i[sh_i < 1e-6] = 0.0
    sh_b = optimized.sh.sum(axis=0) / 1e3; sh_b[sh_b < 1e-6] = 0.0
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(months - 0.2, sh_i, width=0.4, label="Initial",   color="#ff7f0e", alpha=0.8)
    ax.bar(months + 0.2, sh_b, width=0.4, label="ALNS Best", color="#2ca02c", alpha=0.8)
    ax.set_title("National Shortage by Month", fontsize=11)
    ax.set_xlabel("Month"); ax.set_ylabel("Shortage (×1 000 ton)")
    ax.set_xticks(months); ax.set_xticklabels(MONTHS_LBL, fontsize=9)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "alns_shortage_monthly.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [PNG] alns_shortage_monthly.png")

    imp_i = initial.x_imp.sum(axis=(1, 2)) / 1e3
    imp_b = optimized.x_imp.sum(axis=(1, 2)) / 1e3
    loc_i = initial.x_loc.sum() / 1e3
    loc_b = optimized.x_loc.sum() / 1e3
    x_pos = np.arange(p.N_IMP + 1)
    pal   = list(plt.cm.tab10(np.linspace(0, 0.8, p.N_IMP + 1)))
    bars_i = list(imp_i) + [loc_i]
    bars_b = list(imp_b) + [loc_b]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x_pos - 0.2, bars_i, width=0.4, color=pal, alpha=0.6, label="Initial")
    ax.bar(x_pos + 0.2, bars_b, width=0.4, color=pal, alpha=0.95, label="ALNS Best",
           edgecolor="black", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(p.IMP_NAMES + ["Prod. Lokal"], rotation=15, fontsize=9)
    ax.set_title("Supply Volume by Source (Initial vs Best)", fontsize=11)
    ax.set_ylabel("Volume (×1 000 ton)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "alns_supply_source.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [PNG] alns_supply_source.png")

    d_names = list(destroy_counts.keys()); d_vals = list(destroy_counts.values())
    r_names = list(repair_counts.keys());  r_vals = list(repair_counts.values())
    labels_op = [n.replace("destroy_", "D:").replace("repair_", "R:").replace("_tabu", "★")
                 for n in d_names + r_names]
    vals_op   = d_vals + r_vals
    colors_op = ["#9467bd"] * len(d_names) + ["#8c564b"] * len(r_names)
    fig, ax = plt.subplots(figsize=(12, max(4, len(labels_op) * 0.45)))
    ax.barh(range(len(labels_op)), vals_op, color=colors_op, alpha=0.85)
    ax.set_yticks(range(len(labels_op))); ax.set_yticklabels(labels_op, fontsize=9)
    ax.set_title("Operator Usage Counts", fontsize=11); ax.set_xlabel("Times selected")
    ax.invert_yaxis(); ax.grid(True, alpha=0.3, axis="x")
    ax.legend(handles=[Patch(color="#9467bd", label="Destroy"),
                       Patch(color="#8c564b", label="Repair")], fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "alns_operator_usage.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [PNG] alns_operator_usage.png")


# ─── Fig 2: Province supply-demand balance ────────────────────────────────
def _plot_province_balance(initial, optimized, PROV_SHORT, out_dir):
    p = optimized.problem
    sort_idx = np.argsort(-p.DEMAND.sum(axis=1))
    pnames_s = [PROV_SHORT[i] for i in sort_idx]

    def _supply_components(st):
        loc = np.array([st.x_loc[:, i, :].sum()  for i in range(p.N_PROV)])
        imp = np.array([st.x_dist[:, i, :].sum() for i in range(p.N_PROV)])
        trn = np.array([st.x_trns[:, i, :].sum() - st.x_trns[i, :, :].sum()
                         for i in range(p.N_PROV)])
        dem = p.DEMAND.sum(axis=1)
        sh  = st.sh.sum(axis=1)
        return loc[sort_idx], imp[sort_idx], trn[sort_idx], dem[sort_idx], sh[sort_idx]

    fig, axes = plt.subplots(1, 2, figsize=(20, 14), sharey=True)
    fig.suptitle("Province Supply–Demand Balance (Annual, ton)",
                 fontsize=13, fontweight="bold")

    for ax, (st, title) in zip(axes, [(initial, "Initial Solution"),
                                       (optimized, "ALNS Optimized")]):
        loc, imp, trn, dem, sh = _supply_components(st)
        y = np.arange(p.N_PROV)
        ax.barh(y, loc / 1e3, height=0.7, color="#2ca02c", alpha=0.85, label="Lokal")
        ax.barh(y, imp / 1e3, height=0.7, left=loc / 1e3,
                color="#1f77b4", alpha=0.85, label="Impor (via pelabuhan)")
        trn_pos = np.maximum(trn, 0)
        trn_neg = np.minimum(trn, 0)
        ax.barh(y, trn_pos / 1e3, height=0.7, left=(loc + imp) / 1e3,
                color="#ff7f0e", alpha=0.8, label="Transfer masuk")
        ax.barh(y, trn_neg / 1e3, height=0.7, left=(loc + imp + trn_pos) / 1e3,
                color="#d62728", alpha=0.6, label="Transfer keluar (net)")
        ax.vlines(dem / 1e3, y - 0.4, y + 0.4,
                  colors="black", linewidths=1.2, linestyles="--", label="Demand")
        sh_mask = sh > 0.5
        ax.scatter(dem[sh_mask] / 1e3 - sh[sh_mask] / 1e3 / 2,
                   y[sh_mask], marker="x", color="red", s=50, zorder=5)
        ax.set_yticks(y); ax.set_yticklabels(pnames_s, fontsize=7)
        ax.set_xlabel("Volume (×1 000 ton)"); ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8, loc="lower right"); ax.grid(True, alpha=0.25, axis="x")
        ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "alns_province_balance.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [PNG] alns_province_balance.png")


# ─── Fig 3: Province service rate ─────────────────────────────────────────
def _plot_province_service(initial, optimized, PROV_SHORT, out_dir):
    p = optimized.problem
    sort_idx = np.argsort(-p.DEMAND.sum(axis=1))
    pnames_s = [PROV_SHORT[i] for i in sort_idx]

    def _service_rates(st):
        sh  = st.sh.sum(axis=1)
        dem = p.DEMAND.sum(axis=1)
        return 1.0 - sh / np.maximum(dem, 1.0)

    sr_i = _service_rates(initial)[sort_idx]
    sr_b = _service_rates(optimized)[sort_idx]

    def _sr_color(arr):
        out = []
        for v in arr:
            if   v >= 0.999: out.append("#2ca02c")
            elif v >= 0.90:  out.append("#98df8a")
            elif v >= 0.75:  out.append("#ffbb78")
            else:            out.append("#d62728")
        return out

    fig, axes = plt.subplots(1, 2, figsize=(16, 14), sharey=True)
    fig.suptitle("Province Service Rate – All 38 Provinces",
                 fontsize=13, fontweight="bold")

    for ax, (sr, title) in zip(axes, [(sr_i, "Initial"), (sr_b, "ALNS Optimized")]):
        y = np.arange(p.N_PROV)
        ax.barh(y, sr * 100, color=_sr_color(sr), alpha=0.88, height=0.7)
        ax.vlines(100, -0.5, p.N_PROV - 0.5, colors="black", lw=1, ls="--")
        for yp, v in zip(y, sr):
            ax.text(min(v * 100 + 0.5, 102), yp, f"{v:.0%}",
                    va="center", fontsize=6.5, color="black")
        ax.set_yticks(y); ax.set_yticklabels(pnames_s, fontsize=7)
        ax.set_xlabel("Service Rate (%)"); ax.set_title(title, fontsize=11)
        ax.set_xlim(0, 108); ax.grid(True, alpha=0.25, axis="x")
        ax.invert_yaxis()
        ax.legend(handles=[
            Patch(color="#2ca02c", label="100%"),
            Patch(color="#98df8a", label="90–99%"),
            Patch(color="#ffbb78", label="75–89%"),
            Patch(color="#d62728", label="<75%"),
        ], fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "alns_province_service.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [PNG] alns_province_service.png")


# ─── Fig 4: Import flows ──────────────────────────────────────────────────
def _plot_import_flows(initial, optimized, PROV_SHORT, out_dir):
    p = optimized.problem
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("Import Flow Distribution (Annual)",
                 fontsize=13, fontweight="bold")

    for col, (st, tag) in enumerate([(initial, "Initial"),
                                      (optimized, "ALNS Optimized")]):
        # Top: import volume by port (stacked by country)
        ax = axes[0, col]
        port_by_ctry = np.array([st.x_imp[s, :, :].sum(axis=1)
                                  for s in range(p.N_IMP)]) / 1e3
        bottom = np.zeros(p.N_PORT)
        pal_c = list(plt.cm.tab10(np.linspace(0, 0.8, p.N_IMP)))
        for s in range(p.N_IMP):
            ax.bar(range(p.N_PORT), port_by_ctry[s], bottom=bottom,
                   color=pal_c[s], alpha=0.85, label=p.IMP_NAMES[s])
            bottom += port_by_ctry[s]
        ax.set_xticks(range(p.N_PORT))
        ax.set_xticklabels([nm.replace("Tanjung ", "Tj.") for nm in p.PORT_NAMES],
                           rotation=30, ha="right", fontsize=8)
        ax.set_title(f"Import Volume by Port — {tag}", fontsize=10)
        ax.set_ylabel("Volume (×1 000 ton)")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

        # Bottom: distribution to top-15 receiving provinces (stacked by port)
        ax = axes[1, col]
        dist_prov = np.array([st.x_dist[:, i, :].sum() for i in range(p.N_PROV)]) / 1e3
        top15_idx   = np.argsort(-dist_prov)[:15]
        top15_names = [PROV_SHORT[i] for i in top15_idx]
        dist_by_port = np.array([st.x_dist[h, :, :].sum(axis=1)[top15_idx]
                                  for h in range(p.N_PORT)]) / 1e3
        bottom2 = np.zeros(15)
        pal_p   = plt.cm.tab20(np.linspace(0, 1, p.N_PORT))
        for h in range(p.N_PORT):
            if dist_by_port[h].sum() < 0.1:
                continue
            ax.bar(range(15), dist_by_port[h], bottom=bottom2,
                   color=pal_p[h], alpha=0.85, label=p.PORT_NAMES[h])
            bottom2 += dist_by_port[h]
        ax.set_xticks(range(15))
        ax.set_xticklabels(top15_names, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"Import Distributed to Provinces (top 15) — {tag}", fontsize=10)
        ax.set_ylabel("Volume (×1 000 ton)")
        ax.legend(fontsize=7, ncol=2, loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "alns_import_flows.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [PNG] alns_import_flows.png")


# ─── Fig 5: What changed ──────────────────────────────────────────────────
def _plot_flow_changes(initial, optimized, PROV_SHORT, out_dir):
    p = optimized.problem
    sort_idx = np.argsort(-p.DEMAND.sum(axis=1))
    pnames_s = [PROV_SHORT[i] for i in sort_idx]

    delta_loc = np.array([optimized.x_loc[:, i, :].sum()
                          - initial.x_loc[:, i, :].sum()
                          for i in range(p.N_PROV)]) / 1e3
    delta_imp = np.array([optimized.x_dist[:, i, :].sum()
                          - initial.x_dist[:, i, :].sum()
                          for i in range(p.N_PROV)]) / 1e3
    delta_sh  = np.array([(optimized.sh[i].sum() - initial.sh[i].sum())
                          for i in range(p.N_PROV)]) / 1e3

    fig, axes = plt.subplots(1, 3, figsize=(22, 13))
    fig.suptitle("What Changed After Optimization (ALNS Best − Initial)",
                 fontsize=13, fontweight="bold")

    for ax, (delta, title, unit, pos_col, neg_col) in zip(axes, [
        (delta_loc[sort_idx], "Δ Local Supply",    "×1 000 ton", "#2ca02c", "#d62728"),
        (delta_imp[sort_idx], "Δ Import Received", "×1 000 ton", "#1f77b4", "#ff7f0e"),
        (delta_sh[sort_idx],  "Δ Shortage",        "×1 000 ton", "#d62728", "#2ca02c"),
    ]):
        y = np.arange(p.N_PROV)
        cols = [pos_col if v >= 0 else neg_col for v in delta]
        ax.barh(y, delta, color=cols, alpha=0.85, height=0.7)
        ax.vlines(0, -0.5, p.N_PROV - 0.5, colors="black", lw=0.8)
        ax.set_yticks(y); ax.set_yticklabels(pnames_s, fontsize=7)
        ax.set_xlabel(unit); ax.set_title(title, fontsize=11)
        ax.invert_yaxis(); ax.grid(True, alpha=0.25, axis="x")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "alns_flow_changes.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [PNG] alns_flow_changes.png")


# ─── Landed Cost Computation Helper ────────────────────────────────────────

def _compute_landed_cost(state: SoybeanState) -> np.ndarray:
    """
    Compute landed cost per province per month (Rp/ton).

    Landed cost = total acquisition + transport cost divided by total tons received.
    Each province receives soybeans through three channels:

      1. LOCAL PRODUCTION (x_loc[k, i, t]):
         Cost = C_PROD[k] + C_SHIP[k, i]
         Includes production cost at source province k plus shipping to province i.

      2. IMPORT DISTRIBUTION (x_dist[h, i, t]):
         Cost = avg_purch[h, t] + C_DIST[h, i]
         Includes volume-weighted average CIF purchase price at port h plus
         distribution cost from port h to province i.

      3. INTER-PROVINCIAL TRANSFER (x_trns[j, i, t]):
         Cost = avg_acq + C_TRANS[j, i]
         Includes the ACQUISITION COST of the transferred goods (avg_acq, the
         system-wide weighted average of production and import purchase costs)
         plus inter-provincial transport cost from province j to province i.

    Why avg_acq is needed for transfers:
      Transfers are fungible — we don't track which specific batch (local or
      imported) was transferred. The goods being transferred were originally
      acquired at some cost (either produced locally at ~8-12M Rp/ton or imported
      at CIF price ~8-9.6M Rp/ton). Without including this acquisition cost,
      provinces receiving most of their supply via transfers would show an
      artificially low landed cost (~1-2M Rp/ton, just the transport cost),
      which is physically unrealistic.

    Mathematical formulation:
      avg_acq = (Σ C_PROD[k] × x_loc[k,:,:] + Σ C_PURCH[s] × x_imp[s,:,:])
                / (Σ x_loc + Σ x_imp)

      landed[i, t] = (local_c + import_c + trns_c) / ton

      where:
        local_c  = Σ_k (C_PROD[k] + C_SHIP[k,i]) × x_loc[k,i,t]
        import_c = Σ_h (avg_purch[h,t] + C_DIST[h,i]) × x_dist[h,i,t]
        trns_c   = Σ_j (avg_acq + C_TRANS[j,i]) × x_trns[j,i,t]
        ton      = Σ_k x_loc[k,i,t] + Σ_h x_dist[h,i,t] + Σ_j x_trns[j,i,t]

    Returns:
      landed[N_PROV, N_PERIOD]: landed cost in Rp/ton. Province-months with
      less than 0.1 tons total inflow are marked as NaN.
    """
    p = state.problem
    landed = np.zeros((p.N_PROV, p.N_PERIOD))

    # ── Step 1: Compute weighted-average CIF purchase price per port per month ──
    # avg_purch[h, t] = volume-weighted average CIF price of normal imports at port h in month t.
    # This reflects the actual mix of import sources (USA, Canada, Brazil, etc.) at each port.
    avg_purch = np.zeros((p.N_PORT, p.N_PERIOD))
    for h in range(p.N_PORT):
        for t in range(p.N_PERIOD):
            total = float(state.x_imp[:, h, t].sum())
            if total > 0:
                avg_purch[h, t] = float(np.sum(p.C_PURCH * state.x_imp[:, h, t]) / total)

    # ── Step 2: Compute system-wide average acquisition cost ──
    # avg_acq represents the average cost of acquiring one ton of soybeans,
    # whether through local production (C_PROD) or import purchase (C_PURCH).
    # This is used as the acquisition cost component for inter-provincial transfers,
    # since we don't track which specific batch was transferred.
    #
    # Formula: weighted average of (production cost × local volume) and (CIF price × import volume)
    total_local_cost = float(np.sum(p.C_PROD[:, None, None] * state.x_loc))
    total_import_cost = float(np.sum(p.C_PURCH[:, None, None] * state.x_imp))
    total_local_ton = float(state.x_loc.sum())
    total_import_ton = float(state.x_imp.sum())
    total_acq_ton = total_local_ton + total_import_ton

    if total_acq_ton > 0:
        avg_acq = (total_local_cost + total_import_cost) / total_acq_ton
    else:
        # Fallback: if no supply exists (edge case), use average of C_PROD and C_PURCH
        avg_acq = float(np.mean(p.C_PROD) + np.mean(p.C_PURCH)) / 2.0

    # ── Step 3: Compute landed cost per province per month ──
    for i in range(p.N_PROV):
        for t in range(p.N_PERIOD):
            # Total tons received by province i in month t (all three channels)
            ton = (state.x_loc[:, i, t].sum()
                   + state.x_dist[:, i, t].sum()
                   + state.x_trns[:, i, t].sum())
            if ton < 0.1:
                landed[i, t] = float('nan')
                continue

            # Channel 1: Local production cost (production + shipping to province i)
            local_c = float(np.sum(
                (p.C_PROD[:, None] + p.C_SHIP)[:, i] * state.x_loc[:, i, t]))

            # Channel 2: Import distribution cost (CIF purchase + port-to-province distribution)
            import_c = 0.0
            for h in range(p.N_PORT):
                if state.x_dist[h, i, t] > 0:
                    import_c += (avg_purch[h, t] + p.C_DIST[h, i]) * state.x_dist[h, i, t]

            # Channel 3: Inter-provincial transfer cost (acquisition + transport)
            # avg_acq ensures transferred goods carry their acquisition cost,
            # preventing artificially low landed cost for transfer-heavy provinces.
            trns_c = float(np.sum((avg_acq + p.C_TRANS[:, i]) * state.x_trns[:, i, t]))

            # Landed cost = total cost / total tons
            landed[i, t] = (local_c + import_c + trns_c) / ton

    return landed


# ─── Fig 6: Landed Cost Overview (national monthly average) ─────────────────

def _plot_landed_cost_overview(initial, optimized, out_dir):
    p = optimized.problem
    li = _compute_landed_cost(initial)
    lo = _compute_landed_cost(optimized)

    avg_i = np.array([np.nanmean(li[:, t]) if np.any(~np.isnan(li[:, t])) else 0.0 for t in range(p.N_PERIOD)])
    avg_o = np.array([np.nanmean(lo[:, t]) if np.any(~np.isnan(lo[:, t])) else 0.0 for t in range(p.N_PERIOD)])
    months = np.arange(1, p.N_PERIOD + 1)

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Landed Cost Overview — Rata-rata Nasional per Bulan",
                 fontsize=12, fontweight="bold")
    ax.plot(months, avg_i / 1e3, "-o", color="#ff7f0e", lw=2, label="Initial")
    ax.plot(months, avg_o / 1e3, "-o", color="#2ca02c", lw=2, label="ALNS Optimized")
    ax.fill_between(months, avg_i / 1e3, avg_o / 1e3, alpha=0.15,
                    color="#2ca02c" if avg_o.mean() <= avg_i.mean() else "#ff7f0e")
    ax.set_ylabel("Rp/kg")
    ax.set_xlabel("Bulan")
    ax.set_xticks(months)
    ax.set_xticklabels(MONTHS_LBL, fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    _save_fig_inline(fig, os.path.join(out_dir, "alns_landed_cost_overview.png"))
    print("  [PNG] alns_landed_cost_overview.png")


# ─── Fig 7: Landed Cost per Cluster ────────────────────────────────────────

def _plot_landed_cost_cluster(initial, optimized, out_dir):
    p = optimized.problem
    li = _compute_landed_cost(initial)
    lo = _compute_landed_cost(optimized)
    months = np.arange(1, p.N_PERIOD + 1)

    CLUSTER_NAMES = ["Sumatera", "Jawa", "Bali & Nusa Tenggara",
                     "Kalimantan", "Sulawesi", "Maluku & Papua"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Landed Cost per Pulau — Rata-rata per Bulan",
                 fontsize=13, fontweight="bold")

    for c, ax in enumerate(axes.flat):
        provs = [i for i in range(p.N_PROV) if p.CLUSTER[i] == c]
        ai = np.array([np.nanmean(li[provs, t]) if np.any(~np.isnan(li[provs, t])) else 0.0 for t in range(p.N_PERIOD)])
        ao = np.array([np.nanmean(lo[provs, t]) if np.any(~np.isnan(lo[provs, t])) else 0.0 for t in range(p.N_PERIOD)])

        ax.plot(months, ai / 1e3, "-o", color="#ff7f0e", lw=1.8, label="Initial")
        ax.plot(months, ao / 1e3, "-o", color="#2ca02c", lw=1.8, label="Optimized")
        ax.set_title(CLUSTER_NAMES[c] if c < len(CLUSTER_NAMES) else f"Cluster {c}",
                     fontsize=10)
        ax.set_xticks(months)
        ax.set_xticklabels(MONTHS_LBL, fontsize=7, rotation=30)
        ax.set_ylabel("Rp/kg", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "alns_landed_cost_cluster.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [PNG] alns_landed_cost_cluster.png")


# ─── Fig 8: Landed Cost Heatmap (province × month) ─────────────────────────

def _plot_landed_cost_heatmap(initial, optimized, out_dir):
    p = optimized.problem
    lo = _compute_landed_cost(optimized)

    prov_names = _short_prov_names(p.PROV_NAMES)
    sort_idx = np.argsort(np.nanmean(lo, axis=1))

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(lo[sort_idx] / 1e3, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest")
    ax.set_xticks(range(p.N_PERIOD))
    ax.set_xticklabels(MONTHS_LBL, fontsize=8, rotation=30)
    ax.set_yticks(range(p.N_PROV))
    ax.set_yticklabels([prov_names[i] for i in sort_idx], fontsize=7)
    ax.set_title("Landed Cost per Provinsi x Bulan (ALNS Optimized, Rp/kg)",
                 fontsize=12, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.78)
    cbar.set_label("Rp/kg", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "alns_landed_cost_heatmap.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [PNG] alns_landed_cost_heatmap.png")


# ─── Fig 9: Landed Cost Annual Average per Province ────────────────────────

def _plot_landed_cost_province(initial, optimized, out_dir):
    p = optimized.problem
    li = _compute_landed_cost(initial)
    lo = _compute_landed_cost(optimized)

    prov_names = _short_prov_names(p.PROV_NAMES)
    avg_i = np.array([np.nanmean(r) if np.any(~np.isnan(r)) else 0.0 for r in li])
    avg_o = np.array([np.nanmean(r) if np.any(~np.isnan(r)) else 0.0 for r in lo])
    sort_idx = np.argsort(-avg_o)

    fig, ax = plt.subplots(figsize=(12, 14))
    width = 0.4
    y = np.arange(p.N_PROV)
    ax.barh(y - width/2, avg_i[sort_idx] / 1e3, width,
            color="#ff7f0e", alpha=0.75, label="Initial")
    ax.barh(y + width/2, avg_o[sort_idx] / 1e3, width,
            color="#2ca02c", alpha=0.85, label="ALNS Optimized", edgecolor="black", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([prov_names[i] for i in sort_idx], fontsize=7)
    ax.set_xlabel("Rp/kg", fontsize=10)
    ax.set_title("Rata-rata Landed Cost Tahunan per Provinsi",
                 fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "alns_landed_cost_province.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [PNG] alns_landed_cost_province.png")


def _save_fig_inline(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── Fig 10: Local Supply Flow — Producer → Province Matrix ──────────────

def _plot_local_supply_matrix(initial, optimized, out_dir):
    """Heatmap: producer (row) × receiving province (col) — annual volume."""
    p = optimized.problem
    months = MONTHS_LBL

    # Annual matrix for optimized
    annual = optimized.x_loc.sum(axis=2)   # (N_PROD, N_PROV) in tons

    prod_names = [f"{p.PROV_NAMES[p.PROD_IDX[k]]}" for k in range(p.N_PROD)]
    prov_names = _short_prov_names(p.PROV_NAMES)
    # Show only provinces that receive > 0 from at least one producer
    receivers = np.where(annual.sum(axis=0) > 0.1)[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, max(6, p.N_PROD * 0.45)),
                                    gridspec_kw={'width_ratios': [1, 2]})
    fig.suptitle("Aliran Kedelai Lokal — Produsen → Provinsi Penerima (tahunan, ton)",
                 fontsize=12, fontweight="bold")

    # Left: per producer total annual production
    prod_total = optimized.x_loc.sum(axis=(1, 2))
    y = np.arange(p.N_PROD)
    ax1.barh(y, prod_total / 1e3, color="#2ca02c", alpha=0.85)
    ax1.set_yticks(y)
    ax1.set_yticklabels(prod_names, fontsize=8)
    ax1.set_xlabel("ribu ton/tahun")
    ax1.set_title("Output per Produsen", fontsize=10)
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis="x")

    # Right: heatmap producer x province
    im = ax2.imshow(annual[:, receivers] / 1e3, aspect="auto", cmap="Greens",
                    interpolation="nearest")
    ax2.set_xticks(range(len(receivers)))
    ax2.set_xticklabels([prov_names[r] for r in receivers], fontsize=7, rotation=45, ha="right")
    ax2.set_yticks(np.arange(p.N_PROD))
    ax2.set_yticklabels(prod_names, fontsize=8)
    ax2.set_title("Volume ke Provinsi Penerima (ribu ton)", fontsize=10)
    ax2.set_xlabel("Provinsi Penerima")
    ax2.set_ylabel("Produsen")
    cbar = fig.colorbar(im, ax=ax2, shrink=0.75)
    cbar.set_label("ribu ton", fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "alns_local_supply_matrix.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [PNG] alns_local_supply_matrix.png")


# ─── Fig 11: Local Supply Monthly Timeline — per Provinsi penerima ──────

def _plot_local_supply_monthly(initial, optimized, out_dir):
    """Monthly local supply by receiving province (optimized only, stacked)."""
    p = optimized.problem

    top_idx = np.argsort(-optimized.x_loc.sum(axis=(0, 2)))[:12]
    top_names = [_short_prov_names(p.PROV_NAMES)[i] for i in top_idx]

    fig, axes = plt.subplots(4, 3, figsize=(18, 14))
    fig.suptitle("Pasokan Kedelai Lokal per Bulan — 12 Provinsi Penerima Terbesar (ALNS Optimized)",
                 fontsize=12, fontweight="bold")
    months = np.arange(1, p.N_PERIOD + 1)

    for idx, (ax, prov_i) in enumerate(zip(axes.flat, top_idx)):
        monthly = optimized.x_loc[:, prov_i, :].sum(axis=0) / 1e3  # total from all producers
        ax.bar(months, monthly, color="#2ca02c", alpha=0.85)
        ax.axhline(y=monthly.mean(), color="red", lw=1, ls="--", alpha=0.6)
        ax.set_title(top_names[idx], fontsize=9)
        ax.set_xticks(months)
        ax.set_xticklabels(MONTHS_LBL, fontsize=6, rotation=30)
        ax.set_ylabel("ribu ton", fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "alns_local_supply_monthly.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [PNG] alns_local_supply_monthly.png")


# ─── Fig 12: Local Supply Monthly — Breakdown by Producer ──────────────

def _plot_local_supply_by_producer(optimized, out_dir):
    """Monthly stacked area: each producer's monthly output."""
    p = optimized.problem

    monthly_by_prod = optimized.x_loc.sum(axis=1) / 1e3  # (N_PROD, N_PERIOD)
    months = np.arange(1, p.N_PERIOD + 1)
    prod_names = [p.PROV_NAMES[p.PROD_IDX[k]] for k in range(p.N_PROD)]
    colors = plt.cm.tab20(np.linspace(0, 1, p.N_PROD))

    fig, ax = plt.subplots(figsize=(14, 7))
    bottom = np.zeros(p.N_PERIOD)
    for k in range(p.N_PROD):
        ax.bar(months, monthly_by_prod[k], bottom=bottom,
               color=colors[k], alpha=0.85, label=_short_prov_names([prod_names[k]])[0])
        bottom += monthly_by_prod[k]

    ax.set_title("Output Kedelai Lokal per Bulan — per Produsen (ALNS Optimized)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(months)
    ax.set_xticklabels(MONTHS_LBL, fontsize=9)
    ax.set_ylabel("ribu ton")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "alns_local_supply_by_producer.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [PNG] alns_local_supply_by_producer.png")
