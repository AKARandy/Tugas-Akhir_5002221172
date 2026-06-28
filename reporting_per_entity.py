"""
reporting_per_entity.py
───────────────────────
Per-entity detailed reports (CSV + PNG) for each province, port, and cluster.

Top-level entry point:
    export_all_per_entity(initial, optimized, base_dir)
        creates {base_dir}/hasil_provinsi/{name}/...
                {base_dir}/hasil_pelabuhan/{name}/...
                {base_dir}/hasil_pulau/{name}/...

Each entity folder contains both CSV time-series and matplotlib PNGs that
visualise initial vs optimized behaviour.
"""
from __future__ import annotations

import csv
import os
import re

import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:  # allow CSV-only per-entity export in minimal runtimes
    plt = None

from state import SoybeanState

MONTHS_LBL = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
              "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _sanitize(name: str) -> str:
    """Make a name filesystem-safe: spaces→_, drop dots, collapse '&'."""
    s = name.replace(" & ", "_").replace("&", "_")
    s = s.replace(" ", "_").replace(".", "")
    s = re.sub(r"[^A-Za-z0-9_]", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _write_csv(path: str, rows: list, header: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _safe_div(a, b, default=1.0):
    return a / b if b > 1e-9 else default


def _save_fig(fig, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _short_prov_names(names):
    """Compact province labels for tight axes."""
    return [n.replace("Kep. ", "").replace("Kepulauan ", "Kep.")
             .replace("Nusa Tenggara", "NT").replace("Kalimantan", "Kal.")
             .replace("Sulawesi", "Sul.").replace("Sumatera", "Sum.")
            for n in names]


# ═══════════════════════════════════════════════════════════════════════════
#  PER-PROVINCE
# ═══════════════════════════════════════════════════════════════════════════

def export_per_province(initial: SoybeanState, optimized: SoybeanState,
                        base_dir: str) -> str:
    """Generate per-province folders under {base_dir}/hasil_provinsi/{name}/."""
    p = optimized.problem
    out_root = os.path.join(base_dir, "hasil_provinsi")
    os.makedirs(out_root, exist_ok=True)

    for i in range(p.N_PROV):
        prov_name = p.PROV_NAMES[i]
        folder = os.path.join(out_root, _sanitize(prov_name))
        os.makedirs(folder, exist_ok=True)

        _write_province_csvs(initial, optimized, i, folder)
        if plt is not None:
            _plot_province_balance(initial, optimized, i, folder)
            _plot_province_source_mix(initial, optimized, i, folder)
            _plot_province_service(initial, optimized, i, folder)
            _plot_province_local_source(optimized, i, folder)

    print(f"  [PER-PROV] {p.N_PROV} province folders -> {out_root}")
    return out_root


def _province_monthly_rows(state: SoybeanState, i: int) -> list:
    """Return per-month dicts for province i."""
    p = state.problem
    rows = []
    for t in range(p.N_PERIOD):
        local      = float(state.x_loc[:, i, t].sum())
        imp_in     = float(state.x_dist[:, i, t].sum())
        trn_in     = float(state.x_trns[:, i, t].sum())
        trn_out    = float(state.x_trns[i, :, t].sum())
        rows.append({
            "t":         t,
            "month":     MONTHS_LBL[t],
            "demand":    float(p.DEMAND[i, t]),
            "local":     local,
            "import":    imp_in,
            "trn_in":    trn_in,
            "trn_out":   trn_out,
            "supply":    local + imp_in + trn_in - trn_out,
            "inv":       float(state.inv[i, t]),
            "shortage":  float(state.sh[i, t]),
            "safe":      int(state.safe[i, t]),
            "service":   1.0 - _safe_div(float(state.sh[i, t]), float(p.DEMAND[i, t]), 0.0),
        })
    return rows


def _write_province_csvs(initial, optimized, i, folder):
    p = optimized.problem
    prov = p.PROV_NAMES[i]

    # 1. Monthly time-series (initial / optimized)
    for tag, st in (("initial", initial), ("optimized", optimized)):
        rows_dict = _province_monthly_rows(st, i)
        rows = [[r["t"]+1, r["month"],
                 f"{r['demand']:.1f}", f"{r['local']:.1f}",
                 f"{r['import']:.1f}", f"{r['trn_in']:.1f}",
                 f"{r['trn_out']:.1f}", f"{r['supply']:.1f}",
                 f"{r['inv']:.1f}", f"{r['shortage']:.1f}",
                 r["safe"], f"{r['service']:.4f}"]
                for r in rows_dict]
        _write_csv(os.path.join(folder, f"monthly_{tag}.csv"), rows,
                   ["periode_t", "bulan", "demand_ton", "lokal_ton",
                    "impor_ton", "transfer_masuk_ton", "transfer_keluar_ton",
                    "total_pasokan_ton", "inventori_akhir_ton",
                    "shortage_ton", "safe_flag", "service_rate"])

    # 2. Annual summary comparing both
    init_rows = _province_monthly_rows(initial, i)
    opt_rows  = _province_monthly_rows(optimized, i)
    summary_rows = []
    fields = [("demand", "Demand"), ("local", "Pasokan_Lokal"),
              ("import", "Impor_Diterima"), ("trn_in", "Transfer_Masuk"),
              ("trn_out", "Transfer_Keluar"), ("shortage", "Shortage")]
    for key, label in fields:
        v_i = sum(r[key] for r in init_rows)
        v_o = sum(r[key] for r in opt_rows)
        delta = v_o - v_i
        pct = (delta / abs(v_i) * 100) if abs(v_i) > 1e-9 else 0.0
        summary_rows.append([label, f"{v_i:.1f}", f"{v_o:.1f}",
                              f"{delta:+.1f}", f"{pct:+.2f}"])

    # Service rates (annual)
    sh_i = sum(r["shortage"] for r in init_rows)
    sh_o = sum(r["shortage"] for r in opt_rows)
    dem  = sum(r["demand"]   for r in init_rows)
    sr_i = 1.0 - _safe_div(sh_i, dem, 0.0)
    sr_o = 1.0 - _safe_div(sh_o, dem, 0.0)
    summary_rows.append(["Service_Rate_Tahunan",
                         f"{sr_i*100:.2f}%", f"{sr_o*100:.2f}%",
                         f"{(sr_o-sr_i)*100:+.2f} pp", "-"])

    _write_csv(os.path.join(folder, "summary.csv"), summary_rows,
               ["metrik", "initial", "optimized", "perubahan", "perubahan_persen"])

    # 3. Import breakdown by port + country (optimized)
    rows = []
    for t in range(p.N_PERIOD):
        for h in range(p.N_PORT):
            vol = float(optimized.x_dist[h, i, t])
            if vol > 0.1:
                rows.append([t+1, MONTHS_LBL[t], p.PORT_NAMES[h], f"{vol:.1f}"])
    _write_csv(os.path.join(folder, "impor_per_pelabuhan_optimized.csv"), rows,
               ["periode_t", "bulan", "pelabuhan_asal", "volume_ton"])

    # 4. Local supply breakdown by producer + month (optimized)
    rows = []
    for k in range(p.N_PROD):
        for t in range(p.N_PERIOD):
            vol = float(optimized.x_loc[k, i, t])
            if vol > 0.1:
                rows.append([t+1, MONTHS_LBL[t], p.PROV_NAMES[p.PROD_IDX[k]], f"{vol:.1f}"])
    if rows:
        _write_csv(os.path.join(folder, "lokal_per_produsen_optimized.csv"), rows,
                   ["periode_t", "bulan", "produsen_asal", "volume_ton"])


def _plot_province_balance(initial, optimized, i, folder):
    p = optimized.problem
    prov = p.PROV_NAMES[i]
    init_rows = _province_monthly_rows(initial, i)
    opt_rows  = _province_monthly_rows(optimized, i)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(f"Neraca Bulanan — {prov}", fontsize=12, fontweight="bold")
    months = np.arange(1, p.N_PERIOD + 1)

    for ax, rows, label, color in (
        (axes[0], init_rows, "Initial",   "#ff7f0e"),
        (axes[1], opt_rows,  "Optimized", "#2ca02c"),
    ):
        demand = [r["demand"]   for r in rows]
        supply = [r["supply"]   for r in rows]
        inv    = [r["inv"]      for r in rows]
        short  = [r["shortage"] for r in rows]
        ax.bar(months, demand, alpha=0.30, color="#888", label="Demand")
        ax.plot(months, supply, "-o", color=color, lw=2, label="Total Pasokan")
        ax.plot(months, inv, "--", color="#1f77b4", lw=1.4, label="Inventori akhir")
        if any(s > 0 for s in short):
            ax2 = ax.twinx()
            ax2.bar(months, short, color="#d62728", alpha=0.55, width=0.4,
                    label="Shortage (kanan)")
            ax2.set_ylabel("Shortage (ton)", color="#d62728", fontsize=9)
            ax2.tick_params(axis='y', labelcolor="#d62728")
        ax.set_title(f"{label}", fontsize=10)
        ax.set_ylabel("Ton")
        ax.set_xticks(months)
        ax.set_xticklabels(MONTHS_LBL, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")

    _save_fig(fig, os.path.join(folder, "balance.png"))


def _plot_province_source_mix(initial, optimized, i, folder):
    p = optimized.problem
    prov = p.PROV_NAMES[i]
    init_rows = _province_monthly_rows(initial, i)
    opt_rows  = _province_monthly_rows(optimized, i)
    months = np.arange(1, p.N_PERIOD + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle(f"Komposisi Sumber Pasokan — {prov}",
                 fontsize=12, fontweight="bold")

    for ax, rows, title in ((axes[0], init_rows, "Initial"),
                            (axes[1], opt_rows,  "Optimized")):
        loc = np.array([r["local"]  for r in rows])
        imp = np.array([r["import"] for r in rows])
        trn = np.array([r["trn_in"] for r in rows])
        dem = [r["demand"] for r in rows]

        ax.bar(months, loc, color="#2ca02c", label="Lokal")
        ax.bar(months, imp, bottom=loc, color="#1f77b4", label="Impor")
        ax.bar(months, trn, bottom=loc + imp, color="#ff7f0e", label="Transfer Masuk")
        ax.plot(months, dem, "k--", lw=1.4, label="Demand")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(months)
        ax.set_xticklabels(MONTHS_LBL, fontsize=9)
        ax.set_ylabel("Ton")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=8)

    _save_fig(fig, os.path.join(folder, "source_mix.png"))


def _plot_province_service(initial, optimized, i, folder):
    p = optimized.problem
    prov = p.PROV_NAMES[i]
    init_rows = _province_monthly_rows(initial, i)
    opt_rows  = _province_monthly_rows(optimized, i)
    months = np.arange(1, p.N_PERIOD + 1)

    sr_i = [r["service"] * 100 for r in init_rows]
    sr_o = [r["service"] * 100 for r in opt_rows]
    sh_i = [r["shortage"] for r in init_rows]
    sh_o = [r["shortage"] for r in opt_rows]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(f"Service Rate & Shortage — {prov}",
                 fontsize=12, fontweight="bold")

    ax = axes[0]
    ax.plot(months, sr_i, "-o", color="#ff7f0e", lw=2, label="Initial")
    ax.plot(months, sr_o, "-o", color="#2ca02c", lw=2, label="Optimized")
    ax.axhline(100, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.set_ylabel("Service rate (%)")
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1]
    width = 0.4
    ax.bar(months - width/2, sh_i, width, color="#ff7f0e", label="Initial")
    ax.bar(months + width/2, sh_o, width, color="#2ca02c", label="Optimized")
    ax.set_ylabel("Shortage (ton)")
    ax.set_xticks(months)
    ax.set_xticklabels(MONTHS_LBL, fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=9)

    _save_fig(fig, os.path.join(folder, "service.png"))


def _plot_province_local_source(state: SoybeanState, i: int, folder: str) -> None:
    """Monthly timeline: which producers supply this province with local soybeans."""
    p = state.problem
    months = np.arange(1, p.N_PERIOD + 1)
    prov = p.PROV_NAMES[i]

    # Find producers that send >0 to this province
    annual_by_prod = state.x_loc[:, i, :].sum(axis=1)  # (N_PROD,)
    active = np.where(annual_by_prod > 0.1)[0]
    if len(active) == 0:
        return  # no local supply to this province

    activa_names = [p.PROV_NAMES[p.PROD_IDX[k]] for k in active]
    colors = plt.cm.tab20(np.linspace(0, 1, len(active)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                    gridspec_kw={'width_ratios': [2, 1]})
    fig.suptitle(f"Asal Kedelai Lokal — {prov}", fontsize=12, fontweight="bold")

    # Left: monthly stacked bar
    bottom = np.zeros(p.N_PERIOD)
    for idx, k in enumerate(active):
        monthly = state.x_loc[k, i, :]
        ax1.bar(months, monthly / 1e3, bottom=bottom / 1e3,
                color=colors[idx], alpha=0.85, label=_short_prov_names([activa_names[idx]])[0])
        bottom += monthly
    ax1.set_title("Pasokan per Bulan (ribu ton)", fontsize=10)
    ax1.set_xticks(months)
    ax1.set_xticklabels(MONTHS_LBL, fontsize=8, rotation=30)
    ax1.set_ylabel("ribu ton")
    ax1.legend(fontsize=7, ncol=2, loc="upper left")
    ax1.grid(True, alpha=0.3, axis="y")

    # Right: annual pie/donut
    wedges, texts, autotexts = ax2.pie(
        annual_by_prod[active] / 1e3,
        labels=activa_names,
        autopct=lambda p: f'{p:.0f}%' if p > 3 else '',
        colors=colors, startangle=90,
        textprops={'fontsize': 7}
    )
    ax2.set_title("Pangsa Tahunan (ribu ton)", fontsize=10)

    _save_fig(fig, os.path.join(folder, "local_source.png"))


# ═══════════════════════════════════════════════════════════════════════════
#  PER-PORT
# ═══════════════════════════════════════════════════════════════════════════

def export_per_port(initial: SoybeanState, optimized: SoybeanState,
                    base_dir: str) -> str:
    p = optimized.problem
    out_root = os.path.join(base_dir, "hasil_pelabuhan")
    os.makedirs(out_root, exist_ok=True)

    for h in range(p.N_PORT):
        port_name = p.PORT_NAMES[h]
        folder = os.path.join(out_root, _sanitize(port_name))
        os.makedirs(folder, exist_ok=True)

        _write_port_csvs(initial, optimized, h, folder)
        if plt is not None:
            _plot_port_throughput(initial, optimized, h, folder)
            _plot_port_country_mix(initial, optimized, h, folder)
            _plot_port_destination(optimized, h, folder)

    print(f"  [PER-PORT] {p.N_PORT} port folders -> {out_root}")
    return out_root


def _port_monthly_rows(state: SoybeanState, h: int) -> list:
    p = state.problem
    rows = []
    for t in range(p.N_PERIOD):
        imports_by_country = {s: float(state.x_imp[s, h, t]) for s in range(p.N_IMP)}
        emerg_by_country   = {s: float(state.x_emg[s, h, t]) for s in range(p.N_IMP)}
        total_in  = sum(imports_by_country.values()) + sum(emerg_by_country.values())
        total_out = float(state.x_dist[h, :, t].sum())
        cap       = float(p.PORT_THRU_CAP[h, t])
        rows.append({
            "t":         t,
            "month":     MONTHS_LBL[t],
            "imports":   imports_by_country,
            "emerg":     emerg_by_country,
            "total_in":  total_in,
            "total_out": total_out,
            "cap":       cap,
            "util":      _safe_div(total_in, cap, 0.0),
        })
    return rows


def _write_port_csvs(initial, optimized, h, folder):
    p = optimized.problem

    # 1. Monthly time-series
    for tag, st in (("initial", initial), ("optimized", optimized)):
        rows_dict = _port_monthly_rows(st, h)
        rows = []
        for r in rows_dict:
            row = [r["t"]+1, r["month"]]
            for s in range(p.N_IMP):
                row.append(f"{r['imports'][s]:.1f}")
            row.extend([f"{r['total_in']:.1f}", f"{r['total_out']:.1f}",
                        f"{r['cap']:.1f}",
                        f"{r['util']*100:.2f}"])
            rows.append(row)
        header = (["periode_t", "bulan"]
                  + [f"impor_{p.IMP_NAMES[s]}_ton" for s in range(p.N_IMP)]
                  + ["total_masuk_ton", "total_keluar_ton",
                     "kapasitas_ton", "utilisasi_persen"])
        _write_csv(os.path.join(folder, f"monthly_{tag}.csv"), rows, header)

    # 2. Annual summary comparison
    init_rows = _port_monthly_rows(initial, h)
    opt_rows  = _port_monthly_rows(optimized, h)
    rows = []
    metrics = [
        ("Total_Masuk", lambda R: sum(r["total_in"] for r in R)),
        ("Total_Keluar", lambda R: sum(r["total_out"] for r in R)),
        ("Kapasitas_Tahunan", lambda R: sum(r["cap"] for r in R)),
        ("Utilisasi_Rata2", lambda R: sum(r["util"] for r in R)/len(R) * 100),
    ]
    for label, fn in metrics:
        v_i, v_o = fn(init_rows), fn(opt_rows)
        delta = v_o - v_i
        rows.append([label, f"{v_i:.2f}", f"{v_o:.2f}", f"{delta:+.2f}"])
    _write_csv(os.path.join(folder, "summary.csv"), rows,
               ["metrik", "initial", "optimized", "perubahan"])

    # 3. Country breakdown (annual, both scenarios)
    rows = []
    for s in range(p.N_IMP):
        v_i = float(initial.x_imp[s, h, :].sum() + initial.x_emg[s, h, :].sum())
        v_o = float(optimized.x_imp[s, h, :].sum() + optimized.x_emg[s, h, :].sum())
        rows.append([p.IMP_NAMES[s], f"{v_i:.1f}", f"{v_o:.1f}",
                      f"{v_o-v_i:+.1f}"])
    _write_csv(os.path.join(folder, "country_breakdown.csv"), rows,
               ["negara", "initial_ton", "optimized_ton", "perubahan_ton"])

    # 4. Destination breakdown (annual, both scenarios)
    rows = []
    for i in range(p.N_PROV):
        v_i = float(initial.x_dist[h, i, :].sum())
        v_o = float(optimized.x_dist[h, i, :].sum())
        if v_i + v_o > 0.1:
            rows.append([p.PROV_NAMES[i], f"{v_i:.1f}", f"{v_o:.1f}",
                          f"{v_o-v_i:+.1f}"])
    rows.sort(key=lambda r: -float(r[2]))
    _write_csv(os.path.join(folder, "destination_breakdown.csv"), rows,
               ["provinsi_tujuan", "initial_ton", "optimized_ton", "perubahan_ton"])


def _plot_port_throughput(initial, optimized, h, folder):
    p = optimized.problem
    port = p.PORT_NAMES[h]
    init_rows = _port_monthly_rows(initial, h)
    opt_rows  = _port_monthly_rows(optimized, h)
    months = np.arange(1, p.N_PERIOD + 1)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(f"Throughput Pelabuhan — {port}",
                 fontsize=12, fontweight="bold")

    for ax, rows, title in ((axes[0], init_rows, "Initial"),
                            (axes[1], opt_rows,  "Optimized")):
        ti  = [r["total_in"]  for r in rows]
        to  = [r["total_out"] for r in rows]
        cap = [r["cap"]       for r in rows]
        ax.fill_between(months, 0, cap, color="#cccccc", alpha=0.3, label="Kapasitas")
        ax.plot(months, ti, "-o", color="#1f77b4", lw=2, label="Masuk (impor)")
        ax.plot(months, to, "-s", color="#ff7f0e", lw=2, label="Keluar (distribusi)")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Ton")
        ax.set_xticks(months)
        ax.set_xticklabels(MONTHS_LBL, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")

    _save_fig(fig, os.path.join(folder, "throughput.png"))


def _plot_port_country_mix(initial, optimized, h, folder):
    p = optimized.problem
    port = p.PORT_NAMES[h]
    months = np.arange(1, p.N_PERIOD + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle(f"Komposisi Negara Asal Impor — {port}",
                 fontsize=12, fontweight="bold")
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for ax, st, title in ((axes[0], initial, "Initial"),
                          (axes[1], optimized, "Optimized")):
        bottom = np.zeros(p.N_PERIOD)
        for s in range(p.N_IMP):
            vals = np.array([float(st.x_imp[s, h, t] + st.x_emg[s, h, t])
                             for t in range(p.N_PERIOD)])
            ax.bar(months, vals, bottom=bottom,
                   color=colors[s % len(colors)], label=p.IMP_NAMES[s])
            bottom += vals
        ax.set_title(title, fontsize=10)
        ax.set_xticks(months)
        ax.set_xticklabels(MONTHS_LBL, fontsize=9)
        ax.set_ylabel("Ton")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=8)

    _save_fig(fig, os.path.join(folder, "country_mix.png"))


def _plot_port_destination(optimized, h, folder):
    p = optimized.problem
    port = p.PORT_NAMES[h]

    annual = [(p.PROV_NAMES[i], float(optimized.x_dist[h, i, :].sum()))
              for i in range(p.N_PROV)]
    annual = [a for a in annual if a[1] > 0.1]
    annual.sort(key=lambda x: -x[1])
    annual = annual[:15]

    if not annual:
        return

    names = [a[0] for a in annual]
    vols  = [a[1] for a in annual]

    fig, ax = plt.subplots(figsize=(9, max(4, len(annual) * 0.35)))
    ax.barh(names, vols, color="#1f77b4")
    ax.invert_yaxis()
    ax.set_xlabel("Volume tahunan (ton)")
    ax.set_title(f"Distribusi Tahunan ke Provinsi — {port} (Optimized)",
                 fontsize=11, fontweight="bold")
    for j, v in enumerate(vols):
        ax.text(v, j, f" {v:,.0f}", va="center", fontsize=8)
    ax.grid(True, alpha=0.3, axis="x")
    _save_fig(fig, os.path.join(folder, "destination.png"))


# ═══════════════════════════════════════════════════════════════════════════
#  PER-CLUSTER (PULAU)
# ═══════════════════════════════════════════════════════════════════════════

CLUSTER_NAMES = ["Sumatera", "Jawa", "Bali_Nusa_Tenggara",
                 "Kalimantan", "Sulawesi", "Maluku_Papua"]
CLUSTER_DISPLAY = ["Sumatera", "Jawa", "Bali & Nusa Tenggara",
                   "Kalimantan", "Sulawesi", "Maluku & Papua"]


def export_per_cluster(initial: SoybeanState, optimized: SoybeanState,
                       base_dir: str) -> str:
    p = optimized.problem
    out_root = os.path.join(base_dir, "hasil_pulau")
    os.makedirs(out_root, exist_ok=True)

    for c in range(6):
        provs_in = [i for i in range(p.N_PROV) if p.CLUSTER[i] == c]
        folder = os.path.join(out_root, CLUSTER_NAMES[c])
        os.makedirs(folder, exist_ok=True)

        _write_cluster_csvs(initial, optimized, c, provs_in, folder)
        if plt is not None:
            _plot_cluster_overview(initial, optimized, c, provs_in, folder)
            _plot_cluster_source_mix(initial, optimized, c, provs_in, folder)
            _plot_cluster_per_province(initial, optimized, c, provs_in, folder)

    print(f"  [PER-CLUSTER] 6 cluster folders -> {out_root}")
    return out_root


def _cluster_monthly_rows(state: SoybeanState, provs: list) -> list:
    p = state.problem
    rows = []
    for t in range(p.N_PERIOD):
        demand = float(p.DEMAND[provs, t].sum())
        local  = float(state.x_loc[:, provs, t].sum())
        imp    = float(state.x_dist[:, provs, t].sum())
        trn_in = float(state.x_trns[:, provs, t].sum())
        trn_out= float(state.x_trns[provs, :, t].sum())
        sh     = float(state.sh[provs, t].sum())
        inv    = float(state.inv[provs, t].sum())
        rows.append({
            "t": t, "month": MONTHS_LBL[t], "demand": demand,
            "local": local, "import": imp, "trn_in": trn_in, "trn_out": trn_out,
            "supply": local + imp + trn_in - trn_out,
            "inv": inv, "shortage": sh,
            "service": 1.0 - _safe_div(sh, demand, 0.0),
        })
    return rows


def _write_cluster_csvs(initial, optimized, c, provs, folder):
    p = optimized.problem

    # 1. Aggregate monthly time-series
    for tag, st in (("initial", initial), ("optimized", optimized)):
        rows_dict = _cluster_monthly_rows(st, provs)
        rows = [[r["t"]+1, r["month"],
                 f"{r['demand']:.1f}", f"{r['local']:.1f}",
                 f"{r['import']:.1f}", f"{r['trn_in']:.1f}",
                 f"{r['trn_out']:.1f}", f"{r['supply']:.1f}",
                 f"{r['inv']:.1f}", f"{r['shortage']:.1f}",
                 f"{r['service']:.4f}"]
                for r in rows_dict]
        _write_csv(os.path.join(folder, f"monthly_{tag}.csv"), rows,
                   ["periode_t", "bulan", "demand_ton", "lokal_ton",
                    "impor_ton", "transfer_masuk_ton", "transfer_keluar_ton",
                    "total_pasokan_ton", "inventori_ton",
                    "shortage_ton", "service_rate"])

    # 2. Per-province breakdown within this cluster (annual)
    rows = []
    for i in provs:
        d_ann   = float(p.DEMAND[i].sum())
        loc_i   = float(initial.x_loc[:, i, :].sum())
        loc_o   = float(optimized.x_loc[:, i, :].sum())
        imp_i   = float(initial.x_dist[:, i, :].sum())
        imp_o   = float(optimized.x_dist[:, i, :].sum())
        sh_i    = float(initial.sh[i, :].sum())
        sh_o    = float(optimized.sh[i, :].sum())
        sr_i    = 1.0 - _safe_div(sh_i, d_ann, 0.0)
        sr_o    = 1.0 - _safe_div(sh_o, d_ann, 0.0)
        rows.append([p.PROV_NAMES[i], f"{d_ann:.1f}",
                      f"{loc_i:.1f}", f"{loc_o:.1f}",
                      f"{imp_i:.1f}", f"{imp_o:.1f}",
                      f"{sh_i:.1f}", f"{sh_o:.1f}",
                      f"{sr_i*100:.2f}", f"{sr_o*100:.2f}"])
    _write_csv(os.path.join(folder, "province_breakdown.csv"), rows,
               ["provinsi", "demand_tahunan",
                "lokal_initial", "lokal_optimized",
                "impor_initial", "impor_optimized",
                "shortage_initial", "shortage_optimized",
                "service_initial_persen", "service_optimized_persen"])


def _plot_cluster_overview(initial, optimized, c, provs, folder):
    p = optimized.problem
    name = CLUSTER_DISPLAY[c]
    init_rows = _cluster_monthly_rows(initial, provs)
    opt_rows  = _cluster_monthly_rows(optimized, provs)
    months = np.arange(1, p.N_PERIOD + 1)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(f"Neraca Agregat — {name} ({len(provs)} provinsi)",
                 fontsize=12, fontweight="bold")

    for ax, rows, title in ((axes[0], init_rows, "Initial"),
                            (axes[1], opt_rows,  "Optimized")):
        d = [r["demand"]   for r in rows]
        s = [r["supply"]   for r in rows]
        v = [r["inv"]      for r in rows]
        h = [r["shortage"] for r in rows]
        ax.bar(months, d, alpha=0.30, color="#888", label="Demand")
        ax.plot(months, s, "-o", color="#2ca02c", lw=2, label="Total Pasokan")
        ax.plot(months, v, "--", color="#1f77b4", lw=1.4, label="Inventori")
        if any(x > 0 for x in h):
            ax2 = ax.twinx()
            ax2.bar(months, h, color="#d62728", alpha=0.55, width=0.4)
            ax2.set_ylabel("Shortage (ton)", color="#d62728", fontsize=9)
            ax2.tick_params(axis='y', labelcolor="#d62728")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Ton")
        ax.set_xticks(months)
        ax.set_xticklabels(MONTHS_LBL, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")

    _save_fig(fig, os.path.join(folder, "overview.png"))


def _plot_cluster_source_mix(initial, optimized, c, provs, folder):
    p = optimized.problem
    name = CLUSTER_DISPLAY[c]
    months = np.arange(1, p.N_PERIOD + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle(f"Komposisi Sumber Pasokan — {name}",
                 fontsize=12, fontweight="bold")

    for ax, rows, title in (
        (axes[0], _cluster_monthly_rows(initial, provs),  "Initial"),
        (axes[1], _cluster_monthly_rows(optimized, provs), "Optimized"),
    ):
        loc = np.array([r["local"]  for r in rows])
        imp = np.array([r["import"] for r in rows])
        trn = np.array([r["trn_in"] for r in rows])
        dem = [r["demand"] for r in rows]

        ax.bar(months, loc, color="#2ca02c", label="Lokal")
        ax.bar(months, imp, bottom=loc, color="#1f77b4", label="Impor")
        ax.bar(months, trn, bottom=loc + imp, color="#ff7f0e", label="Transfer Masuk")
        ax.plot(months, dem, "k--", lw=1.4, label="Demand")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(months)
        ax.set_xticklabels(MONTHS_LBL, fontsize=9)
        ax.set_ylabel("Ton")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=8)

    _save_fig(fig, os.path.join(folder, "source_mix.png"))


def _plot_cluster_per_province(initial, optimized, c, provs, folder):
    p = optimized.problem
    name = CLUSTER_DISPLAY[c]

    names = [p.PROV_NAMES[i] for i in provs]
    sr_i = [(1 - _safe_div(float(initial.sh[i].sum()),
                            float(p.DEMAND[i].sum()), 0.0)) * 100 for i in provs]
    sr_o = [(1 - _safe_div(float(optimized.sh[i].sum()),
                            float(p.DEMAND[i].sum()), 0.0)) * 100 for i in provs]
    dem  = [float(p.DEMAND[i].sum()) for i in provs]

    fig, axes = plt.subplots(2, 1, figsize=(11, max(7, len(provs) * 0.35)))
    fig.suptitle(f"Per Provinsi — {name}", fontsize=12, fontweight="bold")

    # Service rate comparison
    ax = axes[0]
    y = np.arange(len(provs))
    width = 0.4
    ax.barh(y - width/2, sr_i, width, color="#ff7f0e", label="Initial")
    ax.barh(y + width/2, sr_o, width, color="#2ca02c", label="Optimized")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Service rate (%)")
    ax.axvline(100, color="k", lw=0.7, ls="--", alpha=0.5)
    ax.set_xlim(0, 110)
    ax.grid(True, alpha=0.3, axis="x")
    ax.legend(fontsize=9)

    # Annual demand
    ax = axes[1]
    ax.barh(y, dem, color="#1f77b4")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Demand tahunan (ton)")
    ax.grid(True, alpha=0.3, axis="x")
    for j, v in enumerate(dem):
        ax.text(v, j, f" {v:,.0f}", va="center", fontsize=7)

    _save_fig(fig, os.path.join(folder, "per_province.png"))


# ═══════════════════════════════════════════════════════════════════════════
#  TOP-LEVEL ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def export_all_per_entity(initial: SoybeanState, optimized: SoybeanState,
                          base_dir: str) -> dict:
    """
    Generate all per-entity reports (38 + 11 + 6 folders).

    Returns dict with paths to the three top-level subdirs.
    """
    print(f"  [PER-ENTITY] Generating detailed reports under {base_dir}…")
    paths = {
        "provinsi":  export_per_province(initial, optimized, base_dir),
        "pelabuhan": export_per_port(initial, optimized, base_dir),
        "pulau":     export_per_cluster(initial, optimized, base_dir),
    }
    return paths
