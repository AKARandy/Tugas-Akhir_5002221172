"""
run_experiments_multi.py
~~~~~~~~~~~~~~~~~~~~~~~
3-config x 50-seed comparative experiment with multiprocessing.

Configs:
  1. ALNS+TS  | geometric | standard SA (vs S_current)
  2. ALNS+PRS | geometric | standard SA (vs S_current)
  3. ALNS only| geometric | standard SA (vs S_current)

Each config runs 50 independent seeds (500 iters each) on its own core.
All mid-run plotting/CSVs suppressed (silent mode).
Final aggregate plots + CSVs saved to multi_hasil/.

Usage:
  python run_experiments_multi.py
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

import config
from problem import Problem
from result_artifacts import export_multi_summary
from solver import SoybeanALNSSolver

# =====================================================================
#  Experiment parameters
# =====================================================================
N_SEEDS  = 50
MAX_ITER = 500
N_WORKERS = 3
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "baseline")

MONTHS_LBL = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
              "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]

# CONFIGS = [
#     {
#         "label": "Config 1: ALNS+TS | geometric | paper SA",
#         "short": "C1-geo-paper",
#         "figure_id": 1,
#         "use_tabu": True,
#         "SA_COOLING": "geometric",
#         "SA_PAPER": True,
#         "SA_REDU": 0.995,
#         "TABU_TENURE": 12,
#         "SUB_IT": 30,
#     },
#     {
#         "label": "Config 2: ALNS+TS | geometric | standard SA",
#         "short": "C2-geo-std",
#         "figure_id": 2,
#         "use_tabu": True,
#         "SA_COOLING": "geometric",
#         "SA_PAPER": False,
#         "SA_REDU": 0.99,
#         "TABU_TENURE": 12,
#         "SUB_IT": 30,
#     },
#     {
#         "label": "Config 3: ALNS+TS | exponential | standard SA",
#         "short": "C3-exp-std",
#         "figure_id": 3,
#         "use_tabu": True,
#         "SA_COOLING": "exponential",
#         "SA_PAPER": False,
#         "TABU_TENURE": 12,
#         "SUB_IT": 30,
#     },
#     {
#         "label": "Config 4: ALNS only | exponential | standard SA",
#         "short": "C4-nf-exp-std",
#         "figure_id": 4,
#         "use_tabu": False,
#         "SA_COOLING": "exponential",
#         "SA_PAPER": False,
#         "TABU_TENURE": 12,
#         "SUB_IT": 30,
#     },
# ]

CONFIGS = [
    {
        "label": "ALNS-TS | geometric | standard SA",
        "short": "ALNS-TS",
        "figure_id": 1,
        "use_tabu": True,
        "use_prs": False,
        "SA_COOLING": "geometric",
        "SA_PAPER": False,
        "SA_REDU": 0.99,
        "TABU_TENURE": 12,
        "SUB_IT": 30,
    },
    {
        "label": "ALNS-PRS | geometric | standard SA",
        "short": "ALNS-PRS",
        "figure_id": 2,
        "use_tabu": False,
        "use_prs": True,
        "SA_COOLING": "geometric",
        "SA_PAPER": False,
        "SA_REDU": 0.99,
        "TABU_TENURE": 12,
        "SUB_IT": 30,
        "PRS_EVERY_ITER": True,
        "PRS_SUB_IT": 30,
        "PRS_ALPHA": 0.09,
        "PRS_MOVE_MIN_FRAC": 0.05,
        "PRS_MOVE_MAX_FRAC": 0.35,
    },
    {
        "label": "ALNS only | geometric | standard SA",
        "short": "ALNS",
        "figure_id": 3,
        "use_tabu": False,
        "use_prs": False,
        "SA_COOLING": "geometric",
        "SA_PAPER": False,
        "SA_REDU": 0.99,
        "TABU_TENURE": 12,
        "SUB_IT": 30,
    },
]



# =====================================================================
#  Worker process — one config's 50 seeds
# =====================================================================
def _worker(cfg: dict, problem: Problem) -> tuple[list, list, str]:
    """Run one config's 50 seeds. Returns (runs, all_rows, cfg_short).

    Runs in a separate process via multiprocessing.Pool.
    config overrides are process-local — no save/restore needed.
    """
    config.SA_COOLING = cfg["SA_COOLING"]
    config.SA_PAPER   = cfg["SA_PAPER"]
    if "SA_REDU" in cfg:
        config.SA_REDU = cfg["SA_REDU"]
    config.TABU_TENURE = cfg["TABU_TENURE"]
    config.SUB_IT      = cfg["SUB_IT"]
    config.USE_TABU    = cfg["use_tabu"]
    config.USE_PRS     = cfg.get("use_prs", False)
    config.PRS_EVERY_ITER = cfg.get("PRS_EVERY_ITER", config.PRS_EVERY_ITER)
    config.PRS_SUB_IT      = cfg.get("PRS_SUB_IT", config.PRS_SUB_IT)
    config.PRS_ALPHA       = cfg.get("PRS_ALPHA", config.PRS_ALPHA)
    config.PRS_MOVE_MIN_FRAC = cfg.get("PRS_MOVE_MIN_FRAC", config.PRS_MOVE_MIN_FRAC)
    config.PRS_MOVE_MAX_FRAC = cfg.get("PRS_MOVE_MAX_FRAC", config.PRS_MOVE_MAX_FRAC)
    config.EPS_LOCAL_MIN = 0.0

    runs = []
    all_rows = []
    t_start = time.perf_counter()

    for seed in range(1, N_SEEDS + 1):
        config.EPS_LOCAL_MIN = 0.0
        t0 = time.perf_counter()
        solver = SoybeanALNSSolver(
            problem, max_iter=MAX_ITER, seed=seed,
            use_tabu=cfg["use_tabu"],
        )
        result = solver.run(silent=True)
        elapsed = time.perf_counter() - t0

        bd = result["breakdown_best"]
        st = result["best"]
        sh_total = float(st.sh.sum())
        sh_monthly = st.sh.sum(axis=0).tolist()
        total_imp = float((st.x_imp + st.x_emg).sum())
        total_dem = float(st.problem.DEMAND.sum())
        imp_dep = total_imp / max(total_imp + float(st.x_loc.sum()), 1.0)
        srv = 1.0 - sh_total / max(total_dem, 1.0)

        runs.append({
            "seed":                seed,
            "z_cost":              bd["z_cost"],
            "total_penalty":       bd["total_penalty"],
            "objective":           bd["objective"],
            "shortage":            sh_total,
            "shortage_monthly":    sh_monthly,
            "import_dep":          imp_dep,
            "service_rate":        srv,
            "local_prod":          float(st.x_loc.sum()),
            "elapsed_seconds":     elapsed,
            "history_z_cost":      result["history_z_cost"],
            "history_penalty":     result["history_penalty"],
            "history_best":        result["history_best"],
            "op_history":          result["op_history"],
            "destroy_counts":      result["destroy_counts"],
            "repair_counts":       result["repair_counts"],
            "d_weights_final":     result["d_weights_final"],
            "r_weights_final":     result["r_weights_final"],
            "worst_shortage_prov": result["worst_shortage_prov"],
            "worst_shortage_month": result["worst_shortage_month"],
            "worst_shortage_val":  result["worst_shortage_val"],
            "mean_destruction_scale": float(np.mean(result["destruction_scale_history"])),
            "min_destruction_scale": float(np.min(result["destruction_scale_history"])),
            "max_destruction_scale": float(np.max(result["destruction_scale_history"])),
            "std_destruction_scale": float(np.std(result["destruction_scale_history"])),
        })

        row = {
            "config": cfg["short"],
            "label": cfg["label"],
            "use_tabu": cfg["use_tabu"],
            "use_prs": cfg.get("use_prs", False),
            "SA_COOLING": cfg["SA_COOLING"],
            "SA_PAPER": cfg["SA_PAPER"],
            "SA_REDU": cfg.get("SA_REDU", ""),
            "TABU_TENURE": cfg["TABU_TENURE"],
            "SUB_IT": cfg["SUB_IT"],
            "PRS_EVERY_ITER": cfg.get("PRS_EVERY_ITER", ""),
            "PRS_SUB_IT": cfg.get("PRS_SUB_IT", ""),
            "PRS_ALPHA": cfg.get("PRS_ALPHA", ""),
            "PRS_MOVE_MIN_FRAC": cfg.get("PRS_MOVE_MIN_FRAC", ""),
            "PRS_MOVE_MAX_FRAC": cfg.get("PRS_MOVE_MAX_FRAC", ""),
            "seed": seed,
            "z_cost": bd["z_cost"],
            "total_penalty": bd["total_penalty"],
            "objective": bd["objective"],
            "shortage": sh_total,
            "import_dep": imp_dep,
            "service_rate": srv,
            "local_prod": float(st.x_loc.sum()),
            "elapsed_seconds": elapsed,
            "worst_shortage_prov": result["worst_shortage_prov"],
            "worst_shortage_month": result["worst_shortage_month"],
            "worst_shortage_val": result["worst_shortage_val"],
            "mean_destruction_scale": float(np.mean(result["destruction_scale_history"])),
            "min_destruction_scale": float(np.min(result["destruction_scale_history"])),
            "max_destruction_scale": float(np.max(result["destruction_scale_history"])),
            "std_destruction_scale": float(np.std(result["destruction_scale_history"])),
        }
        for month_idx, val in enumerate(sh_monthly, start=1):
            row[f"shortage_m{month_idx:02d}"] = val
        for op_name, cnt in result["destroy_counts"].items():
            row[f"d_{op_name}"] = cnt
        for op_name, cnt in result["repair_counts"].items():
            row[f"r_{op_name}"] = cnt
        for op_name, w in result["d_weights_final"].items():
            row[f"d_w_{op_name}"] = round(w, 6)
        for op_name, w in result["r_weights_final"].items():
            row[f"r_w_{op_name}"] = round(w, 6)
        all_rows.append(row)

        pct = seed / N_SEEDS * 100
        print(f"  [{cfg['short']}] [{pct:5.1f}%] seed {seed:>2} "
              f"| obj={bd['objective']:>16,.0f} "
              f"| z_cost={bd['z_cost']:>16,.0f} | shortage={sh_total:>10,.1f} "
              f"| worst=prov{result['worst_shortage_prov']}m{result['worst_shortage_month']} "
              f"| {elapsed:.1f}s")

    total_elapsed = time.perf_counter() - t_start
    print(f"  [{cfg['short']}] Done in {total_elapsed:.1f}s "
          f"({total_elapsed / N_SEEDS:.1f}s/run avg)")
    return (runs, all_rows, cfg["short"])


# =====================================================================
#  Aggregate helpers
# =====================================================================
def best_worst_avg(runs: list[dict], key: str):
    vals = [r[key] for r in runs]
    best_idx  = int(np.argmin(vals))
    worst_idx = int(np.argmax(vals))
    avg_val   = float(np.mean(vals))
    return best_idx, worst_idx, avg_val


def avg_curve(runs: list[dict], key: str) -> np.ndarray:
    curves = np.array([r[key] for r in runs])
    return curves.mean(axis=0)


def avg_monthly(runs: list[dict]) -> np.ndarray:
    arrays = np.array([r["shortage_monthly"] for r in runs])
    return arrays.mean(axis=0)


def monthly_shortage_envelope(runs: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = np.array([r["shortage_monthly"] for r in runs], dtype=float)
    return arrays.min(axis=0), arrays.mean(axis=0), arrays.max(axis=0)


# =====================================================================
#  Plotting
# =====================================================================
def _plot_config_convergence(cfg_label: str, runs: list[dict],
                             out_dir: str, cfg_idx: int):
    best_obj_idx, worst_obj_idx, _ = best_worst_avg(runs, "objective")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    n = len(runs[0]["history_z_cost"])
    iters = np.arange(n)

    best_z = runs[best_obj_idx]["history_z_cost"]
    worst_z = runs[worst_obj_idx]["history_z_cost"]
    avg_z = avg_curve(runs, "history_z_cost")

    ax1.fill_between(iters, best_z, worst_z, alpha=0.15, color="gray")
    ax1.plot(iters, best_z, lw=1.5, color="#2ca02c", label="Best")
    ax1.plot(iters, avg_z,  lw=1.2, color="black", ls="--", label="Average")
    ax1.plot(iters, worst_z, lw=1.0, color="#d62728", label="Worst")
    ax1.set_ylabel("Z_cost (Rp)")
    ax1.yaxis.get_major_formatter().set_scientific(True)
    ax1.yaxis.get_major_formatter().set_powerlimits((0, 0))
    ax1.set_title(f"{cfg_label} -- Z_cost Convergence (Best / Avg / Worst, {N_SEEDS} seeds)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    best_p = runs[best_obj_idx]["history_penalty"]
    worst_p = runs[worst_obj_idx]["history_penalty"]
    avg_p = avg_curve(runs, "history_penalty")

    ax2.fill_between(iters, best_p, worst_p, alpha=0.15, color="gray")
    ax2.plot(iters, best_p, lw=1.5, color="#2ca02c", label="Best")
    ax2.plot(iters, avg_p,  lw=1.2, color="black", ls="--", label="Average")
    ax2.plot(iters, worst_p, lw=1.0, color="#d62728", label="Worst")
    ax2.set_ylabel("Total Penalty (Rp)")
    ax2.yaxis.get_major_formatter().set_scientific(True)
    ax2.yaxis.get_major_formatter().set_powerlimits((0, 0))
    ax2.set_xlabel("Iteration")
    ax2.set_title(f"{cfg_label} -- Penalty Convergence")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, f"config_{cfg_idx}_convergence.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PNG] {os.path.basename(path)}")


def _plot_config_monthly_shortage(cfg_label: str, runs: list[dict],
                                   out_dir: str, cfg_idx: int):
    months = np.arange(1, 13)

    min_m, avg_m, max_m = monthly_shortage_envelope(runs)
    min_m = min_m / 1e3
    avg_m = avg_m / 1e3
    max_m = max_m / 1e3

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(months, min_m, max_m, alpha=0.15, color="gray")
    ax.plot(months, min_m, "-o", lw=1.5, ms=4, color="#2ca02c", label="Minimum")
    ax.plot(months, avg_m, "--s", lw=1.2, ms=3, color="black", label=f"Average ({N_SEEDS} seeds)")
    ax.plot(months, max_m, "-^", lw=1.0, ms=3, color="#d62728", label="Maximum")
    ax.set_xticks(months)
    ax.set_xticklabels(MONTHS_LBL, fontsize=9)
    ax.set_xlabel("Month")
    ax.set_ylabel("Shortage (x1 000 ton)")
    ax.set_title(f"{cfg_label} -- Monthly National Shortage (Min / Avg / Max)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, f"config_{cfg_idx}_monthly_shortage.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PNG] {os.path.basename(path)}")


def _plot_comparison_boxplot(all_results: list[list[dict]], out_dir: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    z_costs = [[r["z_cost"] for r in runs] for runs in all_results]
    shortages = [[r["shortage"] for r in runs] for runs in all_results]
    labels = [c["short"] for c in CONFIGS]

    bp1 = ax1.boxplot(z_costs, labels=labels, patch_artist=True, showfliers=False)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for patch, c in zip(bp1["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.4)
    ax1.set_ylabel("Z_cost (Rp)")
    ax1.yaxis.get_major_formatter().set_scientific(True)
    ax1.yaxis.get_major_formatter().set_powerlimits((0, 0))
    ax1.set_title(f"Z_cost Distribution ({N_SEEDS} seeds each)")
    ax1.grid(True, alpha=0.3, axis="y")

    bp2 = ax2.boxplot(shortages, labels=labels, patch_artist=True, showfliers=False)
    for patch, c in zip(bp2["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.4)
    
    # Overlay all data points (jittered strip plot)
    for i, (sh_data, color) in enumerate(zip(shortages, colors), start=1):
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(sh_data))
        ax2.scatter(i + jitter, sh_data, alpha=0.6, s=30, color=color, 
                   edgecolors='black', linewidth=0.5, zorder=3)
    
    ax2.set_ylabel("Total Annual Shortage (ton)")
    ax2.set_title(f"Shortage Distribution ({N_SEEDS} seeds each)")
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(out_dir, "comparison_boxplot.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PNG] {os.path.basename(path)}")


def _plot_comparison_bar(all_results: list[list[dict]], out_dir: str):
    labels = [c["short"] for c in CONFIGS]
    x = np.arange(len(labels))
    width = 0.3

    means_z   = [np.mean([r["z_cost"] for r in runs]) for runs in all_results]
    means_pen = [np.mean([r["total_penalty"] for r in runs]) for runs in all_results]
    means_sh  = [np.mean([r["shortage"] for r in runs]) for runs in all_results]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    bars1 = ax1.bar(x - width, [m / 1e12 for m in means_z], width,
                   label="Mean Z_cost", color="#1f77b4", alpha=0.85)
    bars2 = ax1.bar(x, [m / 1e12 for m in means_pen], width,
                   label="Mean Penalty", color="#d62728", alpha=0.85)
    ax1.set_ylabel("Rp (x10^12)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(True, alpha=0.3, axis="y")

    ax2 = ax1.twinx()
    bars3 = ax2.bar(x + width, means_sh, width,
                   label="Mean Shortage (ton)", color="#ff7f0e", alpha=0.85)
    ax2.set_ylabel("Shortage (ton)", fontsize=10)
    ax2.legend(fontsize=9, loc="upper right")

    ax1.set_title(f"Comparison Across {N_SEEDS} Seeds -- Mean Metrics per Config")

    fig.tight_layout()
    path = os.path.join(out_dir, "comparison_bar.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PNG] {os.path.basename(path)}")


# =====================================================================
#  Main
# =====================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print(f"  MULTI-RUN EXPERIMENT -- {len(CONFIGS)} configs x {N_SEEDS} seeds "
          f"x {MAX_ITER} iters  ({N_WORKERS} workers)")
    print("=" * 70)
    print(f"  Output directory: {OUTPUT_DIR}")
    print()

    for ci, cfg in enumerate(CONFIGS):
        red_val = cfg.get("SA_REDU", "N/A")
        print(f"  [{ci + 1}] {cfg['label']}")
        print(f"       SA_COOLING={cfg['SA_COOLING']}  SA_PAPER={cfg['SA_PAPER']}  "
              f"SA_REDU={red_val}  use_tabu={cfg['use_tabu']}  "
              f"use_prs={cfg.get('use_prs', False)}  "
              f"TABU_TENURE={cfg['TABU_TENURE']}  SUB_IT={cfg['SUB_IT']}  "
              f"PRS_SUB_IT={cfg.get('PRS_SUB_IT', 'N/A')}")
    print()

    problem = Problem.load(data_dir=os.path.dirname(os.path.abspath(__file__)),
                           seed=config.SEED)

    print(f"  Launching {N_WORKERS} worker processes...\n")

    with mp.Pool(N_WORKERS) as pool:
        worker_results = pool.starmap(
            _worker,
            [(cfg, problem) for cfg in CONFIGS],
        )

    all_results = []
    all_rows = []
    for runs, rows, _ in worker_results:
        all_results.append(runs)
        all_rows.extend(rows)

    # -- Save per-run CSV --
    print("\nSaving results...")
    df_runs = pd.DataFrame(all_rows)
    df_runs.to_csv(os.path.join(OUTPUT_DIR, "all_runs.csv"), index=False)
    print(f"  [CSV] all_runs.csv  ({len(df_runs)} rows x {len(df_runs.columns)} cols)")

    # -- Operator iteration-history CSV (long format) --
    op_history_rows = []
    for ci, (cfg, runs) in enumerate(zip(CONFIGS, all_results)):
        for r in runs:
            for it, (d_op, r_op, outcome) in enumerate(r["op_history"]):
                op_history_rows.append({
                    "config": cfg["short"],
                    "seed": r["seed"],
                    "iteration": it,
                    "destroy_op": d_op,
                    "repair_op": r_op,
                    "outcome": outcome,
                })
    df_oph = pd.DataFrame(op_history_rows)
    df_oph.to_csv(os.path.join(OUTPUT_DIR, "operator_history.csv"), index=False)
    print(f"  [CSV] operator_history.csv  ({len(df_oph)} rows)")

    # -- Aggregate summary CSV --
    summary_rows = []
    for ci, (cfg, runs) in enumerate(zip(CONFIGS, all_results)):
        z_vals = [r["z_cost"] for r in runs]
        sh_vals = [r["shortage"] for r in runs]
        obj_vals = [r["objective"] for r in runs]
        pen_vals = [r["total_penalty"] for r in runs]
        dep_vals = [r["import_dep"] for r in runs]
        srv_vals = [r["service_rate"] for r in runs]
        loc_vals = [r["local_prod"] for r in runs]
        t_vals = [r["elapsed_seconds"] for r in runs]

        summary_rows.append({
            "config": cfg["short"],
            "label": cfg["label"],
            "use_tabu": cfg["use_tabu"],
            "use_prs": cfg.get("use_prs", False),
            "SA_COOLING": cfg["SA_COOLING"],
            "SA_PAPER": cfg["SA_PAPER"],
            "SA_REDU": cfg.get("SA_REDU", ""),
            "TABU_TENURE": cfg["TABU_TENURE"],
            "SUB_IT": cfg["SUB_IT"],
            "PRS_EVERY_ITER": cfg.get("PRS_EVERY_ITER", ""),
            "PRS_SUB_IT": cfg.get("PRS_SUB_IT", ""),
            "PRS_ALPHA": cfg.get("PRS_ALPHA", ""),
            "PRS_MOVE_MIN_FRAC": cfg.get("PRS_MOVE_MIN_FRAC", ""),
            "PRS_MOVE_MAX_FRAC": cfg.get("PRS_MOVE_MAX_FRAC", ""),
            "mean_z_cost": np.mean(z_vals),
            "std_z_cost": np.std(z_vals),
            "best_z_cost": np.min(z_vals),
            "worst_z_cost": np.max(z_vals),
            "mean_penalty": np.mean(pen_vals),
            "mean_shortage": np.mean(sh_vals),
            "std_shortage": np.std(sh_vals),
            "best_shortage": np.min(sh_vals),
            "worst_shortage": np.max(sh_vals),
            "mean_import_dep": np.mean(dep_vals),
            "mean_service_rate": np.mean(srv_vals),
            "mean_local_prod": np.mean(loc_vals),
            "total_elapsed_s": sum(t_vals),
            "avg_elapsed_s": np.mean(t_vals),
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(OUTPUT_DIR, "summary.csv"), index=False)
    print(f"  [CSV] summary.csv  ({len(df_summary)} rows)")
    summary_json = export_multi_summary(summary_rows, OUTPUT_DIR)
    print(f"  [JSON] {os.path.basename(summary_json)}")

    # -- Generate per-config convergence + monthly shortage plots --
    print("\nGenerating aggregate plots...")
    for ci, (cfg, runs) in enumerate(zip(CONFIGS, all_results)):
        figure_id = cfg.get("figure_id", ci + 1)
        _plot_config_convergence(cfg["label"], runs, OUTPUT_DIR, figure_id)
        _plot_config_monthly_shortage(cfg["label"], runs, OUTPUT_DIR, figure_id)

    # -- Cross-config comparison plots --
    _plot_comparison_boxplot(all_results, OUTPUT_DIR)
    _plot_comparison_bar(all_results, OUTPUT_DIR)

    # -- Console summary --
    print("\n" + "=" * 100)
    print("  MULTI-RUN EXPERIMENT COMPLETE")
    print("=" * 100)
    hdr = (f"  {'Config':<24} {'Mean Z_cost':>20} {'Best Z_cost':>20} "
           f"{'Mean Shortage':>16} {'Mean Svc Rate':>14} {'Time (50r)':>12}")
    print(hdr)
    print("  " + "-" * 106)
    for row in summary_rows:
        print(f"  {row['config']:<24} {row['mean_z_cost']:>20,.0f} "
              f"{row['best_z_cost']:>20,.0f} {row['mean_shortage']:>16,.1f} "
              f"{row['mean_service_rate']:>13.2%} {row['total_elapsed_s']:>11.1f}s")
    print("=" * 100)
    print(f"  Output saved to: {OUTPUT_DIR}/")
    print()


if __name__ == "__main__":
    mp.freeze_support()
    main()
