"""
run_scenarios_best_model.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Scenario experiment runner for the best baseline model: ALNS-PRS.

The experiment treats S1-S4 as operational stress tests under a fixed
multi-objective policy vector.  EPS_IMPORT_DEP stays at 0.90 for those
scenarios.  Only S5 changes EPS_IMPORT_DEP, and S5 is not crossed with
the operational stress scenarios.

Default full run:
    python run_scenarios_best_model.py

Fast smoke test:
    python run_scenarios_best_model.py --scenarios S0_BASE,S5_IMPORT_FORCE_80 --seeds 1 --max-iter 5 --workers 1
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
import warnings
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import config
from problem import Problem
from solver import SoybeanALNSSolver


OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "scenarios"
DEFAULT_N_SEEDS = 10
DEFAULT_MAX_ITER = 500
DEFAULT_N_WORKERS = min(4, os.cpu_count() or 1)

SCENARIOS = [
    {
        "id": "S0_BASE",
        "group": "Baseline",
        "type": "baseline",
        "level": "default",
        "target_scope": "default problem data",
        "eps_import_dep": 0.90,
        "priority_mode": "balanced",
    },
    {
        "id": "S1_PRICE_25",
        "group": "Harga impor naik",
        "type": "operational_stress",
        "level": "+25%",
        "target_scope": "all import purchase costs",
        "eps_import_dep": 0.90,
        "priority_mode": "balanced",
        "change": "price",
        "factor": 1.25,
    },
    {
        "id": "S1_PRICE_50",
        "group": "Harga impor naik",
        "type": "operational_stress",
        "level": "+50%",
        "target_scope": "all import purchase costs",
        "eps_import_dep": 0.90,
        "priority_mode": "balanced",
        "change": "price",
        "factor": 1.50,
    },
    {
        "id": "S2_IMPORT_CAP_25",
        "group": "Kapasitas negara sumber impor turun",
        "type": "operational_stress",
        "level": "-25%",
        "target_scope": "all import source capacities",
        "eps_import_dep": 0.90,
        "priority_mode": "balanced",
        "change": "import_capacity",
        "factor": 0.75,
    },
    {
        "id": "S2_IMPORT_CAP_50",
        "group": "Kapasitas negara sumber impor turun",
        "type": "operational_stress",
        "level": "-50%",
        "target_scope": "all import source capacities",
        "eps_import_dep": 0.90,
        "priority_mode": "balanced",
        "change": "import_capacity",
        "factor": 0.50,
    },
    {
        "id": "S2_IMPORT_CAP_75",
        "group": "Kapasitas negara sumber impor turun",
        "type": "operational_stress",
        "level": "-75%",
        "target_scope": "all import source capacities",
        "eps_import_dep": 0.90,
        "priority_mode": "balanced",
        "change": "import_capacity",
        "factor": 0.25,
    },
    {
        "id": "S3_PORT_CAP_25",
        "group": "Kapasitas pelabuhan terganggu",
        "type": "operational_stress",
        "level": "-25%",
        "target_scope": "all port throughput capacities",
        "eps_import_dep": 0.90,
        "priority_mode": "balanced",
        "change": "port_capacity",
        "factor": 0.75,
    },
    {
        "id": "S3_PORT_CAP_50",
        "group": "Kapasitas pelabuhan terganggu",
        "type": "operational_stress",
        "level": "-50%",
        "target_scope": "all port throughput capacities",
        "eps_import_dep": 0.90,
        "priority_mode": "balanced",
        "change": "port_capacity",
        "factor": 0.50,
    },
    {
        "id": "S3_PORT_CAP_75",
        "group": "Kapasitas pelabuhan terganggu",
        "type": "operational_stress",
        "level": "-75%",
        "target_scope": "all port throughput capacities",
        "eps_import_dep": 0.90,
        "priority_mode": "balanced",
        "change": "port_capacity",
        "factor": 0.25,
    },
    {
        "id": "S4_DEMAND_25",
        "group": "Demand naik",
        "type": "operational_stress",
        "level": "+25%",
        "target_scope": "all provinces",
        "eps_import_dep": 0.90,
        "priority_mode": "balanced",
        "change": "demand",
        "factor": 1.25,
    },
    {
        "id": "S4_DEMAND_50",
        "group": "Demand naik",
        "type": "operational_stress",
        "level": "+50%",
        "target_scope": "all provinces",
        "eps_import_dep": 0.90,
        "priority_mode": "balanced",
        "change": "demand",
        "factor": 1.50,
    },
    {
        "id": "S4_DEMAND_75",
        "group": "Demand naik",
        "type": "operational_stress",
        "level": "+75%",
        "target_scope": "all provinces",
        "eps_import_dep": 0.90,
        "priority_mode": "balanced",
        "change": "demand",
        "factor": 1.75,
    },
    {
        "id": "S5_IMPORT_RELAX_92",
        "group": "Sensitivitas kebijakan impor",
        "type": "policy_sensitivity",
        "level": "92%",
        "target_scope": "EPS_IMPORT_DEP only",
        "eps_import_dep": 0.92,
        "priority_mode": "balanced",
    },
    {
        "id": "S5_IMPORT_RELAX_95",
        "group": "Sensitivitas kebijakan impor",
        "type": "policy_sensitivity",
        "level": "95%",
        "target_scope": "EPS_IMPORT_DEP only",
        "eps_import_dep": 0.95,
        "priority_mode": "balanced",
    },
    {
        "id": "S5_IMPORT_FORCE_85",
        "group": "Sensitivitas kebijakan impor ketat",
        "type": "policy_sensitivity",
        "level": "85%",
        "target_scope": "EPS_IMPORT_DEP only",
        "eps_import_dep": 0.85,
        "priority_mode": "import",
    },
    {
        "id": "S5_IMPORT_FORCE_80",
        "group": "Sensitivitas kebijakan impor sangat ketat",
        "type": "policy_sensitivity",
        "level": "80%",
        "target_scope": "EPS_IMPORT_DEP only",
        "eps_import_dep": 0.80,
        "priority_mode": "import",
    },
]

DISCLAIMER = (
    "Pada eksperimen skenario operasional, batas ketergantungan impor "
    "dipertahankan pada nilai dasar 90% agar setiap skenario dievaluasi "
    "terhadap standar kebijakan yang sama. Dengan demikian, perubahan harga, "
    "kapasitas impor, kapasitas pelabuhan, dan permintaan tidak diikuti oleh "
    "perubahan target ketergantungan impor. Apabila skenario tertentu "
    "menghasilkan pelanggaran terhadap batas 90%, pelanggaran tersebut "
    "diinterpretasikan sebagai tekanan sistem terhadap objektif ketahanan "
    "impor, bukan sebagai perubahan preferensi pengambil keputusan. Variasi "
    "nilai batas impor 92%, 95%, 85%, dan 80% hanya dianalisis dalam "
    "eksperimen sensitivitas kebijakan terpisah dan tidak disilangkan dengan "
    "skenario operasional untuk menjaga ruang lingkup eksperimen tetap "
    "terkendali."
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _configure_best_model(scenario: dict, max_iter: int) -> None:
    """Reset global config for one scenario run in the worker process."""
    config.MAX_ITER = max_iter
    config.EPS_SHORTAGE = 0.0
    config.EPS_IMPORT_DEP = float(scenario["eps_import_dep"])
    config.EPS_LOCAL_MIN = 0.0
    config.INV_MIN_FRAC = 0.10
    config.PRIORITY_MODE = scenario["priority_mode"]
    config.USE_EMERGENCY = False

    config.USE_TABU = False
    config.USE_PRS = True
    config.SA_COOLING = "geometric"
    config.SA_PAPER = False
    config.SA_REDU = 0.99
    config.PRS_EVERY_ITER = True
    config.PRS_SUB_IT = 30
    config.PRS_ALPHA = 0.09
    config.PRS_MOVE_MIN_FRAC = 0.05
    config.PRS_MOVE_MAX_FRAC = 0.35


def _apply_scenario(base: Problem, scenario: dict) -> Problem:
    """Return a scenario-specific immutable Problem clone."""
    change = scenario.get("change")
    factor = float(scenario.get("factor", 1.0))

    if change == "price":
        return replace(
            base,
            C_PURCH=base.C_PURCH.copy() * factor,
            C_EMG=base.C_EMG.copy() * factor,
        )

    if change == "import_capacity":
        return replace(
            base,
            IMP_CAP_NORMAL=base.IMP_CAP_NORMAL.copy() * factor,
            IMP_CAP_EMERG=base.IMP_CAP_EMERG.copy() * factor,
        )

    if change == "port_capacity":
        return replace(
            base,
            PORT_THRU_CAP=base.PORT_THRU_CAP.copy() * factor,
        )

    if change == "demand":
        return replace(
            base,
            DEMAND=base.DEMAND.copy() * factor,
            SAFETY_STOCK=base.SAFETY_STOCK.copy() * factor,
        )

    return base


def _run_one(task: tuple[Problem, dict, int, int]) -> dict:
    problem, scenario, seed, max_iter = task
    _configure_best_model(scenario, max_iter)

    t0 = time.perf_counter()
    solver = SoybeanALNSSolver(
        problem,
        max_iter=max_iter,
        seed=seed,
        use_tabu=False,
    )
    result = solver.run(silent=True)
    elapsed = time.perf_counter() - t0

    bd = result["breakdown_best"]
    st = result["best"]
    total_import = float((st.x_imp + st.x_emg).sum())
    total_local = float(st.x_loc.sum())
    total_supply = max(total_import + total_local, 1.0)
    total_demand = float(st.problem.DEMAND.sum())
    shortage = float(st.sh.sum())
    import_dep = total_import / total_supply
    eps_import_dep = float(scenario["eps_import_dep"])
    import_dep_violation = max(0.0, import_dep - eps_import_dep)
    service_rate = 1.0 - shortage / max(total_demand, 1.0)

    print(
        f"  [{scenario['id']}] seed {seed:>2} "
        f"| obj={bd['objective']:>16,.0f} "
        f"| z={bd['z_cost']:>16,.0f} "
        f"| sh={shortage:>10,.1f} "
        f"| imp={import_dep:>6.2%} "
        f"| {elapsed:>5.1f}s",
        flush=True,
    )

    return {
        "scenario_id": scenario["id"],
        "scenario_group": scenario["group"],
        "scenario_type": scenario["type"],
        "level": scenario["level"],
        "target_scope": scenario["target_scope"],
        "eps_import_dep": eps_import_dep,
        "priority_mode": scenario["priority_mode"],
        "seed": seed,
        "max_iter": max_iter,
        "z_cost": float(bd["z_cost"]),
        "total_penalty": float(bd["total_penalty"]),
        "objective": float(bd["objective"]),
        "pen_shortage": float(bd["pen_shortage"]),
        "pen_import": float(bd["pen_import"]),
        "pen_local": float(bd["pen_local"]),
        "pen_inv_floor": float(bd["pen_inv_floor"]),
        "pen_infeas": float(bd["pen_infeas"]),
        "shortage": shortage,
        "import_dep": import_dep,
        "import_dep_violation": import_dep_violation,
        "service_rate": service_rate,
        "local_prod": total_local,
        "total_import": total_import,
        "total_demand": total_demand,
        "elapsed_seconds": elapsed,
        "worst_shortage_prov": result["worst_shortage_prov"],
        "worst_shortage_month": result["worst_shortage_month"],
        "worst_shortage_val": result["worst_shortage_val"],
    }


def _scenario_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ordered_ids = [s["id"] for s in SCENARIOS]
    for scenario_id, group in df.groupby("scenario_id", sort=False):
        first = group.iloc[0]
        rows.append({
            "scenario_id": scenario_id,
            "scenario_group": first["scenario_group"],
            "scenario_type": first["scenario_type"],
            "level": first["level"],
            "target_scope": first["target_scope"],
            "eps_import_dep": first["eps_import_dep"],
            "priority_mode": first["priority_mode"],
            "n_runs": len(group),
            "mean_z_cost": group["z_cost"].mean(),
            "std_z_cost": group["z_cost"].std(ddof=0),
            "best_z_cost": group["z_cost"].min(),
            "worst_z_cost": group["z_cost"].max(),
            "mean_penalty": group["total_penalty"].mean(),
            "mean_pen_import": group["pen_import"].mean(),
            "mean_shortage": group["shortage"].mean(),
            "std_shortage": group["shortage"].std(ddof=0),
            "best_shortage": group["shortage"].min(),
            "worst_shortage": group["shortage"].max(),
            "mean_import_dep": group["import_dep"].mean(),
            "mean_import_dep_violation": group["import_dep_violation"].mean(),
            "mean_service_rate": group["service_rate"].mean(),
            "mean_local_prod": group["local_prod"].mean(),
            "mean_total_import": group["total_import"].mean(),
            "total_elapsed_s": group["elapsed_seconds"].sum(),
            "avg_elapsed_s": group["elapsed_seconds"].mean(),
        })

    summary = pd.DataFrame(rows)
    order = {scenario_id: idx for idx, scenario_id in enumerate(ordered_ids)}
    summary["_order"] = summary["scenario_id"].map(order)
    summary = summary.sort_values("_order").drop(columns=["_order"])
    return summary


def _delta_vs_baseline(summary: pd.DataFrame) -> pd.DataFrame:
    base_rows = summary[summary["scenario_id"] == "S0_BASE"]
    if base_rows.empty:
        return pd.DataFrame()

    base = base_rows.iloc[0]
    metrics = [
        "mean_z_cost",
        "mean_penalty",
        "mean_shortage",
        "mean_import_dep",
        "mean_import_dep_violation",
        "mean_service_rate",
        "mean_local_prod",
        "mean_total_import",
    ]

    rows = []
    for _, row in summary.iterrows():
        out = {
            "scenario_id": row["scenario_id"],
            "scenario_group": row["scenario_group"],
            "scenario_type": row["scenario_type"],
            "level": row["level"],
            "eps_import_dep": row["eps_import_dep"],
        }
        for metric in metrics:
            base_val = float(base[metric])
            val = float(row[metric])
            out[f"{metric}_baseline"] = base_val
            out[f"{metric}_value"] = val
            out[f"{metric}_delta"] = val - base_val
            out[f"{metric}_delta_pct"] = (
                (val - base_val) / abs(base_val) * 100.0
                if abs(base_val) > 1e-12 else np.nan
            )
        rows.append(out)
    return pd.DataFrame(rows)


def _plot_scenario_comparison(summary: pd.DataFrame, out_dir: Path) -> Path:
    labels = summary["scenario_id"].tolist()
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax_z, ax_sh, ax_imp, ax_srv = axes.ravel()

    ax_z.bar(x, summary["mean_z_cost"] / 1e12, color="#2f6f9f")
    ax_z.set_title("Mean Z_cost")
    ax_z.set_ylabel("Rp x10^12")
    ax_z.grid(True, axis="y", alpha=0.25)

    ax_sh.bar(x, summary["mean_shortage"], color="#c4513b")
    ax_sh.set_title("Mean Shortage")
    ax_sh.set_ylabel("Ton")
    ax_sh.grid(True, axis="y", alpha=0.25)

    ax_imp.plot(x, summary["mean_import_dep"] * 100, marker="o", color="#33415c")
    ax_imp.plot(x, summary["eps_import_dep"] * 100, ls="--", color="#8a8a8a", label="epsilon")
    ax_imp.set_title("Mean Import Dependency")
    ax_imp.set_ylabel("%")
    ax_imp.legend(fontsize=8)
    ax_imp.grid(True, alpha=0.25)

    ax_srv.bar(x, summary["mean_service_rate"] * 100, color="#4d8b57")
    ax_srv.set_title("Mean Service Rate")
    ax_srv.set_ylabel("%")
    ax_srv.set_ylim(max(0, summary["mean_service_rate"].min() * 100 - 1), 100.1)
    ax_srv.grid(True, axis="y", alpha=0.25)

    for ax in axes.ravel():
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    fig.suptitle("Scenario Experiment - ALNS-PRS, 10 Seeds", fontsize=14)
    fig.tight_layout()
    out_path = out_dir / "scenario_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _write_summary_json(summary: pd.DataFrame, out_dir: Path, args: argparse.Namespace) -> Path:
    payload = {
        "schema_version": "scenario-summary-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runner": Path(__file__).name,
        "model": {
            "name": "ALNS-PRS",
            "max_iter": args.max_iter,
            "seeds": list(range(1, args.seeds + 1)),
            "use_tabu": False,
            "use_prs": True,
            "SA_COOLING": "geometric",
            "SA_PAPER": False,
            "SA_REDU": 0.99,
            "PRS_EVERY_ITER": True,
            "PRS_SUB_IT": 30,
            "PRS_ALPHA": 0.09,
        },
        "disclaimer": DISCLAIMER,
        "scenarios": SCENARIOS,
        "summary": summary.to_dict(orient="records"),
    }
    out_path = out_dir / "summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, ensure_ascii=False, indent=1)
        f.write("\n")
    return out_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ALNS-PRS scenario stress tests and policy sensitivity experiments."
    )
    parser.add_argument("--seeds", type=int, default=DEFAULT_N_SEEDS,
                        help=f"Number of seeds per scenario. Default: {DEFAULT_N_SEEDS}.")
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER,
                        help=f"ALNS iterations per run. Default: {DEFAULT_MAX_ITER}.")
    parser.add_argument("--workers", type=int, default=DEFAULT_N_WORKERS,
                        help=f"Worker processes. Default: {DEFAULT_N_WORKERS}.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR),
                        help=f"Output directory. Default: {OUTPUT_DIR}.")
    parser.add_argument("--scenarios", default="",
                        help="Comma-separated scenario IDs for a subset/smoke test.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.seeds < 1:
        raise ValueError("--seeds must be >= 1")
    if args.max_iter < 1:
        raise ValueError("--max-iter must be >= 1")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    selected_ids = None
    if args.scenarios.strip():
        selected_ids = {item.strip() for item in args.scenarios.split(",") if item.strip()}
        known_ids = {scenario["id"] for scenario in SCENARIOS}
        unknown_ids = sorted(selected_ids - known_ids)
        if unknown_ids:
            raise ValueError(f"Unknown scenario IDs: {', '.join(unknown_ids)}")

    scenarios = [
        scenario for scenario in SCENARIOS
        if selected_ids is None or scenario["id"] in selected_ids
    ]

    print("=" * 80)
    print("  SCENARIO EXPERIMENT - BEST MODEL ALNS-PRS")
    print("=" * 80)
    print(f"  Scenarios : {len(scenarios)}")
    print(f"  Seeds     : {args.seeds} per scenario")
    print(f"  Max iter  : {args.max_iter}")
    print(f"  Workers   : {args.workers}")
    print(f"  Output    : {out_dir}")
    print("=" * 80)

    data_dir = Path(__file__).resolve().parent
    base_problem = Problem.load(data_dir=str(data_dir), seed=config.SEED)
    scenario_problems = {
        scenario["id"]: _apply_scenario(base_problem, scenario)
        for scenario in scenarios
    }

    tasks = [
        (scenario_problems[scenario["id"]], scenario, seed, args.max_iter)
        for scenario in scenarios
        for seed in range(1, args.seeds + 1)
    ]

    t_start = time.perf_counter()
    if args.workers == 1:
        rows = [_run_one(task) for task in tasks]
    else:
        with mp.Pool(args.workers) as pool:
            rows = pool.map(_run_one, tasks)
    elapsed = time.perf_counter() - t_start

    df_runs = pd.DataFrame(rows)
    scenario_order = {scenario["id"]: idx for idx, scenario in enumerate(SCENARIOS)}
    df_runs["_order"] = df_runs["scenario_id"].map(scenario_order)
    df_runs = df_runs.sort_values(["_order", "seed"]).drop(columns=["_order"])
    all_runs_path = out_dir / "all_runs.csv"
    df_runs.to_csv(all_runs_path, index=False)

    summary = _scenario_summary(df_runs)
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    delta = _delta_vs_baseline(summary)
    delta_path = out_dir / "scenario_delta_vs_baseline.csv"
    delta.to_csv(delta_path, index=False)

    json_path = _write_summary_json(summary, out_dir, args)
    plot_path = _plot_scenario_comparison(summary, out_dir)

    print("\nSaving results...")
    print(f"  [CSV]  {all_runs_path}")
    print(f"  [CSV]  {summary_path}")
    print(f"  [CSV]  {delta_path}")
    print(f"  [JSON] {json_path}")
    print(f"  [PNG]  {plot_path}")
    print(f"\nDone in {elapsed:.1f}s.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
