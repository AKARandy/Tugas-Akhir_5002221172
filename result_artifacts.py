"""
result_artifacts.py
-------------------
JSON export helpers for solver outputs.

The JSON artifact is the handoff between optimisation and downstream tools
such as the dashboard.  It intentionally stores compact, reusable state arrays
instead of browser-specific presentation data.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import config


SCHEMA_VERSION = "optimization-result-v1"


def _jsonable(value: Any) -> Any:
    """Convert numpy scalars/arrays and nested containers into JSON values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _config_snapshot() -> dict:
    names = [
        "EPS_SHORTAGE", "EPS_IMPORT_DEP", "EPS_LOCAL_MIN", "INV_MIN_FRAC",
        "PENALTY_SCALE", "M_SHORTAGE", "M_IMPORT_DEP", "M_LOCAL",
        "M_INV_FLOOR", "SAFETY_STOCK_BUFFER", "SEED", "MAX_ITER",
        "TABU_TENURE", "SUB_IT", "ALNS_SCORE_GLOBAL_BEST",
        "PRS_SUB_IT", "PRS_ALPHA", "PRS_PRISM_ANGLE",
        "PRS_MOVE_MIN_FRAC", "PRS_MOVE_MAX_FRAC",
        "ALNS_SCORE_BETTER", "ALNS_SCORE_SA_ACCEPT", "ALNS_SCORE_REJECT",
        "ALNS_DECAY", "ALNS_MIN_WEIGHT", "PRIORITY_MODE",
        "TABU_EVERY_ITER", "PRS_EVERY_ITER", "SA_PAPER", "SA_COOLING", "SA_REDU",
        "USE_EMERGENCY", "USE_TABU", "USE_PRS", "USE_HISTORICAL_IC",
    ]
    return {name: _jsonable(getattr(config, name)) for name in names}


def _problem_metadata(state: Any) -> dict:
    p = state.problem
    return {
        "N_PROV": p.N_PROV,
        "N_IMP": p.N_IMP,
        "N_PORT": p.N_PORT,
        "N_PERIOD": p.N_PERIOD,
        "N_PROD": p.N_PROD,
        "EXCH_RATE": p.EXCH_RATE,
        "PROV_NAMES": p.PROV_NAMES,
        "IMP_NAMES": p.IMP_NAMES,
        "PORT_NAMES": p.PORT_NAMES,
        "PROD_IDX": p.PROD_IDX,
        "PORT_SERV": p.PORT_SERV,
        "CLUSTER": p.CLUSTER,
        "CRITICAL_PROV": p.CRITICAL_PROV,
        "DEMAND": p.DEMAND,
        "PROD_CAP": p.PROD_CAP,
        "IMP_CAP_NORMAL": p.IMP_CAP_NORMAL,
        "IMP_CAP_EMERG": p.IMP_CAP_EMERG,
        "PORT_THRU_CAP": p.PORT_THRU_CAP,
        "PROV_STOR_CAP": p.PROV_STOR_CAP,
        "SAFETY_STOCK": p.SAFETY_STOCK,
        "X_THRESHOLD": p.X_THRESHOLD,
    }


def _state_payload(state: Any) -> dict:
    return {
        "x_loc": state.x_loc,
        "x_imp": state.x_imp,
        "x_emg": state.x_emg,
        "x_dist": state.x_dist,
        "x_trns": state.x_trns,
        "inv": state.inv,
        "sh": state.sh,
        "safe": state.safe,
        "w": state.w,
        "y": state.y,
        "z": state.z,
        "penalty": state._penalty,
        "objective": state.objective(),
        "cost_breakdown": state.cost_breakdown(),
    }


def build_optimization_artifact(result: dict, *, run_label: str = "single") -> dict:
    initial = result["initial"]
    best = result["best"]
    problem = best.problem

    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run": {
            "label": run_label,
            "seed": result.get("seed", config.SEED),
            "max_iter": result.get("max_iter", len(result.get("history_best", [])) - 1),
            "use_tabu": result.get("use_tabu", config.USE_TABU),
            "use_prs": result.get("use_prs", config.USE_PRS),
            "silent": False,
        },
        "config": _config_snapshot(),
        "problem": _problem_metadata(best),
        "metrics": {
            "history_best": result.get("history_best", []),
            "history_z_cost": result.get("history_z_cost", []),
            "history_penalty": result.get("history_penalty", []),
            "history_curr": result.get("history_curr", []),
            "breakdown_initial": result.get("breakdown_initial", {}),
            "breakdown_best": result.get("breakdown_best", {}),
            "destroy_counts": result.get("destroy_counts", {}),
            "repair_counts": result.get("repair_counts", {}),
            "d_weights_final": result.get("d_weights_final", {}),
            "r_weights_final": result.get("r_weights_final", {}),
            "worst_shortage_prov": result.get("worst_shortage_prov"),
            "worst_shortage_month": result.get("worst_shortage_month"),
            "worst_shortage_val": result.get("worst_shortage_val"),
            "improvement_pct": result.get("improvement_pct"),
            "elapsed_seconds": result.get("elapsed_seconds"),
        },
        "states": {
            "initial": _state_payload(initial),
            "best": _state_payload(best),
        },
    }
    payload["run"]["total_demand"] = float(problem.DEMAND.sum())
    payload["run"]["total_local_capacity"] = float(problem.PROD_CAP.sum())
    return _jsonable(payload)


def export_optimization_result(result: dict, out_dir: str | os.PathLike | None = None,
                               filename: str = "optimization_result.json") -> Path:
    out_path = Path(out_dir or config.OUTPUT_DIR) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = build_optimization_artifact(result)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False, indent=1)
        f.write("\n")
    return out_path


def load_optimization_result(path: str | os.PathLike) -> dict:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported optimization artifact schema: "
            f"{payload.get('schema_version')!r}"
        )
    return payload


def export_multi_summary(summary_rows: list[dict], out_dir: str | os.PathLike,
                         filename: str = "summary.json") -> Path:
    out_path = Path(out_dir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "multi-summary-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": _config_snapshot(),
        "summary": _jsonable(summary_rows),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
        f.write("\n")
    return out_path
