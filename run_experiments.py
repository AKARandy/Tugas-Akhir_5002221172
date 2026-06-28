import argparse
import time
import os
import pandas as pd

import config
from problem import Problem
from solver import SoybeanALNSSolver

def _reset_config():
    config.EPS_LOCAL_MIN = 0.0

def run_multirun(n_seeds):
    print(f"\n{'='*50}\n1. MULTI-RUN EXPERIMENT ({n_seeds} SEEDS)\n{'='*50}")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    results = []
    
    p = Problem.load()
    
    for i in range(1, n_seeds + 1):
        _reset_config()
        print(f"\n--- Run {i}/{n_seeds} (Seed={i}) ---")
        solver = SoybeanALNSSolver(p, seed=i)
        t0 = time.perf_counter()
        
        res = solver.run()
        elapsed = time.perf_counter() - t0
        
        bd = res['breakdown_best']
        st = res['best']
        sh = float(st.sh.sum())
        imp_dep = float((st.x_imp + st.x_emg).sum()) / max(float(p.DEMAND.sum()), 1.0)
        srv = 1.0 - sh / max(float(p.DEMAND.sum()), 1.0)
        loc = float(st.x_loc.sum())
        
        results.append({
            'seed': i,
            'z_cost': bd['z_cost'],
            'total_penalty': bd['total_penalty'],
            'objective': bd['objective'],
            'shortage': sh,
            'import_dep_ratio': imp_dep,
            'service_rate': srv,
            'local_prod': loc,
            'elapsed_seconds': elapsed
        })
        
    df = pd.DataFrame(results)
    path = os.path.join(config.OUTPUT_DIR, "multirun_results.csv")
    df.to_csv(path, index=False)
    
    print(f"\nMulti-run complete. Saved to {path}")
    print(f"Mean Z_cost : {df['z_cost'].mean():,.0f} ± {df['z_cost'].std():,.0f}")
    print(f"Best Z_cost : {df['z_cost'].min():,.0f}")
    print(f"Worst Z_cost: {df['z_cost'].max():,.0f}")
    print(f"CV Z_cost   : {df['z_cost'].std() / df['z_cost'].mean() * 100:.2f}%")


def run_sensitivity():
    print(f"\n{'='*50}\n2. SENSITIVITY ANALYSIS (EPS_IMPORT_DEP)\n{'='*50}")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    results = []
    
    p = Problem.load()
    original_eps = config.EPS_IMPORT_DEP
    
    sweep_vals = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    for val in sweep_vals:
        _reset_config()
        config.EPS_IMPORT_DEP = val
        print(f"\n--- Sensitivity Run: EPS_IMPORT_DEP = {val} ---")
        solver = SoybeanALNSSolver(p, seed=42)
        res = solver.run()
        
        bd = res['breakdown_best']
        st = res['best']
        sh = float(st.sh.sum())
        imp_dep = float((st.x_imp + st.x_emg).sum()) / max(float(p.DEMAND.sum()), 1.0)
        srv = 1.0 - sh / max(float(p.DEMAND.sum()), 1.0)
        
        results.append({
            'eps_value': val,
            'z_cost': bd['z_cost'],
            'penalty': bd['total_penalty'],
            'objective': bd['objective'],
            'shortage': sh,
            'import_dep': imp_dep,
            'service_rate': srv
        })
        
    config.EPS_IMPORT_DEP = original_eps
    
    df = pd.DataFrame(results)
    path = os.path.join(config.OUTPUT_DIR, "sensitivity_results.csv")
    df.to_csv(path, index=False)
    
    print(f"\nSensitivity complete. Saved to {path}")
    print(df[['eps_value', 'z_cost', 'penalty', 'objective', 'import_dep']].to_string(index=False))


def run_ablation():
    print(f"\n{'='*50}\n3. ABLATION STUDY\n{'='*50}")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    results = []
    
    p = Problem.load()
    
    configs = [
        {"name": "Full", "exclude_ops": set(), "use_tabu": True},
        {"name": "No D7", "exclude_ops": {"destroy_relatedness"}, "use_tabu": True},
        {"name": "No R5", "exclude_ops": {"repair_transfer_focused"}, "use_tabu": True},
        {"name": "No Tabu", "exclude_ops": set(), "use_tabu": False},
        {"name": "Minimal", "exclude_ops": {
            "destroy_cost_based", "destroy_shortage_based", "destroy_geographic",
            "destroy_bottleneck_port", "destroy_relatedness", "destroy_policy_emergency",
            "repair_regret", "repair_balanced", "repair_transfer_focused", "repair_emergency_last"
        }, "use_tabu": False},
    ]
    
    for c in configs:
        _reset_config()
        print(f"\n--- Ablation Run: {c['name']} ---")
        solver = SoybeanALNSSolver(p, seed=42, exclude_ops=c['exclude_ops'], use_tabu=c['use_tabu'])
        res = solver.run()
        
        bd = res['breakdown_best']
        results.append({
            'config_name': c['name'],
            'z_cost': bd['z_cost'],
            'total_penalty': bd['total_penalty'],
            'objective': bd['objective']
        })
        
    df = pd.DataFrame(results)
    path = os.path.join(config.OUTPUT_DIR, "ablation_results.csv")
    df.to_csv(path, index=False)
    
    print(f"\nAblation complete. Saved to {path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ALNS Soybean Experiments")
    parser.add_argument("--multi", type=int, help="Run multi-seed experiment with N seeds")
    parser.add_argument("--sensitivity", action="store_true", help="Run sensitivity analysis")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    
    args = parser.parse_args()
    
    run_all = not (args.multi or args.sensitivity or args.ablation)
    
    if run_all or args.multi:
        n_seeds = args.multi if args.multi else 30
        run_multirun(n_seeds)
        
    if run_all or args.sensitivity:
        run_sensitivity()
        
    if run_all or args.ablation:
        run_ablation()
