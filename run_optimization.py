"""
run_optimization.py
───────────────────
Entry point for the Soybean Supply Chain ALNS optimisation.

Run from VSCode (F5 / ▶ button) or from a terminal:
    python run_optimization.py

To tune ε-bounds, penalties, output directory, or run-time parameters,
edit `config.py` — that's the single file with user-tunable knobs.
"""
import os
import warnings

warnings.filterwarnings("ignore")

import config
from problem import Problem
from solver import SoybeanALNSSolver


def main() -> None:
    data_dir = os.path.dirname(os.path.abspath(__file__))
    problem = Problem.load(data_dir=data_dir, seed=config.SEED)
    solver  = SoybeanALNSSolver(problem,
                                 max_iter=config.MAX_ITER,
                                 seed=config.SEED,
                                 use_tabu=config.USE_TABU)
    solver.run()


if __name__ == "__main__":
    main()
