#!/usr/bin/env python3
"""
Run the optimizer for a single QASM circuit.
"""

import argparse
import os
import sys
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from feniqs_lib.managers.singleton_plugin_manager import get_plugin_manager
from feniqs_optimizer import FeniqsOptimizer


def resolve_mirror_path(qasm_path: str, mirror_arg: str | None) -> str:
    if mirror_arg:
        return mirror_arg
    return str(Path(qasm_path).with_suffix(".mirror"))


def main():
    parser = argparse.ArgumentParser(description="Run quantum backend optimizer for one QASM file.")
    parser.add_argument("--backend", type=str, default="QiskitAerCpu", help="Quantum backend name")
    parser.add_argument("--qasm", type=str, required=True, help="Path to the QASM file")
    parser.add_argument("--mirror", type=str, default=None, help="Path to the mirror file; defaults to <qasm>.mirror")
    parser.add_argument("--config", type=str, default="yaml/optimizator.yaml", help="Path to optimizer config")
    parser.add_argument("--method", type=str, choices=["cmaes", "moead", "nsga2"], default="cmaes", help="Optimization method")
    parser.add_argument("--gens", type=int, default=10, help="Number of generations")
    parser.add_argument("--pop", type=int, default=10, help="Population size")
    parser.add_argument("--num_eval", type=int, default=3, help="Number of evaluations of the fitness function")
    parser.add_argument("--csv-out", type=str, default=None, help="Path to CSV with all evaluated points")

    args = parser.parse_args()

    if not os.path.exists(args.qasm):
        raise FileNotFoundError(f"QASM file {args.qasm} not found!")

    mirror_path = resolve_mirror_path(args.qasm, args.mirror)
    if not os.path.exists(mirror_path):
        raise FileNotFoundError(
            f"Mirror file {mirror_path} not found. Provide --mirror or create a sibling .mirror file."
        )

    plugin_manager = get_plugin_manager()
    plugin_manager.register_all_plugins()

    print(f"\nRunning optimizer with {args.method.upper()} on backend {args.backend}...\n")

    optimizer = FeniqsOptimizer(
        backend_name=args.backend,
        qasm_file=args.qasm,
        mirror_qasm_file=mirror_path,
        plugin_manager=plugin_manager,
        config_path=args.config,
        opt_method=args.method,
        num_evaluations=args.num_eval,
        csv_output_path=args.csv_out,
    )

    best_params = optimizer.optimize(max_generations=args.gens, population_size=args.pop)
    best_row = optimizer.get_best_evaluation()

    print(f"\nBest Parameters Found: {best_params}")
    if best_row is not None:
        print(f"Best row in CSV: runtime={best_row['runtime']}, fidelity={best_row['fidelity']}, is_best={best_row['is_best']}")
    print(f"Evaluation CSV saved to: {optimizer.csv_output_path}")


if __name__ == "__main__":
    main()
