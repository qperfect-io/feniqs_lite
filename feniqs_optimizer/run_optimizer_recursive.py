#!/usr/bin/env python3
"""
Run the optimizer recursively for all .qasm files in an input folder and its subfolders.

For each circuit, this script writes:
- a per-circuit CSV with all evaluations

And at the end it also writes:
- one aggregated CSV with all evaluations from all circuits
- one summary CSV with the best row for each circuit
- one errors CSV for skipped/failed circuits
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from feniqs_lib.managers.singleton_plugin_manager import get_plugin_manager
from feniqs_optimizer import FeniqsOptimizer


def make_slug(path_text: str) -> str:
    out = []
    for ch in path_text:
        out.append(ch if ch.isalnum() else "_")
    slug = "".join(out)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_").lower()


def write_dict_rows(path: str, rows: List[Dict[str, object]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        with open(path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["message"])
            writer.writerow(["no rows"])
        return

    keys = []
    seen = set()
    preferred = ["qasm_path", "runtime", "fidelity"]
    for key in preferred:
        if any(key in row for row in rows):
            keys.append(key)
            seen.add(key)

    for row in rows:
        for key in row.keys():
            if key not in seen:
                keys.append(key)
                seen.add(key)

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def resolve_mirror_path(qasm_path: str) -> str:
    return str(Path(qasm_path).with_suffix(".mirror"))


def main():
    parser = argparse.ArgumentParser(description="Recursively optimize all QASM files in a folder tree.")
    parser.add_argument("--backend", type=str, default="QiskitAerCpu", help="Quantum backend name")
    parser.add_argument("--input", type=str, required=True, help="Input folder containing .qasm files")
    parser.add_argument("--config", type=str, default="yaml/optimizator.yaml", help="Path to optimizer config")
    parser.add_argument("--method", type=str, choices=["cmaes", "moead", "nsga2"], default="cmaes", help="Optimization method")
    parser.add_argument("--gens", type=int, default=10, help="Number of generations")
    parser.add_argument("--pop", type=int, default=10, help="Population size")
    parser.add_argument("--num_eval", type=int, default=3, help="Number of evaluations of the fitness function")
    parser.add_argument("--output-dir", type=str, default="optimizer_batch_results", help="Folder where all CSV outputs will be written")
    parser.add_argument("--skip-missing-mirror", action="store_true", help="Skip QASM files without a sibling .mirror file")

    args = parser.parse_args()

    input_root = Path(args.input)
    if not input_root.exists():
        raise FileNotFoundError(f"Input folder {input_root} not found")
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input path {input_root} is not a directory")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    qasm_files = sorted(input_root.rglob("*.qasm"))
    if not qasm_files:
        raise FileNotFoundError(f"No .qasm files found under {input_root}")

    plugin_manager = get_plugin_manager()
    plugin_manager.register_all_plugins()

    all_rows: List[Dict[str, object]] = []
    best_rows: List[Dict[str, object]] = []
    errors: List[Dict[str, object]] = []

    for qasm_path in qasm_files:
        qasm_for_run = os.path.relpath(str(qasm_path), os.getcwd())
        mirror_path = resolve_mirror_path(qasm_for_run)

        if not os.path.exists(mirror_path):
            message = f"Missing mirror file for {qasm_for_run}: expected {mirror_path}"
            if args.skip_missing_mirror:
                print(f"[SKIP] {message}")
                errors.append({"qasm_path": qasm_for_run, "error": message})
                continue
            raise FileNotFoundError(message)

        slug = make_slug(os.path.relpath(str(qasm_path), str(input_root)))
        per_file_csv = output_dir / f"{slug}.csv"

        print(f"\n=== Optimizing {qasm_for_run} ===")
        try:
            optimizer = FeniqsOptimizer(
                backend_name=args.backend,
                qasm_file=qasm_for_run,
                mirror_qasm_file=mirror_path,
                plugin_manager=plugin_manager,
                config_path=args.config,
                opt_method=args.method,
                num_evaluations=args.num_eval,
                csv_output_path=str(per_file_csv),
            )
            optimizer.optimize(max_generations=args.gens, population_size=args.pop)
            rows = optimizer.get_evaluation_rows()
            best_row = optimizer.get_best_evaluation()

            all_rows.extend(rows)
            if best_row is not None:
                best_rows.append(best_row)

            print(f"Saved per-file CSV to {per_file_csv}")
        except Exception as exc:
            print(f"[ERROR] {qasm_for_run}: {exc}")
            errors.append({"qasm_path": qasm_for_run, "error": str(exc)})

    aggregate_csv = output_dir / "all_evaluations.csv"
    best_csv = output_dir / "best_solutions.csv"
    errors_csv = output_dir / "errors.csv"

    write_dict_rows(str(aggregate_csv), all_rows)
    write_dict_rows(str(best_csv), best_rows)
    write_dict_rows(str(errors_csv), errors)

    print("\nBatch optimization finished.")
    print(f"All evaluations: {aggregate_csv}")
    print(f"Best solutions:  {best_csv}")
    print(f"Errors:          {errors_csv}")


if __name__ == "__main__":
    main()
