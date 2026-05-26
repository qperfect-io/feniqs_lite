#
# Copyright © 2024 QPerfect. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

from __future__ import annotations

import csv
import glob
import logging
import os
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cma
import numpy as np
import yaml
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions

from feniqs_lib.tools import constants


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
CMA_FIDELITY_ACCURACY = 0.99


class FeniqsOptimizer:
    """
    FeniqsOptimizer integrates different optimization algorithms with quantum backends.
    Supported optimizers:
        - CMA-ES
        - MOEA/D
        - NSGA-II

    In addition to optimization, this class stores every evaluated point and can
    write a CSV similar to the attached full database:
        qasm_path, runtime, fidelity, <optimized params...>, is_best
    """

    def __init__(
        self,
        backend_name,
        qasm_file,
        mirror_qasm_file,
        plugin_manager,
        config_path="yaml/optimizator.yaml",
        opt_method="cmaes",
        num_evaluations=3,
        csv_output_path: Optional[str] = None,
    ):
        self.backend_name = backend_name
        self.qasm_file = qasm_file
        self.mirror_qasm_file = mirror_qasm_file
        self.plugin_manager = plugin_manager
        self.opt_method = opt_method.lower()
        self.num_evaluations = num_evaluations
        self.evaluation_cache = OrderedDict()

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        if backend_name not in config["backends"]:
            raise ValueError(f"Backend `{backend_name}` is not defined in {config_path}")

        backend_config = config["backends"][backend_name]
        self.valid_params = backend_config["params"]
        self.optimization_params = backend_config["optimization_params"]

        if self.opt_method not in ["cmaes", "moead", "nsga2"]:
            raise ValueError(
                f"Invalid optimization method `{self.opt_method}`. Choose from: 'cmaes', 'moead', 'nsga2'."
            )

        self.csv_output_path = csv_output_path or self._default_csv_output_path()
        self._evaluation_rows: List[Dict[str, object]] = []
        self._best_row_index: Optional[int] = None
        self._persistent_backend = None
        self._use_persistent_mimiq = str(self.backend_name).lower() in {
            "mimiqjuliacpu",
            "mimiq_julia_cpu",
        }

    def _default_csv_output_path(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        qasm_stem = Path(self.qasm_file).stem
        backend = self._slugify(self.backend_name)
        method = self._slugify(self.opt_method)
        return f"optimizer_eval_{backend}_{method}_{qasm_stem}_{timestamp}.csv"

    @staticmethod
    def _slugify(text: str) -> str:
        out = []
        for ch in str(text):
            out.append(ch if ch.isalnum() else "_")
        slug = "".join(out)
        while "__" in slug:
            slug = slug.replace("__", "_")
        return slug.strip("_").lower()

    def map_discrete(self, x):
        discrete_params = {}
        for i, param in enumerate(self.optimization_params):
            valid_values = self.valid_params.get(param, [])
            if valid_values:
                idx = int(np.clip(np.round(x[i] * (len(valid_values) - 1)), 0, len(valid_values) - 1))
                discrete_params[param] = valid_values[idx]
        return discrete_params

    def map_discrete_with_encoding(self, x):
        discrete_params = {}
        distances = {}
        for i, param in enumerate(self.optimization_params):
            valid_values = self.valid_params.get(param, [])
            if valid_values:
                idx = int(np.clip(np.round(x[i] * (len(valid_values) - 1)), 0, len(valid_values) - 1))
                mapped_value = valid_values[idx]
                discrete_params[param] = mapped_value
                distances[param] = abs(x[i] * (len(valid_values) - 1) - idx)
        return discrete_params, distances

    def apply_margin_correction(self, es, solutions, distances, margin_threshold=0.1):
        from scipy.stats import norm

        dim = len(self.optimization_params)
        sigma = es.sigma
        for _, dist in zip(solutions, distances):
            for param, d in dist.items():
                param_index = self.optimization_params.index(param)
                if d > margin_threshold:
                    sigma_j = np.sqrt(es.C[param_index, param_index]) * sigma
                    if sigma_j == 0:
                        continue
                    d_std = d / sigma_j
                    p = norm.cdf(-d_std)
                    if p < margin_threshold:
                        gamma_alpha = norm.ppf(1 - margin_threshold)
                        factor = 0 if d_std == 0 else (d_std**2 - gamma_alpha**2) / (d_std**2 * gamma_alpha**2)
                        xi = np.zeros(dim)
                        xi[param_index] = 1.0
                        es.C += factor * np.outer(xi, xi)
        return es

    def _ensure_mimiq_backend_env_on_path(self):
        if not self._use_persistent_mimiq:
            return

        env_info = self.plugin_manager.envs[self.backend_name]
        env_root = os.path.join(constants.PLUGIN, "venv", env_info["venv_name"])
        env_root = os.path.abspath(env_root)

        candidates = glob.glob(os.path.join(env_root, "lib", "python*", "site-packages"))
        if not candidates:
            raise RuntimeError(f"Could not locate site-packages inside backend venv: {env_root}")

        site_packages = candidates[0]
        if site_packages not in sys.path:
            sys.path.insert(0, site_packages)

        current_pythonpath = os.environ.get("PYTHONPATH", "")
        paths = [p for p in current_pythonpath.split(os.pathsep) if p]
        if site_packages not in paths:
            os.environ["PYTHONPATH"] = (
                site_packages if not current_pythonpath else site_packages + os.pathsep + current_pythonpath
            )

    def _run_backend_once(self, params_dict: Dict[str, object]) -> Tuple[float, float]:
        if self._use_persistent_mimiq:
            self._ensure_mimiq_backend_env_on_path()
            from feniqs_lib.backends.quantum_backends.mimiq_julia_backend import MimiqJuliaCpuBackend
            from feniqs_lib.backends.task.task import run_task

            if self._persistent_backend is None:
                self._persistent_backend = MimiqJuliaCpuBackend(
                    test_case=self.qasm_file,
                    nb_shots=1000,
                    seed=1234,
                    **params_dict,
                )
            else:
                self._persistent_backend.reconfigure(
                    nb_shots=1000,
                    seed=1234,
                    **params_dict,
                )

            _, metrics, _ = run_task(self._persistent_backend)
            runtime = float(metrics["total"])
            fidelity = float(metrics["fidelity"])
            return runtime, fidelity

        metrics, _ = self.plugin_manager.run_backend(
            self.backend_name,
            self.qasm_file,
            nb_shots=1000,
            **params_dict,
        )
        runtime = float(metrics["total"]["avg_rt"])
        fidelity = float(metrics["fidelity"]["avg_rt"])
        return runtime, fidelity

    def _record_evaluation(self, params_dict: Dict[str, object], runtime: float, fidelity: float, objective_runtime: float):
        row = {
            "qasm_path": self.qasm_file,
            "runtime": float(runtime),
            "fidelity": float(fidelity),
            "is_best": False,
            "_objective_runtime": float(objective_runtime),
        }
        for param in self.optimization_params:
            row[param] = params_dict.get(param)
        self._evaluation_rows.append(row)

    def _select_best_row_index(self) -> Optional[int]:
        if not self._evaluation_rows:
            return None

        feasible = [
            (idx, row)
            for idx, row in enumerate(self._evaluation_rows)
            if row.get("fidelity") is not None and float(row["fidelity"]) >= CMA_FIDELITY_ACCURACY
        ]
        if feasible:
            return min(feasible, key=lambda item: (float(item[1]["runtime"]), -float(item[1]["fidelity"]), item[0]))[0]

        return min(
            enumerate(self._evaluation_rows),
            key=lambda item: (-float(item[1]["fidelity"]), float(item[1]["runtime"]), item[0]),
        )[0]

    def finalize_results(self):
        self._best_row_index = self._select_best_row_index()
        for idx, row in enumerate(self._evaluation_rows):
            row["is_best"] = idx == self._best_row_index

    def get_evaluation_rows(self) -> List[Dict[str, object]]:
        clean_rows = []
        for row in self._evaluation_rows:
            clean_row = {k: v for k, v in row.items() if not k.startswith("_")}
            clean_rows.append(clean_row)
        return clean_rows

    def get_best_evaluation(self) -> Optional[Dict[str, object]]:
        if self._best_row_index is None:
            self.finalize_results()
        if self._best_row_index is None:
            return None
        row = self._evaluation_rows[self._best_row_index]
        return {k: v for k, v in row.items() if not k.startswith("_")}

    def write_evaluations_csv(self, output_path: Optional[str] = None) -> str:
        if self._best_row_index is None:
            self.finalize_results()

        output_path = output_path or self.csv_output_path
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        fieldnames = ["qasm_path", "runtime", "fidelity", *self.optimization_params, "is_best"]
        with open(output_path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.get_evaluation_rows():
                writer.writerow(row)

        logger.info("Saved evaluation CSV to %s", output_path)
        return output_path

    def execute_with_params(self, params):
        params_dict = self.map_discrete(params)

        total_runtime = 0.0
        total_fidelity = 0.0

        for _ in range(self.num_evaluations):
            try:
                runtime, fidelity = self._run_backend_once(params_dict)
            except Exception as e:
                logger.warning(f"Execution failed for {params_dict}. Assigning high penalty. Error: {e}")
                self._record_evaluation(params_dict, runtime=1e6, fidelity=-1.0, objective_runtime=1e6)
                return 1e6, -1

            total_runtime += runtime
            total_fidelity += fidelity

        avg_runtime = total_runtime / self.num_evaluations
        avg_fidelity = total_fidelity / self.num_evaluations

        objective_runtime = avg_runtime
        if self.opt_method == "cmaes" and avg_fidelity < CMA_FIDELITY_ACCURACY:
            objective_runtime = 1e6

        self._record_evaluation(
            params_dict=params_dict,
            runtime=avg_runtime,
            fidelity=avg_fidelity,
            objective_runtime=objective_runtime,
        )

        logger.info(
            "Params: %s → Fidelity: %.5f, Avg Runtime: %.6f s",
            params_dict,
            avg_fidelity,
            avg_runtime,
        )

        return objective_runtime, avg_fidelity

    def optimize(self, max_generations=10, population_size=10):
        if self.opt_method == "cmaes":
            result = self._optimize_cmaes(max_generations, population_size)
        elif self.opt_method in ["moead", "nsga2"]:
            result = self._optimize_moo(max_generations, population_size)
        else:
            raise ValueError(f"Unsupported optimization method: {self.opt_method}")

        self.finalize_results()
        self.write_evaluations_csv(self.csv_output_path)
        return result

    def _optimize_cmaes(self, max_generations, population_size):
        x0 = [0.5] * len(self.optimization_params)
        sigma0 = 0.2

        def constraints(x):
            _, fidelity = self.execute_with_params(x)
            return [CMA_FIDELITY_ACCURACY - fidelity]

        cfun = cma.ConstrainedFitnessAL(lambda x: self.execute_with_params(x)[0], constraints)
        nh = cma.NoiseHandler(len(x0), [2, 5, 10])

        es = cma.CMAEvolutionStrategy(
            x0,
            sigma0,
            {
                "maxiter": max_generations,
                "popsize": population_size,
                "tolx": 1e-5,
                "tolfun": 1e-4,
            },
        )

        while not es.stop():
            solutions, fit_vals = es.ask_and_eval(cfun, evaluations=nh.evaluations)

            all_distances = []
            for sol in solutions:
                _, dist = self.map_discrete_with_encoding(sol)
                all_distances.append(dist)

            es = self.apply_margin_correction(es, solutions, all_distances, margin_threshold=0.1)
            es.tell(solutions, fit_vals)
            es.disp()

        best_params, _ = self.map_discrete_with_encoding(es.result.xbest)
        best_runtime = es.result.fbest
        print(f"\n**Final Best Parameters: {best_params}**")
        print(f"\n**Best Runtime Achieved: {best_runtime:.6f} s**")
        return best_params

    def _optimize_moo(self, max_generations, population_size):
        problem = QuantumOptimizationProblem(self)

        if self.opt_method == "moead":
            try:
                ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=24)
            except Exception as e:
                logger.error(f"Failed to generate reference directions for MOEA/D: {e}")
                raise RuntimeError("Reference direction generation failed for MOEA/D.")

            if ref_dirs is None or len(ref_dirs) == 0:
                raise ValueError("Reference directions for MOEA/D could not be generated. Try adjusting 'n_partitions'.")

            algorithm = MOEAD(ref_dirs=ref_dirs)
        else:
            crossover_operator = SBX(prob=0.9, eta=1)
            mutation_operator = PM(prob=1.0 / problem.n_var, eta=2)
            algorithm = NSGA2(pop_size=population_size, crossover=crossover_operator, mutation=mutation_operator)

        class GenerationLogger:
            def __init__(self, optimizer_name, max_gens):
                self.generation = 0
                self.optimizer_name = optimizer_name
                self.max_gens = max_gens

            def __call__(self, algorithm):
                self.generation += 1
                logger.info("Generation %s/%s completed for %s.", self.generation, self.max_gens, self.optimizer_name.upper())

        generation_callback = GenerationLogger(self.opt_method, max_generations)

        logger.info("Starting %s optimization with %s generations...", self.opt_method.upper(), max_generations)
        res = minimize(
            problem,
            algorithm,
            get_termination("n_gen", max_generations),
            verbose=True,
            callback=generation_callback,
        )

        self._save_pareto_front(res.F, res.X)
        return [self.map_discrete(solution) for solution in res.X]

    def _save_pareto_front(self, fitness_values, parameter_solutions):
        algo_name = self.opt_method.upper()
        filename = f"pareto_res_{algo_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        with open(filename, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Runtime", "Fidelity", "Parameters"])

            for fitness, params in zip(fitness_values, parameter_solutions):
                if not np.isfinite(fitness[0]) or not np.isfinite(fitness[1]):
                    continue
                runtime, neg_fidelity = fitness
                fidelity = -neg_fidelity
                writer.writerow([runtime, fidelity, self.map_discrete(params)])

        logger.info("Pareto front results saved to %s", filename)


class QuantumOptimizationProblem(Problem):
    def __init__(self, optimizer):
        super().__init__(n_var=len(optimizer.optimization_params), n_obj=2, xl=0, xu=1)
        self.optimizer = optimizer

    def _evaluate(self, x, out, *args, **kwargs):
        results = [self.optimizer.execute_with_params(xi) for xi in x]
        out["F"] = np.array([[runtime, -fidelity] for runtime, fidelity in results])
