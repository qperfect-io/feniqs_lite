#
# Copyright © 2024 QPerfect. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cma
import yaml
import numpy as np
import logging
import csv
import os
from datetime import datetime
from collections import OrderedDict
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2

from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
CMA_FIDELITY_ACCURACY = 0.99
class FeniqsOptimizer:
    """
    FeniqsOptimizer integrates different optimization algorithms with quantum backends.
    Supported optimizers:
        - CMA-ES (minimizes runtime while ensuring fidelity >= CMA_FIDELITY_ACCURACY)
        - MOEA/D (minimizes runtime and maximizes fidelity as independent objectives)
        - NSGA-II (minimizes runtime and maximizes fidelity as independent objectives)
    """

    def __init__(self, backend_name, qasm_file, mirror_qasm_file, plugin_manager, 
                 config_path="yaml/optimizator.yaml", opt_method="cmaes", num_evaluations=3):
        """
        Initialize the optimizer with backend-specific parameters.

        :param backend_name: Name of the quantum backend (e.g., "qiskit - QiskitAerCpu").
        :param qasm_file: Path to the main QASM file.
        :param mirror_qasm_file: Path to the mirrored QASM file (for fidelity calculation).
        :param plugin_manager: Feniqs PluginManager instance to run backends.
        :param config_path: Path to the YAML config defining valid backend parameters got optimization.
        :param opt_method: Optimization algorithm ("cmaes", "moead", "nsga2").
        :param num_evaluations: Number of times to run the backend per evaluation (to average out noise).
        """
        self.backend_name = backend_name
        self.qasm_file = qasm_file
        self.mirror_qasm_file = mirror_qasm_file
        self.plugin_manager = plugin_manager
        self.opt_method = opt_method.lower()
        self.num_evaluations = num_evaluations  # Used to average noisy results (for NSGA-2 & MOEA/D)
        self.evaluation_cache = OrderedDict()

        # Load backend-specific configuration
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        if backend_name not in config["backends"]:
            raise ValueError(f"Backend `{backend_name}` is not defined in {config_path}")

        backend_config = config["backends"][backend_name]
        self.valid_params = backend_config["params"]
        self.optimization_params = backend_config["optimization_params"]

        # Ensure valid optimization method
        if self.opt_method not in ["cmaes", "moead", "nsga2"]:
            raise ValueError(f"Invalid optimization method `{self.opt_method}`. Choose from: 'cmaes', 'moead', 'nsga2'.")

    def map_discrete(self, x):
        """
        Simple function to map continuous values from optimization space to the nearest valid discrete values.

        :param x: List of continuous values from optimization.
        :return: Dictionary of mapped discrete parameters.
        """
        discrete_params = {}
        for i, param in enumerate(self.optimization_params):
            valid_values = self.valid_params.get(param, [])
            if valid_values:
                discrete_params[param] = valid_values[int(np.clip(np.round(x[i] * (len(valid_values) - 1)), 0, len(valid_values) - 1))]
        return discrete_params
    
    def map_discrete_with_encoding(self, x):
        """
        Maps continuous values from optimization space to the nearest valid discrete values.
        This is done by selecting the nearest point in the given set.
        Here, we record the difference (or distance) for potential margin correction.

        :param x: List of continuous values from optimization. 
        :return: A tuple: (mapped_parameters, distances) - Dictionary of mapped discrete parameters and distances for margin correction
        """
        discrete_params = {}
        distances = {}    
        for i, param in enumerate(self.optimization_params):
            valid_values = self.valid_params.get(param, [])
            if valid_values:
                # Index on the continuous scale:
                idx = int(np.clip(np.round(x[i] * (len(valid_values) - 1)), 0, len(valid_values) - 1))
                mapped_value = valid_values[idx]
                discrete_params[param] = mapped_value
                # Distance between x[i] (scaled) and the chosen index (normalized difference)
                distances[param] = abs(x[i] * (len(valid_values) - 1) - idx)
        return discrete_params, distances

    def apply_margin_correction(self, es, solutions, distances, margin_threshold=0.1):
        """
        Applies margin correction to the covariance matrix for discrete parameters.
        For each parameter, the method computes the standardized distance and the marginal
        probability. If the marginal probability falls below the margin_threshold, the covariance
        matrix is adjusted to increase the likelihood of sampling neighboring discrete values.
        
        :param es: The CMA-ES object with attributes es.C and es.sigma.
        :param solutions: Candidate solutions used for iterating over distances.
        :param distances: List of dictionaries with distances for each parameter from the encoding.
        :param margin_threshold: The probability threshold below which correction is applied.
        :return: Updated CMA-ES object with corrected covariance matrix.
        """
        from scipy.stats import norm
        dim = len(self.optimization_params)
        sigma = es.sigma  # current step size
        for _, dist in zip(solutions, distances):
            for param, d in dist.items():
                param_index = self.optimization_params.index(param)
                # Only consider applying correction if d > margin_threshold to avoid unnecessary updates.
                if d > margin_threshold:
                    # Compute the standard deviation along the dimension.
                    sigma_j = np.sqrt(es.C[param_index, param_index]) * sigma
                    if sigma_j == 0:
                        continue  # avoid division by zero
                    # Standardize the distance.
                    d_std = d / sigma_j
                    # Compute the marginal probability for this parameter.
                    p = norm.cdf(-d_std)
                    # If the probability is below the threshold, calculate correction.
                    if p < margin_threshold:
                        gamma_alpha = norm.ppf(1 - margin_threshold)
                        if d_std != 0:
                            factor = (d_std**2 - gamma_alpha**2) / (d_std**2 * gamma_alpha**2)
                        else:
                            factor = 0
                        # Create a unit vector for the parameter's direction.
                        xi = np.zeros(dim)
                        xi[param_index] = 1.0
                        # Update the covariance matrix.
                        es.C += factor * np.outer(xi, xi)
        return es

    def execute_with_params(self, params):
        """
        Executes the quantum simulation with given parameters **multiple times** and averages the results.

        :param params: List of parameters in continuous form.
        :return: (Avg Runtime, Avg Fidelity)
        """
        params_dict = self.map_discrete(params)

        total_runtime = 0
        total_fidelity = 0

        for _ in range(self.num_evaluations):  # Running multiple times to average noisy results
            try:
                metrics, _ = self.plugin_manager.run_backend(
                    self.backend_name, self.qasm_file, nb_shots=1000, **params_dict
                )
                runtime = metrics["total"]["avg_rt"]
                fidelity = metrics["fidelity"]["avg_rt"]
            except Exception as e:
                logger.warning(f"Execution failed for {params_dict}. Assigning high penalty. Error: {e}")
                return 1e6, -1  # Assign worst case values if execution fails

            total_runtime += runtime
            total_fidelity += fidelity

        avg_runtime = total_runtime / self.num_evaluations
        avg_fidelity = total_fidelity / self.num_evaluations

        # **Only for CMA-ES: Penalize low-fidelity solutions**
        if self.opt_method == "cmaes" and avg_fidelity < CMA_FIDELITY_ACCURACY:
            avg_runtime = 1e6  

        # Print evaluated parameters and results
        logger.info(f"**Params: {params_dict} → Fidelity: {avg_fidelity:.5f}, Avg Runtime: {avg_runtime:.2f} s**")

        return avg_runtime, avg_fidelity

    def optimize(self, max_generations=10, population_size=10):
        """
        Runs the selected optimization algorithm.

        :param max_generations: Number of iterations.
        :param population_size: Population size per generation.
        :return: Optimized parameters as a dictionary.
        """
        if self.opt_method == "cmaes":
            return self._optimize_cmaes(max_generations, population_size)
        elif self.opt_method in ["moead", "nsga2"]:
            return self._optimize_moo(max_generations, population_size)

    def _optimize_cmaes(self, max_generations, population_size):
        """
        CMA-ES optimization (minimizes runtime while enforcing fidelity constraint).
        Incorporates sample encoding and margin correction steps, and saves per-generation
        results to a CSV file.
        """
        x0 = [0.5] * len(self.optimization_params)  # Start in the middle
        sigma0 = 0.2  # Initial search radius

        def constraints(x):
            _, fidelity = self.execute_with_params(x)
            return [CMA_FIDELITY_ACCURACY - fidelity]

        cfun = cma.ConstrainedFitnessAL(lambda x: self.execute_with_params(x)[0], constraints)
        nh = cma.NoiseHandler(len(x0), [2, 5, 10])  # Evaluations per stage

        es = cma.CMAEvolutionStrategy(x0, sigma0, {
            "maxiter": max_generations,
            "popsize": population_size,
            "tolx": 1e-5,
            "tolfun": 1e-4,
        })

        # Prepare to store generation results in a CSV file
        results = []
        filename = f"cmaes_res_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        with open(filename, 'a+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Generation", "Best Runtime", "Best Parameters"])

        gen = 0
        while not es.stop():
            solutions, fit_vals = es.ask_and_eval(cfun, evaluations=nh.evaluations)
            
            # Apply the new sample encoding and record distances for margin correction.
            mapped_solutions = []
            all_distances = []
            for sol in solutions:
                mapped, dist = self.map_discrete_with_encoding(sol)
                mapped_solutions.append(mapped)
                all_distances.append(dist)
            
            # Apply margin correction based on the recorded distances.
            es = self.apply_margin_correction(es, solutions, all_distances, margin_threshold=0.1)
            
            es.tell(solutions, fit_vals)
            
            # Save best parameters from the current generation.
            best_params, _ = self.map_discrete_with_encoding(es.result.xbest)
            best_runtime = es.result.fbest
            results.append((gen, best_runtime, best_params))
            
            with open(filename, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([gen, best_runtime, best_params])
            
            gen += 1
            es.disp()

        # Final best solution.
        best_params, _ = self.map_discrete_with_encoding(es.result.xbest)
        best_runtime = es.result.fbest
        results.append(("Final", best_runtime, best_params))

        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Final", best_runtime, best_params])

        print(f"\n**Final Best Parameters: {best_params}**")
        print(f"\n**Best Runtime Achieved: {best_runtime:.2f} s**")
        
        return best_params


    def _optimize_moo(self, max_generations, population_size):
        """
        Multi-objective optimization (MOEA/D, NSGA-II).
        Minimizes runtime and maximizes fidelity as independent objectives.

        ### MOEA/D and NSGA-II Parameter Explanation

        #### MOEA/D Parameters:
        MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition), 
        where different objectives (e.g., minimizing runtime while maximizing fidelity) are optimized simultaneously.

        - **ref_dirs (`get_reference_directions("das-dennis", 2, n_partitions=12)`)**:
            - Defines **reference directions** for decomposition-based optimization.
            - `"das-dennis"` refers to the **Das and Dennis method**, which distributes reference points in the objective space.
            - `2`: Specifies that we have **two objectives** (runtime and fidelity).
            - `n_partitions=12`: Controls the **granularity** of reference directions. Higher values mean finer divisions, more diversity.
              If MOEA/D fails due to bad reference directions, try increasing/decreasing `n_partitions`.

        - **Alternative reference direction methods**:
            - `"uniform"`: Generates uniformly distributed reference directions.
            - `"energy"`: Uses an energy-based approach for better performance in some cases.

        - **How to Choose the Right Reference Directions?**
            - If **solutions are poorly spread**, **increase** `n_partitions` to improve diversity.
            - If **too many solutions cluster together**, **reduce** `n_partitions`.

        - **Performance Indicators for MOEA/D (`indicator="eps"` or `"hv"`)**:
            - **"eps" (Epsilon Indicator)**:
              - Measures how much one solution must be improved to **dominate another**.
              - Works well when we need **fine control over convergence**.
            - **"hv" (Hypervolume Indicator)**:
              - Measures the **size of the dominated space** in objective space.
              - Good for problems where Pareto fronts are complex and need wider exploration.

        #### NSGA-II Parameters:
        NSGA-II (Non-dominated Sorting Genetic Algorithm) is an **elitist, fast sorting genetic algorithm** that does not require 
        decomposition like MOEA/D. Instead, it uses **Pareto dominance** and crowding distance sorting.

        - **pop_size (population size)**:
            - Defines the **number of candidate solutions** in each generation.
            - A larger population **increases diversity** but **slows down convergence**.
            - Typically, **higher population sizes (50-200)** work better for **complex multi-objective problems**.
        
         - **Crossover Operators (Recombination)**
          - `"SBX"` (Simulated Binary Crossover): `SBX(prob=0.9, eta=15)`
            - `prob=0.9` → Crossover probability.
            - `eta=15` → Distribution index (higher values create offspring closer to parents).

        - **Mutation Operators**
          - `"Polynomial Mutation"`: `PM(prob=1.0/n_var, eta=20)`
            - `prob=1.0/n_var` → Each variable has `1/n_var` probability to mutate.
            - `eta=20` → Controls mutation spread (higher values lead to smaller changes).

      
        - **When to Use MOEA/D vs. NSGA-II?**
            - MOEA/D if understand problem structure is clear and can design good reference directions.
            - NSGA-II if the problem is complex and needs general Pareto-based optimization.
            - MOEA/D is faster when properly tuned, but NSGA-II can be more robust for diverse problems.
        """
        problem = QuantumOptimizationProblem(self)

        # Ensure reference directions are properly generated for MOEA/D
        if self.opt_method == "moead":
            try:
                ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=24)
            except Exception as e:
                logger.error(f"Failed to generate reference directions for MOEA/D: {e}")
                raise RuntimeError("Reference direction generation failed for MOEA/D.")
            
            if ref_dirs is None or len(ref_dirs) == 0:
                raise ValueError("Reference directions for MOEA/D could not be generated. Try adjusting 'n_partitions'.")

            algorithm = MOEAD(
                ref_dirs=ref_dirs
            )
        else:
            crossover_operator = SBX(prob=0.9, eta=1)
            mutation_operator = PM(prob=1.0/problem.n_var, eta=2)
            algorithm = NSGA2(pop_size=population_size, crossover=crossover_operator, mutation=mutation_operator)

        # Callback function to log each generation
        class GenerationLogger:
            def __init__(self, optimizer_name, max_gens):
                self.generation = 0
                self.optimizer_name = optimizer_name
                self.max_gens = max_gens

            def __call__(self, algorithm):
                self.generation += 1
                logger.info(f" Generation {self.generation}/{self.max_gens} completed for {self.optimizer_name.upper()}.")

        # Instantiate the generation logger
        generation_callback = GenerationLogger(self.opt_method, max_generations)

        # Run multi-objective optimization with generation logging
        logger.info(f" Starting {self.opt_method.upper()} optimization with {max_generations} generations...")
        res = minimize(
            problem, algorithm, get_termination("n_gen", max_generations),
            verbose=True, callback=generation_callback
        )

        # Save all Pareto front solutions
        self._save_pareto_front(res.F, res.X)

        # Return the full Pareto front instead of just one solution
        return [self.map_discrete(solution) for solution in res.X]

    def _save_pareto_front(self, fitness_values, parameter_solutions):
        """
        Saves only the final Pareto front for MOEA/D and NSGA-II.
        """
        algo_name = self.opt_method.upper()  # To distinguish MOEA/D vs NSGA-II results
        filename = f"pareto_res_{algo_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Runtime", "Fidelity", "Parameters"])

            for fitness, params in zip(fitness_values, parameter_solutions):
                if not np.isfinite(fitness[0]) or not np.isfinite(fitness[1]):  # Avoid NaN/inf values
                    continue
                runtime, neg_fidelity = fitness
                fidelity = -neg_fidelity  # Restore fidelity to positive value
                writer.writerow([runtime, fidelity, self.map_discrete(params)])

            logger.info(f" Pareto front results saved to {filename}")

   

class QuantumOptimizationProblem(Problem):
    """
    Wrapper for MOEA/D and NSGA-II.
    This problem is multi-objective:
    - Objective 1: Minimize Runtime
    - Objective 2: Maximize Fidelity (converted to minimization)
    """
    def __init__(self, optimizer):
        super().__init__(n_var=len(optimizer.optimization_params), n_obj=2, xl=0, xu=1)
        self.optimizer = optimizer

    # def _evaluate(self, x, out, *args, **kwargs):
    #     """
    #     Evaluation function for multi-objective optimization.
    #     - Runtime should be minimized.
    #     - Fidelity should be maximized, so we use `-fidelity` to convert it to minimization.
    #     - Solutions with fidelity < 0.99 are penalized by adding a constraint.
    #     """
    #     results = [self.optimizer.execute_with_params(xi) for xi in x]
        
    #     # Extract runtime and fidelity
    #     runtimes = [r for r, f in results]
    #     fidelities = [f for r, f in results]

    #     # Convert fidelity to minimization (-fidelity)
    #     out["F"] = np.column_stack([runtimes, -np.array(fidelities)])

    #     # Constraint: Fidelity should be >= 0.99 (Negative values mean constraint violation)
    #     out["G"] = np.array([0.9999 - f for f in fidelities])
    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluation function for multi-objective optimization.
        - Runtime should be minimized.
        - Fidelity should be maximized, so we use `-fidelity` to convert it to minimization.
        """
        results = [self.optimizer.execute_with_params(xi) for xi in x]
        
        # Convert fidelity to minimization (-fidelity)
        out["F"] = np.array([[runtime, -fidelity] for runtime, fidelity in results])

