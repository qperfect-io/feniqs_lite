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

"""
The **FeniqsOptimizer** integrates 3 optimization algorithms with quantum backends to find the best execution parameters of backends
for quantum circuits by optimizing execution runtime while ensuring high fidelity of simulation result.

Example to run:

python feniqs_optimizer/run_optimizer.py --backend QiskitAerCpu
                                         --qasm data/random_ucx_120.qasm
                                         --mirror data/random_ucx_120.mirror
                                         --config yaml/optimizator.yaml
                                         --method cmaes
                                         --gens 10
                                         --pop 10
                                         --num_eval 3

"""
from random import shuffle
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import csv
import re
from feniqs_lib.managers.singleton_plugin_manager import get_plugin_manager
from feniqs_lib.managers.plugin_manager import PluginManager
import argparse
from feniqs_optimizer import FeniqsOptimizer


def main():
    """
    Main function to run the optimizer with the chosen algorithm.
    Supports both single-objective (CMA-ES) and multi-objective (MOEA/D, NSGA-II) optimizations.
    """

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run quantum backend optimizer.")
    parser.add_argument("--backend", type=str,
                        default="QiskitAerCpu", help="Quantum backend name")
    parser.add_argument("--qasm", type=str, help="QASM files folder")
    parser.add_argument(
        "--config", type=str, default="yaml/optimizator.yaml", help="Path to optimizer config")
    parser.add_argument("--method", type=str, choices=[
                        "cmaes", "moead", "nsga2"], default="cmaes", help="Optimization method")
    parser.add_argument("--gens", type=int, default=10,
                        help="Number of generations")
    parser.add_argument("--pop", type=int, default=10, help="Population size")
    parser.add_argument("--num_eval", type=int, default=3,
                        help="Number of evaluations of fitness function (for avoiding a noise effect)")

    args = parser.parse_args()

    # Validate QASM file existence
    if not os.path.exists(args.qasm):
        raise FileNotFoundError(f"QASM file {args.qasm} not found!")

    # Initialize plugin manager (from FENIQS)
    plugin_manager = get_plugin_manager()
    plugin_manager.register_all_plugins()
    files_seen = []

    database = f"{args.backend}_{args.method}_db.csv"
    new_db = not os.path.exists(database)
    if not new_db:
        with open(database, 'r') as csv_file:
            reader = csv.reader(csv_file)
            first_ite = True
            for row in reader:
                if first_ite:
                    first_ite = False
                    continue
                files_seen.append(row[0])
        print(" Here are all the files alreday handled:")
        print(files_seen)

    directory = args.qasm
    reg = r".*\.qasm$"

    all_files = []

    # flatten the files
    for root, _, files in os.walk(directory):
        for filename in files:
            if re.search(reg, filename):
                file_path = os.path.join(root, filename)
                if file_path in files_seen:
                    continue
                mirror_path = file_path[:-4] + "mirror"
                if not os.path.exists(mirror_path):
                    Warning(f"The file {filename} won't be used because no file {mirror_path} was found")
                    continue
                
                all_files.append((root, filename))
            
            

    shuffle(all_files)

    for root, filename in all_files:
        if re.search(reg, filename):
            file_path = os.path.join(root, filename)
            mirror_path = file_path[:-4] + "mirror"
            print(f"---------------------------------{filename}---------------------------------------------")
            print(f"\n Running optimizer with {args.method.upper()} on backend {args.backend}...\n")

            # Create and run optimizer
            optimizer = FeniqsOptimizer(
                backend_name=args.backend,
                qasm_file=file_path,
                mirror_qasm_file=mirror_path,
                plugin_manager=plugin_manager,
                config_path=args.config,
                opt_method=args.method,
                num_evaluations=args.num_eval
            )

            best_params = optimizer.optimize(
                max_generations=args.gens, population_size=args.pop)

            print(f"\n**Best Parameters Found:** {best_params}")

            if new_db:
                with open(database, 'a+', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["qasm_path"] + list(best_params.keys()))
                new_db = False

            with open(database, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([file_path] + list(best_params.values()))
            
            files_seen.append(file_path)
                    





    



if __name__ == "__main__":
    main()
