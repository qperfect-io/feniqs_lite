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

import os
import json
import random
import copy
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
"""
Post-Processing: Quantum Simulator Elo Ranking
------------------------------------------
This script computes Elo ratings for various quantum simulators based on their performance
across multiple benchmark tasks. It processes simulation results stored as JSON files in a
predefined folder structure, filters valid candidate results based on fidelity (>= 0.99) and
execution time (<= 300 seconds), and aggregates the best performance for each simulator on
13 benchmark tasks - "twolocal" is excluded.

Key Components:
- **Folder Structure and Configuration**:  
  The script uses a mapping (simulator_base_dir) to define base directories for each simulator.
  A list of benchmark tasks is provided.
  Candidate dimensions (number of qubits) are specified in dimension_list.

- **Data Aggregation**:  
  The function get_aggregated_benchmark_result scans each task folder for JSON files following a 
  specific naming pattern, extracts the candidate dimension and run time, and filters results 
  based on the defined thresholds. It returns valid run time for each dimension (qubit number).
  If there is no results - replace by a fake run time = 1000 s.

- **Performance Comparison and Elo Rating**:  
  The function compare_perf is used to compare performance tuples between two simulators.
  The main simulation function, `simulate_elo_ranking`, runs multiple trials (configurable via `n_trials`)
  where, for each randomized benchmark order, pairwise comparisons are made. Elo ratings are updated
  simultaneously (to avoid order dependency), and a fixed penalty is applied for missing valid results.

- **Output and Visualization**:  
  After the simulation, the script computes and prints a final Elo ratings table, and generates
  a heatmap of pairwise win rates that is saved as "elo_heatmap.pdf".

Prerequisites:
- Python packages: os, json, random, math, copy, time, numpy, matplotlib, pandas.
- Simulation result files organized as JSON files in the directories defined by simulator_base_dir.

Result: ELO Table (printed in terminal) PDF file - "elo_heatmap.pdf"

Usage:
1. Adjust configuration parameters (folder paths, tasks, dimension_list, thresholds, etc.) as needed.
2. Run the script to execute the Elo simulation over the defined number of trials.
3. Analyze the printed Elo ratings table and review the generated win rate heatmap for insights.

"""

import os
import json
import random
import time
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Folder structure for each simulator.
simulator_base_dir = {
    "Qiskit-MPS": "results/qiskit",
    "QMatchaTea-MPS": "results/qmt",
    "MQT-DDS": "results/mqt",
    "Quimb-MPS": "results/quimb",
    "Quimb-TN": "results/quimb_nt",
    "MIMIQ-MPS": "results/mimiq",
    "Pyqrack": "results/pyqrack",
}

# Benchmark tasks (13 tasks; "twolocal" is excluded).
tasks = [ "ae", "ghz", "qft", "qftentangled", "graphstate", "qnn", 
          "qpeexact", "qpeinexact", "qwalk", "random", "realqmp", "su2rand", "wstate" ]

simulator_task_dirs = {
    sim: {
        task: os.path.join(base_dir, "realamp" if task == "realqmp" else task)
        for task in tasks
    }
    for sim, base_dir in simulator_base_dir.items()
}

simulators = list(simulator_base_dir.keys())
benchmarks = tasks  # Each task is treated as a benchmark


def get_aggregated_benchmark_results(simulator, benchmark):
    """
    Scan the corresponding folder for JSON files with names following the pattern:
         <benchmark>_ucx_<dim>_<...>.json

    For each file with a candidate dimension (>= 4), the function:
      - Reads the JSON file and extracts:
          - run_time: from metrics.total.min_rt
          - fidelity: from metrics.fidelity.median_rt
      - If fidelity >= 0.99 and run_time <= 300, the run time is taken;
        otherwise, a penalty value for run-time (1000) is used.
        
    Only the result corresponding to the highest candidate (qubit) number is kept.
    """
    best_dim = 0
    best_rt = 1000
    base_path = simulator_task_dirs[simulator][benchmark]
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Folder {base_path} does not exist")
    for filename in os.listdir(base_path):
        if not filename.endswith(".json"):
            continue  # Skip non-json files.
        parts = filename.split("_")
        if len(parts) < 3:
            continue
        try:
            candidate_dim = int(parts[2])
        except Exception as e:
            print(f"Error parsing dimension from {filename}: {e}")
            continue
        if candidate_dim < 4:
            continue
        file_path = os.path.join(base_path, filename)
        try:
            with open(file_path, "r") as f:
                result = json.load(f)
        except Exception as e:
            current_rt = 1000
            print(f"Error reading {file_path}: {e}")
        else:
            metrics = result.get("metrics", {})
            run_time = metrics.get("total", {}).get("min_rt", 9999)
            fidelity = metrics.get("fidelity", {}).get("median_rt", 0)
            if fidelity >= 0.99 and run_time <= 300:
                current_rt = run_time
            else:
                current_rt = 1000
        # Update if a higher candidate is found.
        if candidate_dim > best_dim:
            best_dim = candidate_dim
            best_rt = current_rt

    return (best_dim, best_rt)


def compare_perf(perfA, perfB):
    """
    Compares two performance tuples (candidate_dim, run_time).

    Rules:
      - The simulator with the higher qubit number wins.
      - If both simulators have the same dimension (qubit number),
        the simulator with the lower run time wins.
      - If both metrics are equal, the game is skipped.
    
    Returns:
         1.0 if A wins, 0.0 if A loses, or None if the game is skipped.
    """
    dim_A, rt_A = perfA
    dim_B, rt_B = perfB

    if dim_A > dim_B:
        return 1.0
    elif dim_A < dim_B:
        return 0.0
    else:
        if rt_A < rt_B:
            return 1.0
        elif rt_A > rt_B:
            return 0.0
        else:
            return None


def simulate_elo_ranking(n_trials=10000, k=32):
    INITIAL_RATING = 1200
    ratings_over_trials = {sim: [] for sim in simulators}

    # Pre-compute best performance for each simulator and each benchmark.
    aggregated_results = {sim: {} for sim in simulators}
    for sim in simulators:
        for bench in benchmarks:
            aggregated_results[sim][bench] = get_aggregated_benchmark_results(sim, bench)

    # Build tournament "game" list: each game corresponds to one benchmark.
    games = benchmarks[:]  # one game per benchmark
    random.shuffle(games)

    N = len(simulators)
    win_sum = np.zeros((N, N))
    win_count = np.zeros((N, N))
    start_time = time.time()
 
    for trial in range(1, n_trials + 1):
        ratings = {sim: INITIAL_RATING for sim in simulators}
        random.shuffle(games)

        for bench in games:
            # Get the pre-computed best performance for each simulator.
            performance = {sim: aggregated_results[sim][bench] for sim in simulators}

            # Compare every unique pair.
            for i in range(N):
                for j in range(i + 1, N):
                    sim_A = simulators[i]
                    sim_B = simulators[j]
                    outcome = compare_perf(performance[sim_A], performance[sim_B])
                    if outcome is None:
                        continue
                    win_sum[i, j] += outcome
                    win_count[i, j] += 1
                    win_sum[j, i] += (1 - outcome)
                    win_count[j, i] += 1
                    # Compute Elo updates.
                    E_A = 1 / (1 + 10 ** ((ratings[sim_B] - ratings[sim_A]) / 400))
                    E_B = 1 - E_A
                    ratings[sim_A] += k * (outcome - E_A)
                    ratings[sim_B] += k * ((1 - outcome) - E_B)

        for sim in simulators:
            ratings_over_trials[sim].append(ratings[sim])
        if trial % (n_trials // 10) == 0:
            print(f"Completed {trial} trials...")
    end_time = time.time()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds.")

    elo_avg = {sim: np.mean(ratings_over_trials[sim]) for sim in simulators}
    elo_std = {sim: np.std(ratings_over_trials[sim]) for sim in simulators}
    win_rate = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                win_rate[i, j] = np.nan
            elif win_count[i, j] > 0:
                win_rate[i, j] = win_sum[i, j] / win_count[i, j]
            else:
                win_rate[i, j] = 0.5
    return elo_avg, elo_std, win_rate, ratings_over_trials


if __name__ == "__main__":
    n_trials = 200000
    elo_avg, elo_std, win_rate, ratings_over_trials = simulate_elo_ranking(
        n_trials=n_trials, k=32
    )
    
    # Build and display the final Elo ratings table.
    elo_table = pd.DataFrame({
        "Simulator": simulators,
        "Elo Average": [elo_avg[sim] for sim in simulators],
        "Elo Std": [elo_std[sim] for sim in simulators]
    })
    elo_table.sort_values(by="Elo Average", ascending=False, inplace=True)
    print("\nFinal Elo Ratings Table:")
    print(elo_table)
    
    # Reorder win_rate matrix according to Elo ranking order.
    sorted_simulators = elo_table["Simulator"].tolist()
    sorted_indices = [simulators.index(sim) for sim in sorted_simulators]
    win_rate_sorted = win_rate[np.ix_(sorted_indices, sorted_indices)]

    # Plot and save the pairwise win rate heatmap.
    custom_cmap = LinearSegmentedColormap.from_list("custom", ["blue", "yellow", "green"])
    N_sim = len(simulators)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(win_rate_sorted, cmap=custom_cmap, vmin=0, vmax=1)
    ax.set_xticks(np.arange(N_sim))
    ax.set_yticks(np.arange(N_sim))
    ax.set_xticklabels(sorted_simulators, rotation=45, ha="right", fontsize=10, fontweight='bold')
    ax.set_yticklabels(sorted_simulators, fontsize=10, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Win Rate", fontsize=12)
   # ax.set_title("Pairwise Win Rate Heatmap (Ordered by Elo Rating)", fontsize=14, fontweight='bold')

    for i in range(N_sim):
        for j in range(N_sim):
            if not np.isnan(win_rate_sorted[i, j]):
                # Get the RGBA color for the current cell value:
                color = im.cmap(im.norm(win_rate_sorted[i, j]))
                r, g, b, _ = color
                # Calculate luminance:
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                # Use black text if the background is light; otherwise, use white:
                text_color = "black" if luminance > 0.5 else "white"
                ax.text(j, i, f"{win_rate_sorted[i, j]:.2f}", ha="center", va="center", 
                        color=text_color, fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig("elo_heatmap.pdf")
    plt.show()


