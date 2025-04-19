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
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.colors as mc

"""
Post-processing: Complexity Heatmap Generation for Benchmark Circuits
-----------------------------------------------------

This script processes simulation result data from various quantum simulators to evaluate
circuits complexity according different simualtors. It reads JSON files stored in a predefined
folder structure, extracts key metrics (number of qubits, runtime, and fidelity), and determines
whether a simulation case is solved based on specified runtime and fidelity thresholds.

Key Components:
- Folder Structure:
  The script uses the simulator_task_dirs dictionary to define result directories for each simulator.
  Each simulator's folder contains subdirectories for various tasks.
  
- Data Extraction:
  For each simulator and task, the script reads JSON files to extract:
    - `nb_qubits`: Number of qubits used in the simulation.
    - `Runtime`: The runtime of the simulation.
    - `Fidelity`: The fidelity of the simulation result.
  A simulation is marked as solved if the runtime is below MAX_RUNTIME =  300 s and fidelity is at least MIN_FIDELITY = 0.99.

- Categorization and Aggregation:
  Simulators are grouped into categories (MPS, TN, DDS) defined in simulator_categories.
  The script aggregates data by algorithm and qubit count, counting the number of valid solvers per category
  and identifying the simulator with the best (minimum) runtime.

- Complexity Determination:
  Based on the percentage of simulators that solved each (Algorithm, Qubits) case, a complexity level is
  assigned ("Easy", "Medium", "Hard", or "Very Hard").

- Visualization:
  A heatmap is generated using Seaborn to display the complexity levels across different algorithms and qubit counts.
  The heatmap is annotated with detailed results including solver counts and the best performing simulator.

Prerequisites:
- Python packages: os, glob, json, numpy, matplotlib, seaborn, pandas, matplotlib.colors.
- A proper directory structure with JSON simulation result files.

Result: PDF file - "complexity_heatmap.png"

Usage:
1. Verify that the result directories specified in simulator_task_dirs are correctly set up - adjust if needed.
2. Adjust configuration parameters (e.g., preferred_qubits, MAX_RUNTIME, MIN_FIDELITY) as needed.
3. Run the script to generate and save the complexity heatmap as "complexity_heatmap.png".

Notes:
- The heatmap's color scheme is based on predefined base colors for each complexity level.
"""

# Convert number of qubits for qwalk to original number, by reducing ancilla
def normalize_qwalk(qubits):
    if 0 <= qubits < 6:
        return 4
    elif 6 <= qubits < 14:
        return 8
    elif 14 <= qubits < 30:
        return 16
    elif 30 <= qubits < 46:
        return 24
    elif 46 <= qubits < 63:
        return 32
    elif 63 <= qubits < 126:
        return 64
    elif 126 <= qubits < 254:
        return 128
    elif 254 <= qubits < 286:
        return 256
    else:
        return qubits



# Define the folder with results structure for each simulator and task.
simulator_task_dirs = {
    "Qiskit": "results/qiskit/",
    "QMatchaTea": "results/qmt/",
    "MQT-DDS": "results/mqt/",
    "Quimb-MPS": "results/quimb/",
    "Quimb-TN": "results/quimb_nt/",
    "MIMIQ": "results/mimiq/",
    "Pyqrack": "results/pyqrack/"
}

# List of benchmark tasks to consider.
simulator_categories = {
    "MPS": ["Qiskit", "QMatchaTea", "Quimb-MPS", "MIMIQ"],
    "NT": ["Quimb-TN", "Pyqrack"],
    "DDS": ["MQT-DDS"]
}

# Define benchmarks to delete.
delete_bench = ["twolocal"]

preferred_qubits = [4, 8, 16, 24, 32, 64, 128, 256, 512, 1024]
metric_name = "total"
rt_key = "min_rt"
MAX_RUNTIME = 300
MIN_FIDELITY = 0.99

data = []
for sim, base_folder in simulator_task_dirs.items():
    for task in os.listdir(base_folder):
        task_folder = os.path.join(base_folder, task)
        if not os.path.isdir(task_folder):
            continue
        
        for file in glob.glob(os.path.join(task_folder, "*.json")):
            try:
                with open(file, "r") as f:
                    jobj = json.load(f)
                nb_qubits = jobj.get("config", {}).get("nb_qubits", None)
                runtime = jobj.get("metrics", {}).get(metric_name, {}).get(rt_key, None)
                fidelity = jobj.get("metrics", {}).get("fidelity", {}).get("median_rt", None)
                
                if nb_qubits and runtime is not None and fidelity is not None:
                    solved = runtime < MAX_RUNTIME and fidelity >= MIN_FIDELITY
                    data.append([sim, task, nb_qubits, runtime, solved])
            except Exception as e:
                print(f"Error processing {file}: {e}")

# Convert to DataFrame
df = pd.DataFrame(data, columns=["Simulator", "Algorithm", "Qubits", "Runtime", "Solved"])
df = df[~df["Algorithm"].isin(delete_bench)]
df.loc[df["Algorithm"] == "qwalk", "Qubits"] = df.loc[df["Algorithm"] == "qwalk", "Qubits"].apply(normalize_qwalk)


def classify_simulator(sim):
    for category, sims in simulator_categories.items():
        if sim in sims:
            return category
    return "Unknown"

df["Category"] = df["Simulator"].apply(classify_simulator)

# Ensure valid counts per category
def count_valid_solvers(group):
    counts = {cat: 0 for cat in simulator_categories.keys()}
    for sim in group["Simulator"].unique():
        category = classify_simulator(sim)
        if category in counts:
            counts[category] += 1
    return pd.Series(counts)

category_counts = df[df["Solved"]].groupby(["Algorithm", "Qubits"]).apply(count_valid_solvers).reset_index()

# Compute the best runtime simulator per (Algorithm, Qubit)
best_runtime_sim = df[df["Solved"]].groupby(["Algorithm", "Qubits"]).apply(lambda x: x.loc[x["Runtime"].idxmin(), "Simulator"] if not x.empty else "None").reset_index(name="Best_Simulator")
category_counts = category_counts.merge(best_runtime_sim, on=["Algorithm", "Qubits"], how="left")

# Compute the percentage of simulators solving each (Algorithm, Qubit) case
category_counts["Solved_Rate"] = category_counts[list(simulator_categories.keys())].sum(axis=1) / len(simulator_task_dirs)
category_counts["Complexity"] = "Very Hard"
category_counts.loc[category_counts["Solved_Rate"] >= 0.1, "Complexity"] = "Hard"
category_counts.loc[category_counts["Solved_Rate"] >= 0.3, "Complexity"] = "Medium"
category_counts.loc[category_counts["Solved_Rate"] >= 0.6, "Complexity"] = "Easy"

# Generate formatted strings for each (Algorithm, Qubits) pair
category_counts["Result"] = category_counts.apply(
    lambda row: f"{row['MPS']}/{row['NT']}/{row['DDS']}\n* {row['Best_Simulator']}",
    axis=1
)
# Pivot DataFrame for heatmap
success_summary_pivot = category_counts.pivot(index="Algorithm", columns="Qubits", values="Complexity").fillna("Very Hard")
annotation_pivot = category_counts.pivot(index="Algorithm", columns="Qubits", values="Result").fillna("0/0/0/0\n* None")

# Ensure preferred_qubits order
success_summary_pivot = success_summary_pivot.reindex(columns=preferred_qubits, fill_value="Very Hard")
annotation_pivot = annotation_pivot.reindex(columns=preferred_qubits, fill_value="0/0/0/0\n* None")

plt.figure(figsize=(15, 8))

# Define base colors
complexity_levels = ["Easy", "Medium", "Hard", "Very Hard"]
base_colors = {
    "Easy": "#2ca02c",
    "Medium": "#ff7f0e",
    "Hard": "#d62728",
    "Very Hard": "#800000"
}

# Convert complexity values to numeric for visualization
complexity_mapping = {k: i for i, k in enumerate(complexity_levels)}
numeric_matrix = success_summary_pivot.replace(complexity_mapping)

# Apply color palette based on complexity levels
cmap = sns.color_palette([base_colors[k] for k in complexity_levels], as_cmap=True)

ax = sns.heatmap(
    numeric_matrix,
    annot=annotation_pivot,
    fmt="",
    linewidths=0.5,
    cmap=cmap,
    cbar=False,
    mask=numeric_matrix.isnull(),
    annot_kws={"size": 12, "weight": "bold"}     
)

# Set axis labels with fontsize=14
ax.set_xlabel("Qubits", fontsize=14)
ax.set_ylabel("Algorithm", fontsize=14)

# Increase tick label sizes to 14
ax.tick_params(axis='x', labelsize=14, rotation=45)
ax.tick_params(axis='y', labelsize=14, rotation=0)

plt.tight_layout()
plt.savefig("complexity_heatmap.png", dpi=300)
plt.show()

