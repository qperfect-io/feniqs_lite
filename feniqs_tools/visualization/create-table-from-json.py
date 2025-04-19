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
import pandas as pd

"""
Summarizing Results: Maximum Qubit Achieved by each Simulator on each Benchmark Circuit 
-----------------------------------

This script generates a table summarizing the maximum number of qubits achieved by each simulator for every test circuit. 
It leverages simulation results saved in JSON files to compile and display this scalability metric.

Prerequisites:
- JSON files containing simulation results.

Configuration:
- simulator_task_dirs:
  Update the paths to the result directories for each simulator. If you add a new simulator, include its path here.

- benchmark_tasks:
  Modify this list to add new test circuits or exclude existing ones from the results table as needed.

Results: "benchmark_max_qubits.csv"  

Usage:
1. Update Configurations:
   Edit the 'simulator_task_dirs' and 'benchmark_tasks' in the script to match your setup.
2. Run the Script:
   Execute the script. It will read the JSON result files and generate a table summarizing the maximum number of 
   qubits achieved by each simulator across the selected test circuits.

Notes:
- Ensure your JSON simulation result files are correctly formatted and exist.
- Double-check paths and circuit selections to avoid any discrepancies in the results.
"""

# Define the folder with results structure for each simulator and task.
simulator_task_dirs = {
    "Qiskit-MPS": "results/qiskit",
    "MIMIQ-MPS": "results/mimiq",
    "QMatchaTea-MPS": "results/qmt",
    "Quimb-MPS": "results/quimb",
    "Quimb-TN": "results/quimb_nt",
    "MQT-DDS": "results/mqt",
    "Pyqrack": "results/pyqrack"
}

# List of benchmark tasks to consider.
benchmark_tasks = [
    "ae", "ghz", "graphstate", "qft", "qftentangled", "qnn", 
    "qpeexact", "qpeinexact", "qwalk", "random", 
    "realamp", "su2rand", "twolocal", "wstate"
]

# Initialize a list to store the results (one row per benchmark task).
results = []

# Iterate over benchmark tasks.
for i, task in enumerate(benchmark_tasks, start=1):
    row = {"n": i, "alg": task}
    
    # For each simulator, check its folder for this task.
    for sim_label, base_dir in simulator_task_dirs.items():
        task_dir = os.path.join(base_dir, task)
        best_qubits = 0  # Default if no valid file is found.
        
        if os.path.exists(task_dir):
            json_files = glob.glob(os.path.join(task_dir, "*.json"))
            # Build a list of (nb_qubits, fidelity) for each JSON file.
            valid_entries = []
            for jf in json_files:
                try:
                    with open(jf, "r") as f:
                        data = json.load(f)
                    nb_qubits = data.get("config", {}).get("nb_qubits", 0)
                    fidelity = data.get("metrics", {}).get("fidelity", {}).get("median_rt", 0.0)
                    # Accept only files with fidelity >= 0.99.
                    if fidelity >= 0.99:
                        valid_entries.append(nb_qubits)
                except Exception as e:
                    print(f"Error processing {jf}: {e}")
            if valid_entries:
                best_qubits = max(valid_entries)
        else:
            print(f"Warning: Directory '{task_dir}' does not exist for simulator '{sim_label}'.")
        
        row[sim_label] = best_qubits
    results.append(row)

# Create a DataFrame from the results.
df = pd.DataFrame(results)

# Optional: print the table.
print(df.to_string(index=False))

# Save the table as a CSV file (semicolon-delimited).
csv_filename = "benchmark_max_qubits.csv"
df.to_csv(csv_filename, index=False, sep=";")
print(f"Saved results to '{csv_filename}'.")

