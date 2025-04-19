

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
import sys
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines

"""
Simulator Benchmark Analysis: Runtime vs. Qubits
-------------------------------------------------
"""

def darken_color(color, amount=0.7):
    """Return a darker shade of the given hex color."""
    color = color.lstrip('#')
    lv = len(color)
    rgb = tuple(int(color[i:i+lv//3], 16) for i in range(0, lv, lv//3))
    dark_rgb = tuple(max(0, int(c * amount)) for c in rgb)
    return '#' + ''.join(f'{v:02x}' for v in dark_rgb)


simulator_base_dir = {
    "Qiskit-MPS": "results/qiskit",
    "QMatchaTea-MPS": "results/qmt",
    "MQT-DDS": "results/mqt",
    "Quimb-MPS": "results/quimb",
    "Quimb-TN": "results/quimb_nt",
    "MIMIQ-MPS": "results/mimiq",
    "Pyqrack": "results/pyqrack",
}

tasks = [
    "ae", "ghz", "qft", "qftentangled", "graphstate", "qnn", 
    "qpeexact", "qpeinexact", "qwalk", "random", "realqmp", "su2rand", 
    "twolocal", "wstate"
]

SIMULATOR_TASK_DIRS = {
    sim: {
        task: os.path.join(base_dir, "realamp" if task == "realqmp" else task)
        for task in tasks
    }
    for sim, base_dir in simulator_base_dir.items()
}

# Qubits set
PREFERRED_QUBITS = [4, 8, 16, 24, 32, 64, 128, 256,  512, 1024]

# Define fixed runtime range across all plots.
EPSILON = 1e-3       # Lower bound for runtime (seconds)
MAX_RUNTIME = 400    # Upper bound for runtime (seconds)


for sim_label, task_dirs in SIMULATOR_TASK_DIRS.items():
    for task_name, folder in task_dirs.items():
        if not os.path.isdir(folder):
            print(f"Warning: Directory '{folder}' for simulator '{sim_label}' and task '{task_name}' does not exist. It will be skipped.")


points = []
for sim_label, task_dirs in SIMULATOR_TASK_DIRS.items():
    for task_name, folder in task_dirs.items():
        if not os.path.isdir(folder):
            print(f"Skipping missing folder: '{folder}' for simulator '{sim_label}' and task '{task_name}'.")
            continue
        json_files = glob.glob(os.path.join(folder, "*.json"))
        for jf in json_files:
            try:
                with open(jf, "r") as f:
                    jobj = json.load(f)
                nb_qubits = jobj.get("config", {}).get("nb_qubits")
                if nb_qubits is None:
                    continue
                fidelity = jobj.get("metrics", {}).get("fidelity", {}).get("median_rt")
                if fidelity is None or fidelity < 0.99:
                    continue
                runtime_val = jobj.get("metrics", {}).get("total", {}).get("min_rt")
                if runtime_val is None:
                    continue
                runtime_val = runtime_val if runtime_val > 0 else EPSILON
                if runtime_val > MAX_RUNTIME:
                    continue
          
                points.append((runtime_val, nb_qubits, sim_label, task_name))
            except Exception as e:
                print(f"Error processing file {jf}: {e}")

points = np.array(points, dtype=object)
df = pd.DataFrame(points, columns=["runtime", "qubits", "simulator", "algorithm"])
df["algorithm"] = df["algorithm"].astype("category")

sns.set_context("paper", font_scale=0.8)
sns.set_style("whitegrid")

palette = {
    "Qiskit-MPS": "#1f77b4",
    "QMatchaTea-MPS": "#2ca02c",
    "MQT-DDS": "#17becf",
    "Quimb-MPS": "#d62728",
    "Quimb-TN": "#9467bd",
    "MIMIQ-MPS": "#ff7f0e",
    "Pyqrack": "#e377c2"
}
marker_shapes = {
    "Qiskit-MPS": "o",
    "QMatchaTea-MPS": "s",
    "MQT-DDS": "D",
    "Quimb-MPS": "^",
    "Quimb-TN": "v",
    "MIMIQ-MPS": "h",
    "Pyqrack": "*"
}

output_dir = "fig"
os.makedirs(output_dir, exist_ok=True)

x_min_fixed = min(PREFERRED_QUBITS)       
x_max_fixed = max(PREFERRED_QUBITS)    
y_min_fixed = EPSILON                     
y_max_fixed = MAX_RUNTIME                


x_vals_fixed = np.logspace(np.log10(x_min_fixed), np.log10(x_max_fixed), 100)

lin_const_fixed = y_min_fixed / x_min_fixed
quad_const_fixed = y_min_fixed / (x_min_fixed**2)
y_lin_fixed = lin_const_fixed * x_vals_fixed
y_quad_fixed = quad_const_fixed * x_vals_fixed**2

algorithms = df["algorithm"].cat.categories
algorithms = df["algorithm"].cat.categories
for alg in algorithms:
    df_alg = df[df["algorithm"] == alg]
    # Increase figure size for a large picture
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each simulator's data: x = number of qubits, y = runtime
    for sim, group in df_alg.groupby("simulator"):
        group = group.sort_values("qubits")
        sim_color = palette.get(sim, "black")
        ax.plot(
            group["qubits"],
            group["runtime"],
            marker=marker_shapes.get(sim, "o"),
            color=sim_color,
            linewidth=2,
            markersize=4,
            markerfacecolor=sim_color,
            markeredgecolor=darken_color(sim_color),
            markeredgewidth=0.5,
            label=sim
        )
    
    # Plot the reference lines (linear and quadratic)
    ax.plot(x_vals_fixed, y_lin_fixed, linestyle="--", color="grey", linewidth=1.5, zorder=0)
    ax.plot(x_vals_fixed, y_quad_fixed, linestyle=":", color="grey", linewidth=1.5, zorder=0)
    
    # Set log-scales and fixed axis limits
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_min_fixed, x_max_fixed)
    ax.set_ylim(y_min_fixed, y_max_fixed)
    
    # Set tick labels (smaller font size)
    ax.set_xticks(PREFERRED_QUBITS)
    ax.set_xticklabels(PREFERRED_QUBITS, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    
    # Set axis labels with smaller font size
    ax.set_xlabel("Number of Qubits [log scale]", fontsize=14)
    ax.set_ylabel("Run time (seconds) [log scale]", fontsize=14)
    
    # Create a smaller legend. Adjust bbox_to_anchor to allocate ~35% of the figure width for the legend.
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1., 0.5),
              fontsize=12, title="Simulator", title_fontsize=12, frameon=True)
    
    fig.tight_layout()
    filename = os.path.join(output_dir, f"{alg}_runtime.pdf")
    fig.savefig(filename, format="pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)


selected_tasks = {"wstate", "qnn", "random", "realqmp"}
df_filtered = df[df["algorithm"].isin(selected_tasks)].copy()


df_filtered["task_display"] = df_filtered["algorithm"].replace({"realqmp": "realamp"})
tasks_to_plot = ["wstate", "realamp", "qnn", "random"]



fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
axes = axes.flatten()

for ax, task in zip(axes, tasks_to_plot):
    df_task = df_filtered[df_filtered["task_display"] == task]
    for sim, group in df_task.groupby("simulator"):
        group = group.sort_values("qubits")
        sim_color = palette.get(sim, "black")
        ax.plot(
            group["qubits"],     # x: number of qubits
            group["runtime"],    # y: runtime
            marker=marker_shapes.get(sim, "o"),
            color=sim_color,
            linewidth=2,
            markersize=4,
            markerfacecolor=sim_color,
            markeredgecolor=darken_color(sim_color),
            markeredgewidth=0.5,
            label=sim
        )
    # Plot the "fixed" reference lines in every subplot.
    ax.plot(x_vals_fixed, y_lin_fixed, linestyle="--", color="grey", linewidth=1.5, zorder=0)
    ax.plot(x_vals_fixed, y_quad_fixed, linestyle=":", color="grey", linewidth=1.5, zorder=0)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='x', labelsize=12)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_min_fixed, x_max_fixed)
    ax.set_ylim(y_min_fixed, y_max_fixed)
    ax.set_xticks(PREFERRED_QUBITS)
    ax.set_xticklabels(PREFERRED_QUBITS, fontsize=12)
    ax.set_ylabel("Run time (seconds) [log scale]", fontsize=14)
    ax.set_xlabel("Number of Qubits [log scale]", fontsize=14)
    ax.set_title(f"Task: {task}", fontsize=12)
    ax.grid(True, which="both", ls="--", lw=0.3,  alpha=0.5)

# Create a common legend 
simulators_sorted = sorted(df_filtered["simulator"].unique())
handles = [
    mlines.Line2D([], [],
                  color=palette[sim],
                  marker=marker_shapes[sim],
                  linestyle='-',
                  markersize=4,
                  markerfacecolor=palette[sim],
                  markeredgecolor=darken_color(palette[sim]),
                  markeredgewidth=0.5,
                  label=sim)
    for sim in simulators_sorted
]

lin_handle = mlines.Line2D([], [], color="grey", linestyle="--", linewidth=1.5, label="Linear ~ nqubits")
quad_handle = mlines.Line2D([], [], color="grey", linestyle=":", linewidth=1.5, label="Quadratic ~ nqubits²")
handles.extend([lin_handle, quad_handle])


fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.05),
           ncol=4, fontsize=12, title="Simulator", title_fontsize=12,  frameon=True)

fig.tight_layout(rect=[0, 0, 1, 0.95])
filename = os.path.join(output_dir, "four_panel_runtime.pdf")
fig.savefig(filename, format="pdf", dpi=600, bbox_inches="tight")
plt.close(fig)

