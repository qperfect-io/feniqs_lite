
<div style="display: flex; align-items: center; justify-content: space-between;">
  <h1 style="margin: 0;">feniqs_lite: Optimizer of Quantum Backend Hyperparameters</h1>
  <img src="../assets/logo.png" alt="Feniqs Lite Logo" style="width: 100px;">
</div>

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Optimizer](#running-the-optimizer)
- [Important Notes](#important-notes)
- [Command-line Arguments](#command-line-arguments)
- [Optimization Methods](#optimization-methods)
  - [CMA-ES (Single-Objective Optimization)](#cma-es-single-objective-optimization)
  - [MOEA/D (Multi-Objective Optimization)](#moead-multi-objective-optimization)
  - [NSGA-II (Multi-Objective Optimization)](#nsga-ii-multi-objective-optimization)
- [Configuration File (YAML)](#configuration-file-yaml)
- [Output Files](#output-files)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)
- [ML-based MIMIQ-MPS Hyperparameter Recommender](#ml-based-mimiq-mps-hyperparameter-recommender)

## Overview
**FeniqsOptimizer**  is designed to automatically tune the hyperparameters for Matrix Product State (MPS) simulators to achieve optimal performance. By systematically adjusting settings such as bond dimensions, truncation thresholds, and other simulation-specific parameters, it aims to balance accuracy with computational efficiency. This enables researchers to benchmark and compare simulator performance reliably, ensuring high-fidelity results while minimizing run time.

**FeniqsOptimizer**  contains three following optimization algorithms:

- **CMA-ES (Covariance Matrix Adaptation Evolution Strategy)** - Single-objective optimization minimizing runtime while ensuring defined fidelity (e.g., >= 0.9999).
- **MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition)** - Optimizes runtime and fidelity as independent objectives.
- **NSGA-II (Non-dominated Sorting Genetic Algorithm II)** - Another multi-objective optimization method for runtime and fidelity.


- **Other features**
    - Ensures noise reduction by special CMA-ES noise handler and running each evaluation multiple times and averaging results.
    - Supports quantum backend configuration through a YAML file (`yaml/optimizator.yaml`)
    - Logs each generation's progress and saves optimization results automatically.
---

## Installation

Ensure you have the required dependencies installed:

```bash
pip install cma pymoo 
```

Additionally, ensure that **feniqs_lite** is properly installed and accessible.

---

## Usage

### Running the Optimizer

The optimizer can be executed using the command-line interface:

```bash
python feniqs_optimizer/run_optimizer.py --backend QiskitAerCpu \
               --qasm paper_data/ae/ae_ucx_8.qasm \
               --mirror paper_data/ae/ae_ucx_8.mirror \
               --config yaml/optimizator.yaml \
               --method cmaes \
               --gens 10 \
               --pop 10 \
               --num_eval 3
```
List of possible backends: `QiskitAerCpu` (MPS),  `QmatchateaCpu` (MPS), `QuimbMpsCpu` (MPS), `MimiqJuliaCpu` (MPS).

## Important Notes

1. **Backends:**  
  Currently, hyperparameter optimization is only supported for 4 backends: `QiskitAerCpu` (MPS),  `QmatchateaCpu` (MPS), `QuimbMpsCpu` (MPS), `MimiqJuliaCpu` (MPS).
   
  Optimization of hyperparameters on other backends is not supported at the current version.

2. **Configuration:**  
   If you have selected a backend for optimization, make sure to set the `"install"` option to `true` in the file `yaml/venv_deps.yaml` before proceeding.

3. **MIMIQ Simulator Access:**  
   To use MIMIQ, you must have access to the local version of the simulator. Note that MIMIQ-MPS is a commercial simulator.




### Command-line Arguments

| Argument     | Description                                                          |
|--------------|----------------------------------------------------------------------|
| `--backend`  | The quantum backend to be optimized (e.g., `QiskitAerCpu`).          |
| `--qasm`     | Path to the main QASM file to be executed.                           |
| `--mirror`   | Path to the mirrored QASM file (for fidelity calculation).           |
| `--config`   | Path to the YAML configuration file for optimization parameters.     |
| `--method`   | Optimization method (`cmaes`, `moead`, `nsga2`).                     |
| `--gens`     | Number of generations (iterations) for optimization.                 |
| `--pop`      | Population size for evolutionary algorithms.                         |
| `--num_eval` | Number of evaluations per parameter set (averaging to reduce noise). |

---

## Optimization Methods

### CMA-ES (Single-Objective Optimization)

- **Objective:** Minimize execution time while ensuring defined fidelity.
- **Penalty:** If fidelity is below than defined, the runtime is set to `1e6` to discard the solution.
- **Noise Handling:** Uses multiple evaluations per solution (`num_eval`) to reduce randomness.

### MOEA/D (Multi-Objective Optimization)
- **Objectives:** Minimize execution time and maximize fidelity as independent objectives.
- **Reference Directions:** Uses the **Das-Dennis method** to generate diverse search directions.
- **Diversity Control:** Adjust `n_partitions` for better Pareto front coverage.

### NSGA-II (Multi-Objective Optimization)
- **Objectives:**  Minimize execution time and maximize fidelity as independent objectives.
- **Crossover & Mutation:** Uses **SBX (Simulated Binary Crossover)** and **Polynomial Mutation**.
- **Crowding Distance Sorting:** Ensures diverse solutions along the Pareto front.

---

## Configuration File (YAML)

The optimizer loads backend-specific parameter ranges from a YAML configuration file - `yaml/optimizator.yaml`. 

Here is an example for a configuration of Qiskit backend:

```yaml
backends:
  QiskitAerCpu:
    params:
      matrix_product_state_max_bond_dimension: [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
      opt_level: [1, 2, 3]
      matrix_product_state_truncation_threshold: [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
      mps_sample_measure_algorithm: ['mps_probabilities', 'mps_apply_measure']
    optimization_params:
      - matrix_product_state_max_bond_dimension
      - opt_level
      - matrix_product_state_truncation_threshold
      - mps_sample_measure_algorithm
```
This file defines **valid discrete values** for each parameter that the optimizer can select for each backend.

---

## Output Files

The optimizer saves results in CSV format:

- **CMA-ES Results:**
  ```csv
  Generation, Best Runtime, Best Parameters
  0, 0.11, "{...}"
  1, 0.10, "{...}"
  ...i
  Final, 0.09, "{...}"
  ```

- **Pareto Front Results (MOEA/D & NSGA-II):**
  ```csv
  Runtime, Fidelity, Parameters
  0.09, 0.999, "{...}"
  0.11, 0.998, "{...}"
  ```

---

## Troubleshooting

1. **MOEA/D generates duplicate solutions**
   - Increase `n_partitions` for better diversity.
   - Ensure proper mutation/crossover rates.

2. **NSGA-II solutions do not spread well**
   - Increase `pop_size` to allow better coverage.
   - Adjust mutation probability.


---

## Future Improvements
- Parallel execution for large-scale simulations.

---

## ML-based MIMIQ-MPS Hyperparameter Recommender
This tool trains a machine-learning model that predicts optimal **MIMIQ–MPS**
simulation parameters directly from an **OpenQASM 2.0** circuit.

It supports:
- automatic feature extraction from QASM  
- learning from labeled benchmark data  
- prediction for new circuits  
 

The predicted parameters are: `bond_dim`, `ent_dim`, `trunc_eps`, `meth`, `fuse`, `perm`.

### Input Format for Training
The training CSV file is summarizes the results obtained from evolutionary optimization step. 
It must contain the following columns: qasm_path, bond_dim, ent_dim, trunc_eps, meth, fuse, perm

Each row represents a benchmarked quantum circuit with its optimal MIMIQ–MPS
hyperparameters.

| Column     | Meaning                                    |
|------------|--------------------------------------------|
| `qasm_path`| Path to qasm file (training circuit)       |
| `bond_dim` | MPS bond dimension (4 - 4096)              |
| `ent_dim`  | Entanglement dimension (4 - 4096)          |
| `trunc_eps`| truncation threshold (1e-1 -1e-12)         |
| `meth`     | MPS method (`vmpoa`, `dmpo`)               |
| `fuse`     | Whether gate fusion is enabled (T/F)       |
| `perm`     | Whether qubit permutation is enabled  (T/F)|

### Workflow Overview
Recommended workflow:

1. Train the model on known benchmark circuits   
2. Inspect extracted features for a given circuit  
3. Predict hyperparameters for new circuits   


#### Step 1 — Train the Model

Run:

```bash
python get_mps_param.py train --labels data/labels.csv --outdir mimiq_model

```
This command:
- loads all QASM files listed in labels.csv
- extracts numerical features from each circuit
- trains Random Forest models for all parameters
- stores the trained model in mimiq_model/
- It also builds an exact-match cache so training circuits are always predicted
exactly.

`mimiq_model` is the name of the folder where the models will be saved. You can name it differently if needed.

#### Step 3 - Predict Hyperparameters
To predict hyperparameters for a new circuit:
```bash
python get_mps_param.py predict --modeldir mimiq_model --qasm twolocalrandom_ucx_100.qasm

```
Just change name `twolocalrandom_ucx_100.qasm` to name of your circuit.

#### Step 3 - Inspect Features
To see what the model “sees” for a circuit:

```bash
python get_mps_param.py features --qasm data/train/graphstate_ucx_100.qasm

```
This prints a JSON object containing extracted features such as: number of qubits, number of gates, two-qubit gate fraction, interaction graph density and etc.
If predictions look strange, always inspect the extracted features.

