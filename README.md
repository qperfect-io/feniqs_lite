

<div style="display: flex; align-items: center; justify-content: space-between;">
  <h1 style="margin: 0;">feniqs_lite: Quantum Simulator Benchmarking Framework</h1>
  <img src="assets/logo.png" alt="Feniqs Lite Logo" style="width: 100px;">
</div>

## Table of Contents
- [Motivation](#motivation)
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Design](#design)
  - [Daemon – feniqs_daemon](#daemon--feniqs_daemon)
  - [Library – feniqs_lib](#library--feniqs_lib)
  - [Tools – feniqs_tools](#tools--feniqs_tools)
  - [Optimizer of Quantum Backend Hyperparameters – feniqs_optimizer](#optimizer-of-quantum-backend-hyperparameters)
- [Input API](#input-api)
- [Benchmark Metrics](#benchmark-metrics)
- [Benchmark Suite](#benchmark-suite)
- [Configuration](#configuration)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
   - [PostgreSQL Installation](#postgresql-installation)
- [Adding New Benchmark Cases](#adding-new-benchmark-cases)
- [Additional Notes](#additional-notes)

## Motivation 
Quantum emulators are indispensable for developing and testing quantum algorithms before deploying them on real hardware. Various open‐source and commercial emulators employ different numerical methods to simulate quantum circuits, and as these techniques continue to evolve and push performance limits, systematic benchmarking becomes crucial.

Different emulation methods often exhibit distinct performance characteristics — some are highly optimized for specific problem classes, while others are designed for broader applications. By benchmarking quantum emulators, we can gain valuable insights into their run time efficiency, fidelity, and scalability across diverse application cases and computing environments (e.g., CPU and GPU). These insights are essential for researchers and developers as they work to fine-tune and optimize quantum computing technologies for a wide range of applications.

## Project Overview

**feniqs_lite** is a transparent, user-friendly benchmarking tool for quantum emulators, which could eventually be extended to quantum computers in the future. It simplifies the execution of benchmarks circuits in OpenQASm 2.0 format across diverse quantum emulators via unified API, while presenting the results in a clear format (PostgreSQL and/or JSON files). This tool allows users evaluate and compare emulator performance using three metrics: total run time, accuracy, and scalabilit and to make informed decisions for optimizing quantum computing technologies.

## Key Features

- **Multi-Backend Support via Plugins:**
  - **Cirq:** StateVector (CPU, GPU)
  - **Qiskit:** StateVector, MPS (CPU, GPU)
  - **Mimiq:** StateVector, MPS (CPU)
  - **Pennylane:** StateVector (CPU, GPU)
  - **Qulacs:** StateVector (CPU, GPU)
  - **Yao:** StateVector (CPU, GPU)
  - **ProjectQ:** StateVector (CPU)
  - **QMatchTea:** MPS (CPU, GPU)
  - **MQT-DDS:** Decision Diagrams (CPU)
  - **Pyqrack:** Factorized Ket (CPU)
  - **Qrisp:** Sparse Matrix (CPU)
  - **Quimb:** MPS, NT (CPU)

A flexible plugin system allows for the easy integration of new backends as needed, 
so the presented list of backends can be extended.


- **Unified API:**  
  A single, consistent API for all supported backends.
  It means that users can run simulations on any circuit without needing deep knowledge of the underlying simulator or advanced programming skills. 
  In order to run simulation, there are only three general steps:

   - Activate the Simulator:  
      Open the `yaml/venv_deps.yaml` configuration file and set the `"install"` option to `true` for the desired simulator.

   - Configure Tuning Parameters (Optional):  
      If needed, adjust the tuning parameters for the selected simulator in the `yaml/config_backend.yaml` file.

   - Run the Daemon:  
      Start the daemon as a Python script, and it will automatically execute benchmarks across the active backends.

This streamlined process ensures that even users with limited expertise in individual simulators can efficiently run benchmarks, as all supported backends are accessible through the same simple, unified interface.



- **OpenQASM 2.0 Support:**  
  Compatibility with the OpenQASM 2.0 format.

- **Isolated Backend Management:**  
  Each backend operates within its own virtual environment to prevent dependency conflicts.

- **Benchmark Data Storage:**  
  Results are stored in a PostgreSQL database and as JSON files for flexible access and analysis.

- **Automated Benchmark Management:**  
  Continuous monitoring and automatic execution of new benchmark files.

- **Visualization & Post-Processing Tools:**  
  Scripts for data visualization and post-processing of benchmark results.


## Design

**feniqs_lite** is comprised of the following components:

### Daemon – feniqs_daemon

The Daemon component (`feniqs_daemon/main.py`) allows for background/foreground monitoring and benchmark task execution. 

Key Features:
   - Automatically executes benchmark circuits in OpenQASM 2.0 format across all supported backends.
   - Save results in PostgreSQL database and JSON files.
   - Monitors a directory for changes.

For more details, please, see README in [README in `feniqs_daemon/`](./feniqs_daemon/README.md).

### Library – feniqs_lib
A core components (`feniqs_lib`), including backends and their plugins.

### Tools – feniqs_tools

1. Visualisation - five following scripts:
   - A table of scalability - `create-table-from-json.py`
   - A radar chart for scalability comparing - `radar-scalability.py`
   - A heat map of circuit complexity - `heatmap-qc-complexity.py`
   - A heat map for ELO ratings - `heatmap-elo-nt-sim.py`
   - Detailed plots showing run time versus qubit number for each circuit and a pannel with 4 plots (run time versus qubit number) covering four circuits with different complexity levels - `plot4panel.py`

For more details, please, see README in [README in `feniqs_tools/visualization`](./feniqs_tools/visualization/README.md).

2. Database handling - `helper_db.py`

For more details, please, see README in [README in `feniqs_tools/db`](./feniqs_tools/db/README.md).


### Optimizer of Quantum Backend Hyperparameters 
Optimizer (`feniqs_optimizer`) is designed to automatically tune the hyper parameters for Matrix Product State (MPS) simulators to achieve optimal performance. By systematically adjusting settings such as bond dimensions, truncation thresholds, and other simulation-specific parameters, it aims to balance accuracy with computational efficiency. This enables researchers to benchmark and compare simulator performance reliably, ensuring high-fidelity results while minimizing run time.

For more details, please, see README in [README in `feniqs_optimizer`](./feniqs_optimizer/README.md).

## Input API
The benchmark input API is OpenQASM 2.0 format. All benchmark problems/algorithms have to be presented as circuits in this format.

## Benchmark Metrics
Each benchmark test case should be run at least 3-4 times (when feasible), and metrics are defined in terms of mean, average, min and max values and standard deviations. The defined metrics include:
- run time:
  - suppoted profiling:
   - loading time;
   - execution times;
   - sampling time;
  - total time;
- fidelity.


## Benchmark Suite
The benchmark suite can be download from Zenodo: https://doi.org/10.5281/zenodo.15220683.

This benchmark suite was used in the paper "Comparative Benchmarking of Utility-Scale Quantum Emulators". It generated by the open-source MQTBench library - a standardized
framework designed to assess the performance of quantum computing platforms by providing
scalable quantum circuits in OpenQASM 2.0. The suite covers a range of commonly
used quantum circuit primitives and application-orientated tasks, from which we identify 13 circuit classes capable of scaling beyond 100 qubits. We extended the circuit sizes beyond the 130-qubit constraint of the default MQTBench suite, and (by using their library) generated circuits from 4 to 1024 qubits. To ensure compatibility across all selected emulators, each algorithm was transpiled to a minimal gate set (comprising u and cx gates) and sanitized by removing gates with negligible rotation angles and normalizing large angles modulo 4𝜋.
 
## Configuration

**feniqs_lite** uses three main YAML configuration files to manage the database, virtual environments, and backend settings:

1. `yaml/db_config.yaml`:
   - This file configures the database settings for storing benchmark results.
   - It is not recommended to modify this file.

2. `yaml/venv_deps.yaml`:
   - This file defines the dependencies for each backend and specifies whether to install the backend.
   - Users can set the `install` option to `true` or `false` for each backend, depending on whether they want to use it. 
   - Other options in this file are predefined and generally should not be modified.

3. `yaml/config_backend.yaml`:
   - This file contains configuration parameters for all backends.
   - Parameters include `device_type`, `fusion`, `precision`, `seed`, and more specific options depending on the device type, such as `mps_parallel_threshold` or `tensor_network_num_sampling_qubits`.
   - Each backend has its own set of configuration options. We support all major options, which are already defined in this YAML file.
   - Feel free to modify these parameters to suit your needs, but do not change the option names.


## Prerequisites
Python 3.10 or later

## Installation 

1. **Clone the Repository:**
First, clone the **feniqs_lite** of the repository to your local machine. This will create a local copy of the project.
In order to do it, please type the code below in the terminal window:

```bash
   git clone https://github.com/qperfect-io/feniqs_lite.git
```
2. **Navigate to the Project Directory:**
Change into the directory of the cloned repository. This is where all your project files are located:  

```bash
   cd feniqs_lite
```
3. **Create a Virtual Environment:**
Create a Python virtual environment to isolate your project dependencies. This ensures that any libraries you install for this project won’t interfere with other Python projects on your system.
Please note, that this virtual environement is only for the poject, but not for backends (simulators).
Their virtual evironements (one for each selected in yaml config backend) will be created automatically by a manager of backend's virtual enviroments.

```bash
   python -m venv feniqs_env
```
4. **Activate the Virtual Environment:**
Activate the virtual environment so that any Python commands you run will use the environment's Python interpreter and installed packages.
For Linux type OS, please use the following command:

```bash
   source feniqs_env/bin/activate
```

5. **Install the Project Dependencies:**
Install the project dependencies in editable mode. This will install all necessary packages listed in the project's setup while allowing you to make changes to the codebase that take effect without reinstalling the package.
```bash
   pip install -e .
```

### PostgreSQL Installation
All parts of **feniqs_lite** use PostgreQSL database, so PostgreSQL Server has to be installed on your local computer.
1. **Check if PostgreSQL is Installed:**
In order to check if if's already installed on your system, please tun the following command in the terminal window:
```bash
   psql --version
```
If PostgreSQL is not installed, you will see an error message and you need to follow the next steps below.
2. **Install PostgreSQL Server:**
```bash
sudo apt-get update
sudo apt-get install postgresql postgresql-contrib
```
3. **Start  PostgreSQL Service:**
PostgreSQL usually starts automatically after installation. To check if PostgreSQL is running, use:
```bash
   sudo systemctl status postgresql
```
If it is not running, you can start it with:
```bash
   sudo systemctl start postgresql
```
4. **Create a new SQL user and Database:** 
You need to create a new SQL user if one doesn't exist (if it is an installation). This can be done with the following commands:

Go to PostgreSQL Promt with user 'postgres'
```
sudo -u postgres psql
```

Create a new user named 'feniqs_user' with password 'nopassword' in PostgreSQL promt
```sql
CREATE ROLE feniqs_user WITH LOGIN PASSWORD 'nopassword';
```

Create a new database in PostgreSQL promt
```sql
CREATE DATABASE feniqs OWNER feniqs_user;
```

Exit from PostgreSQL promt
```sql
\q
```
Return to the root folder of the project and run the SQL script to set up the database schema in shell:
```bash
   psql -U feniqs_user -d feniqs -f db/timescale_benchmarks.sql
```


## Adding New Benchmark Cases
Create `data` directory (if does not exist yet) in the root folder of this project and copy the QASM files to it. It will be an observed directory for benchmark QASM files by default. The daemon will automatically detect and process these new files, and the results will be saved to a PostgreSQL database and/or JSON.

## Additional Notes
**feniqs_lite** was used in the paper "Comparative Benchmarking of Utility-Scale Quantum Emulators".




