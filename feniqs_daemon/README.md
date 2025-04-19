<div style="display: flex; align-items: center; justify-content: space-between;">
  <h1 style="margin: 0;">feniqs_lite: Daemon</h1>
  <img src="../assets/logo.png" alt="Feniqs Lite Logo" style="width: 100px;">
</div>

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Input Options Details](#input-options-details)
- [Configuration Files](#configuration-files)
- [Usage](#usage)
  - [Running Daemon](#running-daemon)
  - [Stopping Daemon](#stopping-daemon)
- [Troubleshooting](#troubleshooting)

## Overview

**Feniqs Daemon** is a linux service for benchmarking different quantum backends and can work in console application mode. 
It is the main module responsible for continuously monitoring benchmark input files (i.e., OpenQASM circuits) and processing them using registered quantum backend plugins. 

## Key Features
- **Multi-Backend Access:**  
  Supports 12 quantum frameworks, enabling benchmarking.

- **Configurable Backend Support:**  
  All supported quantum backends are specified in `yaml/venv_deps.yaml`. Users can enable one, two, or all backends by setting the `"install"` option to `true` for the desired emulators. Detailed configuration options for each backend are provided in `yaml/config_backend.yaml`.

- **Isolated Virtual Environments:**  
  Each backend is installed in its own virtual environment, preventing dependency conflicts. This isolation is critical for maintaining system stability and reliability, especially when different backends require different versions of the same libraries.

- **Foreground Mode Support:**  
  The daemon can run in the foreground, displaying messages directly in the console. To stop the daemon in this mode, simply press `Ctrl+C`.

- **Background Mode Support:**  
  The daemon can also run in the background, continuously monitoring directories and executing tasks without user intervention. To stop the daemon in background mode, locate the process ID (PID) using `ps -A | grep feniqs_daemon` and terminate it with `kill -s INT <PID>`.

- **Watch Directory:**  
  The system monitors a specified folder for changes, automatically processing new QASM files with the selected quantum backends.

- **Timeout Handling:**  
  Each benchmark simulation has a defined timeout (300 sec by default). If a simulation exceeds this time limit, it is automatically terminated and a new simulation is initiated.

- **Result Saving:**  
  Benchmark results are saved to a database and/or as JSON files, ensuring that performance data is securely stored and easily accessible for further analysis.

## Installation
Please, see Installation in main [README](../README.md).

## Input Options Details

All input options are defined via command-line arguments. Key options include:

| Argument       | Description                                                                                                          |
|----------------|----------------------------------------------------------------------------------------------------------------------|
| `--logfile`    | Path to the log file. Logs are saved to this file if provided; otherwise, they are printed to stderr.                |
| `--level`      | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). This option controls the verbosity of log messages.           |
| `--foreground` | Run the daemon in the foreground to print messages directly to the console.                                          |
| `--watchfolder`| Folder path to monitor for new benchmark files (typically containing QASM files).                                    |
| `--db`         | Enable database usage (default: true). Set to false if a PostgreSQL database is not available.                       |
| `--json`       | Save benchmark results in JSON format (default: false).                                                              |
| `--mf`         | Enable mirror fidelity functionality (default: false).                                                               |
| `--timeout`    | Set the simulation timeout in seconds (default: 300 seconds).                                                        |
| `--runs`       | Number of runs per benchmark to average out noise (default: 3).                                                      |
| `--incl_first` | Option related to including the first iteration (warmup run) of benchmark.                                           |

These input options empower users to precisely control the daemon’s behavior, including:

- The simulation timeout duration,
- The choice to enable or disable database usage,
- Running the daemon in either background or foreground mode,
- Configuring output formats for saved results.


## Configuration Files

The daemon relies on several YAML configuration files to manage backend settings and virtual environments:
- **Backend Configuration:**  
  `yaml/config_backend.yaml` defines parameters and valid ranges for each backend.
- **Virtual Environment Dependencies:**  
  `yaml/venv_deps.yaml` contains settings to control the installation of backend-specific environments.
**Note:** the configs file must be modified before running Daemon. 

## Usage
The simple command to get all available options options of Daemon:
```sh
python feniqs_daemon/main.py --help
```
**Note:** Before running benchmarks, please ensure that your OpenQASM 2.0 benchmark suite is located in the `data` directory (the default directory). You can use any circuits in OpenQASM format. If you do not have your own benchmark suite, please refer to the **Benchmark Suite** section in the main README for instructions.


### Running Daemon
To run Daemon in foreground mode, you can use:
```sh
python feniqs_daemon/main.py --foreground --timeout=300 --json=true --db=false --runs=10
```

To run  Daemon in background mode, you can use:
```sh
python feniqs_daemon/main.py
```
The running daemon will automatically detect and process these new OpenQASM files, and the results will be saved to a PostgreSQL database and/or JSON files.

### Stopping Daemon
To stop the daemon running in foreground mode, simply use `Ctrl+C` in the terminal where the daemon is running. This will send an interrupt signal to the process, causing it to terminate gracefully.

If you are running the daemon in the background (i.e., without the --foreground option), you will need to find the process ID (PID) and kill it. Here are the steps:

- Find the PID: You can find the PID of the daemon using ps:
```sh
ps -A | grep feniqs_daemon
```
This command will return the PID of the running daemon process.
- Kill the Process: Once you have the PID, you can stop the daemon using the `kill` command:
```sh
kill -s INT <PID>
```

## Troubleshooting

- **PID File Issues:**  
  The daemon checks for existing PID files and will refuse to start if a stale PID file is detected.
- **Process Killing:**  
  If there are orphan processes or issues with hanging processes, the daemon uses dedicated functions to kill them.
- **"inotify watch limit reached"**
If you encounter the error "inotify watch limit reached" after running the daemon, please run the following command to increase the inotify watch limit:
```sh
echo 524288 | sudo tee /proc/sys/fs/inotify/max_user_watches
```






























