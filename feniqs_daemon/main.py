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
#
import os
import sys
import time
import json
import logging
import signal
import csv
import yaml
import errno
import psutil
import setproctitle

from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from daemon import DaemonContext
try:
    from daemon.pidfile import PIDLockFile
except ImportError:
    from daemon.pidlockfile import PIDLockFile

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import feniqs_lib.tools.constants as constants
from feniqs_lib.tools.constants import GlobalConfig
from feniqs_lib.tools import welcome
from feniqs_lib.managers.plugin_manager import PluginManager
from feniqs_lib.managers.venv_manager import VenvManager
from feniqs_lib.managers.singleton_plugin_manager import get_plugin_manager
from feniqs_lib.tools.logger_api import config_logger


from feniqs_daemon.tools.input_arguments import parse_options, LOG_FILE, BACKEND_CONFIG, VENV_CONFIG, version, QASM_FOLDER

PID_FILE = "/tmp/feniqs_daemon.pid"
PID_FILE = os.path.abspath(PID_FILE)


# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler(LOG_FILE),
    logging.StreamHandler()
])
logger = logging.getLogger(__name__)

# Signal handler for graceful shutdown
def signal_handler(signum, frame):
    logger.info(f"Received signal {signum}, shutting down...")
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)
        logger.info(f"Deleted PID file: {PID_FILE}")
    sys.exit(0)

# This function is required to define the subprocess to kill after timeout - if not it will be zombie in os
def kill_process_by_name_or_command(process_name_starts_with=None, command_starts_with = None, timeout = 1):
    victims = []
    try:
        # Iterate over all running processes
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                proc_name = proc.info['name']
                proc_cmdline = proc.info['cmdline']
                name_matches = process_name_starts_with and proc_name and proc_name.startswith(process_name_starts_with)
                cmdline_matches = command_starts_with and proc_cmdline and len(proc_cmdline) > 0 and proc_cmdline[0].startswith(command_starts_with)
                if name_matches and cmdline_matches:
                    victims.append(proc)
                    sys.stderr.write(f"Killing process {proc.info['pid']} ({proc.info['name']})\n")
                    proc.kill()  # Try to kill the process
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        n = 0
        while victims:
            for proc in victims[:]:
                if not proc.is_running():
                    sys.stderr.write(f"Process {proc.pid} terminated successfully\n")
                    victims.remove(proc)
            if victims:
                n += 1
                if n > timeout:
                    raise RuntimeError(f"Failed to kill processes {[proc.pid for proc in victims]} after {timeout} seconds")
                time.sleep(1)
    except Exception as e:
        logger.error(f"Unexpected error while killing processes: {e}")

def kill_orphans_process(parent_pid, timeout = 1):
    victims = []
    try:
        parent = psutil.Process(parent_pid)
        for child in parent.children(recursive=True):
            victims.append(child)
            sys.stderr.write(f"Killing orphan process {child.pid} from parent {parent_pid}\n")
            try:
                child.kill()
            except Exception as e:
                logger.error(f"Failed to kill process {child.pid}: {e}")
        n = 0
        while victims:
            for child in victims[:]:
                if not child.is_running():
                    sys.stderr.write(f"Process {child.pid} terminated successfully\n")
                    victims.remove(child)
            if victims:
                n += 1
                if n > timeout:
                    raise RuntimeError(f"Failed to kill orphan processes {[child.pid for child in victims]} after {timeout} seconds")
                time.sleep(1)
    except psutil.NoSuchProcess:
        logger.warning(f"Parent process {parent_pid} no longer exists.")
    except Exception as e:
        logger.error(f"Unexpected error while killing orphan processes: {e}")

# Timeout handler for simulation timeout
def timeout_handler(signum, frame):
    raise TimeoutError("Simulation exceeded the time limit of 300 seconds.")

# Save results to a unique CSV file for each QASM file
def save_to_csv(data, metric, qasm_file_name):
    if metric in ["total", "fidelity"]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(qasm_file_name))[0]
        csv_file_name = f"{base_name}_{metric}_{timestamp}.csv"
        with open(csv_file_name, mode='w') as file:
            writer = csv.DictWriter(file, fieldnames=data.keys())
            writer.writeheader()
            writer.writerow(data)

# Process QASM file with timeout and save results
def process_qasm_file(qasm_file_path, backend_name, plugin_manager, venv_manager):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(global_config.timeout)  # Set the timeout

    try:
        with open(qasm_file_path, 'r') as file:
            qasm_content = file.read()

        with open(BACKEND_CONFIG, 'r') as backend_file:
            backend_config = yaml.safe_load(backend_file)

        if backend_name in venv_manager.installed_envs:
            try:
                logger.info(f"Benchmark from {qasm_file_path} is run by {backend_name}")
                current_name = plugin_manager.find_key_by_value(backend_name)
                parent_pid = os.getpid()
                metrics, config = plugin_manager.run_backend(
                    current_name, qasm_file_path, nb_shots=1000)

                # Save results to DB if enabled
                if global_config.db:
                    try:
                        from feniqs_lib.db.benchmark_db import Benchmark_db
                        db = Benchmark_db()
                        register_date = datetime.now().isoformat()
                        qasm_file = config.get('test_case', 'unknown')
                        if qasm_file != 'unknown':
                            qasm_file_split = qasm_file.split('/')
                            qasm_file = qasm_file_split[-1]
                            benchmark_group = qasm_file_split[-3]
                        else:
                            benchmark_group = 'Unprecised'
                        for metric, values in metrics.items():
                            metrics_json = json.dumps(values)
                            result_data = {
                                "time": register_date,
                                "backend": backend_name,
                                "package_version": config.get('package_version', 'unknown'),
                                "backend_type": config.get('device_type', 'unknown'),
                                "bench_file": qasm_file,
                                "function_name": metric,
                                "nb_qubits": config.get('nb_qubits', 0),
                                "exception": None,
                                "settings": json.dumps(config),
                                "metrics": metrics_json,
                                "benchmark_group": benchmark_group
                            }
                            db.insert_result(result_data)
                    except Exception as db_error:
                        logger.error(f"Database error: {db_error}", exc_info=True)
                else:
                    logger.info("DB usage is disabled. Skipping database insertion.")

                last_name_of_qasm_folder = os.path.basename(global_config.watchfolder) #QASM_FOLDER)
                target_folder_name = f"{backend_name}-{last_name_of_qasm_folder}"
                target_folder = os.path.join(os.getcwd(), target_folder_name)
                os.makedirs(target_folder, exist_ok=True)
                with open(constants.BACKEND_CONFIG, 'r') as file:
                    path_to_results = yaml.safe_load(file)["result_file"]
                if global_config.json:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    base_name = os.path.splitext(os.path.basename(qasm_file_path))[0]
                    new_json_file_name = f"{base_name}_{timestamp}.json"
                    sys.stdout.write(f"{target_folder}\n")
                    new_json_file_path = os.path.join(target_folder, new_json_file_name)
                    os.rename(path_to_results, new_json_file_path)
                    sys.stdout.write(f"{new_json_file_path}\n")
                    sys.stdout.write(f"{path_to_results}\n")
                logger.info(f"Benchmark from {qasm_file_path} was finished by {backend_name}")
                signal.alarm(0)
                return metrics, config

            except TimeoutError:
                logger.error(f"Benchmark from {qasm_file_path} took too long and was terminated after {global_config.timeout} seconds.")
                kill_process_by_name_or_command(process_name_starts_with = "python", command_starts_with = "feniqs_lib")
                result_data = {
                    "time": datetime.now().isoformat(),
                    "backend": backend_name,
                    "package_version": 'unknown',
                    "backend_type": 'unknown',
                    "bench_file": qasm_file_path,
                    "function_name": 'N/A',
                    "nb_qubits": 0,
                    "exception": "TimeoutError",
                    "settings": 'N/A',
                    "metrics": 'N/A',
                    "benchmark_group": 'Timeout'
                }
                # Optionally, call save_to_csv(result_data, "Timeout", qasm_file_path)
                return None, None
            except Exception as e:
                logger.error(f"Failed to process with backend {backend_name}: {e}", exc_info=True)
                kill_orphans_process(os.getpid())
                result_data = {
                    "time": datetime.now().isoformat(),
                    "backend": backend_name,
                    "package_version": 'unknown',
                    "backend_type": 'unknown',
                    "bench_file": qasm_file_path,
                    "function_name": 'N/A',
                    "nb_qubits": 0,
                    "exception": str(e),
                    "settings": 'N/A',
                    "metrics": 'N/A',
                    "benchmark_group": 'Exception'
                }
                # Optionally, call save_to_csv(result_data, "Exception", qasm_file_path)
    except Exception as e:
        logger.error(f"Failed to process QASM file {qasm_file_path}: {e}", exc_info=True)
        kill_orphans_process(os.getpid())
    finally:
        signal.alarm(0)
    return None, None

class DaemonCore:
    def __init__(self, path, plugin_manager, venv_manager):
        self.observer = Observer()
        self.path = path
        self.plugin_manager = plugin_manager
        self.venv_manager = venv_manager

    def run(self):
        if not os.path.exists(self.path):
            logger.error(f"Directory does not exist: {self.path}")
            os.makedirs(self.path)
            logger.info(f"Created directory: {self.path}")
        event_handler = Handler(self.plugin_manager, self.venv_manager)
        self.observer.schedule(event_handler, self.path, recursive=True)
        self.observer.start()
        start_time = time.perf_counter()
        self.process_existing_files()
        try:
            while True:
                time.sleep(1)
        except Exception as e:
            logger.error(f"Observer error: {e}", exc_info=True)
            self.observer.stop()
        self.observer.join()
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logger.info(f"Total simulation time for all files: {total_time:.2f} seconds")

    def process_existing_files(self):
        files = []
        for root, _, file_names in os.walk(self.path):
            for file_name in file_names:
                files.append(os.path.join(root, file_name))
        with open(BACKEND_CONFIG, 'r') as backend_file:
            backend_config = yaml.safe_load(backend_file)
        backends = backend_config.get('backends', {})
        for backend_name in backends:
            process_files_for_backend(files, backend_name, self.plugin_manager, self.venv_manager)
        logger.info("Finished all simulations. Waiting for a change in the target folder.")

class Handler(FileSystemEventHandler):
    def __init__(self, plugin_manager, venv_manager):
        self.plugin_manager = plugin_manager
        self.venv_manager = venv_manager

    def on_created(self, event):
        logger.debug("Creating new benchmark file")
        if event.is_directory:
            return
        files = [event.src_path]
        if event.src_path.endswith(".qasm"):
            with open(BACKEND_CONFIG, 'r') as backend_file:
                backend_config = yaml.safe_load(backend_file)
            backends = backend_config.get('backends', {})
            for backend_name in backends:
                process_files_for_backend(files, backend_name, self.plugin_manager, self.venv_manager)
        logger.info("Finished all simulations. Waiting for a change in the target folder.")

    def on_deleted(self, event):
        logger.debug("Delete benchmark file")

def process_files_for_backend(files, backend_name, plugin_manager, venv_manager):
    for file_path in files:
        if file_path.endswith(".qasm"):
            process_qasm_file(file_path, backend_name, plugin_manager, venv_manager)

def ensure_directory_exists(directory):
    logger.debug(f"Ensuring directory exists: {directory}")
    if not os.path.exists(directory):
        logger.debug(f"Creating directory: {directory}")
        os.makedirs(directory)
    logger.debug(f"Directory exists or created: {directory}")

def daemon_main(plugin_manager, venv_manager):
    core = DaemonCore(global_config.watchfolder, plugin_manager, venv_manager) 
    try:
        core.run()
    except KeyboardInterrupt as e:
        logger.info("A Keyboard interruption was detected")
    except Exception as e:
        logger.error(f"Unexpected Exception has been detected: {e}")

if __name__ == "__main__":
    welcome.print_welcome_message()
    options = parse_options()
    
    global_config = GlobalConfig()
    global_config.db = getattr(options, "db", True)
    global_config.json = getattr(options, "json", True)
    global_config.incl_first = getattr(options, "incl_first", False)
    global_config.timeout = getattr(options, "timeout", 300)
    global_config.runs = getattr(options, "runs", 5)
    global_config.mf = getattr(options, "mf", True)
    os.environ["MF_ENABLED"] = str(global_config.mf)
    os.environ["FI_ENABLED"] = str(global_config.incl_first)
    os.environ["RUNS"] = str(global_config.runs)


    pid_dir = os.path.dirname(PID_FILE)
    ensure_directory_exists(pid_dir)
    log_dir = os.path.dirname(options.logfile)
    ensure_directory_exists(log_dir)

    if options.level.upper() in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        config_logger(options.level, options.logfile)
    else:
        sys.stderr.write("Invalid --level. Valid values are: DEBUG-INFO-WARNING-ERROR-CRITICAL\n")
        sys.exit(1)
    ensure_directory_exists(options.watchfolder)
    global_config.watchfolder = getattr(options, "watchfolder", True)

    venv_manager = VenvManager(VENV_CONFIG)
    venv_manager.create_envs()
    plugin_manager = get_plugin_manager()
    plugin_manager.register_all_plugins()
    setproctitle.setproctitle("feniqs_daemon")
    
    if not options.foreground:
        if os.path.exists(PID_FILE):
            logger.warning("Non-deleted PID file is found!")
            pid = None
            try:
                pid = int(open(PID_FILE).read())
                os.kill(pid, 0)
            except (ValueError, OSError, IOError):
                logger.warning("Removing old PID file\n")
                try:
                    os.remove(PID_FILE)
                except OSError as e:
                    if e.errno != errno.ENOENT:
                        raise e
                if pid is not None:
                    kill_orphans_process(pid)
            else:
                logger.warning("Daemon is already running...")
                logger.warning("Running of feniqs_daemon is stopped")
                sys.exit(1)
        logger.info("Starting feniqs daemon in the background...")
        try:
            with DaemonContext(
                pidfile=PIDLockFile(PID_FILE),
                files_preserve=[logging.getLogger().handlers[0].stream]
            ):
                logger.debug("Daemon context started")
                signal.signal(signal.SIGTERM, signal_handler)
                signal.signal(signal.SIGINT, signal_handler)
                signal.signal(signal.SIGHUP, signal_handler)
                logger.info("feniqs_daemon started.")
                daemon_main(plugin_manager, venv_manager)
        except Exception as e:
            logger.error(f"Failed to start daemon context: {e}", exc_info=True)
    else:
        logger.info("feniqs_daemon started in the foreground.")
        daemon_main(plugin_manager, venv_manager)
    logger.info("Feniqs Daemon is stopped")

