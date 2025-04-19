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

'''
Main constants of FENIQS
'''

class GlobalConfig:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalConfig, cls).__new__(cls)
            cls._instance.db = True          # Enable or disable database usage (default: true)
            cls._instance.json = False       # Save benchmark results in JSON format (default: false)
            cls._instance.mf = True         # Enable or disable mirror fidelity usage (default: true)
            cls._instance.timeout = 300      # Timeout value for simulation (default: 300 seconds)  
            cls._instance.watchfolder = 'data/test'
            cls._instance.runs = 3           # Number of runs for each banchmark (default: 3) - must be increased for circuits with short run time 
            cls._instance.incl_first = False # Include or not the first run (default: false) - it allows to make a warm up first seesion - very important for julia based code 
        return cls._instance

def update_global_config_from_env() -> GlobalConfig:

    global_config = GlobalConfig()
    fi_env = os.environ.get("FI_ENABLED", "false")
    mf_env = os.environ.get("MF_ENABLED", "true")
    run_env = int(os.environ.get("RUNS", 3))
    global_config.incl_first = fi_env.lower() in ["true", "1", "yes"]
    global_config.mf = mf_env.lower() in ["true", "1", "yes"]
    global_config.runs = run_env

    return global_config

# Path to folder with qasm files
QASM_FOLDER = 'data/test'

# Path to plugin folder
PLUGIN = 'feniqs_lib/backends/plugins'

# Config file of viratual environements for backends: i.e., simulators, clouds
VENV_CONFIG = 'yaml/venv_deps.yaml'

# Config file of backend for fedining its special parameters: e.g., fusion, threading and other optimisations
BACKEND_CONFIG = 'yaml/config_backend.yaml'

# Config file of PostgeSQL database
DB_CONFIG = 'yaml/db_config.yaml'


