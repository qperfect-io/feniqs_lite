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

import json
import yaml

from feniqs_lib.tools.constants import GlobalConfig
from feniqs_lib.tools.constants import update_global_config_from_env
import feniqs_lib.tools.constants as constants

"""
A general function for running the backends and getting the metrics: run time and fidelity (for non statevector emulators)
"""

def run_task(backend):
    """
    This function runs one simulation of quantum circuit by selected backend (through its plugin).
    One simualtion includes the following operations: 
    parse, execute, sample, (format the result if needed), calculat mirrow fidelity for non stateector emulators (if selected this option), 
    and return the metrics (run time and fidelity).

    Args: 
        backend (AbstractBackend): Backend object, selected for benchmatking 
    Returns:
        dict, dict: the metrics & the configuration used for the simulation
    """

    backend_name = type(backend).__name__
    # The results to fill
    metrics = {}
    total = 0
    # BAckend was already intanciated so we can just get the timimgs value
    metrics["loading"] = backend._loading_time

    metrics["Intantiating"] = backend.time_value
    total += backend.time_value

    # Parsing the qasm file
    backend.parse()
    metrics["parsing"] = backend.time_value
    total += backend.time_value

    # Execute and sample

    samples = backend.execute_and_sample()
    metrics["execution_and_sampling"] = backend.time_value

    
    total += backend.time_value
    
    # Formatting the samples
    samples = backend.format_sample(samples)
    metrics["fomatting"] = backend.time_value
    total += backend.time_value
  
    # Print the results
    metrics["total"] = total

    global_config = update_global_config_from_env()

    if global_config.mf: 
        qasm_str_mir = backend.generate_mirror_qasm()
        backend.fidelity = backend.get_mirror_fidelity(backend._qasm_file, qasm_str_mir)
        metrics["fidelity"] = backend.fidelity
    else:
        metrics["fidelity"] = None

    config = backend.get_config_dict()
    results_to_json(samples, metrics, config)
    return samples, metrics, config


def results_to_json(samples, metrics, config):
    """
    Writes the results from run_task to a json file in one single object

    Args:
        samples (dict): The dictionnary of the samples resulting from the simulation.
        metrics (dict(json.dump({"samples": samples, "metrics": metrics, "config": config}, f, indent=4)): 
                        The dictionnary of metrics retrieve from the simulation.
        config (dict): The configuration used in the simulation.
    """
    with open(constants.BACKEND_CONFIG, "r") as f:
        path_to_results = yaml.safe_load(f)["result_file"]

    # Save the results
    with open(path_to_results, "w") as f:
        json.dump({"metrics": metrics, "config": config}, f, indent=4)
  

