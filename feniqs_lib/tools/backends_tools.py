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

import re
import statistics
from feniqs_lib.backends.task.task import run_task, results_to_json


def remove_barriers(qasm_str):
    """
    Check for barriers in the QASM string, print a warning if found, and remove them.

    Args:
        qasm_str (str): The QASM string.

    Returns:
        str: The QASM string with barriers removed.
    """
    barrier_pattern = re.compile(r'\bbarrier\b[^\n]*')
    if barrier_pattern.search(qasm_str):
        print("Warning: Barriers found and removed from the QASM string.")
        qasm_str = barrier_pattern.sub('', qasm_str)
    return qasm_str


def run_backend_common(BackendClass, test_case: str, nb_shots: int, seed: int, env: str = None, include_first_run: bool = False, runs_number: int = 10, **kwargs):
    if env is not None:
        backend = BackendClass(test_case=test_case,
                               nb_shots=nb_shots, seed=seed, env=env, **kwargs)
    else:
        backend = BackendClass(test_case=test_case,
                               nb_shots=nb_shots, seed=seed, **kwargs)

    metric_values = {}

    samples = None
    config = None

    start_index = 0 if include_first_run else 2

    for i in range(runs_number):
        samples, metrics, config = run_task(backend)
        if i >= start_index:
            for key in metrics.keys():
                if key not in metric_values.keys():
                    metric_values[key] = []
                metric_values[key].append(metrics.get(key, None))

    decomposed_metrics = {}
    for key in metrics.keys():
        valid_values = [v for v in metric_values[key] if v is not None]
        # This might happen for the fidelity, using statevector the fidelity will be set to none
        if len(valid_values) == 0:
            continue
        decomposed_metrics[key] = {}
        if valid_values:
            decomposed_metrics[key]['median_rt'] = statistics.median(
                valid_values)
            decomposed_metrics[key]['avg_rt'] = statistics.mean(
                valid_values)
            decomposed_metrics[key]['min_rt'] = min(valid_values)
            decomposed_metrics[key]['max_rt'] = max(valid_values)
        else:
            decomposed_metrics[key]['median_rt'] = None
            decomposed_metrics[key]['avg_rt'] = None
            decomposed_metrics[key]['min_rt'] = None
            decomposed_metrics[key]['max_rt'] = None

    results_to_json(samples, decomposed_metrics, config)
    return samples, decomposed_metrics, config
