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

from .cirq_abstract_backend import CirqSimulatorAbstractBackend
from .abstract_backend import AbstractBackend
import cirq


class CirqCpuBackend(CirqSimulatorAbstractBackend):
    def __init__(self, split_untangled_state: bool = True, run_async=False, **kwargs):
        """
        Constructor for the Cirq backend

        :param split_untangled_state: If True, optimizes simulation by running unentangled qubit sets independently and merging those states at the end, defaults to True
        :type split_untangled_state: bool, optional
        :param run_async: Asynchronously samples from the given Circuit, defaults to False
        :type run_async: bool, optional

        Here is a full example to use this backend:

        .. code-block:: python

            # to generate the backend and pass it options
            backend = CirqCpuBackend(test_case='path_to_qasm.qasm', nb_shots=1, split_untangled_state=True, run_async=False)
            # To parse the qasm file
            backend.parse()
            # To execute and sample the results
            samples = backend.execute_and_sample()
            # To foramt the samples in a json format
            samples = backend.format_sample(samples)
            # To retrieve all the options used in the simulation
            config = backend.get_config_dict()
        """

        # Forced config:
        kwargs['fusion_enable'] = False
        kwargs['max_parallel_threads'] = 1
        super().__init__(device='Cirq_Cpu', device_type='sparse_matrix_state',
                         package_version=cirq.__version__, **kwargs)

        # Cirq specific config
        self.add_config_attr('split_untangled_state', split_untangled_state)
        self.add_config_attr('run_async', run_async)

        # instanciate backend
        self._backend = self.generate_backend()

    @AbstractBackend._measure_time
    def generate_backend(self):
        """
        Generates the Cirq simulator backend.

        Returns:
            cirq.Simulator: The backend to use for the simulation
        """
        return cirq.Simulator(dtype=self.get_precision(), seed=self.config.seed, split_untangled_states=self.config.split_untangled_state)


if __name__ == '__main__':
    print("Running cirq")
    file_path = '' # this is a placeholder, please fill this field by a path to your qasm file
    backend = CirqCpuBackend(nb_shots=10, test_case=file_path)
    backend.parse()
    samples = backend.execute_and_sample()
    samples_formatted = backend.format_sample(samples)
    print(samples_formatted)
