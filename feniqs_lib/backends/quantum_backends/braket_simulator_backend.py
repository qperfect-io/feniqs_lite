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

from .braket_abstract_backend import BraketAbstractBackend
from .abstract_backend import AbstractBackend
from .abstract_config import SimulatorConfig

from braket.devices import LocalSimulator
import numpy as np


class BraketCpuBackend(BraketAbstractBackend):
    """The Backend used to run Braket simulation on CPU
    """

    def __init__(self, device_type="braket_sv", **kwargs):
        """Initialize the backend to handle braket simultaion on CPU

        :param device_type: The name of the simulator to use, defaults to "braket_sv", alternative: braket_sv(up to 25 qubits), braket_dm (12).
        :type device_type: str, optional

        Here is a full example to use this backend:

        .. code-block:: python

            # to generate the backend and pass it options
            backend = BraketCpuBackend(device_type='braket_sv', test_case='path_to_qasm.qasm', nb_shots=1)
            # To parse the qasm file
            backend.parse()
            # To execute and sample the results
            samples = backend.execute_and_sample()
            # To foramt the samples in a json format
            samples = backend.format_sample(samples)
            # To retrieve all the options used in the simulation
            config = backend.get_config_dict()
        """

        # Cannot change this parameter for braket
        # To check
        kwargs['precision'] = 'double'
        package_ver = self._get_version('amazon-braket-sdk')
        config = SimulatorConfig(
            device='Braket_Cpu', package_version=package_ver, device_type=device_type, **kwargs)
        np.random.seed(config.seed)
        super().__init__(config)

    @AbstractBackend._measure_time
    def generate_backend(self):
        return LocalSimulator(backend=self.config.device_type)


if __name__ == '__main__':
    print("Running cirq")
    file_path = ''  # this is a placeholder, please fill this field by a path to your qasm file
    backend = BraketCpuBackend(nb_shots=10, test_case=file_path)
    backend.parse()
    samples = backend.execute_and_sample()
    samples_formatted = backend.format_sample(samples)
    print(samples_formatted)
