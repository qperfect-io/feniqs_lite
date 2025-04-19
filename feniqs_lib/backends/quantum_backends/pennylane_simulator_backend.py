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

from .pennylane_abstract_backend import PennyLaneAbstractBackend
from .abstract_config import SimulatorConfig
from abc import ABCMeta
import pennylane as qml
import numpy as np
import os


class PennyLaneSimulatorAbstractBackend(PennyLaneAbstractBackend, metaclass=ABCMeta):

    def __init__(self, simulator_name='default.qubit', **kwargs):
        """
        Constructor for the Pennylane backends

        :param simulator_name: Indicates which pennylane to use, defaults to 'default.qubit' Can be (default.qubits, default.mixed, default.gaussian, lightning.qubit, lightning.kokkos)
        :type simulator_name: str, optional
        """

        # Forced config
        kwargs['fusion_enable'] = False
        kwargs['precision'] = 'double'
        kwargs['max_parallel_threads'] = 1

        device_type = 'statevector' if simulator_name != 'default.Clifford' else 'clifford'

        config = SimulatorConfig(device_type=device_type,
                                 package_version=qml.__version__, **kwargs)
        super().__init__(config)
        self.add_config_attr('simulator_name', simulator_name)
        self._simulator_name = simulator_name
        # PennyLane specific
        np.random.seed(self.config.seed)
        self._qnode_circuit = None
        self._result = None
        self._dev = None
        # For other pennylane backend it will be instantiated in execute because we need the number of qubits
        self._dev = self.generate_backend()

    def generate_backend(self):
        """Generates the PennyLane backend.

        Returns:
            qml.Device: The PennyLane backend to use for the simulation
        """
        return qml.device(self._simulator_name, wires=self.config.nb_qubits, shots=self.config.nb_shots)


class PennyLaneCpuBackend(PennyLaneSimulatorAbstractBackend):
    """PennyLane implementation of the AbstractBackend for CPU."""

    def __init__(self, **kwargs):
        """Constructor for the pennylane backedn on cpus.`

        Here is a full example to use this backend:

        .. code-block:: python

            # to generate the backend and pass it options
            backend = PennyLaneCpuBackend(simulator_name='default.qubit', test_case='path_to_qasm.qasm', nb_shots=1)
            # To parse the qasm file
            backend.parse()
            # To execute and sample the results
            samples = backend.execute_and_sample()
            # To foramt the samples in a json format
            samples = backend.format_sample(samples)
            # To retrieve all the options used in the simulation
            config = backend.get_config_dict()
        """
        super().__init__(device='Pennylane_Cpu', **kwargs)


class PennyLaneGpuBackend(PennyLaneSimulatorAbstractBackend):
    """PennyLane implementation of the AbstractBackend for GPU."""

    def __init__(self, simulator_name='lightning.gpu', **kwargs):
        """Initializes the PennyLane backend."""
        super().__init__(device='Pennylane_Gpu', simulator_name=simulator_name, **kwargs)


if __name__ == '__main__':
    file_path = '' # this is a placeholder, please fill this field by the path to your qasm file
    backend = PennyLaneCpuBackend(nb_shots=10, test_case=file_path)
    backend.parse()
    samples = backend.execute_and_sample()
    samples_formatted = backend.format_sample(samples)
    print(samples_formatted)
