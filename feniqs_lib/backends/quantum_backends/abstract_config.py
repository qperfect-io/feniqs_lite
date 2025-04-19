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

from __future__ import annotations
from random import randint

"""
Basic Configuration Class for All Quantum Backend.
"""

class BasicConfig:

    def __init__(self, device: str, device_type: str, package_version: str, test_case: str, nb_shots: int = 1) -> None:
        """
        Constructor for basic configuration class

        :param device: The provider and device used to run the simulation example: cirq_cpu, pennylane_cloud
        :type device: str
        :param device_type: The Strategy used for the simulation: statevector, matrix_product_state, tensor_network
        :type device_type: str
        :param package_version: The version of the provider's package
        :type package_version: str
        :param test_case: The path to the file used for the simulation
        :type test_case: str
        :param nb_shots: The number of shots to use for the simulation, defaults to 1
        :type nb_shots: int, optional
        """
        self.device = device
        self.device_type = device_type
        self.package_version = package_version
        # Keep it empty for now
        self.backend_version = ''
        self.test_case = test_case
        self.nb_shots = nb_shots
        # Attention: number of qubits should be determined after loading the QASM file with circuit
        self.nb_qubits = None

    def add_attr(self, attr: str, value):
        """
        Some specific attributes can be added to the configuration if needed by the backends.

        Args:
            attr (str): The name of the attribute to add
            value (Any): th value of the attribute
        """
        setattr(self, attr, value)

    def add_attrs(self, **kwargs):
        """
        Add multiple attributes to the config
        """

        for key, value in kwargs.items():
            setattr(self, key, value)


class CloudConfig(BasicConfig):
    def __init__(self, **kwargs) -> None:
        """
        Constructor for a backen config using the cloud
        For Now is the same as Basic config
        """
        
        super().__init__(**kwargs)


class SimulatorConfig(BasicConfig):
    def __init__(self, seed: int = None, precision: str = 'double', max_parallel_threads: int = 1, fusion_enable: bool = False, **kwargs):
        """
        Constructor for a backend config running a simulation
        all argument expected by Basic C

        :param seed: _description_, defaults to None
        :type seed: int, optional
        :param precision: _description_, defaults to 'double'
        :type precision: str, optional
        :param max_parallel_threads: _description_, defaults to 1
        :type max_parallel_threads: int, optional
        :param fusion_enable: _description_, defaults to False
        :type fusion_enable: bool, optional
        """
        
        super().__init__(kwargs['device'], kwargs['device_type'],
                         kwargs['package_version'], kwargs['test_case'], kwargs['nb_shots'])

        self.seed = seed if seed is not None else randint(1, 1000)
        self.fusion_enable = fusion_enable
        self.precision = precision
        self.max_parallel_threads = max_parallel_threads
        self.add_attrs(**kwargs)


class AbstractConfig():
    def __init__(self, config: BasicConfig):
        self.config = config

    def get_config_dict(self) -> dict:
        return vars(self.config)

    def add_config_attr(self, attr: str, value):
        self.config.add_attr(attr, value)
