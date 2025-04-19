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
from .abstract_cloud_backend import AbstractCloudBackend
from .abstract_config import CloudConfig
import pennylane as qml
from abc import ABCMeta, abstractmethod


class PennyLaneCloudAbstractBackend(PennyLaneAbstractBackend, AbstractCloudBackend, metaclass=ABCMeta):
    """PennyLane implementation of the AbstractBackend for cloud-based simulators."""

    @abstractmethod
    def __init__(self, simulator_name='default.qubit', **kwargs):
        """Initializes the PennyLane backend."""

        config = CloudConfig(
            device='PennyLane_Cloud', device_type=simulator_name, package_version=qml.__version__, **kwargs)
        super().__init__(config)
        self._simulator_name = simulator_name

    def generate_backend(self):
        """Generates the PennyLane backend.

        Returns:
            qml.Device: The PennyLane backend to use for the simulation
        """
        return qml.device(self._simulator_name, wires=self.config.nb_qubits, shots=self.config.nb_shots)

    def close_connection(self):
        """Nothong to do for PennyLane."""
        return super().close_connection()


class PennyLaneCloudBackend(PennyLaneCloudAbstractBackend):
    """PennyLane implementation of the AbstractBackend for cloud-based simulators.
    I don't see anything specific to do here yet, but included for consistency and future extensibility."""

    def __init__(self, simulator_name='default.qubit', **kwargs):
        """Initializes the PennyLane backend."""
        super().__init__(simulator_name=simulator_name, **kwargs)
