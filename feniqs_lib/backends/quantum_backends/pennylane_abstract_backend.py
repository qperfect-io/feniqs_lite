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

from .abstract_backend import AbstractBackend
from .abstract_config import BasicConfig
import pennylane as qml
import numpy
from abc import ABCMeta, abstractmethod


class PennyLaneAbstractBackend(AbstractBackend, metaclass=ABCMeta):
    """
    A quantum computing backend for PennyLane. Handles following operations:
    loading, parsing, executing, and sampling quantum circuits.
    """

    @abstractmethod
    def __init__(self, config: BasicConfig):
        """
        Intantiate the PennyLane backend handler.

        Args:
            simulator_name (str, optional): The PennyLane backend to use. Defaults to 'default.qubit'.
            All options supported by the AbstractBackend constructor can be passed here.
        """
        super().__init__(config)
        self._simulator_name = 'To define in child class'

    @AbstractBackend._measure_time
    def generate_backend(self):
        """
        Generates the PennyLane backend.

        Returns:
            qml.Device: The PennyLane backend to use for the simulation
        """
        return qml.device(self._simulator_name, wires=self.config.nb_qubits, shots=self.config.nb_shots)

    @AbstractBackend._measure_time
    def parse(self):
        """
        Parses the QASM string. For PennyLane, parsing occurs during execution.\n
            Results stored in self._qnode_circuit
        """
        if not self._qasm_str:
            raise ValueError(
                "No QASM content loaded. Please load a QASM file first.")

        self._qnode_circuit = qml.from_qasm(self._qasm_str)

    def plot_circuit(self):

        print(qml.draw(self._qnode_circuit,
              wire_order=range(self.config.nb_qubits))())

    @ staticmethod
    def _from_penny_to_common_value(x):
        """
        Translates the unusual -1 and 1 to the commonly use 0 and 1 sates
        """
        return '0' if x == 1 else '1'

    @AbstractBackend._measure_time
    def format_sample(self, samples):
        """
        Formats the result of a PennyLane simulation into a dictionary of counts.

        Args:
            samples (list(Tensor)): The raw samples from the PennyLane simulation.
            each tensor is a list of 1 and -1 representing the measure of 1 qubit for nb_shots

        Returns:
            dict: The dictionary of counts
        """
        # from tensors to list of lists (and inverse results)
        samples = list(map(lambda l: l.tolist(), samples))
        # pennylane handle one shot as int
        if type(samples[0]) == int:
            key = ''.join(map(str, samples))
            return {key: 1}
        # from list of lists to list of strings
        samples = map(lambda l: ''.join(map(str, l)), samples)
        # from list of strings to dict
        return self._from_sample_list_to_dict(samples)

    @AbstractBackend._measure_time
    def sample_only(self):
        """
        Samples measurement outcomes from the quantum circuit.
        """

        # Get the samples
        return self._result

    @AbstractBackend._measure_time
    def execute_only(self):
        """
        Executes the quantum circuit on the backend.\n
        Results stored in self._result.
        """
        # !IMPORTANT: the execution and sampling cannot be separated for this backend
        # This method is in place simply to keep the structure of the code consistent

        @ qml.qnode(self._dev)
        def inner_circuit():
            self._qnode_circuit()
            return qml.sample()

        self._result = inner_circuit()

    @AbstractBackend._measure_time
    def execute_and_sample(self):
        """
        Executes and samples the circuit on the backend.
        Cannot divide the execution and sampling in PennyLane, so they are combined.
        """

        self.execute_only()
        return self.sample_only()

    def get_precision(self):
        """
        Returns the precision of the backend.
        """
        if self.config.precision == "double":
            return numpy.double
        return numpy.single
