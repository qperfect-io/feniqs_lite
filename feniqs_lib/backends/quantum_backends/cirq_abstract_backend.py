
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
from .abstract_backend import AbstractBackend
from .abstract_simulator_backend import AbstractSimulatorBackend
from .abstract_config import BasicConfig, SimulatorConfig
import cirq
import numpy
from cirq.contrib.qasm_import import circuit_from_qasm
from abc import ABCMeta, abstractmethod
import asyncio
from feniqs_lib.tools.backends_tools import remove_barriers


class CirqAbstractBackend(AbstractBackend, metaclass=ABCMeta):
    """
    Cirq implementation of the AbstractBackend. Handles following operations: 
    loading, parsing, executing, and sampling quantum circuits.
    """
    
    @abstractmethod
    def __init__(self, config: BasicConfig):
        """
        Initializes the Cirq backend.
        """
        super().__init__(config)

    @AbstractBackend._measure_time
    def parse(self):
        """
           Parses the loaded QASM into a Cirq circuit.\n
           Results stored in self._cirq_circuit.
        """

        if not self._qasm_str:
            raise ValueError(
                "Error: QASM content is empty. Please load a QASM file first.")

        self._qasm_str = remove_barriers(self._qasm_str)

        self._cirq_circuit = circuit_from_qasm(self._qasm_str)

        if len(self._cirq_circuit.all_measurement_key_names()) == 0:
            self._cirq_circuit.append(cirq.measure(
                *self._cirq_circuit.all_qubits()))

    def plot_circuit(self):
        print(self._cirq_circuit)

    @staticmethod
    def get_measurement_keys(cirq_circuit):
        measurement_keys = set()
        for moment in cirq_circuit:
            for op in moment:
                if isinstance(op.gate, cirq.MeasurementGate):
                    measurement_keys.add(op.gate.key)
        return list(measurement_keys)

    @staticmethod
    def cirq_results_to_histogram(result, circuit):
        measurement_keys = sorted(
            CirqAbstractBackend.get_measurement_keys(circuit))
        result_dict = dict(
            result.multi_measurement_histogram(keys=measurement_keys))
        keys = [''.join(map(str, map(int, arr[::-1])))
                for arr in result_dict.keys()]
        return {k[::-1]: result_dict[v] for k, v in zip(keys, result_dict.keys())}

    @AbstractBackend._measure_time
    def format_sample(self, samples):
        """Formats the result of a Cirq simulation into a dictionary of counts.

        Args:
            samples (?): The raw samples from the Cirq simulation.

        Returns:
            dict: a dictionary of counts
        """
        return CirqAbstractBackend.cirq_results_to_histogram(samples, self._cirq_circuit)

    def sample_only(self):
        """
        Samples measurement outcomes from the quantum circuit.
        """
        
        # !IMPORTANT: the execution and sampling cannot be separated for qiskit
        # This method is in place simply to keep the structure of the code consistent

        # Get the samples
        return self._result

    async def execute_async(self):
        """
        Executes the quantum circuit on the backend in an asynchronous manner.
        """
        self._result = await self._backend.run_async(
            self._cirq_circuit, repetitions=self.config.nb_shots)

    def execute_only(self):
        """
        Executes the quantum circuit on the backend.\n
        results stored in self._result.
        """

        if not self._cirq_circuit:
            raise ValueError(
                "Error: Cirq circuit is not defined. Please parse the QASM content first.")
        if getattr(self.config, 'run_async', False):
            asyncio.run(self.execute_async())
        else:
            self._result = self._backend.run(
                self._cirq_circuit, repetitions=self.config.nb_shots)

    @AbstractBackend._measure_time
    def execute_and_sample(self):
        """
        Executes and samples the circuit  @abstract.AbstractBackend._measure_timeon the backend.
        Cannot divide the execution and sampling in Cirq, so they are combined.
        """
        self.execute_only()
        return self.sample_only()

    def get_precision(self):
        if self.config.precision == 'double':
            return numpy.complex128
        elif self.config.precision == 'single':
            return numpy.complex64
        else:
            raise ValueError(
                f"Error: Precision {self.config.precision} is not supported.")


class CirqSimulatorAbstractBackend(CirqAbstractBackend, AbstractSimulatorBackend, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, device: str, package_version: str, device_type: str, **kwargs):

        config = SimulatorConfig(device=device, device_type=device_type,
                                 package_version=package_version, **kwargs)

        # Parent constructor
        super().__init__(config)

        # Cirq specific
        self._cirq_circuit = None
        self._result = None
