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
from .qasm_parser import QasmParser
from feniqs_lib.tools.backends_tools import remove_barriers
from .abstract_simulator_backend import AbstractSimulatorBackend
from .abstract_config import SimulatorConfig
import numpy as np
import qulacs
import os
from collections import defaultdict
from abc import ABCMeta, abstractmethod


class QulacsAbstractBackend(AbstractSimulatorBackend, QasmParser, metaclass=ABCMeta):
    """
    Abstract Class of Qulacs Backend
    Type of simulator: StateVector 
    Implementation: CPU, GPU
    """
    @abstractmethod
    def __init__(self, device: str = 'Qulacs', max_parallel_threads: int = 1, **kwargs):
        """
        Constructor for The Qulacs Backend
        :param device: Name of the provider and device also indicates CPU or GPU or Cloud, defaults to 'Qulacs'
        :type device: str, optional
        :param max_parallel_threads: Number of threads to use, defaults to 1
        :type max_parallel_threads: int, optional
        """

        # Forced config:
        kwargs['fusion_enable'] = False
        kwargs['precision'] = 'double'
        os.environ['QULACS_NUM_THREADS'] = str(max_parallel_threads)

        config = SimulatorConfig(device=device, device_type='statevector', package_version=qulacs.__version__,
                                 max_parallel_threads=max_parallel_threads, **kwargs)
        super().__init__(config)

        # Qulacs specific
        self._circuit = None

        # Default
        self.separates_execution_and_sampling = True
        self.custom_parser = False

        self._state = self.generate_backend()

    def __getattr__(self, x):
        return getattr(qulacs.gate, x)

    def __getitem__(self, x):
        return getattr(qulacs.gate, x)

    @AbstractSimulatorBackend._measure_time
    def parse(self):
        """
        Parses the loaded QASM string into a Qulacs quantum circuit by Feniqs Parser Backend.\n
        Results stored in self._circuit.
        """
        if not self._qasm_str:
            raise ValueError(
                "No QASM content loaded. Please load a QASM file first.")

        self._qasm_str = remove_barriers(self._qasm_str)

        nb_qubits, gatelist = self.parse_qasm(self._qasm_str)

        # We have to regenerate the backend if the number of qubits has changed
        if nb_qubits != self.config.nb_qubits:
            self.config.nb_qubits = nb_qubits
            self._state = self.generate_backend()

        self._circuit = qulacs.QuantumCircuit(self.config.nb_qubits)
        for gatename, qubits, params in gatelist:
            # IMPOTANT we have to remove measurements gate for the sampling to work
            if gatename == "M":
                continue
            curr_gate = self._construct_gate(gatename, qubits, params)
            if curr_gate:
                self._circuit.add_gate(curr_gate)

    def plot_circuit(self):
        pass


    @AbstractSimulatorBackend._measure_time
    def format_sample(self, samples):
        """
        Formats the result of a Qulacs simulation into a dictionary of counts.

        Args:
            samples (list[int]): List of integers representing sampled states.

        Returns:
            dict: A dictionary where keys are binary strings of qubit states and values are counts of these states.
        """
        res = defaultdict(int)    # Initialize defaultdict to handle integers
        for sample in samples:
            # Srting formating
            bin_sample = f'{sample:0{self.config.nb_qubits}b}'[::-1]
            res[bin_sample] += 1  # Automatically handles new keys
        # Convert defaultdict back to a regular dict (may be not required in our case)
        return dict(res)

    @AbstractSimulatorBackend._measure_time
    def sample_only(self):
        """
        Samples measurement outcomes from the quantum circuit.
        """

        if not self._state:
            raise ValueError(
                "No execution result found. Please execute the circuit first.")
        # Samples the data and trims it from the leading 0b
        return self._state.sampling(self.config.nb_shots)

    @AbstractSimulatorBackend._measure_time
    def execute_only(self):
        """
        Executes the quantum circuit on Qulacs.\n
        Results stored in self._state.
        """
        if not self._circuit:
            raise ValueError(
                "No Qulacs circuit defined. Please parse the QASM content first.")
        self._state = qulacs.QuantumState(self.config.nb_qubits)
        self._circuit.update_quantum_state(self._state)

    @AbstractSimulatorBackend._measure_time
    def execute_and_sample(self):
        """
        Executes and samples the quantum circuit.
        """
        self.execute_only()
        return self.sample_only()

    def set_precision(self, precision):
        """
        Define double precision.
        """
        if precision != "double":
            raise NotImplementedError(
                f"Cannot set {precision} precision for {self.config.device} backend.")

    def get_precision(self):
        return "double"

    def _construct_gate(self, gatename, qubits, params=None):
        """
        Constructs a Qulacs qulacs.gate from the qulacs.gate name, qubits, and optional parameters.
        """
        if hasattr(self, gatename):
            gate_func = getattr(self, gatename)
            args = qubits
            if params is not None:
                args.extend(params)
            return gate_func(*args)
        else:
            raise NotImplementedError(
                f"Gate {gatename} is not implemented in this backend.")

    def M(self, qb_index, c_index, qb_register, c_register):
        # Apply the measurement qulacs.gate directly
        return qulacs.gate.Measurement(qb_index, c_index)

    def RX(self, target_qubit, angle):
        # Apply the RX qulacs.gate with a negative angle directly
        return qulacs.gate.RX(target_qubit, -angle)

    def RY(self, target_qubit, angle):
        # Apply the RY qulacs.gate with a negative angle directly
        return qulacs.gate.RY(target_qubit, -angle)

    def RZ(self, target_qubit, angle):
        # Apply the RZ qulacs.gate with a negative angle directly
        return qulacs.gate.RZ(target_qubit, -angle)

    def CU1(self, control_qubit, target_qubit, angle):
        # Use numpy to create a diagonal matrix directly with the required phase shift
        gate_matrix = np.diag([1, np.exp(1j * angle)])
        curr_gate = qulacs.gate.DenseMatrix([target_qubit], gate_matrix)
        curr_gate.add_control_qubit(control_qubit, 1)
        return curr_gate

    def CU3(self, control_qubit, target_qubit, angle, phi, lam):
        # Precalculate trigonometric functions and phase factors to optimize performance
        cos_tr, sin_tr = np.cos(angle / 2.0), np.sin(angle / 2.0)
        ph_plus, ph_minus = np.exp(
            0.5j * (phi + lam)), np.exp(0.5j * (phi - lam))
        gate_matrix = np.array([
            [np.conj(ph_plus) * cos_tr, -np.conj(ph_minus) * sin_tr],
            [ph_minus * sin_tr, ph_plus * cos_tr]
        ])
        curr_gate = qulacs.gate.DenseMatrix([target_qubit], gate_matrix)
        curr_gate.add_control_qubit(control_qubit, 1)
        return curr_gate

    def SQRTX(self, target_qubit):
        # Apply the SQRTX qulacs.gate directly
        return qulacs.gate.sqrtX(target_qubit)

    def SQRTXDG(self, target_qubit):
        # Apply the SQRTXDG qulacs.gate directly
        return qulacs.gate.sqrtXdag(target_qubit)

    def TDG(self, *args):
        return self.Tdag(*args)

    def RZZ(self, target_qubit1, target_qubit2, angle):
        # Directly construct the diagonal matrix for RZZ to minimize computation
        phase = np.exp(0.5j * angle)
        gate_matrix = np.diag([np.conj(phase), phase, phase, np.conj(phase)])
        curr_gate = qulacs.gate.DenseMatrix(
            [target_qubit1, target_qubit2], gate_matrix)
        return curr_gate


class QulacsCpuBackend(QulacsAbstractBackend):
    def __init__(self, **kwargs):
        """
        Constructor for the Qulacs BAckend using a CPU.
        Refer to the parent class  QulacsAbstractBackend to see accepted options.

        Here is a full example to use this backend:

        .. code-block:: python

            # to generate the backend and pass it options
            backend = QulacsCpuBackend(test_case='path_to_qasm.qasm', nb_shots=1, max_parallel_threads=1)
            # To parse the qasm file
            backend.parse()
            # To execute and sample the results
            samples = backend.execute_and_sample()
            # To foramt the samples in a json format
            samples = backend.format_sample(samples)
            # To retrieve all the options used in the simulation
            config = backend.get_config_dict()
        """
        super().__init__(device='Qulacs_Cpu', **kwargs)

    @AbstractSimulatorBackend._measure_time
    def generate_backend(self):
        """No need for qulacs
        """
        return qulacs.QuantumState(self.config.nb_qubits)


class QulacsGpuBackend(QulacsAbstractBackend):
    """
    A subclass of Qulacs that initializes a quantum simulation environment optimized for GPU.
    """

    def __init__(self, **kwargs):
        """
        Initializes the Qulacs GPU backend with configurations inherited from the base Qulacs class.

        Args:
            config (Config): A configuration object containing simulation parameters.
        """
        super().__init__(device='Qulacs_Gpu', **kwargs)

    @AbstractSimulatorBackend._measure_time
    def generate_backend(self):
        return qulacs.QuantumStateGpu(self.config.nb_qubits)


if __name__ == '__main__':
    file_path = '' # this is a placeholder, please fill this field by the path to your qasm file
    backend = QulacsCpuBackend(nb_shots=10, test_case=file_path)
    backend.parse()
    samples = backend.execute_and_sample()
    samples_formatted = backend.format_sample(samples)
    print(samples_formatted)
