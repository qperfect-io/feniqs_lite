
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

import os
import numpy as np
import qrisp as qr
import pkg_resources
from abc import ABCMeta, abstractmethod
from .abstract_backend import AbstractBackend
from .abstract_simulator_backend import AbstractSimulatorBackend
from .abstract_config import SimulatorConfig

class QrispCpuBackend(AbstractSimulatorBackend, metaclass=ABCMeta):
    """        
    Class of Qrisp Backend
    Type of simulator: Sparse Matrix (SM)
    Implementation: CPU
    Documentation: https://www.qrisp.eu/general/tutorial/index.html
    """
    def __init__(self, device='Qrisp', **kwargs):
        """
        Backend for Qrisp SM simulations.
        param: None
        """
        config = SimulatorConfig(
            device=device, device_type="sm", package_version=pkg_resources.get_distribution("qrisp").version, **kwargs)
        super().__init__(config)

    @ AbstractBackend._measure_time
    def generate_backend(self):
        return None   

    @AbstractBackend._measure_time
    def parse(self):
        """
        Parses the loaded QASM string into a Quimb circuit.
        """
        if self._qasm_str is None:
            raise ValueError("Error: No QASM loaded. Please load a QASM file first.")
     
        self._qrisp_circuit = qr.QuantumCircuit().from_qasm_str(
            self._qasm_str
        )
        return self._qrisp_circuit

    @AbstractBackend._measure_time
    def format_sample(self, samples):
        """
        Formats the samples into a dictionary of counts.
        """
        counts = {"0": 0, "1": 0}
        for sample in samples:
            counts["0"] += sum(1 for bit in sample if bit == "0")
            counts["1"] += sum(1 for bit in sample if bit == "1")
        return counts

    @AbstractSimulatorBackend._measure_time
    def sample_only(self):
        """
        Samples the circuit.
        """
        return self._result

    @AbstractSimulatorBackend._measure_time
    def execute_only(self):
        """
        Executes the quantum circuit on the backend.
        """
        self._qrisp_circuit.measure(self._qrisp_circuit.qubits)
        self._result = self._qrisp_circuit.run(shots=self.config.nb_shots)

    def _import_qasm(self, qasm_file):
        """
        Import a QASM file, removing measurement operations.
        """
        with open(qasm_file, "r") as file:
            qasm_str = file.read()
        self._qasm_str = qasm_str
        circuit = qr.QuantumCircuit().from_qasm_str(
            self._qasm_str
        )
        return circuit

    def _circuit_stats(self, circuit):
        return circuit.num_qubits(), sum(value for key, value in circuit.count_ops().items() if key!='measure')

    def _compile(self, circuit):
        return circuit

    def compute_fidelity(self, singular_values_cut):
        """
        Compute fidelity based on singular value cuts.
        """
        return np.prod(1 - np.array(singular_values_cut))
    
    def execute(self):
        """Executes the parsed file and computes the fidelity
        """
        self.execute_only()

    def get_mirror_fidelity(self, qasm_file, mirror_qasm_file):
        """
        Calculate the mirror fidelity of a circuit.
        """
        circuit = self._compile(self._import_qasm(qasm_file))
        mirror_circuit = self._compile(self._import_qasm(mirror_qasm_file))
        circuit.barrier()
        circuit.extend(mirror_circuit)
        self._qrisp_circuit = circuit
        nq, ng = self._circuit_stats(circuit)
        self.execute()   
        zero_prob = self._result['0'*nq]/self.config.nb_shots
        return zero_prob


    @AbstractSimulatorBackend._measure_time
    def execute_and_sample(self):
        """
        Runs the simulation and gets the samples.

        Returns:
            dict: the dictionary of counts
        """
        self.execute_only()
        return self.sample_only()



        



