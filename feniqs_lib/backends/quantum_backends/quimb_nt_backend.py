
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
import quimb
import quimb.tensor as qtn
from abc import ABCMeta, abstractmethod
from .abstract_backend import AbstractBackend
from .abstract_simulator_backend import AbstractSimulatorBackend
from .abstract_config import SimulatorConfig


class QuimbNTCpuBackend(AbstractSimulatorBackend, metaclass=ABCMeta):
    """
    Class of Quimb Tensor Network (TN)Backend
    Type of simulator: Tensor Network 
    Implementation: CPU
    Documentation: https://quimb.readthedocs.io/en/latest/autoapi/quimb/tensor/circuit/index.html#quimb.tensor.circuit.Circuit
    """

    def __init__(self, device='QuimbTN',contract: bool = False,  **kwargs):
        """
        Backend for Quimb TN simulations.
        param: contract - Flag to perform gate optimization (F for False, T for True)
        """
        config = SimulatorConfig(
            device=device, device_type="nt", package_version=quimb.__version__, **kwargs)
        super().__init__(config)
        self.add_config_attr("contract", contract)

    @ AbstractBackend._measure_time
    def generate_backend(self):
        return None   

    @AbstractBackend._measure_time
    def parse(self):
        """
        Parses the loaded QASM string into a Quimb circuit.
        """
        if self._qasm_file is None:
            raise ValueError("Error: No QASM loaded. Please load a QASM file first.")
        with open(self._qasm_file, "r") as file:
            qasm_lines = file.readlines()
        self._qasm_str = ''.join( line for line in qasm_lines if 'measure' not in line.lower()
                and 'creg' not in line.lower() and 'barrier' not in line.lower() )
        self._quimb_circuit = qtn.Circuit(1).from_openqasm2_str(
            self._qasm_str,
            gate_opts ={'contract': self.config.contract})
        
        return self._quimb_circuit

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
        if not hasattr(self._result, "sample"):
            raise AttributeError("The circuit does not support sampling.")
        return [str(s) for s in self._result.sample(self.config.nb_shots)]

    @AbstractSimulatorBackend._measure_time
    def execute_only(self):
        """
        Executes the quantum circuit on the backend.
        """
        self._result = self._quimb_circuit

    def _import_qasm(self, qasm_file):
        """
        Import a QASM file, removing measurement operations.
        """
        with open(qasm_file, "r") as file:
            qasm_lines = file.readlines()

        self._qasm_str = ''.join( line for line in qasm_lines if 'measure' not in line.lower()
                and 'creg' not in line.lower() and 'barrier' not in line.lower() )
        
        circuit = qtn.Circuit(1)        
        return circuit.from_openqasm2_str(
            self._qasm_str,
            gate_opts ={ 'contract': self.config.contract})
        

    def _circuit_stats(self, circuit):
        return len(circuit.calc_qubit_ordering()), circuit.num_gates

    def _compile(self, circuit): 
        return circuit

    def compute_fidelity(self, singular_values_cut):
        """
        Compute fidelity based on singular value cuts.
        """
        return np.prod(1 - np.array(singular_values_cut))

    def get_mirror_fidelity(self, qasm_file, mirror_qasm_file):
        """
        Calculate the mirror fidelity of a circuit.
        """
        circuit = self._compile(self._import_qasm(qasm_file))
        mirror_circuit = self._compile(self._import_qasm(mirror_qasm_file))
        circuit.apply_gates(mirror_circuit.gates)
        nq, ng = self._circuit_stats(circuit)
        zero_prob = abs(circuit.amplitude('0'*nq))**2
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



