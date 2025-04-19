
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


class QuimbMPSCpuBackend(AbstractSimulatorBackend, metaclass=ABCMeta):
    """        
    Class of Quimb Matrix Product State (MPS) Backend
    Type of simulator: Matrix Product State
    Implementation: CPU
    Documentation: https://quimb.readthedocs.io/en/latest/tensor-circuit-mps.html
    """
    def __init__(self, device='QuimbMPSCpu',  bonddim = 256, cutoff: float = 1e-10, permute: bool = False, gate_contract: str ='auto-mps', **kwargs):
        """
        Backend for Quimb MPS simulations.
        :param device: Name of the device, defaults to 'QuimbMPS'.
        :type device: str, optional
        :param bonddim: Maximum bond dimension for MPS or QNT, defaults to 512.
        :type bonddim: int, optional
        :param nshots: Number of shots for sampling, defaults to 1000.
        :type nshots: int, optional
        :param gate_contract: Contraction mode; possible values: "auto-mps", "swap+split" (defaults - 'auto-mps').
        :type gate_contract: str, optional
        :param cutoff: Truncation threshold.
        :type cutoff: float
        :param permute: Flag for a permutation-based strategy that applies non-local two-qubit gate contractions using swap+split (F for False, T for True)
        :type permute: bool
        """
        config = SimulatorConfig(
            device=device, device_type="mps", package_version=quimb.__version__, **kwargs)
        super().__init__(config)
        self.add_config_attr("bonddim", bonddim)
        self.add_config_attr("gate_contract", gate_contract)
        self.add_config_attr("cutoff", cutoff)
        self.add_config_attr("permute", permute)
        
       

    @ AbstractBackend._measure_time
    def generate_backend(self):
        return null    

    @AbstractBackend._measure_time
    def parse(self):
        """
        Parses the loaded QASM string into a Quimb circuit.
        """
        if self._qasm_str is None:
            raise ValueError("Error: No QASM loaded. Please load a QASM file first.")
     
        self._quimb_circuit = qtn.CircuitMPS(1).from_openqasm2_str(
            self._qasm_str,
            max_bond=int(self.config.bonddim),
            cutoff=float(self.config.cutoff),
            gate_contract=str(self.config.gate_contract),
        )
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
            qasm_str = file.read()
        self._qasm_str = qasm_str
        circuit = qtn.CircuitMPS(1).from_openqasm2_str(
            self._qasm_str,
            max_bond=int(self.config.bonddim),
            cutoff=float(self.config.cutoff),
            gate_contract=str(self.config.gate_contract)
        )
        return circuit

    def _circuit_stats(self, circuit):
        return len(circuit.calc_qubit_ordering()), circuit.num_gates

    def _compile(self, circuit):
        if self.config.permute:
            nq, ng = self._circuit_stats(circuit)
            circuit2 = qtn.CircuitPermMPS(nq,
            max_bond=int(self.config.bonddim),
            cutoff=float(self.config.cutoff),
            gate_contract=str(self.config.gate_contract),
            )
            circuit2. apply_gates(circuit.gates)
            return circuit2
        else: 
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
        circuit = self._import_qasm(qasm_file)
        mirror_circuit = self._import_qasm(mirror_qasm_file)
        circuit.apply_gates(mirror_circuit.gates)
        circuit = self._compile(circuit)
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



        



