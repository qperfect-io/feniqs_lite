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
import sys
import glob
import json
import tempfile
from abc import ABCMeta, abstractmethod
from .abstract_backend import AbstractBackend
from .abstract_simulator_backend import AbstractSimulatorBackend
from .abstract_config import SimulatorConfig
import pkg_resources
# Import Qrack API classes from your Qrack Python package.
from pyqrack.qrack_circuit import QrackCircuit
from pyqrack.qrack_simulator import QrackSimulator
import pyqrack
from qiskit import QuantumCircuit
from pyqrack.qrack_system import Qrack

class PyqrackCpuBackend(AbstractSimulatorBackend, metaclass=ABCMeta):
    def __init__(self, device='Pyqrack', qubitCount=None,
                 isTensorNetwork=True,
                 isSchmidtDecomposeMulti=True,
                 isSchmidtDecompose=True,
                 isStabilizerHybrid=True,
                 isBinaryDecisionTree=False,
                 isPaged=False,
                 isCpuGpuHybrid=False,
                 isOpenCL=True,
                 isHostPointer=False,
                 noise=0,
                 nb_shots=1024,
                 **kwargs):
        """
        Qrack CPU backend with configurable parameters.
        
        :param device: Device name.
        :param qubitCount: Number of qubits (if known; otherwise determined from QASM).
        :param isTensorNetwork: Use tensor network methods.
        :param isSchmidtDecomposeMulti: Enable multi-Schmidt decomposition.
        :param isSchmidtDecompose: Enable Schmidt decomposition.
        :param isStabilizerHybrid: Use a stabilizer hybrid method.
        :param isBinaryDecisionTree: Use binary decision tree method.
        :param isPaged: Use paged simulation.
        :param isCpuGpuHybrid: Enable CPU/GPU hybrid mode.
        :param isOpenCL: Enable OpenCL.
        :param isHostPointer: Enable host pointer mode.
        :param noise: Noise level.
        :param nb_shots: Number of shots for sampling.
        :param kwargs: Additional parameters.
        """
        config = SimulatorConfig(
            device=device, device_type="tn", package_version=pkg_resources.get_distribution("pyqrack").version,
            qubitCount=qubitCount,
            is_tensor_network=isTensorNetwork,
            is_schmidt_decompose_multi=isSchmidtDecomposeMulti,
            is_schmidt_decompose=isSchmidtDecompose,
            is_stabilizer_hybrid=isStabilizerHybrid,
            is_binary_decision_tree=isBinaryDecisionTree,
            is_paged=isPaged,
            is_cpu_gpu_hybrid=isCpuGpuHybrid,
            is_opencl=isOpenCL,
            is_host_pointer=isHostPointer,
            noise=noise,
            nb_shots=nb_shots, **kwargs)

        super().__init__(config)

    @AbstractBackend._measure_time
    def generate_backend(self):
        # Qrack does not require an explicit backend generation step.
        return None

    @AbstractBackend._measure_time
    def parse(self):
        if self._qasm_str is None:
            raise ValueError("Error: No QASM loaded. Please load a QASM file first.")
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp:
            tmp.write(self._qasm_str)
            tmp_filename = tmp.name
        circ = QuantumCircuit.from_qasm_file(tmp_filename)
        n = circ.num_qubits
        self._qrack_circuit = QrackCircuit.in_from_qiskit_circuit(circ)
        os.remove(tmp_filename)
        return self._qrack_circuit

    def _import_qasm(self, qasm_file):
        """
        Imports a QASM file into a Qrack circuit using the file interface.
        """
        with open(qasm_file, "r") as file:
           qasm_str = file.read()

        return self.parse()

    @AbstractSimulatorBackend._measure_time
    def execute_only(self):
        """
        Executes the Qrack circuit simulation.
        Constructs a QrackSimulator with the specified parameters and runs the circuit.
        """
        if self._qrack_circuit is None:
            raise ValueError("Error: No QASM loaded. Please run parse() first.")
        nq = self._qrack_circuit.get_qubit_count()

        self._qrack_sim = QrackSimulator(
             qubitCount=nq,
             isTensorNetwork=True, 
             isSchmidtDecomposeMulti=True,
             isSchmidtDecompose=True,
             isStabilizerHybrid=True,
             isBinaryDecisionTree=False, 
             isPaged=False, 
             isCpuGpuHybrid=False, 
             isOpenCL=False, 
            isHostPointer=False, 
            noise=False 
        )
        self._qrack_circuit.run(self._qrack_sim)
        fidelity = self._qrack_sim.get_unitary_fidelity()
        self._result = self._qrack_sim

    @AbstractSimulatorBackend._measure_time
    def sample_only(self):
        """
        Samples measurement outcomes from the Qrack simulation result.
        Assumes that the QrackSimulator object provides a sample(nb_shots) method.
        """
        if self._result is None:
            raise ValueError("Error: Simulation result not available. Please run execute_only() first.")
        return self._qrack_sim.m_all() 

    @AbstractSimulatorBackend._measure_time
    def execute_and_sample(self):
        """
        Executes the simulation and obtains measurement samples.
        """
        self.execute_only()
        return self.sample_only()

    def format_sample(self, samples):
        """
        Formats the measurement outcomes into a dictionary of counts.
        This method implements the abstract format_sample method.
        
        :param samples: List of sample outcomes (e.g., as strings).
        :return: Dictionary mapping each outcome to its count.
        """
        return 0 
    def get_mirror_fidelity(self, qasm_file, mirror_qasm_file):
        """
        Computes the mirror fidelity of a circuit.
        
        Loads the original circuit from qasm_file and the mirror circuit from mirror_qasm_file,
        composes them into a single circuit, runs the combined circuit on a QrackSimulator,
        then retrieves the state vector via the simulator's out_ket() method.
        The mirror fidelity is defined as the squared magnitude of the amplitude of the 
        all-zero state.
        
        :param qasm_file: Path to the QASM file for the original circuit.
        :param mirror_qasm_file: Path to the QASM file for the mirror circuit.
        :return: Mirror fidelity, i.e. |⟨0...0|ψ⟩|².
        """
        from qiskit import QuantumCircuit

        # Load the circuits from QASM files.
        mirror_qcirc = QuantumCircuit.from_qasm_file(mirror_qasm_file)
        normal_qcirc = QuantumCircuit.from_qasm_file(qasm_file)

        # Compose the circuits (mirror_qcirc followed by normal_qcirc).
        combined_qcirc = mirror_qcirc.compose(normal_qcirc)

        # Convert the combined Qiskit circuit into a QrackCircuit.
        combined_qrack = QrackCircuit.in_from_qiskit_circuit(combined_qcirc)

        # Determine the number of qubits.
        nq = combined_qrack.get_qubit_count()

        # Create a new QrackSimulator with the specified number of qubits.
        sim = QrackSimulator(
            qubitCount=nq,
            isTensorNetwork=True, 
            isSchmidtDecomposeMulti=True,
            isSchmidtDecompose=True,
            isStabilizerHybrid=True,
            isBinaryDecisionTree=False, 
            isPaged=False, 
            isCpuGpuHybrid=False, 
            isOpenCL=False, 
            isHostPointer=False, 
            noise=False 
        )

        # Run the combined circuit on the simulator.
        combined_qrack.run(sim)
        # The all-zero state corresponds to the first element (index 0) of the state vector.
        shots = self.config.nb_shots if hasattr(self, 'config') and hasattr(self.config, 'nb_shots') else 1024
        # Measure all qubits for the given number of shots.
        results = sim.measure_shots(list(range(nq)), shots)
        # Count the number of outcomes that are all zeros.
        count_all_zero = results.count(0)
        # Mirror fidelity is the fraction of shots with the all-zero outcome.
        fidelity = count_all_zero / shots
        return fidelity
