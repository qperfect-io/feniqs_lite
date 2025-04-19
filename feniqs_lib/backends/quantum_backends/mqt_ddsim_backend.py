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
from abc import ABCMeta, abstractmethod
from .abstract_backend import AbstractBackend
from .abstract_simulator_backend import AbstractSimulatorBackend
from .abstract_config import SimulatorConfig
from .qasm_parser import QasmParser
from collections import defaultdict
import mqt.ddsim as ddsim
from qiskit import QuantumCircuit
from mqt.ddsim import CircuitSimulator, UnitarySimulator


class DdsimAbstractBackend(AbstractSimulatorBackend, QasmParser, metaclass=ABCMeta):
    """
    DDSIM implementation of the AbstractBackend. Handles the following operations:
    loading, parsing, executing, and sampling quantum circuits.
    """
    @abstractmethod
    def __init__(self, device: str = 'MqtDDSim',  **kwargs):
        """
        Constructor for the MQT DDSIM backend 

        :param device: the provider, defaults to 'MqtDDSim'
        :type device: str, optional
        :param simulator_type: The simulator to use, defaults to 'qasm_simulator'
        :type simulator_type: str, optional
        """
        # Ensure only expected config parameters are passed
        # simulator type can be : 'qasm_simulator', 'statevector_simulator', 'hybrid_qasm_simulator', 'hybrid_statevector_simulator', 'path_sim_qasm_simulator', 'path_sim_statevector_simulator', 'unitary_simulator'
        #self.simulator_type = simulator_type
        config_kwargs = {
            'device': device,
            'device_type': kwargs.get('device_type'),
            'package_version': ddsim.__version__,
            'test_case': kwargs.get('test_case'),
            'nb_shots': kwargs.get('nb_shots', 1),
            'seed': kwargs.get('seed'),
            'precision': 'double',
            'max_parallel_threads': kwargs.get('max_parallel_threads', 1),
            'fusion_enable': False
        }
  
        config = SimulatorConfig(**config_kwargs)
        super().__init__(config)
        self.simulator_type = self.config.device_type
        self._qiskit_circuit = None
        self._backend = self.generate_backend()
        self._job = None
        self._result = None

    def generate_backend(self):
        """
        Generate the DDSIM backend based on the simulator type.
        """
        if self.simulator_type == 'qasm_simulator':
            return ddsim.DDSIMProvider().get_backend("qasm_simulator")
        elif self.simulator_type == 'circuit_simulator':
            return CircuitSimulator
        elif self.simulator_type == 'unitary_simulator':
            return UnitarySimulator
        else:
            raise ValueError(
                f"Unsupported simulator type: {self.simulator_type}")

    @AbstractSimulatorBackend._measure_time
    def parse(self):
        """
        Parses the loaded QASM string into a Qiskit quantum circuit.
        Results stored in self._qiskit_circuit.
        """
        if not self._qasm_str:
            raise ValueError(
                "No QASM content loaded. Please load a QASM file first.")

        self._qiskit_circuit = QuantumCircuit.from_qasm_str(self._qasm_str)
        self.config.nb_qubits = self._qiskit_circuit.num_qubits

    def plot_circuit(self):
        """
        Plots the Qiskit circuit.
        """
        print(self._qiskit_circuit.draw(fold=-1))

    @AbstractSimulatorBackend._measure_time
    def format_sample(self, samples):
        """
        Formats the result of a DDSIM simulation into a dictionary of counts.

        Args:
            samples (dict): Dictionary of counts from DDSIM simulation.

        Returns:
            dict: A dictionary where keys are binary strings of qubit states and values are counts of these states.
        """
        res = defaultdict(int)
        for key, val in samples.items():
            bin_sample = key[::-1]
            res[bin_sample] += val
        return dict(res)

    @AbstractSimulatorBackend._measure_time
    def sample_only(self):
        """
        Samples measurement outcomes from the quantum circuit.
        """
        if self.simulator_type in ['qasm_simulator', 'circuit_simulator']:
            if not self._result:
                raise ValueError(
                    "No execution result found. Please execute the circuit first.")
            return self.format_sample(self._result)
        else:
            raise ValueError(
                "Sampling is not supported for the selected simulator type.")

    @AbstractSimulatorBackend._measure_time
    def execute_only(self):
        """
        Executes the quantum circuit on DDSIM.
        Results stored in self._result.
        """
        if not self._qiskit_circuit:
            raise ValueError(
                "No Qiskit circuit defined. Please parse the QASM content first.")

        if self.simulator_type == 'qasm_simulator':
            self._job = self._backend.run(
                self._qiskit_circuit, shots=self.config.nb_shots)
            self._result = self._job.result().get_counts(self._qiskit_circuit)
        elif self.simulator_type == 'circuit_simulator':
            simulator = self._backend(self._qiskit_circuit)
            self._result = simulator.simulate(shots=self.config.nb_shots)
        elif self.simulator_type == 'unitary_simulator':
            simulator = self._backend(self._qiskit_circuit)
            simulator.construct()
            self._result = simulator.export_dd_to_graphviz_str(
                colored=True, edge_labels=True, classic=False)
        else:
            raise ValueError(
                f"Unsupported simulator type: {self.simulator_type}")

    def _import_qasm(self, qasm_file):
        """
        Import a QASM file, removing measurement operations.
        """
        with open(qasm_file, "r") as file:
            qasm_str = file.read()
        self._qasm_str = qasm_str
        circuit = QuantumCircuit().from_qasm_str(
            self._qasm_str
        )
        return circuit
    
    def execute(self):
        """
        Executes the parsed file and computes the fidelity
        """
        self.execute_only()
    
    def _compile(self, circuit): 
        return circuit

    def _circuit_stats(self, circuit):
        return circuit.num_qubits(), sum(circuit.count_ops().values())

    def get_mirror_fidelity(self, qasm_file, mirror_qasm_file):
        """
        Calculate the mirror fidelity of a circuit.
        """
        circuit = self._compile(self._import_qasm(qasm_file))
        circuit.remove_final_measurements(inplace=True)
        mirror_circuit = self._compile(self._import_qasm(mirror_qasm_file))
        mirror_circuit.remove_final_measurements(inplace=True)
        circuit.barrier()
        circuit = circuit.compose(mirror_circuit)
        self._qiskit_circuit = circuit
        self.execute()  
        initial_state = '0' * circuit.num_qubits 
        return  self._result.get(initial_state, 0)/self.config.nb_shots


        
    @AbstractSimulatorBackend._measure_time
    def execute_and_sample(self):
        """
        Executes and samples the quantum circuit.
        """
        self.execute_only()
        return self.sample_only()


class MqtDDSimCPUBackend(DdsimAbstractBackend):
    """
    DDSIM implementation of the AbstractBackend.
    """

    def __init__(self, simulator_type='qasm_simulator', **kwargs):
        """
        Constructor for the mqtddsim backend

        :param simulator_type: the types of simulators to use, defaults to 'qasm_simulator'
        :type simulator_type: str, optional

        Here is a full example to use this backend:

        .. code-block:: python

            # to generate the backend and pass it options
            backend = MqtDDsimCpuBackend(device_type='qasm_simulator', test_case='path_to_qasm.qasm', nb_shots=1)
            # To parse the qasm file
            backend.parse()
            # To execute and sample the results
            samples = backend.execute_and_sample()
            # To foramt the samples in a json format
            samples = backend.format_sample(samples)
            # To retrieve all the options used in the simulation
            config = backend.get_config_dict()
        """
 
        super().__init__(device='MqtDDSim_Cpu',  **kwargs)
