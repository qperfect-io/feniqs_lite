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
import os.path
import unittest
import sys
from shutil import rmtree
import numpy as np
import qtealeaves.emulator
from .abstract_backend import AbstractBackend
from .abstract_simulator_backend import AbstractSimulatorBackend
from .abstract_config import SimulatorConfig
from abc import ABCMeta, abstractmethod

from qiskit import QuantumCircuit, transpiler, transpile, ClassicalRegister
from qiskit.circuit.library import QuantumVolume

from qtealeaves.observables import TNObsProjective, TNObservables, TNState2File
from qmatchatea import QCOperators, QCBackend, QCConvergenceParameters, run_simulation, __version__ as qmatchatea_version
from qmatchatea.py_emulator import QCEmulator

from qmatchatea.utils.qk_utils import qk_transpilation_params


class QMatchaTeaAbstractBackend(AbstractSimulatorBackend, metaclass = ABCMeta):

    def __init__(self, device = 'Qmatchatea', backend = 'PY', linearize_enable = True, optimization_level = 1, 
                 tensor_compilator_enable = True, max_bond_dimension = 1, cut_ratio = 1e-6, svd_ctrl='V', **kwargs):
        """
        Initialize the backend for the Qmatchatea simulation

        :param backend: DEfines if the backend is run using pytho or fortran, defaults to 'PY', alternatives: 'PY', 'FR'
        :type backend: str, optional
        :param linearize_enable: activates linearization optimization, defaults to True
        :type linearize_enable: bool, optional
        :param optimization_level: Optimization level to use for the qiksit transpiler, defaults to 1
        :type optimization_level: _type_, optional
        :param tensor_compilator_enable: Activate the tensor compilator optimization, defaults to True
        :type tensor_compilator_enable: bool, optional
        :param max_bond_dimension: _description_, defaults to 1
        :type max_bond_dimension: int, optional
        """
        # config example can be found in here: https://baltig.infn.it/quantum_matcha_tea/py_api_quantum_matcha_tea/-/tree/master/examples?ref_type=heads

        # overwrite forced config
        kwargs['fusion_enable'] = False

        config = SimulatorConfig(
            device=device, device_type = 'matrix_product_state', package_version = qmatchatea_version, **kwargs)
        super().__init__(config)
        # All none common config
        self.add_config_attr("linearize_enable", linearize_enable)
        self.add_config_attr("optimization_level", optimization_level)
        self.add_config_attr("tensor_compilator_enable",
                             tensor_compilator_enable)
        self.add_config_attr("cut_ratio", cut_ratio)   
        self.add_config_attr("svd_ctrl", svd_ctrl)                    
        self.add_config_attr("laguage_backend", backend)
        self.add_config_attr("max_bond_dimension", max_bond_dimension)
        self._backend = self.generate_backend()

    @abstractmethod
    def generate_backend(self):
        raise NotImplementedError

    @AbstractBackend._measure_time
    def parse(self):
        """
        Parses the loaded QASM string into a Qiskit quantum circuit.\n
        Results stored in self._qiskit_circuit.
        """
        if self._qasm_str is None:
            raise ValueError(
                "Error: No QASM loaded. Please load a QASM file first.")
        self._qiskit_circuit = QuantumCircuit.from_qasm_str(
            self._qasm_str)

        # updates the number of qubits if first naive check was wrong
        self.config.nb_qubits = self._qiskit_circuit.num_qubits

        # Check if measurements are present in the circuit
        ops = self._qiskit_circuit.count_ops()
        if 'measure' not in ops:
            self._qiskit_circuit.measure_all()

    @AbstractBackend._measure_time
    def format_sample(self, samples):
        """
        Returns the formatted samples\n
        Nothing specific to do for qmatchatea

        Args:
            samples (dict): The raw samples from the QmatchaTea simulation.

        Returns:
            dict: The dictionary of counts
        """
        samples = {k: int(v) for k, v in samples.items()}

        return samples

    @AbstractSimulatorBackend._measure_time
    def sample_only(self):
        """
        Gets the samples from the result of the simulation.

        Returns:
            dict: The dictionary of counts
        """
        return self._result.observables['projective_measurements']

    @AbstractSimulatorBackend._measure_time
    def execute_only(self):
        """
        Executes the quantum circuit on the backend.
        """
        # creates the observables
        observables = TNObservables()
        # Projection for everyshot
        observables += TNObsProjective(self.config.nb_shots)

        options = {}
        if getattr(self.config, 'max_bond_dimension', None):
            options['max_bond_dimension'] = self.config.max_bond_dimension

        #conv_params = QCConvergenceParameters(
        #    trunc_tracking_mode="C", **options)

        conv_params = QCConvergenceParameters(
            max_bond_dimension = self.config.max_bond_dimension,
            cut_ratio = self.config.cut_ratio,
            svd_ctrl = self.config.svd_ctrl
        )

       # true_state = qiskit_get_statevect(self._qiskit_circuit)

       # run the simulation
        qk_params = qk_transpilation_params(
            linearize=self.config.linearize_enable, optimization=self.config.optimization_level, tensor_compiler=self.config.tensor_compilator_enable)

        results = run_simulation(self._qiskit_circuit, backend=self._backend,
                                 convergence_parameters=conv_params,
                                 observables=observables, transpilation_parameters=qk_params)

        #self.fidelity = results.fidelity

        return results
    
    def _import_qasm(self, qasm_file):
        """
        Import a QASM file, removing measurement operations.
        """
        with open(qasm_file, 'r') as file:
            qasm_str = ''.join(
                line for line in file if not any(op in line.lower() for op in ['measure', 'creg'])
            )
        return QuantumCircuit.from_qasm_str(qasm_str)

    def _compile(self, circuit):
        """
        Compile the circuit (uses Qiskit).
        """
        return transpile(circuit, optimization_level=1)

    def get_mirror_fidelity(self, qasm_file, mirror_qasm_file):
        """
        Calculate the mirror fidelity of a circuit.
        """
        qc = self._compile(self._import_qasm(qasm_file))
        qc_mirror = self._compile(self._import_qasm(mirror_qasm_file))
        qc = qc.compose(qc_mirror)
        self._qiskit_circuit = qc
        res = self.execute()    
        initial_state = '0' * qc.num_qubits
        if self._qiskit_circuit.num_qubits >= 1024:
            initial_state = '000...000'
        measurements = self._result.observables['projective_measurements']
        count = measurements.get(initial_state, 0)
        return count/self.config.nb_shots

    def execute(self):
        """
        Executes the parsed file and computes the fidelity
        """
        self._result = self.execute_only()

    @abstractmethod
    def compute_fidelity(self, singular_values_cut):
        pass

    @AbstractSimulatorBackend._measure_time
    def execute_and_sample(self):
        """
        Runs the simulation and gets the samples.

        Returns:
            dict: the dictionary of counts
        """
        self.execute()
        return self.sample_only()

    def get_precision(self):
        """
        Returns the precision type
        'Z' - double
        'C' - float
        Returns:
            _type_: _description_
        """

        if self.config.precision == "double":
            return "Z"
        return "C"


class QMatchaTeaCpuBackend(QMatchaTeaAbstractBackend):

    def __init__(self, **kwargs):
        """
        Constructor for the QMatchTeaCpu backend.
        Refer to the parent class QmatachateaAbstractBackend to see all supported argument.

        Here is a full example to use this backend:

        .. code-block:: python
            # to generate the backend and pass it options
            backend = QmatchateaCpuBackend(test_case = 'path_to_qasm.qasm', nb_shots = 1, backend = 'PY', linearize_enable = True, 
                                           optimization_level = 1, tensor_compilator_enable = True, cut_ratio = 1e-6, std_ctrl = 'V', max_bond_dimension = 1)
            # To parse the qasm file
            backend.parse()
            # To execute and sample the results
            samples = backend.execute_and_sample()
            # To foramt the samples in a json format
            samples = backend.format_sample(samples)
            # To retrieve all the options used in the simulation
            config = backend.get_config_dict()
        """
        super().__init__(device='Qmatchatea_Cpu', **kwargs)

    def compute_fidelity(self, singular_values_cut):
        return np.prod(1 - np.array(singular_values_cut))

    def generate_backend(self):
        """
        Generates the backend to use for the simulation
        """
        return QCBackend(backend="PY", ansatz = "MPS", precision=self.get_precision(), device="cpu")  # , num_procs=self.config.max_parallel_threads)


class QMatchaTeaGpuBackend(QMatchaTeaAbstractBackend):
    def __init__(self, **kwargs):
        super().__init__(device='Qmatchatea_Gpu', **kwargs)
        self.cp = self._import_cupy()

    def _import_cupy(self):
        try:
            import cupy as cp
            return cp
        except ImportError:
            raise ImportError(
                "CuPy is not installed. Please install CuPy to use the GPU backend.")

    def compute_fidelity(self, singular_values_cut):
        cp = self._import_cupy()
        return np.prod(1 - cp.array(singular_values_cut).get())

    def generate_backend(self):
        """
        Generates the backend to use for the simulation
        """
        return QCBackend(backend = "PY", precision=self.get_precision(), device = "gpu", num_procs = self.config.max_parallel_threads)
