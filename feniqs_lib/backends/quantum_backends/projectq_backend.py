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

import numpy as np
from feniqs_lib.tools.backends_tools import remove_barriers
from .abstract_simulator_backend import AbstractSimulatorBackend
from .abstract_config import SimulatorConfig
from .qasm_parser import QasmParser
import projectq
import os


class ProjectQBackend(AbstractSimulatorBackend, QasmParser):

    def __init__(self, optimizer: int = 0, max_parallel_threads: int = 1, fusion_enable: bool = True, **kwargs):
        """
        Constructor for the ProjectQ Backend

        :param optimizer: set the level of the Circuit optimization compiler engine, defaults to 0
        :type optimizer: int, optional
        :param max_parallel_threads: sets the number of threads to use, defaults to 1
        :type max_parallel_threads: int, optional
        :param fusion_enable: Activates The fusion optimization on the circuits, defaults to True
        :type fusion_enable: bool, optional

        Here is a full example to use this backend:

        .. code-block:: python

            # to generate the backend and pass it options
            backend = ProjectQBackend(test_case='path_to_qasm.qasm', nb_shots=1, optimizer=0, max_parallel_threads=1, fusion_enable=True)
            # To parse the qasm file
            backend.parse()
            # To execute and sample the results
            samples = backend.execute_and_sample()
            # To foramt the samples in a json format
            samples = backend.format_sample(samples)
            # To retrieve all the options used in the simulation
            config = backend.get_config_dict()
        """
        version = self._get_version(projectq.__name__)
        # cannot be set
        kwargs['precision'] = 'single'
        # TODO: next two line should have been replaced in pluggin manager, check if this is working now
        os.environ['OMP_NUM_THREADS'] = str(max_parallel_threads)
        os.environ['OMP_PROC_BIND'] = 'spread'
        config = SimulatorConfig(device='Projectq_Cpu', device_type='statevector',
                                 package_version=version, fusion_enable=fusion_enable, max_parallel_threads=max_parallel_threads, **kwargs)
        super().__init__(config)
        self.add_config_attr('optimizer', optimizer)

        self._qasm_str = self._load(self.config.test_case)
        self._circuit = None
        self.nb_qubits = self._count_qubits_from_qasm(self._qasm_str)

        self.custom_parser = False

        self._backend, self._eng, self._qureg = self.generate_backend()

    @AbstractSimulatorBackend._measure_time
    def generate_backend(self):
        """
        Generates the backend and engine for ProjectQ.

        Returns:
            tuple: the backend and engine
        """
        # backend
        backend = projectq.backends.Simulator(
            gate_fusion=self.config.fusion_enable, rnd_seed=self.config.seed)
        # engine
        if self.config.optimizer:
            eng = projectq.MainEngine(
                backend=backend, engine_list=[
                    projectq.cengines.LocalOptimizer(self.config.optimizer)]
            )
        else:
            eng = projectq.MainEngine(backend=backend)
        # register
        qureg = eng.allocate_qureg(self.nb_qubits)
        return backend, eng, qureg

    def _print_circuit(self, circuit):
        for gate, target in circuit:
            target = target[0] if type(target) == tuple else target
            print(gate, target.id)

    @AbstractSimulatorBackend._measure_time
    def parse(self):
        """
        Parse the QASM string into a ProjectQ quantum circuit.
        """
        circuit = []

        self._qasm_str = remove_barriers(self._qasm_str)

        nqubits, gatelist = self.parse_qasm(self._qasm_str)

        # in case the first check for nb_qubits failed
        if nqubits != self.nb_qubits:
            self._qureg = self._eng.allocate_qureg(nqubits)

        # iterates over the parsed gates
        for gatename, qubits, params in gatelist:
            if gatename == "M":
                circuit.append((projectq.ops.Measure, self._qureg[qubits[0]]))
            else:
                # all the gates are defined in the class
                gate = getattr(self, gatename)
                if params is not None:
                    parameters = list(params)
                    if len(qubits) > 1:
                        circuit.append(
                            (gate(*parameters), tuple(self._qureg[i] for i in qubits)))
                    else:
                        circuit.append(
                            (gate(*parameters), self._qureg[qubits[0]]))

                # gates with no parameters
                elif len(qubits) > 1:
                    circuit.append(
                        (gate, tuple(self._qureg[i] for i in qubits)))
                else:
                    circuit.append((gate, self._qureg[qubits[0]]))
        self._circuit = circuit

    def _to_common_value(self, x):
        """
        Translates the unusual -1 and 1 to the commonly use 0 and 1 states.
        """
        return '0' if int(x) == 1 else '1'

    @AbstractSimulatorBackend._measure_time
    def _sample_only(self):
        """
        Samples measurement outcomes from the quantum circuit.
        """
        res = {}

        # Samples nb_shots times
        for _ in range(self.config.nb_shots):
            curr_res = ''
            # Measure each qubit
            for qubit in self._qureg:
                exp = self._backend.get_expectation_value(
                    projectq.ops.QubitOperator(f'Z{qubit.id}'), self._qureg)
                curr_res += self._to_common_value(int(exp))

            curr_res = curr_res

            # add to the dict the measurement results
            res[curr_res] = res.get(curr_res, 0) + 1

        return res

    def sample_only(self):
        """
        Gets the samples from the ProjectQ simulation.
        And Measure all qubits in the circuit. for projectq to not be anoying with exceptions

        Returns:
            _type_: _description_
        """
        # Measure all qubits in the circuit. for projectq to not be anoying with exceptions
        for qubits in self._qureg:
            projectq.ops.Measure | qubits
        res = self._sample_only()
        return res

    @AbstractSimulatorBackend._measure_time
    def execute_only(self):
        """
        Executes the quantum circuit on the backend.
        """
        for gate, target in self._circuit:
            gate | target
        self._eng.flush()

    @AbstractSimulatorBackend._measure_time
    def execute_and_sample(self):
        """
        Executes and samples the quantum circuit on the backend.
        """
        self.execute_only()
        return self.sample_only()

    @AbstractSimulatorBackend._measure_time
    def format_sample(self, samples):
        """
        Nothing to do for ProjectQ.
        """
        return samples


    SDG = projectq.ops.Sdag
    TDG = projectq.ops.Tdag
    SWAP = projectq.ops.Swap
    CCX = projectq.ops.Toffoli
    SQRTX = projectq.ops.SqrtX
    SQRTXDG = projectq.ops.DaggeredGate(projectq.ops.SqrtX)

    def RX(self, theta): return projectq.ops.Rx(theta)
    def RY(self, theta): return projectq.ops.Ry(theta)
    def RZ(self, theta): return projectq.ops.Rz(theta)
    def RZZ(self, theta): return projectq.ops.Rzz(theta)
    def CRX(self, theta): return projectq.ops.C(self.RX(theta))
    def CRY(self, theta): return projectq.ops.C(self.RY(theta))
    def CRZ(self, theta): return projectq.ops.CRz(theta)
    def U1(self, theta):
        return projectq.ops.R(theta)

    def U2(self, phi, lam):
        pplus, pminus = np.exp(0.5j * (phi + lam)), np.exp(0.5j * (phi - lam))
        matrix = np.array([[np.conj(pplus), -np.conj(pminus)],
                           [pminus, pplus]])
        matrix /= np.sqrt(2)
        return projectq.ops.MatrixGate(matrix)

    def U3(self, theta, phi, lam):
        cost, sint = np.cos(theta / 2.0), np.sin(theta / 2.0)
        pplus, pminus = np.exp(0.5j * (phi + lam)), np.exp(0.5j * (phi - lam))
        matrix = np.array([[np.conj(pplus) * cost, -np.conj(pminus) * sint],
                           [pminus * sint, pplus * cost]])
        return projectq.ops.MatrixGate(matrix)

    def CU1(self, theta):
        U1 = projectq.ops.R(theta)
        return projectq.ops.C(U1, n_qubits=1)

    def __getattr__(self, x):
        return getattr(projectq.ops, x)

    def __item__(self, x):
        return getattr(projectq.ops, x)

    def set_precision(self, precision):
        if precision != "double":
            raise NotImplementedError(
                f"Cannot set {precision} precision for {self.name} backend.")

    def get_precision(self):
        return "double"

    def get_device(self):
        return None
