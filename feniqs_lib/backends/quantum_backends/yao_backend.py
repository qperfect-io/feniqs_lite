
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
from .abstract_julia_backend import AbstractJuliaBackend
from .abstract_config import SimulatorConfig
from .qasm_parser import QasmParser
from feniqs_lib.tools.backends_tools import remove_barriers
import functools
from abc import ABCMeta, abstractmethod

from juliacall import Main as jl
import os


class YaoAbstractBackend(AbstractJuliaBackend, QasmParser, metaclass=ABCMeta):
    """
    Abstract Class of Yao Backend
    Type of simulator: StateVector 
    Implementation: CPU, GPU
    """
    _gates = ['X', 'Y', 'Z', 'T', 'TDG', 'H', 'phase', 'shift', 'Rx', 'Ry', 'Rz',

              'rot', 'swap', 'I2', 'reflect', 'matrixgate', 'sqrtX', 'sqrtXdg', 'S', 'SDG']

    @abstractmethod
    def __init__(self, device: str, env: str, **kwargs):
        """Constructor of Yao Backend Class
        :param device: Simply indicates if the simulation uses a CPU or GPU
        :type device: str
        :param env: name of the julia environment to use
        :type env: str
        """
        # Forced config:
        kwargs['max_parallel_threads'] = 1
        kwargs['fusion_enable'] = False

        config = SimulatorConfig(
            device=device, device_type='statevector', package_version='MAGIC_VERSION_NUMBER', **kwargs)

        super().__init__(julia_module_name="yao_julia", env=env, config=config)

        self.import_packages()
        self.config.package_version = self._get_julia_package_version("Yao")
        self.create_specific_gates()
        self.create_attr()
        # Yao specific
        self.julia_module.seval(
            f"import Random; Random.seed!({self.config.seed})")
        self._circuit = None
        self.separates_execution_and_sampling = True
        self.custom_parser = False

        # Generates backend
        # called for consistency
        self.generate_backend()

    def apply_gate(self, *args):
        params = '' if len(args) == 2 else '(' + \
            ', '.join(map(lambda x: str(x), args[2:])) + ')'
        return f"put({args[1]}=>{args[0]}{params})"

    def create_attr(self):

        for att in self._gates:
            # Do not try to use lambda and save yourself pythonic trouble
            att_func = functools.partial(self.apply_gate, att)
            setattr(self, att.upper(), att_func)

    def create_specific_gates(self):
        """Creates gates that are not in the default list of gates defined by Yao.
        """
        # single qubit gates
        self.julia_module.seval(
            """sqrtX = GeneralMatrixBlock(1/2 * [[1 + 1im, 1 - 1im] [1-1im, 1+1im]]);
             sqrtXdg = Base.adjoint(sqrtX);
             S = shift(π/2);
             SDG = Base.adjoint(S);
             TDG = Base.adjoint(T);""")

        # used to define u2 and u3
        self.julia_module.seval("""function u3(theta::Real, phi::Real, lam::Real)
                                    return GeneralMatrixBlock([[cos(theta/2), exp(1im*phi)*sin(theta/2)] [-exp(1im*lam)*sin(theta/2), exp(1im*(phi+lam))*cos(theta/2)]])
                                end

                                function u2(phi::Real, lam::Real)
                                    return u3(pi/2, phi, lam)
                                end""")

    def _get_qreg_size(self):
        for line in self._qasm_str.splitlines():
            print(line)

    def _construct_gate(self, gatename, qubits, params):
        if hasattr(self, gatename):
            gate_func = getattr(self, gatename)
            args = qubits
            if params:
                args.extend(params)
            return gate_func(*args)
        else:
            raise NotImplementedError(
                f"Gate {gatename} is not implemented in this backend.")

    @AbstractJuliaBackend._measure_time
    def parse(self):
        """Parses the loaded QASM into a Yao circuit."""
        if not self._qasm_str:
            raise ValueError(
                "No QASM content loaded. Please load a QASM file first.")

        self.julia_module.seval("circuit = chain()")
        gates_count = 0
        max_gates_to_julia_at_once = 3000  # arbitrary number to be fine tuned
        self._qasm_str = remove_barriers(self._qasm_str)
        self.nqubits, gatelist = self.parse_qasm(self._qasm_str)
        circuit = []
        for gatename, qubits, params in gatelist:
            if gatename == 'M':
                continue
            # from string to Yao gate
            qubits = [qubit + 1 for qubit in qubits]
            gate = self._construct_gate(gatename.upper(), qubits, params)
            if gate:
                circuit.append(gate)

            gates_count += 1
            # split the circuit in multiple parts to avoid juliacall segfault
            if gates_count == max_gates_to_julia_at_once:
                self.julia_module.seval(
                    f"circuit = chain(circuit, {', '.join(circuit)})")
                gates_count = 0
                circuit = []

        if len(circuit):
            self.julia_module.seval(
                f"circuit = chain(circuit, {', '.join(circuit)})")

    def plot_circuit(self):
        pass

    @ AbstractJuliaBackend._measure_time
    def format_sample(self, samples):
        return self._from_sample_list_to_dict(map(lambda x: x[:-4][::-1], samples))

    def sample_only(self):
        """
        Samples measurement outcomes from the quantum circuit.
        """

        self.time_value, samples = self.julia_module.seval(f"""function custom_yao_sample()
                                                         res = []
                                                         for _ in 1:{
                                                             self.config.nb_shots}
                                                             push!(res, join(measure(reg)))
                                                         end
                                                         return res
                                                     end
                                                     timing = @elapsed samples = custom_yao_sample()
                                                     timing, samples""")

        return samples

    def execute_only(self):
        """
        Executes the quantum circuit on Yao.
        """
        timing = self.julia_module.seval("@elapsed Yao.apply!(reg, circuit)")
        self.time_value = timing

    def execute_and_sample(self):
        """
        Executes and samples the quantum circuit.

        Args:
            repetitions (int, optional): The number of samples to retrieve. Defaults to 1.

        Returns:
            dict: dict containing the results and the number of apparition of each result.
        """
        self.execute_only()
        exec_timing = self.time_value
        res = self.sample_only()
        self.time_value += exec_timing
        return res

    def get_precision(self):
        if self.config.precision == 'double':
            return 'ComplexF64'
        return 'ComplexF32'

    def M(self, qb_target, c_target, qb_register, c_register):
        return f"measure(reg, {qb_target})"

    def CNOT(self, control, target):
        return f"control({control}, {target}=>X)"

    def U2(self, target, phi, lam):
        return f"put({target}=>u2({phi}, {lam}))"

    def U3(self, target, theta, phi, lam):
        return f"put({target}=>u3({theta}, {phi}, {lam}))"


class YaoCpuBackend(YaoAbstractBackend):

    def __init__(self, env, **kwargs):
        """
        Constructor for the Yao backend on a cpu
        Refer to the parent classs for more information on the options.

        :param env: name of the julia environment created for yao
        :type env: str

        Here is a full example to use this backend:

        .. code-block:: python

            # to generate the backend and pass it options
            backend = YaoCpuBackend(env='yao_julia_cpu_env', test_case='path_to_qasm.qasm', nb_shots=1)
            # To parse the qasm file
            backend.parse()
            # To execute and sample the results
            samples = backend.execute_and_sample()
            # To foramt the samples in a json format
            samples = backend.format_sample(samples)
            # To retrieve all the options used in the simulation
            config = backend.get_config_dict()
        """
        super().__init__(device='Yao_Cpu', env=env, **kwargs)

    def generate_backend(self):
        """
        Creates the register to use for the simulation.
        """
        self.time_value = self.julia_module.seval(
            f"@elapsed reg = zero_state({self.get_precision()}, {self.config.nb_qubits})")
        return

    def import_packages(self):
        self.julia_module.seval("using Yao")


class YaoGpuBackend(YaoAbstractBackend):

    def __init__(self,  **kwargs):
        super().__init__(device='Yao_Gpu', **kwargs)

    def generate_backend(self):
        """
        Creates the register to use for the simulation.
        """
        self.time_value = self.julia_module.seval(
            f"@elapsed reg = cuzero_state({self.get_precision()}, {self.config.nb_qubits})")
        return

    def import_packages(self):
        self.julia_module.seval("using cuYao")
