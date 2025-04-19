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
from .abstract_simulator_backend import AbstractSimulatorBackend
from .qasm_parser import QasmParser
from .abstract_config import SimulatorConfig
import stim
import functools
from collections import Counter


class StimCpuBackend(AbstractSimulatorBackend, QasmParser):
    """
    Abstract Class of Stim Backend
    Type of simulator: StateVector 
    Implementation: CPU
    """
    _gates = ['X', 'Y', 'Z', 'H', 'S', 'S_DAG', 'CX', 'CNOT', 'CY', 'CZ',
              'SWAP', 'I', 'M', 'R', 'MR', 'MX', 'MRX', 'MRY']

    def __init__(self, **kwargs):
        """
        Constructor for Stim
        No specific parameters for stim

        Here is a full example to use this backend:

        .. code-block:: python

            # to generate the backend and pass it options
            backend = StimBackend(test_case='path_to_qasm.qasm', nb_shots=1)
            # To parse the qasm file
            backend.parse()
            # To execute and sample the results
            samples = backend.execute_and_sample()
            # To foramt the samples in a json format
            samples = backend.format_sample(samples)
            # To retrieve all the options used in the simulation
            config = backend.get_config_dict()
        """

        # Forced config
        kwargs['precision'] = 'double'  # TODO: check this
        kwargs['fusion_enable'] = False
        kwargs['max_parallel_threads'] = 1

        config = SimulatorConfig(device='Stim_Cpu', device_type='clifford',
                                 package_version=stim.__version__, **kwargs)
        super().__init__(config)

        self._backend = self.generate_backend()
        # Creates the attributes for the gates applyable
        self.create_attr()

    def apply_gate(self, *args):
        """
        Defines how to apply the most basic gates in Stim.

        Returns:
            _type_: _description_
        """
        if args[0] == 'M':
            self._backend.append('M', args[1][0])
        else:
            self._backend.append(args[0], args[1])

    def create_attr(self):
        """
        Creates the attributes for the gates in Stim.
        """
        for att in self._gates:
            att_func = functools.partial(self.apply_gate, att)
            setattr(self, att.upper(), att_func)

    @AbstractSimulatorBackend._measure_time
    def generate_backend(self):
        """
        Generate the fast circuit simulator backend.

        Returns:
            Circuit: The backend to use for the simulation
        """
        return stim.Circuit()

    @AbstractSimulatorBackend._measure_time
    def parse(self):
        """
        Custom parsing for Stim.\n Results stored in self._circuit.
        """
        nqubits, gatelist = self.parse_qasm(self._qasm_str)
        for gate in gatelist:
            getattr(self, gate[0].upper())(gate[1])

    def plot_circuit(self):
        print(self._backend.diagram())

    @AbstractSimulatorBackend._measure_time
    def sample_only(self):
        """
        Get the samples from the Stim simulation.

        Returns:
            list: list of the samples
        """
        return self._sampler.sample(self.config.nb_shots)

    @AbstractSimulatorBackend._measure_time
    def execute_only(self):
        """
        Not sure this is actually applying the gates to the circuit. TODO: check
        """
        self._sampler = self._backend.compile_sampler(seed=self.config.seed)

    @AbstractSimulatorBackend._measure_time
    def execute_and_sample(self):
        self.execute_only()
        return self.sample_only()

    @AbstractSimulatorBackend._measure_time
    def format_sample(self, samples):
        """
        Formats the result of a Stim simulation into a dictionary of counts.

        Args:
            samples (list): The raw samples from the Stim simulation.

        Returns:
            dict: The dictionary of counts
        """
        res = []
        for sample in samples:
            res.append(''.join(map(lambda x: str(int(x)), sample)))

        return dict(Counter(res))
