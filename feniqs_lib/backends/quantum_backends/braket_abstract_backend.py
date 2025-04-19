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
from .abstract_backend import AbstractBackend
from .abstract_simulator_backend import AbstractSimulatorBackend
from .abstract_config import BasicConfig, SimulatorConfig
from abc import ABCMeta, abstractmethod

from braket.ir.openqasm import Program
import re


class BraketAbstractBackend(AbstractBackend, metaclass=ABCMeta):
    """
    This class implements the abstract interface for the Brakets backend.
    """

    def __init__(self, config: BasicConfig):
        """
        Initialize the Braket backend.

        Args:
            config (BasicConfig): The configuration object for the backend w/o the specifics of Braket.
        """
        super().__init__(config)
        with open("feniqs_empiriqa/feniqs/backends/utils/qelib1.inc", "r") as f:
            self._qelib1_inc = f.read()
        self._backend = self.generate_backend()

    @AbstractBackend._measure_time
    def parse(self):
        """
        Parse the loaded QASM into a Braket program.
        """
        # We need to remove the include lines for braket
        self._qasm_str = str(re.sub(r'^include.*\n?', self._qelib1_inc,
                                    self._qasm_str, flags=re.MULTILINE))
        # braket can't interpret CX so replaced by cnot
        self._qasm_str = str(re.sub(r'CX', 'cnot',
                                    self._qasm_str, flags=re.MULTILINE))

        if not self._qasm_str:
            raise ValueError(
                "Error: QASM content is empty. Please load a QASM file first.")

        self._braket_program = Program(source=self._qasm_str)

    @AbstractBackend._measure_time
    def execute_only(self):
        """
        Executes the circuit on the backend.
        """
        self._results = self._backend.run(
            task_specification=self._braket_program, shots=self.config.nb_shots)

    @AbstractBackend._measure_time
    def format_sample(self, samples):
        """
        Foramts the raw samples from the backend into a dictionary of counts.

        Args:
            samples (Any): The raw samples from the simulation directly from sample_only or execute_and_sample

        Raises:
            NotImplementedError: Implementation has to be backend specific
        """
        return dict(samples)

    @AbstractBackend._measure_time
    def sample_only(self):
        """
        Samples measurement outcomes from the executed circuit.
        This is not needed for Braket's backends
        """
        return self._results.result().measurement_counts

    @AbstractBackend._measure_time
    def execute_and_sample(self):
        """
        Executes and samples the circuit on the backend.
        Is the default way to execute and sample on the benchamrk as some simulators do not allow for differentiations between the two.
        """
        self.execute_only()
        return self.sample_only()

    def run(self):
        """
        Runs everything in the benchmark after in initialization

        Returns:
            Any: the raw samples from the simulation
        """
        self.parse()
        timing = self.time_value
        res = self.execute_and_sample()
        self.time_value += timing
        return res
