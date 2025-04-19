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

from .abstract_simulator_backend import AbstractSimulatorBackend
from .abstract_config import SimulatorConfig
from Quantanium import Quantanium


class QuantaniumCpuBackend(AbstractSimulatorBackend):

    def __init__(self, device_type='statevector', bonddim=None, entdim=None, qasmincludes=None, **kwargs):
        """
        Constructor for quantanium

        :param device_type: The type of simulator to use (will be used as 'algorithm' in execute), defaults to 'statevector'
        :type device_type: str, optional
        :param bonddim: The bond dimension for the MPS algorithm, default to None
        :type bonddim: int, optional
        :param entdim: The entangling dimension for the MPS algorithm., default to None
        :type entdim: int, optional
        :param qasminclude: List of OPENQASM files to include in the execution
        :type qasminclude: list, optional

        Here is a full example to use this backend:

        .. code-block:: python

            # to generate the backend and pass it options
            backend = QuantaniumCpuBackend(test_case='path_to_qasm.qasm', nb_shots=1)
            # To parse the qasm file
            backend.parse()
            # To execute and sample the results
            samples = backend.execute_and_sample()
            # To foramt the samples in a json format
            samples = backend.format_sample(samples)
            # To retrieve all the options used in the simulation
            config = backend.get_config_dict()
        """

        config = SimulatorConfig(
            device="Quantanium_Cpu", device_type=device_type, package_version=self._get_version('quantaniumpy'),
            # Quantanium specific
            bonddim=bonddim, entdim=entdim, **kwargs)
        self._qasms = qasmincludes
        super().__init__(config)
        self._backend = self.generate_backend()

    @AbstractSimulatorBackend._measure_time
    def generate_backend(self):
        """
        Generates the backend for Quanatanium
        """
        return Quantanium()

    @AbstractSimulatorBackend._measure_time
    def parse(self):
        """
        Parses the loaded QASM into a quantum circuit.
        """
        self._circuit = self.backend.convert_qasm_to_qua_circuit(
            self.config.test_case)

    @AbstractSimulatorBackend._measure_time
    def format_sample(samples):
        """
        Format the samples to be the commonly used dictionnary
        :param samples: a dictionnary of frozenbitarray
        :type samples: dict()
        """
        samples = {''.join(map(str, key.tolist()))
                           : val for key, val in samples.items()}
        return super().format_sample()

    @AbstractSimulatorBackend._measure_time
    def sample_only(self):
        """
        At this time cstate is broken, so we can't sample FIXME

        Returns:
            _type_: The raw samples from the execution
        """
        return self._result.histogram()

    @AbstractSimulatorBackend._measure_time
    def execute_only(self):
        """
        Executes the circuit on the backend.
        """
        self._result = self._backend.execute(
            self._circuit, nsamples=self.config.nb_shots, seed=self.config.see,
            bonddim=self.config.bonddim, entdim=self.config.entdim, qasmincludes=self._qasms)

    @AbstractSimulatorBackend._measure_time
    def execute_and_sample(self):
        """
        Executes the circuit and samples the results.
        """
        self.execute_only()
        return self.sample_only()


if __name__ == '__main__':
    # test_case is a placeholder, please fill this field by a path to your qasm file
    backend = QuantaniumCpuBackend(nb_shots=10, test_case='')
    backend.parse()
    backend.execute_and_sample()
