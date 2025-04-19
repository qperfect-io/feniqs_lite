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

from .cirq_abstract_backend import CirqSimulatorAbstractBackend
from .abstract_backend import AbstractBackend
import qsimcirq


class QsimCirqCpuBackend(CirqSimulatorAbstractBackend):
    def __init__(self, fusion_enable: bool = True, fusion: int = 10, **kwargs):
        """Constructor for the QsimCirq backend

        :param fusion_enable: Activates the fusion option, defaults to True
        :type fusion_enable: bool, optional
        :param fusion: Max Number of qubit for gate fuse, defaults to 10
        :type fusion: int, optional

        Here is a full example to use this backend:

        .. code-block:: python

            # to generate the backend and pass it options
            backend = QmatchateaCpuBackend(test_case='path_to_qasm.qasm', nb_shots=1,fusion_enable=True, fusion=10)
            # To parse the qasm file
            backend.parse()
            # To execute and sample the results
            samples = backend.execute_and_sample()
            # To foramt the samples in a json format
            samples = backend.format_sample(samples)
            # To retrieve all the options used in the simulation
            config = backend.get_config_dict()
        """
        # For more information on qsimcirq options:   https://quantumai.google/reference/python/qsimcirq/QSimSimulator

        kwargs['precision'] = 'single'
        super().__init__(package_version=qsimcirq.__version__, device='QsimCirq_Cpu', device_type='full_wave_function',
                         fusion_enable=fusion_enable, **kwargs)
        self.add_config_attr('fusion', fusion)

        # instanciate backend
        self._backend = self.generate_backend()

    @AbstractBackend._measure_time
    def generate_backend(self):
        """Generates the Cirq simulator backend.

        Returns:
            cirq.Simulator: The backend to use for the simulation
        """
        options = {}
        if not self.config.fusion_enable:
            options['max_fused_gate_size'] = 1
        else:
            options['max_fused_gate_size'] = self.config.fusion
        options['cpu_threads'] = self.config.max_parallel_threads

        options = qsimcirq.QSimOptions(**options)
        return qsimcirq.QSimSimulator(seed=self.config.seed, qsim_options=options)


class QsimCirqGpuBackend(CirqSimulatorAbstractBackend):
    def __init__(self, fusion_enable: bool = True, gpu_mode: int = 0, fusion: int = 10, **kwargs):
        """Constructor for Gpu backend for Cirq.

        Args:
            split_untangled_state (bool, optional): _description_. Defaults to True.
            run_async (bool, optional): _description_. Defaults to False.
            gpu_mode (int, optional): use CUDA if 0 else uses NVDIA cuStateVec. Defaults to 0.
            fusion (int, optional): _description_. Defaults to None.
        """
        # For more information: https://quantumai.google/qsim/cirq_interface
        kwargs['split_untangled_state'] = False
        kwargs['precision'] = 'single'
        super().__init__(device='QsimCirq_Gpu', fusion_enable=fusion_enable,
                         device_type='full_wave_function', **kwargs)

        self.add_config_attr('fusion', fusion)
        self.add_config_attr('gpu_mode', gpu_mode)

        # instanciate backend
        self._backend = self.generate_backend()

    @AbstractBackend._measure_time
    def generate_backend(self):
        """Generates the Cirq simulator backend.

        Returns:
            cirq.Simulator: The backend to use for the simulation
        """
        options = {'use_gpu': True}
        # if 0 - gpu ; 1 - cuStateVec
        options = {'gpu_mode': 1}

        options = qsimcirq.QSimOptions(**options)

        return qsimcirq.QSimSimulator(seed=self.config.seed, qsim_options=options)


if __name__ == '__main__':
    file_path = '' # this is a placeholder, please fill this field by the path to your qasm file
    backend = QsimCirqCpuBackend(nb_shots=10, test_case=file_path) 
    backend.parse()
    samples = backend.execute_and_sample()
    samples_formatted = backend.format_sample(samples)
    print(samples_formatted)
