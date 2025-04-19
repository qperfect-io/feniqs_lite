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
from .qiskit_abstract_backend import QiskitAbstractBackend
from .abstract_simulator_backend import AbstractSimulatorBackend
from .abstract_config import SimulatorConfig
from .abstract_backend import AbstractBackend
from abc import ABCMeta         
import multiprocessing
import qiskit
from qiskit_aer import AerSimulator


class QiskitSimulatorAbstractBackend(QiskitAbstractBackend, AbstractSimulatorBackend, metaclass=ABCMeta):

    def __init__(self, device='Qiskit_Aer', device_type: str = "statevector",
                 max_parallel_threads=1, max_parallel_shots=0, max_parallel_experiments=0,
                 fusion: int = 10, fusion_threshold: int = 10, fusion_enable: bool = True,
                 opt_level: int = 1,
                 # Statevector only param
                 statevector_parallel_threshold=14, statevector_sample_measure_opt=10,
                 # MPS only options
                 matrix_product_state_max_bond_dimension=None, mps_parallel_threshold=14, mps_omp_threads=1, mps_lapack=False,
                 matrix_product_state_truncation_threshold = 1e-5, mps_sample_measure_algorithm = 'mps_apply_measure', 
                 # tensor_network only option
                 tensor_network_num_sampling_qubits = 10,
                 **kwargs):
        """
        Here is the constructor for qiskit's backend
        To find more information on the available options: https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html

        :param device_type: Set the simulation method, defaults to "statevector" Can be: (statevector, matrix_product_state, tensor_network, qasm_simulator)
        :type device_type: str, optional
        :param max_parallel_threads: Sets the maximum number of CPU cores used by OpenMP for parallelization. If set to 0 the maximum will be set to the number of CPU cores, defaults to 1
        :type max_parallel_threads: int, optional
        :param max_parallel_shots: Sets the maximum number of shots that may be executed in parallel during each experiment execution, up to the max_parallel_threads value. If set to 1 parallel shot execution will be disabled. If set to 0 the maximum will be automatically set to max_parallel_threads. Note that this cannot be enabled at the same time as parallel experiment execution, defaults to 0
        :type max_parallel_shots: int, optional
        :param max_parallel_experiments: Sets the maximum number of qobj experiments that may be executed in parallel up to the max_parallel_threads value. If set to 1 parallel circuit execution will be disabled. If set to 0 the maximum will be automatically set to max_parallel_threads, defaults to 0
        :type max_parallel_experiments: int, optional
        :param fusion_enable: Enable fusion optimization in circuit optimization passes, defaults to True
        :type fusion_enable: bool, optional
        :param fusion: Maximum number of qubits for a operation generated in a fusion optimization. A default value (None) automatically sets a value depending on the simulation method, defaults to 10
        :type fusion: int, optional
        :param fusion_threshold: Threshold that number of qubits must be greater than or equal to enable fusion optimization. A default value automatically sets a value depending on the simulation method, defaults to 10
        :type fusion_threshold: int, optional
        :param statevector_parallel_threshold: Sets the threshold that the number of qubits must be greater than to enable OpenMP parallelization for matrix multiplication during execution of an experiment. If parallel circuit or shot execution is enabled this will only use unallocated CPU cores up to max_parallel_threads. Note that setting this too low can reduce performance, defaults to 14
        :type statevector_parallel_threshold: int, optional
        :param statevector_sample_measure_opt: Sets the threshold that the number of qubits must be greater than to enable a large qubit optimized implementation of measurement sampling. Note that setting this two low can reduce performance, defaults to 10
        :type statevector_sample_measure_opt: int, optional
        :param matrix_product_state_max_bond_dimension:  Sets a limit on the number of Schmidt coefficients retained at the end of the svd algorithm. Coefficients beyond this limit will be discarded. (Default: None, i.e., no limit on the bond dimension).
        :type matrix_product_state_max_bond_dimension: int, optional
        :param mps_parallel_threshold: This option sets OMP number threshold (Default: 14).
        :type mps_max_parallele_threshold: int, optional
        :param mps_opm_threads: This option sets the number of OMP threads (Default: 1).
        :type mps_opm_threads: int, optional
        :param mps_lapack: This option indicates to compute the SVD function using OpenBLAS/Lapack interface (Default: False).
        :type mps_lapack: bool, optional
        :param tensor_network_num_sampling_qubits: is used to set number of qubits to be sampled in single tensor network contraction when using sampling measure. (Default: 10)
        :type tensor_network_num_sampling_qubits: int, optional
        """
        # Forced config:
        kwargs['package_version'] = qiskit.__version__
        kwargs['device_type'] = device_type
        config = SimulatorConfig(device=device, fusion_enable=fusion_enable,
                                 max_parallel_threads=max_parallel_threads, **kwargs)

        # qiskit simulator specific option
        config.add_attr('fusion', fusion)
        config.add_attr('opt_level', opt_level)
        config.add_attr('fusion_threshold', fusion_threshold)
        config.add_attr('max_parallel_shots', max_parallel_shots)
        config.add_attr('max_parallel_experiments',
                        max_parallel_experiments)

        # qiskit specific
        self._qiskit_circuit = None
        self._backend = None
        self._job = None

        self._run_options = dict(
            method=device_type,
            precision=config.precision,
            # Fusion
            fusion_enable=fusion_enable,
            fusion_threshold=fusion_threshold,
            fusion_max_qubit=fusion,
            # Parallel
            max_parallel_shots=max_parallel_shots,
            max_parallel_experiments=max_parallel_experiments,
            max_parallel_threads=max_parallel_threads  )
        if device_type == 'statevector':
            # updating config
            config.add_attrs(statevector_parallel_threshold=statevector_parallel_threshold,
                             statevector_sample_measure_opt=statevector_sample_measure_opt)
            # uptdating running options
            self._run_options['statevector_parallel_threshold'] = config.statevector_parallel_threshold
            self._run_options['statevector_sample_measure_opt'] = config.statevector_sample_measure_opt
        if device_type == 'matrix_product_state':
            # ipdating config
            config.add_attrs(matrix_product_state_max_bond_dimension=matrix_product_state_max_bond_dimension,
                             mps_parallel_threshold=mps_parallel_threshold, mps_omp_threads=mps_omp_threads, mps_lapack=mps_lapack,
                             matrix_product_state_truncation_threshold = matrix_product_state_truncation_threshold, mps_sample_measure_algorithm=mps_sample_measure_algorithm)
            # updating running options
            self._run_options['matrix_product_state_max_bond_dimension'] = config.matrix_product_state_max_bond_dimension
            self._run_options['mps_parallel_threshold'] = config.mps_parallel_threshold
            self._run_options['mps_omp_threads'] = config.mps_omp_threads
            self._run_options['mps_lapack'] = config.mps_lapack
            self._run_options['matrix_product_state_truncation_threshold'] = config.matrix_product_state_truncation_threshold
            self._run_options['mps_sample_measure_algorithm'] = config.mps_sample_measure_algorithm
         
        if device_type == 'tensor_network':
            # updating config
            config.add_attrs(
                tensor_network_num_sampling_qubits=tensor_network_num_sampling_qubits)
            # updating running options
            self._run_options['tensor_network_num_sampling_qubits'] = config.tensor_network_num_sampling_qubits
        
        super().__init__(config)

    @ AbstractBackend._measure_time
    def generate_backend(self):
        """
        Generate and returns the backend to use for the qiskit simulation.

        :return: The backend to use for the simulation
        :rtype: AerSimulator
        """
        
        num_threads = multiprocessing.cpu_count()

        self._run_options['mps_omp_threads'] = num_threads #int(num_threads * 0.85)  # Use 85% of threads for MPS
        self._run_options['max_parallel_threads'] = num_threads
        backend = AerSimulator(**self._run_options)

        backend.set_options(seed_simulator=self.config.seed)
        # For config purpose 
        self.config.backend_version = backend.configuration().backend_version
        return backend


class QiskitAerBackend(QiskitSimulatorAbstractBackend):
    """
    Qiskit implementation of the AbstractBackend. Handles following operations:
    loading, parsing, executing, and sampling quantum circuits.
    """

    def __init__(self, **kwargs):
        """
        Initializes the Qiskit backend.

        Here is a full example to use this backend:

        .. code-block:: python

            # to generate the backend and pass it options
            backend = QiskitAerBackend(device_type='matrix_product_state', test_case='path_to_qasm.qasm', nb_shots=1)
            # To parse the qasm file
            backend.parse()
            # To execute and sample the results
            samples = backend.execute_and_sample()
            # To foramt the samples in a json format
            samples = backend.format_sample(samples)
            # To retrieve all the options used in the simulation
            config = backend.get_config_dict()
            """
        # Forced config:
        super().__init__(device='Qiskit_Aer_Cpu', **kwargs)
        self._backend = self.generate_backend()


class QiskitAerGpu(QiskitSimulatorAbstractBackend):

    def __init__(self, use_cuTensorNet_autotuning=False, **kwargs):
        """
        Constructor for Qiskit using GPU

        :param use_cuTensorNet_autotuning: enables auto tuning of plan in cuTensorNet API. It takes some time for tuning, so enable if the circuit is very large, defaults to False
        :type use_cuTensorNet_autotuning: bool, optional
        """

        # Forced config:
        super().__init__(device='Qiskit_Aer_Gpu', **kwargs)

        if self.config.device_type == 'tensor_network':
            self.add_config_attr('use_cuTensorNet_autotuning',
                                 use_cuTensorNet_autotuning)
            self._run_options['use_cuTensorNet_autotuning'] = use_cuTensorNet_autotuning

        self._run_options["cuStateVec_enable"] = True
        self._run_options = {'device': 'GPU'}
   
        self._backend = self.generate_backend()
