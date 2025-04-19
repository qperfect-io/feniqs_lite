
from __future__ import annotations
from .qiskit_abstract_backend import QiskitAbstractBackend
from .abstract_cloud_backend import AbstractCloudBackend
from .abstract_config import CloudConfig
from abc import ABCMeta, abstractmethod
import qiskit
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


class qiskitCloudAbstractBackend(QiskitAbstractBackend, AbstractCloudBackend, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, token: str, backend: str, optimization_level: int = 1, **kwargs):
        """
        Constructor for the Qiskit Cloud backend.

        Args:
            token (str): The token to use for the cloud
            backend (str): The backend to use
            optimization_level (int, optional): The optimization level to use. Defaults to 1.
            **kwargs: Additional arguments to pass to the backend
        """
        # from qiskit.providers.aer import StatevectorSimulator
        # Forced config:
        kwargs['package_version'] = qiskit.__version__
        kwargs['device_type'] = 'quantum_computer'
        config = CloudConfig(device=backend, **kwargs)

        kwargs['device'] = backend
        super().__init__(config)
        # qiskit cloud specific
        self._token = token
        self._backend = None
        self._pm = None

        self.execution_and_sampling_time = 0

        self.config.add_attr('optimization_level', optimization_level)
        self._backend = self.generate_backend()

    def generate_backend(self):
        """
        Uses the run_option to instanciate the backend to use for the simulation

        Returns:
            AerSimulator: The simulator to use
        """
        service = QiskitRuntimeService(
            channel='ibm_quantum', token=self._token)
        backend = service.get_backend(self.config.device)
        return backend

    def parse(self):
        super().parse()
        self._pm = generate_preset_pass_manager(
            optimization_level=1, target=self._backend.target)
        self._qiskit_circuit = self._pm.run(self._qiskit_circuit)

    def execute_and_sample(self):
        """
        Since we use the cloud we need to extract the time for the execution and the sampling separately form the totla time.

        Returns:
            qiskit.Results: Raw samples from qiskit
        """
        samples = super().execute_and_sample()
        self.execution_and_sampling_time = samples.time_taken
        # next line Needs testing
        if samples.header:
            self.config.backend_version = samples.header.backend_version
        return samples

    def close_connection(self):
        """Nothing to do for qiskit"""
        return super().close_connection()


class QiskitCloudBackend(qiskitCloudAbstractBackend):

    def __init__(self, token: str, backend: str, optimization_level: int = 1, **kwargs):
        """
        Constructor for the Qiskit Cloud backend.

        Args:
            token (str): The token to use for the cloud
            backend (str): The backend to use
            optimization_level (int, optional): The optimization level to use. Defaults to 1.
            **kwargs: Additional arguments to pass to the backend
        """
        # Forced config:
        super().__init__(device='Qiskit_Cloud', token=token, backend=backend,
                         optimization_level=optimization_level, **kwargs)
