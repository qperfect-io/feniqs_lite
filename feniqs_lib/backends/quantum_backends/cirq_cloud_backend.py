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
from .cirq_abstract_backend import CirqAbstractBackend
from .abstract_cloud_backend import AbstractCloudBackend
from .abstract_backend import AbstractBackend
from .abstract_config import BasicConfig, CloudConfig
import cirq
from azure.quantum.cirq import AzureQuantumService


class CirqCloudAbstract(CirqAbstractBackend, AbstractCloudBackend, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, config: BasicConfig):
        super().__init__(config)

        # Obviously not available for the cloud
        self.add_config_attr('run_async', False)

    def close_connection(self):
        # nothing to do for cirq
        return super().close_connection()


class CirqCloudAzureBackend(CirqCloudAbstract):
    """
    Class implemeting the Azure backend for Cirq.
    Be carefull of the fees ;)
    """

    def __init__(self, backend: str, ressource_id: str, location: str,  **kwargs):
        """
        Initializes the Cirq Azure Cloud backend.
        for details about the args: https://quantumai.google/cirq/hardware/azure-quantum/getting_started_ionq

        Args:
            backend (str): Name of the target backend (eg 'ionq.qpu')
            ressource_id (str): _description_
            location (str): _description_
        """
        config = CloudConfig(
            device='Cirq_Cloud', device_type=backend, package_version=cirq.__version__, **kwargs)
        self._ressource_id = ressource_id
        self._location = location
        super().__init__(config)

    @AbstractBackend._measure_time
    def generate_backend(self):
        """
        Generates the backend for the Azure cloud.

        Returns:
            _type_: _description_
        """
        return AzureQuantumService(ressource_id=self._ressource_id, location=self._location, default_target=self.config.device_type)
