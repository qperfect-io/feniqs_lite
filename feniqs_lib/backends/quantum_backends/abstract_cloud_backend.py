
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
from .abstract_config import CloudConfig
from .abstract_backend import AbstractBackend
from abc import ABCMeta, abstractmethod


class AbstractCloudBackend(AbstractBackend, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, config: CloudConfig):
        super().__init__(config)

    @abstractmethod
    def close_connection(self):
        """
        To implement for some backends that need to close the connection to the cloud after the job is done (ex: mimiq).
        """
        pass
