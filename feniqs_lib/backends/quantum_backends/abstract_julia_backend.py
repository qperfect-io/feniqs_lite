
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
from abc import abstractmethod, ABCMeta
import os
from juliacall import newmodule
from juliacall import Main as jl


class AbstractJuliaBackend(AbstractSimulatorBackend, metaclass=ABCMeta):

    def __init__(self,  julia_module_name,  config):
        """
        Constructor for the backend running in julia.

        Args:
            julia_module_name (str): The name to use for the new julia module.
            package_name (str): The name of the package from which to get the version.
            config (Config): The configuration for the backend.
        """

        self.julia_module = newmodule(julia_module_name)
        self.env = "feniqs_lib/backends/plugins/venv/.mimiq_julia_cpu_env"
        if self.env:
            jl.seval(f'using Pkg; Pkg.activate("{self.env}")')

        self._common_import()
        super().__init__(config)

    def _common_import(self):

        self.julia_module.seval("using PkgVersion")

    def _get_julia_package_version(self, package_name):
        """
        Gets the version of the given package from Julia.

        Args:
            package_name (str): The name of the package to get the version of.

        Returns:
            str: The version of the package.
        """
        return str(self.julia_module.seval(f"PkgVersion.Version({package_name})"))

    def _load(self, qasm_path):
        """Loads QASM from the given file path."""
        self.julia_module.seval(
            f"timing = @elapsed file = open(\"{qasm_path}\");\
              timing += @elapsed qasm_str = read(file, String);\
              timing += @elapsed close(file)")
        self._loading_time = self.julia_module.seval("timing")
        qasm_str = self.julia_module.seval("qasm_str")

        self.nb_qubits = self._count_qubits_from_qasm(qasm_str)
        return qasm_str

    @abstractmethod
    def import_packages(self):
        raise NotImplementedError

