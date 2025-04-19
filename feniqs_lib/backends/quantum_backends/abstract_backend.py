
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
from abc import ABC, abstractmethod
import time
from .abstract_config import BasicConfig, AbstractConfig
from importlib.metadata import version, PackageNotFoundError
from collections import Counter
from feniqs_lib.tools.logger_api import greenlet_exception_logger, config_logger
from logging import getLogger
from functools import wraps


class AbstractBackend(ABC, AbstractConfig):
    """
    An abstract base class for quantum simulator/hardware backends. Defines the essential
    operations for their benchmarking.
    """

    def __init__(self, config: BasicConfig):
        """
        Constructor for The Abstract class implementing most backend
        To never instanciate directly

        :param config: Config object containing all the configuration related to the simulation
        :type config: BasicConfig

        The non-abstract backends implementing this interface should be used this way:

        .. code-block:: python

            # This is an abstract and, therefore, cannot be intanciated.
            # This example supposed you've already instanciated a backend.
            # something like this: backend = BackendClass(...)

            # You can retrieve the time taken to instanciate a backend using backend._instanciate
            # You can retrieve the time taken to load the file and read it using backend._time_taken
            # parse the file
            backend.parse()
            # You can retrive the time taken to parse the file it using backend._time_taken
            # execute and sample
            samples = backend.execute_and_sample()
            # You can retrive the time taken to execute and sample the simulation using backend._time_taken
            # format the samples
            samples = backend.format_sample(samples)
            # As usual you can get the formatting time using backend._time_taken

        """
        super().__init__(config)
        config_logger('INFO')
        self.logger = getLogger(str(config.device))
        # deactivates all qiskit logger for debug purposes
        getLogger('qiskit').propagate = False
        greenlet_exception_logger(self.logger)
        self._loading_time = 0
        self.qasm_string = ""
        self.time_value = 0
        self.fidelity = None
        self._qasm_file = config.test_case
        # Load the QASM file
        tik = time.time()
        self._qasm_str = self._load(config.test_case)
        if self._loading_time == 0:
            self._loading_time = time.time()-tik
        # gets the number of qubits
        self.config.add_attr(
            'nb_qubits', self._count_qubits_from_qasm(self._qasm_str))

        self.execution_and_sampling_separation = False

    def _load(self, qasm_path):
        """
        Extract the text from the file given

        :param qasm_path: path to the qasm file to use
        :type qasm_path: str
        :return: the content of the file
        :rtype: str
        """
        with open(qasm_path, 'r') as file:
            qasm_str = file.read()
        return qasm_str

    def _count_qubits_from_qasm(self, qasm_string):
        """
        Naive qubit count looking for the first qreg definition and extracting the number of aubits

        :param qasm_string: The qasm file as a string
        :type qasm_string: str
        :return: The number of qubits extracted
        :rtype: int
        """
        for line in qasm_string.splitlines():
            if line.startswith('qreg'):
                return int(line.strip().split('[')[1].split(']')[0])
        return 0

    def plot_circuit(self):
        """
        Method used to plot the circuit once the circuit has been parsed.
        """
        pass

    def _from_sample_list_to_dict(self, samples):
        """
        Method used to convert the raw samples obtained after running a simulation with execute_and_sample(). 
        """
        return dict(Counter(samples))

    def _get_version(self, package):
        """
        Gets the version of the module used by the backend

        :param package: Name of the module used by the backend
        :type package: str
        :return: The version of the backend, ex: 1.0.0
        :rtype: str
        """
        package = package.lower()
        try:
            package_version = version(package)
        except PackageNotFoundError as e:
            return "unknown_version"
        return package_version

    @abstractmethod
    def generate_backend(self):
        """
        Generates and returns the backend used for the simulation
        :raises NotImplementedError: This should be implemented in the child classes
        """
        raise NotImplementedError

    @abstractmethod
    def parse(self):
        """
        After loading the file in the constructor the file need to be parse to create a circuit executable on the backend.
        The circuit will be stored in a local attributes usually called _circuit
        :raises NotImplementedError: This should be implmented in the children classes
        """
        raise NotImplementedError

    def execute_only(self):
        """
        Can be use to separate the actual execution time from everything else that might be neccessary to setup.
        :raises NotImplementedError: This should be implement in the children classes or not used at all
        """
        raise NotImplementedError

    @abstractmethod
    def format_sample(samples):
        """
        Method used to convert the raw samples obtained after running a simulation with execute_and_sample().

        :param samples: The direct result from execute_and_sample() it should be the result of the simulation of the current backend
        :type samples: Any
        :return: A dictonnary of counts with the measurements as key and the frequency of apparition as value
        :rtype: dict

        How to use:

        .. code-block:: python 

            # first you need samples to format
            samples = current_backend.execute_and_sample()
            # Then to format the samples
            format_samples = backend.format_sample(samples)
            print(format_samples)
        """
        raise NotImplementedError

    def sample_only(self):
        """
        Can be used to separate the sampling run time from everything else

        :raises NotImplementedError: This should be either implmented in the children classes or never called
        """
        raise NotImplementedError

    @abstractmethod
    def execute_and_sample(self):
        """
        Once the circuit has been parsed this method allows the backend to execute and retrieve samples from the circuit.
        This is the most important part of the benchmarking process.
        :raises NotImplementedError: This must be implemented in the children classes
        How to use:
        .. code-block:: python

            # First generate the backend. select the backend you want instanciate and the options you want to use.
            backend = ChildBackend(...)
            # parse the file
            backend.parse()
            # execute and sample
            samples = backend.execute_and_sample()
        """
        raise NotImplementedError

            

    def generate_mirror_qasm(self):
        """
        Generate a new string where '.qasm' is replaced by '.mirror' in self._qasm_str.
        """
  
        if self._qasm_file.endswith('.qasm'):
            return self._qasm_file.replace('.qasm', '.mirror')
        else:
            raise ValueError("The input string does not end with .qasm.")


    def _measure_time(method):
        """
        Decorator to measure the execution time of methods.
        """

        def timed(self, *args, **kw):
            start_time = time.time()
            result = method(self, *args, **kw)
            end_time = time.time()
            self.time_value = end_time-start_time

            return result
        return timed

AbstractBackend._measure_time = staticmethod(AbstractBackend._measure_time)
