
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

from .abstract_config import CloudConfig
from .abstract_config import BasicConfig
from .abstract_backend import AbstractBackend
from random import randint
import mimiqcircuits


class MimiqCloudBackend(AbstractBackend):

    def __init__(self, token_path: str, fast: bool = False, device_type: str = 'statevector', **kwargs):
        """
        Constructor for Mimiq using cloud

        :param token_path: path to the file with the token to use
        :type token_path: str
        :param fast: activate the dev server, defaults to False
        :type fast: bool, optional
        :param device_type: indicates which simulator to use, defaults to 'auto'
        :type device_type: str, optional
        """
        config = CloudConfig(
            device='Mimiq_Cloud', device_type=device_type, package_version=mimiqcircuits.__version__, **kwargs)
        super().__init__(config)
        self._token_path = token_path
        fast = '' if not fast else 'fast'
        self._conn = mimiqcircuits.MimiqConnection(
            f"https://mimiq{fast}.qperfect.io/api")
        self._conn.loadtoken(filepath=token_path)
        self.execution_and_sampling_time = 0
        self._backend = self.generate_backend()

    @AbstractBackend._measure_time
    def generate_backend(self):

        return

    def parse(self):
        """
        Parsing is done in the execute method for mimiq
        """
        pass

    def sample_only(self):
        """
        Samples are retrieved in the execute method for mimiq

        Returns:
            list: the list of samples
        """
        return self._result

    def execute_only(self):
        job = self._conn.execute(circuit=self.config.test_case,
                                 nsamples=self.config.nb_shots)
        self._result = self._conn.get_results(job)
        self.execution_and_sampling_time = self._result.timings['apply'] + \
            self._result.timings['sample']

    @AbstractBackend._measure_time
    def execute_and_sample(self):
        self.execute_only()
        return self.sample_only()

    @AbstractBackend._measure_time
    def format_sample(self, samples):
        """
        From the raw samples given by sample_only to a dictionary of counts.

        Args:
            samples (list): list of the samples

        Returns:
            dict: the dictionary of counts
        """
        samples = samples.histogram()
        res = {}
        for key, val in samples.items():
            res[key.to01()] = val
        return res

    def close_connection(self):
        return self._conn.close()
