
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

from ..quantum_backends.quantanium_backend import QuantaniumCpuBackend
from feniqs_lib.managers.hook_specs import hookimpl
from ..task.task import run_task


class QuantaniumCpuPluggin:
    @hookimpl
    def run_backend(self, test_case: str, nb_shots: int, seed: int, **kwargs):

        backend = QuantaniumCpuBackend(
            test_case=test_case, nb_shots=nb_shots, seed=seed, **kwargs)
        samples, timings, config = run_task(backend)

        return samples, timings, config


def plugin():
    return QuantaniumCpuBackend()
