
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

from ..quantum_backends.mimiq_julia_backend import MimiqJuliaCpuBackend
from feniqs_lib.managers.hook_specs import hookimpl
from feniqs_lib.tools.backends_tools import run_backend_common
from feniqs_lib.tools.constants import GlobalConfig
from ..task.task import run_task, results_to_json

class MimiqJuliaCpuPlugin:
    @hookimpl
    def run_backend(self, test_case: str, nb_shots: int, seed: int, env: str, **kwargs):
        global_config = GlobalConfig()
        fi_env = os.environ.get("FI_ENABLED", "False")
        global_config.incl_first = fi_env.lower() in ["true", "1", "yes"]
        run_env = int(os.environ.get("RUNS", 3))
        global_config.runs = run_env  
        return run_backend_common(MimiqJuliaCpuBackend, test_case, nb_shots, seed, env, global_config.incl_first, global_config.runs, **kwargs)


def plugin():
    return MimiqJuliaCpuPlugin()



