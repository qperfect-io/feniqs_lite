
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

from ..quantum_backends.projectq_backend import ProjectQBackend
from feniqs_lib.managers.hook_specs import hookimpl
from feniqs_lib.tools.backends_tools import run_backend_common
from feniqs_lib.tools.constants import GlobalConfig
from feniqs_lib.tools.constants import update_global_config_from_env
from ..task.task import run_task

class ProjectCpuPlugin:
    @hookimpl
    def run_backend(self, test_case: str, nb_shots: int, seed: int, **kwargs):
        global_config = update_global_config_from_env()
        return run_backend_common(ProjectQBackend, test_case, nb_shots, seed, None, global_config.incl_first, global_config.runs, **kwargs)

def plugin():
    return ProjectCpuPlugin()
