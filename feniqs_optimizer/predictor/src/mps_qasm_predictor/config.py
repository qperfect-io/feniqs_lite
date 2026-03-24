#
# Copyright © 2026 QPerfect. All Rights Reserved.
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
from dataclasses import dataclass

RUNTIME_COL = "runtime"
FIDELITY_COL = "fidelity"
QASM_PATH_COL = "qasm_path"
CANDIDATE_COLS = [
    "matrix_product_state_max_bond_dimension",
    "mps_lapack",
    "opt_level",
    "matrix_product_state_truncation_threshold",
    "mps_sample_measure_algorithm",
]

@dataclass(frozen=True)
class ValidationConfig:
    fidelity_tol: float = 1e-3
    runtime_tol: float = 0.05
    severe_runtime_tol: float = 0.20
    random_state: int = 42
    n_bootstrap: int = 1000
    drop_unevaluated_from_metrics: bool = True
