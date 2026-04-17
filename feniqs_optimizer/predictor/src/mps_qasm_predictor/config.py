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
from typing import Optional

RUNTIME_COL = "runtime"
FIDELITY_COL = "fidelity"
QASM_PATH_COL = "qasm_path"
BACKEND_COL = "backend"
CANDIDATE_COLS = [
    "bond_dimension",
    "method",
    "opt_level",
    "entdim",
    "sample_algorithm",
]

@dataclass(frozen=True)
class ValidationConfig:
    fidelity_tol: float = 1e-3
    runtime_tol: float = 0.05
    severe_runtime_tol: float = 0.20
    random_state: int = 42
    n_bootstrap: int = 100
    drop_unevaluated_from_metrics: bool = True
    validation_tree_n_jobs: int = 1
    validation_model_profile: str = "lite"
    max_k: int = 5
    save_rank_tables: bool = False
    save_pdf_figures: bool = True
    run_random_split: bool = True
    run_lofo: bool = True
    run_size_based: bool = False
    run_backend_transfer: bool = False
    size_split_stride: int = 2
    max_size_splits_per_family: Optional[int] = 2
    max_lofo_families: Optional[int] = None
