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
from pathlib import Path
from typing import Iterable, Optional
import pandas as pd
from .config import CANDIDATE_COLS, QASM_PATH_COL
from .features import extract_qasm_features, read_qasm, qasm_sha256


def infer_family_from_path(path: str) -> str:
    parts = Path(path).parts
    if "paper_data" in parts:
        idx = parts.index("paper_data")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return Path(path).name.split("_")[0]


def resolve_qasm_path(raw_path: str, qasm_root: str | Path) -> Path:
    raw = Path(raw_path)
    root = Path(qasm_root)
    if raw.is_absolute() and raw.exists() and raw.suffix == ".qasm":
        return raw
    parts = raw.parts
    if "paper_data" in parts:
        idx = parts.index("paper_data")
        candidate = root / Path(*parts[idx+1:])
        if candidate.exists() and candidate.suffix == '.qasm':
            return candidate
        candidate = root / Path(*parts[idx:])
        if candidate.exists() and candidate.suffix == '.qasm':
            return candidate
    candidate = root / raw.name
    if candidate.exists() and candidate.suffix == ".qasm":
        return candidate
    matches = [m for m in root.rglob(raw.name) if m.suffix == '.qasm']
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Could not resolve QASM path {raw_path!r} under {str(qasm_root)!r}")


def build_circuit_feature_table(qasm_paths: Iterable[str], qasm_root: str | Path) -> pd.DataFrame:
    rows=[]
    for qp in sorted(set(qasm_paths)):
        resolved=resolve_qasm_path(qp, qasm_root)
        qasm=read_qasm(resolved)
        feats=extract_qasm_features(qasm)
        feats[QASM_PATH_COL]=qp
        feats["resolved_qasm_path"]=str(resolved)
        feats["family"]=infer_family_from_path(qp)
        feats["qasm_sha256"]=qasm_sha256(qasm)
        rows.append(feats)
    return pd.DataFrame(rows)


def load_datasets(full_eval_csv: str | Path, best_csv: Optional[str | Path], qasm_root: str | Path):
    full = pd.read_csv(full_eval_csv)
    best = pd.read_csv(best_csv) if best_csv else pd.DataFrame(columns=full.columns)
    circuit_features = build_circuit_feature_table(full[QASM_PATH_COL].unique(), qasm_root)
    full = full.merge(circuit_features, on=QASM_PATH_COL, how="inner", validate="many_to_one")
    if len(best):
        best = best.merge(circuit_features, on=QASM_PATH_COL, how="inner", validate="many_to_one")
    return full, best, circuit_features


def build_candidate_catalogue(full_df: pd.DataFrame) -> pd.DataFrame:
    cat = full_df[CANDIDATE_COLS].drop_duplicates().sort_values(CANDIDATE_COLS).reset_index(drop=True)
    cat["candidate_id"] = range(len(cat))
    return cat
