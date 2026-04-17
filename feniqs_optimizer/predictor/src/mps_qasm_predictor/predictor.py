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
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .features import extract_qasm_features, read_qasm, qasm_sha256
from .model import build_candidate_matrix_for_circuit


@dataclass
class Prediction:
    candidate_id: int
    bond_dimension: int
    method: str
    opt_level: int
    entdim: float
    sample_algorithm: str
    predicted_score: float
    predicted_runtime_penalty: float | None = None
    predicted_fidelity_gap: float | None = None
    near_optimal_probability: float | None = None
    prediction_uncertainty: float | None = None


class MPSPredictor:
    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        self.model = joblib.load(model_dir / 'ranker.joblib')
        self.candidates = pd.read_csv(model_dir / 'candidate_catalogue.csv')
        metadata_path = model_dir / 'metadata.json'
        if metadata_path.exists():
            meta = json.loads(metadata_path.read_text(encoding='utf-8'))
            self.backend = meta.get('backend', 'unknown')
            self.feature_set = meta.get('feature_set', 'advanced')
        else:
            self.backend = 'unknown'
            self.feature_set = 'advanced'
        schema_path = model_dir / 'schema.json'
        if schema_path.exists():
            schema = json.loads(schema_path.read_text(encoding='utf-8'))
            self.feature_set = schema.get('feature_set', self.feature_set)

    def rank_qasm(self, qasm_path: str | Path) -> pd.DataFrame:
        qasm = read_qasm(qasm_path)
        feats = extract_qasm_features(qasm)
        feats['resolved_qasm_path'] = str(qasm_path)
        feats['qasm_sha256'] = qasm_sha256(qasm)
        feats['backend'] = self.backend
        X = build_candidate_matrix_for_circuit(pd.Series(feats), self.candidates)

        if hasattr(self.model, 'predict_components'):
            comp = self.model.predict_components(X)
            out = self.candidates.copy()
            for key, values in comp.items():
                out[key] = values
        else:
            out = self.candidates.copy()
            out['predicted_score'] = self.model.predict(X)

        out = out.sort_values('predicted_score', ascending=False).reset_index(drop=True)
        return out

    def predict_qasm(self, qasm_path: str | Path) -> Prediction:
        ranked = self.rank_qasm(qasm_path)
        top = ranked.iloc[0]
        d: dict[str, Any] = top.to_dict()
        keep = {k: d.get(k) for k in Prediction.__annotations__.keys()}
        return Prediction(**keep)

    def predict_json(self, qasm_path: str | Path) -> str:
        return json.dumps(asdict(self.predict_qasm(qasm_path)), indent=2)
