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
import joblib
import pandas as pd
from .features import extract_qasm_features, read_qasm, qasm_sha256
from .model import build_candidate_matrix_for_circuit

@dataclass
class Prediction:
    candidate_id: int
    matrix_product_state_max_bond_dimension: int
    mps_lapack: int
    opt_level: int
    matrix_product_state_truncation_threshold: float
    mps_sample_measure_algorithm: str
    predicted_score: float

class MPSPredictor:
    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        self.model = joblib.load(model_dir/'ranker.joblib')
        self.candidates = pd.read_csv(model_dir/'candidate_catalogue.csv')

    def rank_qasm(self, qasm_path: str | Path) -> pd.DataFrame:
        qasm = read_qasm(qasm_path)
        feats = extract_qasm_features(qasm)
        feats['resolved_qasm_path'] = str(qasm_path)
        feats['qasm_sha256'] = qasm_sha256(qasm)
        X = build_candidate_matrix_for_circuit(pd.Series(feats), self.candidates)
        scores = self.model.predict(X)
        out = self.candidates.copy()
        out['predicted_score'] = scores
        out = out.sort_values('predicted_score', ascending=False).reset_index(drop=True)
        return out

    def predict_qasm(self, qasm_path: str | Path) -> Prediction:
        ranked = self.rank_qasm(qasm_path)
        top = ranked.iloc[0]
        d = top.to_dict()
        return Prediction(**d)

    def predict_json(self, qasm_path: str | Path) -> str:
        return json.dumps(asdict(self.predict_qasm(qasm_path)), indent=2)
