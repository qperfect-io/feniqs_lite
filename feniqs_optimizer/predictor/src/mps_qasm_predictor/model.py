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
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .config import CANDIDATE_COLS, RUNTIME_COL, FIDELITY_COL, QASM_PATH_COL

def make_training_targets(df: pd.DataFrame, fidelity_tol: float = 1e-3) -> pd.DataFrame:
    out = df.copy()
    grouped = out.groupby(QASM_PATH_COL, sort=False)
    best_fidelity = grouped[FIDELITY_COL].transform('max')
    max_feasible_runtime = (
        out.groupby(QASM_PATH_COL, sort=False)[[RUNTIME_COL, FIDELITY_COL]]
        .apply(lambda g: g.loc[g[FIDELITY_COL] >= g[FIDELITY_COL].max() - fidelity_tol, RUNTIME_COL].min())
        .rename('best_feasible_runtime')
    )
    out = out.merge(max_feasible_runtime, left_on=QASM_PATH_COL, right_index=True, how='left')
    runtime_ratio = out[RUNTIME_COL] / out['best_feasible_runtime'].clip(lower=1e-12)
    fidelity_gap = (best_fidelity - out[FIDELITY_COL]).clip(lower=0.0)
    out['runtime_ratio_to_feasible'] = runtime_ratio
    out['fidelity_gap_to_best'] = fidelity_gap
    out['near_optimal'] = ((runtime_ratio <= 1.05) & (fidelity_gap <= fidelity_tol)).astype(int)
    out['target_score'] = -(np.log(runtime_ratio.clip(lower=1.0)) + 20.0 * fidelity_gap)
    return out

@dataclass
class FittedRanker:
    model: object
    numeric_cols: List[str]
    categorical_cols: List[str]
    candidate_catalogue: pd.DataFrame
    def save(self, outdir: str | Path) -> None:
        out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, out / 'ranker.joblib')
        self.candidate_catalogue.to_csv(out / 'candidate_catalogue.csv', index=False)
        with open(out / 'schema.json','w',encoding='utf-8') as f:
            json.dump({'numeric_cols': self.numeric_cols, 'categorical_cols': self.categorical_cols}, f, indent=2)

def _candidate_feature_frame(catalogue: pd.DataFrame) -> pd.DataFrame:
    c = catalogue.copy()
    c['log2_bond_dim'] = np.log2(c['matrix_product_state_max_bond_dimension'])
    c['lapack'] = c['mps_lapack'].astype(int)
    c['trunc_log10'] = np.log10(c['matrix_product_state_truncation_threshold'])
    c['bond_trunc_interaction'] = c['log2_bond_dim'] * c['trunc_log10']
    c['opt_lapack_interaction'] = c['opt_level'].astype(float) * c['lapack']
    return c

def _build_design(df: pd.DataFrame):
    cols_exclude = {QASM_PATH_COL,RUNTIME_COL,FIDELITY_COL,'resolved_qasm_path','qasm_sha256','family','near_optimal','target_score','runtime_ratio_to_feasible','fidelity_gap_to_best','best_feasible_runtime'}
    base_cols = [c for c in df.columns if c not in cols_exclude and c not in CANDIDATE_COLS]
    cand = _candidate_feature_frame(df[CANDIDATE_COLS])
    X = pd.concat([df[base_cols].reset_index(drop=True), cand.reset_index(drop=True)], axis=1)
    cat_cols = ['mps_sample_measure_algorithm']
    num_cols = [c for c in X.columns if c not in cat_cols]
    return X, num_cols, cat_cols

class HybridRanker:
    def __init__(self, reg_estimators: Sequence[object], clf: object, beta: float = 0.35):
        self.reg_estimators = list(reg_estimators)
        self.clf = clf
        self.beta = beta
    def fit(self, X, y_reg, y_cls, sample_weight=None):
        for est in self.reg_estimators:
            est.fit(X, y_reg, reg__sample_weight=sample_weight)
        self.clf.fit(X, y_cls, clf__sample_weight=sample_weight)
        return self
    def predict(self, X):
        reg_preds = np.column_stack([est.predict(X) for est in self.reg_estimators]).mean(axis=1)
        p = self.clf.predict_proba(X)[:, 1]
        return reg_preds + self.beta * (p - 0.5)

def fit_ranker(train_df: pd.DataFrame, candidate_catalogue: pd.DataFrame) -> FittedRanker:
    train_df = make_training_targets(train_df)
    X, num_cols, cat_cols = _build_design(train_df)
    y = train_df['target_score'].to_numpy(dtype=float)
    y_cls = train_df['near_optimal'].to_numpy(dtype=int)
    fam_counts = train_df['family'].value_counts()
    sample_weight = train_df['family'].map(lambda f: 1.0 / fam_counts[f]).to_numpy(dtype=float)
    pre = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), num_cols),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols),
    ])
    reg1 = Pipeline([('pre', pre), ('reg', ExtraTreesRegressor(n_estimators=420, random_state=42, n_jobs=-1, min_samples_leaf=2, max_features='sqrt'))])
    reg2 = Pipeline([('pre', pre), ('reg', RandomForestRegressor(n_estimators=320, random_state=17, n_jobs=-1, min_samples_leaf=2, max_features='sqrt'))])
    clf = Pipeline([('pre', pre), ('clf', ExtraTreesClassifier(n_estimators=320, random_state=7, n_jobs=-1, min_samples_leaf=2, class_weight='balanced'))])
    ensemble = HybridRanker([reg1, reg2], clf, beta=0.40)
    ensemble.fit(X, y, y_cls, sample_weight=sample_weight)
    return FittedRanker(ensemble, num_cols, cat_cols, candidate_catalogue)

def build_candidate_matrix_for_circuit(circuit_feature_row: pd.Series, candidate_catalogue: pd.DataFrame) -> pd.DataFrame:
    base = pd.DataFrame([circuit_feature_row.to_dict()] * len(candidate_catalogue))
    mat = pd.concat([base.reset_index(drop=True), candidate_catalogue.reset_index(drop=True)], axis=1)
    cand = _candidate_feature_frame(mat[CANDIDATE_COLS])
    cols_exclude = set(CANDIDATE_COLS)
    base_cols = [c for c in mat.columns if c not in cols_exclude]
    X = pd.concat([mat[base_cols].reset_index(drop=True), cand.reset_index(drop=True)], axis=1)
    return X
