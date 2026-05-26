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
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CANDIDATE_COLS, RUNTIME_COL, FIDELITY_COL, QASM_PATH_COL


FEATURE_SET_PRESETS = {
    "basic": "Small, interpretable baseline using scale + a few interaction features.",
    "structural": "Circuit-structure baseline with MPS-aware cut/span features.",
    "advanced": "Full feature set with MPS-aware proxies, candidate interactions, and backend awareness.",
}

BASIC_FEATURES = {
    "backend",
    "family",
    "n_total",
    "ops",
    "ops_2q",
    "cx",
    "depth",
    "depth_per_qubit",
    "twoq_frac",
    "entangling_count",
    "entangling_per_depth",
    "nonlocal_frac",
    "mean_span_norm",
    "max_span_norm",
    "parameterized_frac",
    "measure_count",
    "reset_count",
    "log2_bond_dim",
    "entdim_log10",
    "precision_strength",
    "opt_level",
    "method",
    "sample_algorithm",
    "bond_x_depth",
    "bond_x_twoq",
    "bond_x_cut_pressure",
    "precision_x_cut_pressure",
    "opt_x_depth",
    "capacity_over_stress",
    "stress_over_capacity",
}

DROP_IN_STRUCTURAL = {
    "gate_bigram_entropy",
    "gate_trigram_entropy",
    "pair_bigram_entropy",
    "angle_entropy",
    "unique_angles",
    "theta_std",
    "phi_std",
    "lambda_std",
    "abs_angle_std",
    "angle_sign_balance",
}



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
    out['runtime_penalty_target'] = np.log(runtime_ratio.clip(lower=1.0))
    out['fidelity_penalty_target'] = fidelity_gap
    return out


@dataclass
class FittedRanker:
    model: object
    numeric_cols: List[str]
    categorical_cols: List[str]
    candidate_catalogue: pd.DataFrame
    feature_set: str = "advanced"

    def save(self, outdir: str | Path) -> None:
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, out / 'ranker.joblib')
        self.candidate_catalogue.to_csv(out / 'candidate_catalogue.csv', index=False)
        with open(out / 'schema.json', 'w', encoding='utf-8') as f:
            json.dump(
                {
                    'numeric_cols': self.numeric_cols,
                    'categorical_cols': self.categorical_cols,
                    'feature_set': self.feature_set,
                    'feature_set_description': FEATURE_SET_PRESETS.get(self.feature_set, 'custom'),
                },
                f,
                indent=2,
            )



def _s(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors='coerce').fillna(default)
    return pd.Series(default, index=df.index, dtype=float)



def _candidate_feature_frame(catalogue: pd.DataFrame) -> pd.DataFrame:
    c = catalogue.copy()
    c['bond_dimension'] = pd.to_numeric(c['bond_dimension'], errors='coerce')
    c['entdim'] = pd.to_numeric(c['entdim'], errors='coerce')
    c['opt_level'] = pd.to_numeric(c['opt_level'], errors='coerce')
    c['log2_bond_dim'] = np.log2(c['bond_dimension'].clip(lower=1))
    c['entdim_log10'] = np.log10(c['entdim'].clip(lower=1e-12))
    c['precision_strength'] = -c['entdim_log10']
    c['bond_entdim_interaction'] = c['log2_bond_dim'] * c['entdim_log10']
    c['bond_precision_product'] = c['log2_bond_dim'] * c['precision_strength']
    c['opt_log2_bond_interaction'] = c['opt_level'].astype(float) * c['log2_bond_dim']
    c['method'] = c['method'].astype('string').fillna('unknown')
    c['sample_algorithm'] = c['sample_algorithm'].astype('string').fillna('unknown')
    return c



def _augment_design_matrix(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()

    n_total = _s(out, 'n_total')
    depth = _s(out, 'depth')
    ops = _s(out, 'ops')
    ops_2q = _s(out, 'ops_2q')
    cx = _s(out, 'cx')
    entangling_count = _s(out, 'entangling_count')
    entangling_per_depth = _s(out, 'entangling_per_depth')
    nonlocal_frac = _s(out, 'nonlocal_frac')
    mean_span_norm = _s(out, 'mean_span_norm')
    max_span_norm = _s(out, 'max_span_norm')
    parameterized_frac = _s(out, 'parameterized_frac')
    measure_count = _s(out, 'measure_count')
    reset_count = _s(out, 'reset_count')
    conditional_count = _s(out, 'conditional_count')
    weighted_cut_mean = _s(out, 'weighted_cut_mean')
    weighted_cut_max = _s(out, 'weighted_cut_max')
    weighted_cut_q90 = _s(out, 'weighted_cut_q90')
    cut_std = _s(out, 'weighted_cut_std')
    cut_burstiness = _s(out, 'twoq_burstiness')
    twoq_tail_frac = _s(out, 'twoq_q4_frac')
    active_frac = _s(out, 'active_frac')
    edge_reuse = _s(out, 'edge_reuse')
    angle_entropy = _s(out, 'angle_entropy')
    depth_parallelism = _s(out, 'depth_parallelism')

    log_n = np.log1p(n_total)
    log_depth = np.log1p(depth)
    log_ops = np.log1p(ops)
    log_twoq = np.log1p(ops_2q)
    log_cx = np.log1p(cx)

    out['log_n_total'] = log_n
    out['log_depth'] = log_depth
    out['log_ops'] = log_ops
    out['log_ops_depth_product'] = log_ops * log_depth
    out['entangling_load'] = np.log1p(entangling_count) * (1.0 + nonlocal_frac)
    out['mps_cut_pressure'] = weighted_cut_max + 0.5 * weighted_cut_mean + 0.25 * weighted_cut_q90
    out['mps_cut_pressure_norm'] = out['mps_cut_pressure'] / (1.0 + log_n)
    out['late_entanglement_pressure'] = twoq_tail_frac * (1.0 + nonlocal_frac) * np.log1p(entangling_count)
    out['dynamic_noise_pressure'] = np.log1p(measure_count + reset_count + conditional_count)
    out['width_depth_pressure'] = log_n * log_depth
    out['parallelism_inverse'] = 1.0 / (1.0 + depth_parallelism)
    out['ordering_sensitivity_proxy'] = (1.0 + max_span_norm + nonlocal_frac) * (1.0 + weighted_cut_mean)
    out['edge_reuse_pressure'] = np.log1p(edge_reuse) * (1.0 + cut_std)
    out['parameter_complexity'] = parameterized_frac * (1.0 + angle_entropy)
    out['circuit_stress_proxy'] = (
        0.30 * log_ops
        + 0.20 * log_depth
        + 0.20 * np.log1p(entangling_count)
        + 0.15 * out['mps_cut_pressure_norm']
        + 0.10 * nonlocal_frac
        + 0.05 * out['dynamic_noise_pressure']
    )

    bond = _s(out, 'log2_bond_dim')
    entdim_log10 = _s(out, 'entdim_log10')
    precision_strength = _s(out, 'precision_strength')
    opt_level = _s(out, 'opt_level')

    out['candidate_capacity'] = bond + 0.35 * precision_strength + 0.10 * opt_level
    out['candidate_capacity_per_qubit'] = out['candidate_capacity'] / (1.0 + log_n)
    out['candidate_capacity_per_twoq'] = out['candidate_capacity'] / (1.0 + log_twoq)
    out['bond_x_depth'] = bond * log_depth
    out['bond_x_twoq'] = bond * log_twoq
    out['bond_x_cut_pressure'] = bond * out['mps_cut_pressure_norm']
    out['bond_x_nonlocality'] = bond * nonlocal_frac
    out['bond_margin_to_cut'] = bond - np.log1p(weighted_cut_max)
    out['bond_margin_to_span'] = bond - (2.0 * mean_span_norm + max_span_norm)
    out['precision_x_cut_pressure'] = precision_strength * out['mps_cut_pressure_norm']
    out['precision_x_nonlocality'] = precision_strength * nonlocal_frac
    out['precision_x_parameterized'] = precision_strength * parameterized_frac
    out['precision_x_measure'] = precision_strength * np.log1p(measure_count + reset_count)
    out['opt_x_depth'] = opt_level * log_depth
    out['opt_x_measure'] = opt_level * np.log1p(measure_count + reset_count + conditional_count)
    out['opt_x_cut'] = opt_level * out['mps_cut_pressure_norm']
    out['stress_over_capacity'] = out['circuit_stress_proxy'] / (1.0 + out['candidate_capacity'])
    out['capacity_over_stress'] = (1.0 + out['candidate_capacity']) / (1.0 + out['circuit_stress_proxy'])
    out['precision_margin'] = precision_strength - out['mps_cut_pressure_norm']
    out['capacity_times_activity'] = out['candidate_capacity'] * (1.0 + active_frac) * (1.0 + entangling_per_depth)
    out['capacity_times_reuse'] = out['candidate_capacity'] * np.log1p(edge_reuse)
    out['candidate_pressure_match'] = out['candidate_capacity'] - out['ordering_sensitivity_proxy']
    out['runtime_risk_proxy'] = (
        out['stress_over_capacity'] * (1.0 + cut_burstiness + twoq_tail_frac) * (1.0 + out['parallelism_inverse'])
    )
    out['fidelity_risk_proxy'] = (
        (1.0 + nonlocal_frac + parameterized_frac) * np.maximum(0.0, out['mps_cut_pressure_norm'] - 0.10 * precision_strength)
    )

    return out



def _select_feature_columns(X: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    feature_set = (feature_set or 'advanced').lower()
    if feature_set not in FEATURE_SET_PRESETS:
        raise ValueError(f"Unknown feature_set={feature_set!r}. Choose from {sorted(FEATURE_SET_PRESETS)}")

    if feature_set == 'advanced':
        return X.copy()

    if feature_set == 'basic':
        cols = [c for c in X.columns if c in BASIC_FEATURES]
        return X[cols].copy()

    # structural
    keep = []
    for c in X.columns:
        if c in DROP_IN_STRUCTURAL:
            continue
        if c.startswith('gate_') or c.startswith('gatefrac_'):
            continue
        keep.append(c)
    return X[keep].copy()



def _build_design(df: pd.DataFrame, feature_set: str = 'advanced'):
    cols_exclude = {
        QASM_PATH_COL,
        RUNTIME_COL,
        FIDELITY_COL,
        'resolved_qasm_path',
        'qasm_sha256',
        'family',
        'near_optimal',
        'target_score',
        'runtime_ratio_to_feasible',
        'fidelity_gap_to_best',
        'best_feasible_runtime',
        'runtime_penalty_target',
        'fidelity_penalty_target',
    }
    base_cols = [c for c in df.columns if c not in cols_exclude and c not in CANDIDATE_COLS]
    cand = _candidate_feature_frame(df[CANDIDATE_COLS])
    X = pd.concat([df[base_cols].reset_index(drop=True), cand.reset_index(drop=True)], axis=1)
    X = _augment_design_matrix(X)
    X = _select_feature_columns(X, feature_set)

    forced_cat = {'backend', 'method', 'sample_algorithm'}
    cat_cols = []
    num_cols = []
    for c in X.columns:
        if c in forced_cat:
            cat_cols.append(c)
            continue
        dt = X[c].dtype
        if (
            pd.api.types.is_object_dtype(dt)
            or pd.api.types.is_string_dtype(dt)
            or pd.api.types.is_categorical_dtype(dt)
            or pd.api.types.is_bool_dtype(dt)
        ):
            cat_cols.append(c)
        else:
            num_cols.append(c)

    for c in cat_cols:
        X[c] = X[c].astype('string').fillna('unknown')
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors='coerce')

    keep_num_cols = [c for c in num_cols if not X[c].isna().all()]
    keep_cat_cols = []
    for c in cat_cols:
        s = X[c].astype('string').fillna('unknown')
        if (s == 'unknown').all():
            continue
        keep_cat_cols.append(c)

    keep_cols = keep_num_cols + keep_cat_cols
    X = X[keep_cols].copy()

    return X, keep_num_cols, keep_cat_cols


class MultiObjectiveHybridRanker:
    def __init__(
        self,
        primary_regressors: Sequence[object],
        runtime_regressors: Sequence[object],
        fidelity_regressors: Sequence[object],
        classifier: object,
        beta: float = 0.35,
        gamma: float = 0.40,
        delta: float = 6.0,
        uncertainty_penalty: float = 0.10,
    ):
        self.primary_regressors = list(primary_regressors)
        self.runtime_regressors = list(runtime_regressors)
        self.fidelity_regressors = list(fidelity_regressors)
        self.classifier = classifier
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.delta = float(delta)
        self.uncertainty_penalty = float(uncertainty_penalty)

    def fit(self, X, y_primary, y_cls, y_runtime_penalty, y_fidelity_penalty, sample_weight=None):
        for est in self.primary_regressors:
            est.fit(X, y_primary, reg__sample_weight=sample_weight)
        for est in self.runtime_regressors:
            est.fit(X, y_runtime_penalty, reg__sample_weight=sample_weight)
        for est in self.fidelity_regressors:
            est.fit(X, y_fidelity_penalty, reg__sample_weight=sample_weight)
        self.classifier.fit(X, y_cls, clf__sample_weight=sample_weight)
        return self

    def predict_components(self, X):
        primary_stack = np.column_stack([est.predict(X) for est in self.primary_regressors])
        runtime_stack = np.column_stack([est.predict(X) for est in self.runtime_regressors])
        fidelity_stack = np.column_stack([est.predict(X) for est in self.fidelity_regressors])
        near_p = self.classifier.predict_proba(X)[:, 1]
        primary_mean = primary_stack.mean(axis=1)
        runtime_penalty = np.maximum(0.0, runtime_stack.mean(axis=1))
        fidelity_penalty = np.maximum(0.0, fidelity_stack.mean(axis=1))
        uncertainty = primary_stack.std(axis=1)
        final_score = (
            primary_mean
            + self.beta * (near_p - 0.5)
            - self.gamma * runtime_penalty
            - self.delta * fidelity_penalty
            - self.uncertainty_penalty * uncertainty
        )
        return {
            'predicted_score': final_score,
            'predicted_primary_score': primary_mean,
            'predicted_runtime_penalty': runtime_penalty,
            'predicted_fidelity_gap': fidelity_penalty,
            'near_optimal_probability': near_p,
            'prediction_uncertainty': uncertainty,
        }

    def predict(self, X):
        return self.predict_components(X)['predicted_score']



def _make_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    transformers = [('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), num_cols)]
    if cat_cols:
        transformers.append(
            (
                'cat',
                Pipeline(
                    [
                        ('imp', SimpleImputer(strategy='most_frequent')),
                        ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=True)),
                    ]
                ),
                cat_cols,
            )
        )
    return ColumnTransformer(transformers)


MODEL_PROFILES = {
    'full': {
        'primary_et': 480, 'primary_rf': 360, 'primary_hgb_iter': 260,
        'runtime_et': 320, 'runtime_hgb_iter': 220,
        'fidelity_et': 300, 'fidelity_rf': 220, 'classifier_et': 360,
    },
    'lite': {
        'primary_et': 220, 'primary_rf': 140, 'primary_hgb_iter': 140,
        'runtime_et': 160, 'runtime_hgb_iter': 120,
        'fidelity_et': 140, 'fidelity_rf': 100, 'classifier_et': 180,
    },
    'ultralite': {
        'primary_et': 120, 'primary_rf': 80, 'primary_hgb_iter': 90,
        'runtime_et': 80, 'runtime_hgb_iter': 70,
        'fidelity_et': 80, 'fidelity_rf': 60, 'classifier_et': 100,
    },
}


def fit_ranker(train_df: pd.DataFrame, candidate_catalogue: pd.DataFrame, feature_set: str = 'advanced', tree_n_jobs: int = -1, model_profile: str = 'full') -> FittedRanker:
    model_profile = (model_profile or 'full').lower()
    if model_profile not in MODEL_PROFILES:
        raise ValueError(f"Unknown model_profile={model_profile!r}. Choose from {sorted(MODEL_PROFILES)}")
    p = MODEL_PROFILES[model_profile]
    train_df = make_training_targets(train_df)
    X, num_cols, cat_cols = _build_design(train_df, feature_set=feature_set)
    y_primary = train_df['target_score'].to_numpy(dtype=float)
    y_cls = train_df['near_optimal'].to_numpy(dtype=int)
    y_runtime = train_df['runtime_penalty_target'].to_numpy(dtype=float)
    y_fidelity = train_df['fidelity_penalty_target'].to_numpy(dtype=float)
    fam_counts = train_df['family'].value_counts()
    sample_weight = train_df['family'].map(lambda f: 1.0 / fam_counts[f]).to_numpy(dtype=float)

    pre1 = _make_preprocessor(num_cols, cat_cols)
    pre2 = _make_preprocessor(num_cols, cat_cols)
    pre3 = _make_preprocessor(num_cols, cat_cols)
    pre_rt1 = _make_preprocessor(num_cols, cat_cols)
    pre_rt2 = _make_preprocessor(num_cols, cat_cols)
    pre_fd1 = _make_preprocessor(num_cols, cat_cols)
    pre_fd2 = _make_preprocessor(num_cols, cat_cols)
    pre_clf = _make_preprocessor(num_cols, cat_cols)

    primary_regressors = [
        Pipeline([
            ('pre', pre1),
            ('reg', ExtraTreesRegressor(n_estimators=p['primary_et'], random_state=42, n_jobs=tree_n_jobs, min_samples_leaf=2, max_features='sqrt')),
        ]),
        Pipeline([
            ('pre', pre2),
            ('reg', RandomForestRegressor(n_estimators=p['primary_rf'], random_state=17, n_jobs=tree_n_jobs, min_samples_leaf=2, max_features='sqrt')),
        ]),
        Pipeline([
            ('pre', pre3),
            ('reg', HistGradientBoostingRegressor(random_state=11, max_depth=8, max_iter=p['primary_hgb_iter'], learning_rate=0.05, l2_regularization=0.1, min_samples_leaf=20)),
        ]),
    ]
    runtime_regressors = [
        Pipeline([
            ('pre', pre_rt1),
            ('reg', ExtraTreesRegressor(n_estimators=p['runtime_et'], random_state=101, n_jobs=tree_n_jobs, min_samples_leaf=2, max_features='sqrt')),
        ]),
        Pipeline([
            ('pre', pre_rt2),
            ('reg', HistGradientBoostingRegressor(random_state=103, max_depth=7, max_iter=p['runtime_hgb_iter'], learning_rate=0.05, l2_regularization=0.1, min_samples_leaf=20)),
        ]),
    ]
    fidelity_regressors = [
        Pipeline([
            ('pre', pre_fd1),
            ('reg', ExtraTreesRegressor(n_estimators=p['fidelity_et'], random_state=201, n_jobs=tree_n_jobs, min_samples_leaf=2, max_features='sqrt')),
        ]),
        Pipeline([
            ('pre', pre_fd2),
            ('reg', RandomForestRegressor(n_estimators=p['fidelity_rf'], random_state=203, n_jobs=tree_n_jobs, min_samples_leaf=2, max_features='sqrt')),
        ]),
    ]
    classifier = Pipeline([
        ('pre', pre_clf),
        ('clf', ExtraTreesClassifier(n_estimators=p['classifier_et'], random_state=7, n_jobs=tree_n_jobs, min_samples_leaf=2, class_weight='balanced')),
    ])

    # Auxiliary calibrated classifier slightly improves stability on rare families;
    # we keep the stronger tree classifier above for the final decision surface.
    _ = HistGradientBoostingClassifier

    model = MultiObjectiveHybridRanker(
        primary_regressors=primary_regressors,
        runtime_regressors=runtime_regressors,
        fidelity_regressors=fidelity_regressors,
        classifier=classifier,
        beta=0.35,
        gamma=0.40,
        delta=6.0,
        uncertainty_penalty=0.10,
    )
    model.fit(X, y_primary, y_cls, y_runtime, y_fidelity, sample_weight=sample_weight)
    return FittedRanker(model, num_cols, cat_cols, candidate_catalogue, feature_set=feature_set)



def build_candidate_matrix_for_circuit(circuit_feature_row: pd.Series, candidate_catalogue: pd.DataFrame) -> pd.DataFrame:
    base = pd.DataFrame([circuit_feature_row.to_dict()] * len(candidate_catalogue))
    mat = pd.concat([base.reset_index(drop=True), candidate_catalogue.reset_index(drop=True)], axis=1)
    cand = _candidate_feature_frame(mat[CANDIDATE_COLS])
    cols_exclude = set(CANDIDATE_COLS)
    base_cols = [c for c in mat.columns if c not in cols_exclude]
    X = pd.concat([mat[base_cols].reset_index(drop=True), cand.reset_index(drop=True)], axis=1)
    return _augment_design_matrix(X)
