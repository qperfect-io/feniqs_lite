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

import gc
import json
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .compare_default import compare_predicted_vs_default
from .backend import normalize_backend_df
from .config import ValidationConfig, QASM_PATH_COL, CANDIDATE_COLS, RUNTIME_COL, FIDELITY_COL
from .data import load_datasets, build_candidate_catalogue
from .model import fit_ranker, make_training_targets, build_candidate_matrix_for_circuit
from .plots import (
    plot_summary_heatmap,
    plot_runtime_distribution_both_regimes,
    plot_runtime_vs_default_distribution_both,
    plot_median_speedup_vs_default_both,
    plot_ecdf_speedup_vs_default_both,
    plot_family_topk_bars,
    plot_family_tradeoff_panel,
    plot_feature_set_comparison,
)

SIZE_COL = "n_total"
EVAL_COLUMNS = [
    QASM_PATH_COL, "pred_rank", "is_offline_evaluable", "near_optimal",
    "runtime_ratio_to_feasible", "fidelity_gap_to_best", "exact_match",
    "eval_runtime", "eval_fidelity", "heldout_family", "heldout_n_qubits", "heldout_backend"
]


def _empty_eval_df() -> pd.DataFrame:
    return pd.DataFrame(columns=EVAL_COLUMNS)


def _select_rankings_from_available_candidates(model, circuit_df: pd.DataFrame, max_k: int = 5) -> pd.DataFrame:
    rows = []
    for qasm_path, g in circuit_df.groupby(QASM_PATH_COL, sort=False):
        row = g.iloc[0]
        feats = row.drop(labels=[c for c in [RUNTIME_COL, FIDELITY_COL] + CANDIDATE_COLS if c in row.index])
        available_candidates = g[CANDIDATE_COLS].drop_duplicates().reset_index(drop=True)
        X = build_candidate_matrix_for_circuit(feats, available_candidates)
        if hasattr(model, 'predict_components'):
            comp = model.predict_components(X)
            scores = np.asarray(comp['predicted_score'], dtype=float)
        else:
            comp = {}
            scores = np.asarray(model.predict(X), dtype=float)
        order = np.argsort(scores)[::-1]
        for rank, idx in enumerate(order[:max_k], start=1):
            idx = int(idx)
            choice = available_candidates.iloc[idx].to_dict()
            choice[QASM_PATH_COL] = qasm_path
            choice['predicted_score'] = float(scores[idx])
            for key, values in comp.items():
                if key == 'predicted_score':
                    continue
                choice[key] = float(values[idx])
            choice['pred_rank'] = int(rank)
            rows.append(choice)
    return pd.DataFrame(rows)


def _attach_offline_validation_status(merged: pd.DataFrame) -> pd.DataFrame:
    out = merged.copy()
    missing = out[RUNTIME_COL].isna() | out[FIDELITY_COL].isna()
    out["is_offline_evaluable"] = (~missing).astype(int)
    out["validation_status"] = np.where(missing, "not_evaluated_offline", "validated_offline")
    out["validation_message"] = np.where(
        missing,
        "Predicted hyperparameter tuple is not present among evaluated configurations for this circuit; runtime and fidelity require a new simulator run.",
        "Predicted hyperparameter tuple is present in the evaluation table; runtime and fidelity were validated offline.",
    )
    return out


def _filter_evaluable(eval_df: pd.DataFrame | None, cfg: ValidationConfig) -> pd.DataFrame:
    if eval_df is None:
        return pd.DataFrame()
    if len(eval_df) == 0:
        return eval_df.copy()
    if not cfg.drop_unevaluated_from_metrics or "is_offline_evaluable" not in eval_df.columns:
        return eval_df.copy()
    return eval_df[eval_df["is_offline_evaluable"] == 1].copy()


def evaluate_predictions(full_df: pd.DataFrame, pred_df: pd.DataFrame, cfg: ValidationConfig) -> pd.DataFrame:
    full_df = normalize_backend_df(full_df, backend='auto') if not set(CANDIDATE_COLS).issubset(full_df.columns) else full_df.copy()
    pred_df = normalize_backend_df(pred_df, backend='auto') if not set(CANDIDATE_COLS).issubset(pred_df.columns) else pred_df.copy()
    merge_keys = [QASM_PATH_COL] + CANDIDATE_COLS
    clean_full = full_df.dropna(subset=merge_keys + [RUNTIME_COL, FIDELITY_COL]).copy()
    clean_pred = pred_df.dropna(subset=merge_keys).copy()
    scored = make_training_targets(clean_full, fidelity_tol=cfg.fidelity_tol)
    if not scored.empty:
        scored = (
            scored.sort_values([QASM_PATH_COL, 'near_optimal', 'fidelity_gap_to_best', 'runtime_ratio_to_feasible', RUNTIME_COL],
                               ascending=[True, False, True, True, True])
                  .drop_duplicates(subset=merge_keys, keep='first')
        )
    merged = clean_pred.merge(scored, on=merge_keys, how="left", validate="one_to_one")
    merged = _attach_offline_validation_status(merged)
    oracle = (
        scored.sort_values([QASM_PATH_COL, "target_score"], ascending=[True, False])
        .groupby(QASM_PATH_COL, as_index=False)
        .first()
    )
    merged = merged.merge(
        oracle[[QASM_PATH_COL] + CANDIDATE_COLS].rename(columns={c: f"oracle_{c}" for c in CANDIDATE_COLS}),
        on=QASM_PATH_COL,
        how="left",
    )
    merged["exact_match"] = np.logical_and.reduce([merged[c] == merged[f"oracle_{c}"] for c in CANDIDATE_COLS]).astype(int)
    merged["exact_match"] = np.where(merged["is_offline_evaluable"] == 1, merged["exact_match"], np.nan)
    merged["eval_runtime"] = np.where(merged["is_offline_evaluable"] == 1, merged[RUNTIME_COL], np.nan)
    merged["eval_fidelity"] = np.where(merged["is_offline_evaluable"] == 1, merged[FIDELITY_COL], np.nan)
    for c in ["near_optimal", "runtime_ratio_to_feasible", "fidelity_gap_to_best", "best_feasible_runtime"]:
        merged[c] = np.where(merged["is_offline_evaluable"] == 1, merged[c], np.nan)
    return merged


def summarize_eval(eval_df: pd.DataFrame, cfg: ValidationConfig) -> Dict[str, float]:
    if eval_df is None or len(eval_df) == 0 or QASM_PATH_COL not in eval_df.columns:
        return {
            "n_circuits_total": 0, "n_circuits_validated": 0, "n_circuits_not_evaluable": 0,
            "offline_coverage": 0.0, "exact_match_rate": np.nan, "near_optimal_rate": np.nan,
            "median_runtime_ratio": np.nan, "p90_runtime_ratio": np.nan, "p95_runtime_ratio": np.nan,
            "median_fidelity_gap": np.nan, "p90_fidelity_gap": np.nan, "p95_fidelity_gap": np.nan,
            "runtime_fail_rate_1p2": np.nan, "runtime_fail_rate_1p5": np.nan,
            "fidelity_fail_rate_tol": np.nan, "fidelity_fail_rate_10xtol": np.nan,
        }
    valid = _filter_evaluable(eval_df, cfg)
    n_total = int(eval_df[QASM_PATH_COL].nunique())
    n_valid = int(valid[QASM_PATH_COL].nunique()) if not valid.empty else 0
    n_missing = n_total - n_valid
    if valid.empty:
        out = {
            "n_circuits_total": n_total, "n_circuits_validated": 0, "n_circuits_not_evaluable": n_missing, "offline_coverage": 0.0,
            "exact_match_rate": np.nan, "near_optimal_rate": np.nan, "median_runtime_ratio": np.nan, "p90_runtime_ratio": np.nan, "p95_runtime_ratio": np.nan,
            "median_fidelity_gap": np.nan, "p90_fidelity_gap": np.nan, "p95_fidelity_gap": np.nan, "runtime_fail_rate_1p2": np.nan, "runtime_fail_rate_1p5": np.nan,
            "fidelity_fail_rate_tol": np.nan, "fidelity_fail_rate_10xtol": np.nan,
        }
        return out
    rr = valid["runtime_ratio_to_feasible"].astype(float)
    fg = valid["fidelity_gap_to_best"].astype(float)
    return {
        "n_circuits_total": n_total,
        "n_circuits_validated": n_valid,
        "n_circuits_not_evaluable": n_missing,
        "offline_coverage": float(n_valid / max(n_total, 1)),
        "exact_match_rate": float(valid["exact_match"].astype(float).mean()),
        "near_optimal_rate": float(valid["near_optimal"].astype(float).mean()),
        "median_runtime_ratio": float(rr.median()),
        "p90_runtime_ratio": float(rr.quantile(0.9)),
        "p95_runtime_ratio": float(rr.quantile(0.95)),
        "median_fidelity_gap": float(fg.median()),
        "p90_fidelity_gap": float(fg.quantile(0.9)),
        "p95_fidelity_gap": float(fg.quantile(0.95)),
        "runtime_fail_rate_1p2": float((rr > 1.2).mean()),
        "runtime_fail_rate_1p5": float((rr > 1.5).mean()),
        "fidelity_fail_rate_tol": float((fg > cfg.fidelity_tol).mean()),
        "fidelity_fail_rate_10xtol": float((fg > 10 * cfg.fidelity_tol).mean()),
    }


def summarize_topk(rank_eval_df: pd.DataFrame, cfg: ValidationConfig, ks: Iterable[int] = (1, 3, 5)) -> Dict[str, float]:
    valid = _filter_evaluable(rank_eval_df, cfg)
    out: Dict[str, float] = {}
    if valid.empty or QASM_PATH_COL not in valid.columns or 'pred_rank' not in valid.columns:
        for k in ks:
            out[f"top_{k}_near_optimal_rate"] = np.nan
        return out
    grouped = valid.groupby(QASM_PATH_COL, sort=False)
    def _hit_for_k(g: pd.DataFrame, k: int) -> int:
        sel = g.loc[g["pred_rank"] <= k, "near_optimal"]
        if sel.empty:
            return 0
        sel = sel.fillna(0)
        m = sel.max()
        if pd.isna(m):
            return 0
        return int(m > 0)
    for k in ks:
        hit = grouped.apply(lambda g: _hit_for_k(g, k), include_groups=False)
        out[f"top_{k}_near_optimal_rate"] = float(hit.mean()) if len(hit) else 0.0
    return out


def topk_by_family(rank_eval_df: pd.DataFrame, cfg: ValidationConfig, ks: Iterable[int] = (1, 3, 5), family_col: str = "family") -> pd.DataFrame:
    valid = _filter_evaluable(rank_eval_df, cfg)
    if valid.empty or family_col not in rank_eval_df.columns:
        return pd.DataFrame(columns=[family_col, "n_circuits"] + [f"top_{k}_near_optimal_rate" for k in ks])
    fam_df = rank_eval_df[[QASM_PATH_COL, family_col]].drop_duplicates()
    merged = valid.merge(fam_df, on=QASM_PATH_COL, suffixes=("", "_dup"))
    dup_col = f"{family_col}_dup"
    if dup_col in merged.columns:
        merged = merged.drop(columns=[dup_col])
    rows = []
    for fam, g in merged.groupby(family_col, sort=False):
        record = {family_col: fam, "n_circuits": int(g[QASM_PATH_COL].nunique())}
        by_circuit = g.groupby(QASM_PATH_COL, sort=False)
        def _hit(x: pd.DataFrame, k: int) -> int:
            sel = x.loc[x["pred_rank"] <= k, "near_optimal"]
            if sel.empty:
                return 0
            sel = sel.fillna(0)
            m = sel.max()
            if pd.isna(m):
                return 0
            return int(m > 0)
        for k in ks:
            hit = by_circuit.apply(lambda x: _hit(x, k), include_groups=False)
            record[f"top_{k}_near_optimal_rate"] = float(hit.mean()) if len(hit) else 0.0
        rows.append(record)
    return pd.DataFrame(rows)


def _bootstrap_ci(values: np.ndarray, metric: str = "mean", n_bootstrap: int = 100, seed: int = 42):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(vals), len(vals))
        sample = vals[idx]
        stats.append(sample.mean() if metric == "mean" else np.median(sample))
    return float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def run_random_group_split(full_df: pd.DataFrame, cfg: ValidationConfig, feature_set: str = "advanced"):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=cfg.random_state)
    groups = full_df[QASM_PATH_COL].astype(str)
    train_idx, test_idx = next(gss.split(full_df, groups=groups))
    train = full_df.iloc[train_idx].copy()
    test = full_df.iloc[test_idx].copy()
    candidate_catalogue = build_candidate_catalogue(train)
    ranker = fit_ranker(train, candidate_catalogue, feature_set=feature_set, tree_n_jobs=cfg.validation_tree_n_jobs, model_profile=cfg.validation_model_profile)
    pred_rank = _select_rankings_from_available_candidates(ranker.model, test, max_k=cfg.max_k)
    ev_rank = evaluate_predictions(test, pred_rank, cfg)
    ev_top1 = ev_rank[ev_rank["pred_rank"] == 1].copy()
    return ranker, ev_top1, ev_rank


def run_leave_one_family_out(full_df: pd.DataFrame, cfg: ValidationConfig, feature_set: str = "advanced"):
    eval_rows = []
    rank_rows = []
    summaries = []
    families = sorted(full_df["family"].dropna().unique())
    if cfg.max_lofo_families is not None:
        families = families[:cfg.max_lofo_families]
    for fam in families:
        train = full_df[full_df["family"] != fam].copy()
        test = full_df[full_df["family"] == fam].copy()
        if train.empty or test.empty:
            continue
        candidate_catalogue = build_candidate_catalogue(train)
        ranker = fit_ranker(train, candidate_catalogue, feature_set=feature_set, tree_n_jobs=cfg.validation_tree_n_jobs, model_profile=cfg.validation_model_profile)
        pred_rank = _select_rankings_from_available_candidates(ranker.model, test, max_k=cfg.max_k)
        ev = evaluate_predictions(test, pred_rank, cfg)
        ev["heldout_family"] = fam
        ev_top1 = ev[ev["pred_rank"] == 1].copy()
        eval_rows.append(ev_top1)
        rank_rows.append(ev)
        s = summarize_eval(ev_top1, cfg)
        s.update(summarize_topk(ev, cfg))
        s["heldout_family"] = fam
        summaries.append(s)
        del candidate_catalogue, ranker, pred_rank, ev, ev_top1, train, test
        gc.collect()
    lofo_eval = pd.concat(eval_rows, ignore_index=True) if eval_rows else _empty_eval_df()
    lofo_rank = pd.concat(rank_rows, ignore_index=True) if rank_rows else _empty_eval_df()
    lofo_summary = pd.DataFrame(summaries)
    return lofo_eval, lofo_rank, lofo_summary


def run_size_based_test(full_df: pd.DataFrame, cfg: ValidationConfig, min_circuits_per_size: int = 1, feature_set: str = "advanced"):
    eval_rows = []
    rank_rows = []
    summaries = []
    circuits = full_df[[QASM_PATH_COL, "family", SIZE_COL]].drop_duplicates()
    for fam, fam_df in circuits.groupby("family", sort=False):
        sizes = fam_df[SIZE_COL].dropna().unique()
        if len(sizes) < 2:
            continue
        selected_sizes = sorted(sizes)[::max(1, cfg.size_split_stride)]
        if cfg.max_size_splits_per_family is not None:
            selected_sizes = selected_sizes[:cfg.max_size_splits_per_family]
        for size in selected_sizes:
            held_paths = fam_df.loc[fam_df[SIZE_COL] == size, QASM_PATH_COL].unique().tolist()
            if len(held_paths) < min_circuits_per_size:
                continue
            train = full_df[~full_df[QASM_PATH_COL].isin(held_paths)].copy()
            test = full_df[full_df[QASM_PATH_COL].isin(held_paths)].copy()
            if train.empty or test.empty:
                continue
            candidate_catalogue = build_candidate_catalogue(train)
            ranker = fit_ranker(train, candidate_catalogue, feature_set=feature_set, tree_n_jobs=cfg.validation_tree_n_jobs, model_profile=cfg.validation_model_profile)
            pred_rank = _select_rankings_from_available_candidates(ranker.model, test, max_k=cfg.max_k)
            ev = evaluate_predictions(test, pred_rank, cfg)
            ev["heldout_family"] = fam
            ev["heldout_n_qubits"] = size
            ev_top1 = ev[ev["pred_rank"] == 1].copy()
            eval_rows.append(ev_top1)
            rank_rows.append(ev)
            s = summarize_eval(ev_top1, cfg)
            s.update(summarize_topk(ev, cfg))
            s["heldout_family"] = fam
            s["heldout_n_qubits"] = float(size)
            summaries.append(s)
            del candidate_catalogue, ranker, pred_rank, ev, ev_top1, train, test
            gc.collect()
    size_eval = pd.concat(eval_rows, ignore_index=True) if eval_rows else _empty_eval_df()
    size_rank = pd.concat(rank_rows, ignore_index=True) if rank_rows else _empty_eval_df()
    size_summary = pd.DataFrame(summaries)
    return size_eval, size_rank, size_summary


def run_leave_one_backend_out(full_df: pd.DataFrame, cfg: ValidationConfig, feature_set: str = "advanced"):
    if 'backend' not in full_df.columns:
        return _empty_eval_df(), _empty_eval_df(), pd.DataFrame()
    backends = [b for b in sorted(full_df['backend'].dropna().astype(str).unique()) if b]
    if len(backends) < 2:
        return _empty_eval_df(), _empty_eval_df(), pd.DataFrame()
    eval_rows = []
    rank_rows = []
    summaries = []
    for heldout_backend in backends:
        train = full_df[full_df['backend'].astype(str) != heldout_backend].copy()
        test = full_df[full_df['backend'].astype(str) == heldout_backend].copy()
        if train.empty or test.empty:
            continue
        candidate_catalogue = build_candidate_catalogue(train)
        ranker = fit_ranker(train, candidate_catalogue, feature_set=feature_set, tree_n_jobs=cfg.validation_tree_n_jobs, model_profile=cfg.validation_model_profile)
        pred_rank = _select_rankings_from_available_candidates(ranker.model, test, max_k=cfg.max_k)
        ev = evaluate_predictions(test, pred_rank, cfg)
        ev['heldout_backend'] = heldout_backend
        ev_top1 = ev[ev['pred_rank'] == 1].copy()
        eval_rows.append(ev_top1)
        rank_rows.append(ev)
        s = summarize_eval(ev_top1, cfg)
        s.update(summarize_topk(ev, cfg))
        s['heldout_backend'] = heldout_backend
        summaries.append(s)
        del candidate_catalogue, ranker, pred_rank, ev, ev_top1, train, test
        gc.collect()
    backend_eval = pd.concat(eval_rows, ignore_index=True) if eval_rows else _empty_eval_df()
    backend_rank = pd.concat(rank_rows, ignore_index=True) if rank_rows else _empty_eval_df()
    backend_summary = pd.DataFrame(summaries)
    return backend_eval, backend_rank, backend_summary


def run_feature_set_comparison(full_df: pd.DataFrame, cfg: ValidationConfig, feature_sets: Iterable[str] = ("basic", "structural", "advanced")) -> pd.DataFrame:
    rows = []
    for feature_set in feature_sets:
        try:
            _, random_eval, random_rank = run_random_group_split(full_df, cfg, feature_set=feature_set)
            lofo_eval, lofo_rank, _ = run_leave_one_family_out(full_df, cfg, feature_set=feature_set)
            size_eval, size_rank, _ = (_empty_eval_df(), _empty_eval_df(), pd.DataFrame())
            if cfg.run_size_based:
                size_eval, size_rank, _ = run_size_based_test(full_df, cfg, feature_set=feature_set)
        except Exception as exc:
            rows.append({'feature_set': feature_set, 'status': f'failed: {exc}'})
            continue
        rec = {'feature_set': feature_set, 'status': 'ok'}
        for prefix, ev, rank in [('random', random_eval, random_rank), ('lofo', lofo_eval, lofo_rank), ('size', size_eval, size_rank)]:
            s = summarize_eval(ev, cfg)
            s.update(summarize_topk(rank, cfg))
            rec[f'{prefix}_near_optimal_rate'] = s.get('near_optimal_rate')
            rec[f'{prefix}_median_runtime_ratio'] = s.get('median_runtime_ratio')
            rec[f'{prefix}_median_fidelity_gap'] = s.get('median_fidelity_gap')
            rec[f'{prefix}_top_3_near_optimal_rate'] = s.get('top_3_near_optimal_rate')
            rec[f'{prefix}_top_5_near_optimal_rate'] = s.get('top_5_near_optimal_rate')
        rows.append(rec)
    return pd.DataFrame(rows)


def _prepare_predicted_for_default_compare(eval_df: pd.DataFrame, family_col: str) -> pd.DataFrame:
    cols = [c for c in [QASM_PATH_COL, "eval_runtime", "eval_fidelity", family_col] if c in eval_df.columns]
    out = eval_df[cols].copy()
    rename = {"eval_runtime": "pred_runtime", "eval_fidelity": "pred_fidelity"}
    if family_col in out.columns:
        rename[family_col] = "family"
    return out.rename(columns=rename)


def _add_ci_block(metrics: dict, key: str, df: pd.DataFrame, cfg: ValidationConfig):
    valid_df = _filter_evaluable(df, cfg)
    if valid_df.empty:
        return
    metrics[key]["near_optimal_rate_ci95"] = _bootstrap_ci(valid_df["near_optimal"].to_numpy(float), metric="mean", n_bootstrap=cfg.n_bootstrap, seed=cfg.random_state)
    metrics[key]["median_runtime_ratio_ci95"] = _bootstrap_ci(valid_df["runtime_ratio_to_feasible"].to_numpy(float), metric="median", n_bootstrap=cfg.n_bootstrap, seed=cfg.random_state)
    metrics[key]["median_fidelity_gap_ci95"] = _bootstrap_ci(valid_df["fidelity_gap_to_best"].to_numpy(float), metric="median", n_bootstrap=cfg.n_bootstrap, seed=cfg.random_state)


def _write_metrics_json(outpath: Path, cfg: ValidationConfig, payloads: dict[str, tuple[pd.DataFrame, pd.DataFrame]]):
    metrics = {}
    for key, (top1_df, rank_df) in payloads.items():
        metrics[key] = summarize_eval(top1_df, cfg)
        metrics[key].update(summarize_topk(rank_df, cfg, ks=(1, 3, 5)))
        _add_ci_block(metrics, key, top1_df, cfg)
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def run_validation(full_eval_csv: str | Path, best_csv: str | Path | None, qasm_root: str | Path, outdir: str | Path, cfg: ValidationConfig = ValidationConfig(), default_csv: Optional[str | Path] = None, backend: str = "auto", feature_set: str = "advanced", compare_feature_sets: bool = False, best_flag_col: str | None = "is_best", feature_cache_csv: str | Path | None = None):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    full, best, circuit_features = load_datasets(full_eval_csv, best_csv, qasm_root, backend=backend, best_flag_col=best_flag_col, feature_cache_csv=feature_cache_csv)

    ranker = None
    random_eval = _empty_eval_df(); random_rank_eval = _empty_eval_df()
    lofo_eval = _empty_eval_df(); lofo_rank_eval = _empty_eval_df(); lofo_summary = pd.DataFrame()
    size_eval = _empty_eval_df(); size_rank_eval = _empty_eval_df(); size_summary = pd.DataFrame()
    backend_transfer_eval = _empty_eval_df(); backend_transfer_rank_eval = _empty_eval_df(); backend_transfer_summary = pd.DataFrame()

    if cfg.run_random_split:
        ranker, random_eval, random_rank_eval = run_random_group_split(full, cfg, feature_set=feature_set)
        if ranker is not None:
            ranker.save(outdir / "model")
        random_eval.to_csv(outdir / "random_grouped_eval.csv", index=False)
        if cfg.save_rank_tables:
            random_rank_eval.to_csv(outdir / "random_grouped_rank_eval.csv", index=False)
        gc.collect()

    if cfg.run_lofo:
        lofo_eval, lofo_rank_eval, lofo_summary = run_leave_one_family_out(full, cfg, feature_set=feature_set)
        lofo_eval.to_csv(outdir / "lofo_eval.csv", index=False)
        if cfg.save_rank_tables:
            lofo_rank_eval.to_csv(outdir / "lofo_rank_eval.csv", index=False)
        lofo_summary.to_csv(outdir / "lofo_summary.csv", index=False)
        gc.collect()

    if cfg.run_size_based:
        size_eval, size_rank_eval, size_summary = run_size_based_test(full, cfg, feature_set=feature_set)
        size_eval.to_csv(outdir / "size_based_eval.csv", index=False)
        if cfg.save_rank_tables:
            size_rank_eval.to_csv(outdir / "size_based_rank_eval.csv", index=False)
        size_summary.to_csv(outdir / "size_based_summary.csv", index=False)
        gc.collect()

    if cfg.run_backend_transfer:
        backend_transfer_eval, backend_transfer_rank_eval, backend_transfer_summary = run_leave_one_backend_out(full, cfg, feature_set=feature_set)
        if not backend_transfer_eval.empty:
            backend_transfer_eval.to_csv(outdir / "backend_transfer_eval.csv", index=False)
            if cfg.save_rank_tables:
                backend_transfer_rank_eval.to_csv(outdir / "backend_transfer_rank_eval.csv", index=False)
            backend_transfer_summary.to_csv(outdir / "backend_transfer_summary.csv", index=False)
        gc.collect()

    feature_comp = pd.DataFrame()
    if compare_feature_sets:
        feature_comp = run_feature_set_comparison(full, cfg)
        feature_comp.to_csv(outdir / "feature_set_comparison.csv", index=False)

    if cfg.run_lofo and not lofo_rank_eval.empty:
        topk_by_family(lofo_rank_eval, cfg, ks=(1, 3, 5), family_col="heldout_family").to_csv(outdir / "lofo_topk_by_family.csv", index=False)
    if cfg.run_size_based and not size_rank_eval.empty:
        topk_by_family(size_rank_eval, cfg, ks=(1, 3, 5), family_col="heldout_family").to_csv(outdir / "size_based_topk_by_family.csv", index=False)

    figures_dir = outdir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    if cfg.run_lofo and not lofo_summary.empty:
        plot_summary_heatmap(lofo_summary, figures_dir / "lofo_summary_heatmap.png", title="Leave-one-family-out summary", family_col="heldout_family", save_pdf=cfg.save_pdf_figures)
        plot_family_topk_bars(lofo_summary, figures_dir / "lofo_topk_by_family.png", family_col="heldout_family", save_pdf=cfg.save_pdf_figures)
        plot_family_tradeoff_panel(lofo_summary, figures_dir / "lofo_tradeoff_panel.png", family_col="heldout_family", save_pdf=cfg.save_pdf_figures)
    if cfg.run_size_based and not size_summary.empty:
        plot_summary_heatmap(size_summary, figures_dir / "size_based_summary_heatmap.png", title="Size-based validation summary", family_col="heldout_family", save_pdf=cfg.save_pdf_figures)
    if cfg.run_lofo and cfg.run_size_based and not lofo_eval.empty and not size_eval.empty:
        plot_runtime_distribution_both_regimes(lofo_eval.rename(columns={"runtime_ratio_to_feasible": "runtime_ratio"}), size_eval.rename(columns={"runtime_ratio_to_feasible": "runtime_ratio"}), figures_dir / "runtime_distribution_lofo_vs_size_based.png", runtime_col="runtime_ratio", family_col="heldout_family", save_pdf=cfg.save_pdf_figures)

    if default_csv is not None and cfg.run_lofo and cfg.run_size_based and not lofo_eval.empty and not size_eval.empty:
        lofo_pred = _prepare_predicted_for_default_compare(lofo_eval, "heldout_family")
        size_pred = _prepare_predicted_for_default_compare(size_eval, "heldout_family")
        lofo_compare_dir = outdir / "compare_default_lofo"
        size_compare_dir = outdir / "compare_default_size_based"
        lofo_pairs = compare_predicted_vs_default(lofo_pred, default_csv, lofo_compare_dir, backend=backend)
        size_pairs = compare_predicted_vs_default(size_pred, default_csv, size_compare_dir, backend=backend)
        plot_runtime_vs_default_distribution_both(lofo_pairs, size_pairs, figures_dir / "predicted_vs_default_runtime_distribution_both.png", save_pdf=cfg.save_pdf_figures)
        plot_median_speedup_vs_default_both(lofo_pairs, size_pairs, figures_dir / "predicted_vs_default_median_speedup_both.png", save_pdf=cfg.save_pdf_figures)
        plot_ecdf_speedup_vs_default_both(lofo_pairs, size_pairs, figures_dir / "predicted_vs_default_ecdf_both.png", save_pdf=cfg.save_pdf_figures)

    if compare_feature_sets and not feature_comp.empty:
        plot_feature_set_comparison(feature_comp, figures_dir / "feature_set_comparison.png", save_pdf=cfg.save_pdf_figures)

    payloads = {}
    if cfg.run_random_split:
        payloads["random_grouped_split"] = (random_eval, random_rank_eval)
    if cfg.run_lofo:
        payloads["leave_one_family_out"] = (lofo_eval, lofo_rank_eval)
    if cfg.run_size_based:
        payloads["size_based_test"] = (size_eval, size_rank_eval)
    if cfg.run_backend_transfer:
        payloads["leave_one_backend_out"] = (backend_transfer_eval, backend_transfer_rank_eval)
    _write_metrics_json(outdir / "metrics_summary.json", cfg, payloads)

    return {"ranker": ranker, "random_top1": random_eval, "lofo_top1": lofo_eval, "size_top1": size_eval}
