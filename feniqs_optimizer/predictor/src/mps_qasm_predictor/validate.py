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
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .compare_default import compare_predicted_vs_default
from .config import ValidationConfig, QASM_PATH_COL, CANDIDATE_COLS, RUNTIME_COL, FIDELITY_COL
from .data import load_datasets, build_candidate_catalogue
from .model import fit_ranker, make_training_targets, build_candidate_matrix_for_circuit
from .plots import (
    plot_summary_heatmap,
    plot_runtime_distribution_both_regimes,
    plot_runtime_vs_default_distribution_both,
    plot_median_speedup_vs_default_both,
    plot_ecdf_speedup_vs_default_both,
)

SIZE_COL = "n_total"


def _select_rankings_from_available_candidates(model, circuit_df: pd.DataFrame, max_k: int = 5) -> pd.DataFrame:
    rows = []
    for qasm_path, g in circuit_df.groupby(QASM_PATH_COL, sort=False):
        row = g.iloc[0]
        feats = row.drop(labels=[c for c in [RUNTIME_COL, FIDELITY_COL] + CANDIDATE_COLS if c in row.index])
        available_candidates = g[CANDIDATE_COLS].drop_duplicates().reset_index(drop=True)
        X = build_candidate_matrix_for_circuit(feats, available_candidates)
        scores = model.predict(X)
        order = np.argsort(scores)[::-1]
        for rank, idx in enumerate(order[:max_k], start=1):
            choice = available_candidates.iloc[int(idx)].to_dict()
            choice[QASM_PATH_COL] = qasm_path
            choice["predicted_score"] = float(scores[int(idx)])
            choice["pred_rank"] = int(rank)
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


def _filter_evaluable(eval_df: pd.DataFrame, cfg: ValidationConfig) -> pd.DataFrame:
    if not cfg.drop_unevaluated_from_metrics:
        return eval_df.copy()
    return eval_df[eval_df["is_offline_evaluable"] == 1].copy()


def evaluate_predictions(full_df: pd.DataFrame, pred_df: pd.DataFrame, cfg: ValidationConfig) -> pd.DataFrame:
    scored = make_training_targets(full_df, fidelity_tol=cfg.fidelity_tol)
    merged = pred_df.merge(scored, on=[QASM_PATH_COL] + CANDIDATE_COLS, how="left", validate="one_to_one")
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
    merged["exact_match"] = np.logical_and.reduce(
        [merged[c] == merged[f"oracle_{c}"] for c in CANDIDATE_COLS]
    ).astype(int)
    merged["exact_match"] = np.where(merged["is_offline_evaluable"] == 1, merged["exact_match"], np.nan)
    merged["eval_runtime"] = np.where(merged["is_offline_evaluable"] == 1, merged[RUNTIME_COL], np.nan)
    merged["eval_fidelity"] = np.where(merged["is_offline_evaluable"] == 1, merged[FIDELITY_COL], np.nan)

    for c in ["near_optimal", "runtime_ratio_to_feasible", "fidelity_gap_to_best", "best_feasible_runtime"]:
        merged[c] = np.where(merged["is_offline_evaluable"] == 1, merged[c], np.nan)

    return merged


def summarize_eval(eval_df: pd.DataFrame, cfg: ValidationConfig) -> Dict[str, float]:
    valid = _filter_evaluable(eval_df, cfg)
    n_total = int(eval_df[QASM_PATH_COL].nunique())
    n_valid = int(valid[QASM_PATH_COL].nunique())
    n_missing = n_total - n_valid

    if valid.empty:
        return {
            "n_circuits_total": n_total,
            "n_circuits_validated": 0,
            "n_circuits_not_evaluable": n_missing,
            "offline_coverage": 0.0,
            "exact_match_rate": np.nan,
            "near_optimal_rate": np.nan,
            "median_runtime_ratio": np.nan,
            "p90_runtime_ratio": np.nan,
            "p95_runtime_ratio": np.nan,
            "median_fidelity_gap": np.nan,
            "p90_fidelity_gap": np.nan,
            "p95_fidelity_gap": np.nan,
            "runtime_fail_rate_1p2": np.nan,
            "runtime_fail_rate_1p5": np.nan,
            "fidelity_fail_rate_tol": np.nan,
            "fidelity_fail_rate_10xtol": np.nan,
        }

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

    if valid.empty:
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


def topk_by_family(
    rank_eval_df: pd.DataFrame,
    cfg: ValidationConfig,
    ks: Iterable[int] = (1, 3, 5),
    family_col: str = "family",
) -> pd.DataFrame:
    valid = _filter_evaluable(rank_eval_df, cfg)
    fam_df = rank_eval_df[[QASM_PATH_COL, family_col]].drop_duplicates()

    if valid.empty:
        return pd.DataFrame(columns=[family_col, "n_circuits"] + [f"top_{k}_near_optimal_rate" for k in ks])

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


def _bootstrap_ci(values: np.ndarray, metric: str = "mean", n_bootstrap: int = 1000, seed: int = 42):
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


def run_random_group_split(full_df: pd.DataFrame, cfg: ValidationConfig):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=cfg.random_state)
    groups = full_df[QASM_PATH_COL].astype(str)
    train_idx, test_idx = next(gss.split(full_df, groups=groups))
    train = full_df.iloc[train_idx].copy()
    test = full_df.iloc[test_idx].copy()

    candidate_catalogue = build_candidate_catalogue(train)
    ranker = fit_ranker(train, candidate_catalogue)

    pred_rank = _select_rankings_from_available_candidates(ranker.model, test, max_k=5)
    ev_rank = evaluate_predictions(test, pred_rank, cfg)
    ev_top1 = ev_rank[ev_rank["pred_rank"] == 1].copy()

    return ranker, ev_top1, ev_rank


def run_leave_one_family_out(full_df: pd.DataFrame, cfg: ValidationConfig):
    eval_rows = []
    rank_rows = []
    summaries = []

    for fam in sorted(full_df["family"].dropna().unique()):
        train = full_df[full_df["family"] != fam].copy()
        test = full_df[full_df["family"] == fam].copy()
        if train.empty or test.empty:
            continue

        candidate_catalogue = build_candidate_catalogue(train)
        ranker = fit_ranker(train, candidate_catalogue)

        pred_rank = _select_rankings_from_available_candidates(ranker.model, test, max_k=5)
        ev = evaluate_predictions(test, pred_rank, cfg)
        ev["heldout_family"] = fam

        ev_top1 = ev[ev["pred_rank"] == 1].copy()
        eval_rows.append(ev_top1)
        rank_rows.append(ev)

        s = summarize_eval(ev_top1, cfg)
        s.update(summarize_topk(ev, cfg))
        s["heldout_family"] = fam
        summaries.append(s)

    lofo_eval = pd.concat(eval_rows, ignore_index=True) if eval_rows else pd.DataFrame()
    lofo_rank = pd.concat(rank_rows, ignore_index=True) if rank_rows else pd.DataFrame()
    lofo_summary = pd.DataFrame(summaries)

    return lofo_eval, lofo_rank, lofo_summary


def run_size_based_test(full_df: pd.DataFrame, cfg: ValidationConfig, min_circuits_per_size: int = 1):
    eval_rows = []
    rank_rows = []
    summaries = []

    circuits = full_df[[QASM_PATH_COL, "family", SIZE_COL]].drop_duplicates()
    for fam, fam_df in circuits.groupby("family", sort=False):
        sizes = fam_df[SIZE_COL].dropna().unique()
        if len(sizes) < 2:
            continue

        for size in sorted(sizes):
            held_paths = fam_df.loc[fam_df[SIZE_COL] == size, QASM_PATH_COL].unique().tolist()
            if len(held_paths) < min_circuits_per_size:
                continue

            train = full_df[~full_df[QASM_PATH_COL].isin(held_paths)].copy()
            test = full_df[full_df[QASM_PATH_COL].isin(held_paths)].copy()
            if train.empty or test.empty:
                continue

            candidate_catalogue = build_candidate_catalogue(train)
            ranker = fit_ranker(train, candidate_catalogue)

            pred_rank = _select_rankings_from_available_candidates(ranker.model, test, max_k=5)
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

    size_eval = pd.concat(eval_rows, ignore_index=True) if eval_rows else pd.DataFrame()
    size_rank = pd.concat(rank_rows, ignore_index=True) if rank_rows else pd.DataFrame()
    size_summary = pd.DataFrame(summaries)

    return size_eval, size_rank, size_summary


def _prepare_predicted_for_default_compare(eval_df: pd.DataFrame, family_col: str) -> pd.DataFrame:
    cols = [QASM_PATH_COL, "eval_runtime", "eval_fidelity", family_col]
    out = eval_df[cols].copy()
    out = out.rename(
        columns={
            "eval_runtime": "pred_runtime",
            "eval_fidelity": "pred_fidelity",
            family_col: "family",
        }
    )
    return out


def _write_metrics_json(
    outpath: Path,
    cfg: ValidationConfig,
    random_eval: pd.DataFrame,
    random_rank_eval: pd.DataFrame,
    lofo_eval: pd.DataFrame,
    lofo_rank_eval: pd.DataFrame,
    size_eval: pd.DataFrame,
    size_rank_eval: pd.DataFrame,
):
    valid_random_eval = _filter_evaluable(random_eval, cfg)
    valid_lofo_eval = _filter_evaluable(lofo_eval, cfg)
    valid_size_eval = _filter_evaluable(size_eval, cfg)

    metrics = {
        "random_grouped_split": summarize_eval(random_eval, cfg),
        "leave_one_family_out": summarize_eval(lofo_eval, cfg),
        "size_based_test": summarize_eval(size_eval, cfg),
    }

    metrics["random_grouped_split"].update(summarize_topk(random_rank_eval, cfg, ks=(1, 3, 5)))
    metrics["leave_one_family_out"].update(summarize_topk(lofo_rank_eval, cfg, ks=(1, 3, 5)))
    metrics["size_based_test"].update(summarize_topk(size_rank_eval, cfg, ks=(1, 3, 5)))

    metrics["random_grouped_split"]["near_optimal_rate_ci95"] = _bootstrap_ci(
        valid_random_eval["near_optimal"].to_numpy(float),
        metric="mean",
        n_bootstrap=cfg.n_bootstrap,
        seed=cfg.random_state,
    )
    metrics["random_grouped_split"]["median_runtime_ratio_ci95"] = _bootstrap_ci(
        valid_random_eval["runtime_ratio_to_feasible"].to_numpy(float),
        metric="median",
        n_bootstrap=cfg.n_bootstrap,
        seed=cfg.random_state,
    )
    metrics["random_grouped_split"]["median_fidelity_gap_ci95"] = _bootstrap_ci(
        valid_random_eval["fidelity_gap_to_best"].to_numpy(float),
        metric="median",
        n_bootstrap=cfg.n_bootstrap,
        seed=cfg.random_state,
    )

    metrics["leave_one_family_out"]["near_optimal_rate_ci95"] = _bootstrap_ci(
        valid_lofo_eval["near_optimal"].to_numpy(float),
        metric="mean",
        n_bootstrap=cfg.n_bootstrap,
        seed=cfg.random_state,
    )
    metrics["leave_one_family_out"]["median_runtime_ratio_ci95"] = _bootstrap_ci(
        valid_lofo_eval["runtime_ratio_to_feasible"].to_numpy(float),
        metric="median",
        n_bootstrap=cfg.n_bootstrap,
        seed=cfg.random_state,
    )
    metrics["leave_one_family_out"]["median_fidelity_gap_ci95"] = _bootstrap_ci(
        valid_lofo_eval["fidelity_gap_to_best"].to_numpy(float),
        metric="median",
        n_bootstrap=cfg.n_bootstrap,
        seed=cfg.random_state,
    )

    metrics["size_based_test"]["near_optimal_rate_ci95"] = _bootstrap_ci(
        valid_size_eval["near_optimal"].to_numpy(float),
        metric="mean",
        n_bootstrap=cfg.n_bootstrap,
        seed=cfg.random_state,
    )
    metrics["size_based_test"]["median_runtime_ratio_ci95"] = _bootstrap_ci(
        valid_size_eval["runtime_ratio_to_feasible"].to_numpy(float),
        metric="median",
        n_bootstrap=cfg.n_bootstrap,
        seed=cfg.random_state,
    )
    metrics["size_based_test"]["median_fidelity_gap_ci95"] = _bootstrap_ci(
        valid_size_eval["fidelity_gap_to_best"].to_numpy(float),
        metric="median",
        n_bootstrap=cfg.n_bootstrap,
        seed=cfg.random_state,
    )

    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def run_validation(
    full_eval_csv: str | Path,
    best_csv: str | Path | None,
    qasm_root: str | Path,
    outdir: str | Path,
    cfg: ValidationConfig = ValidationConfig(),
    default_csv: Optional[str | Path] = None,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    full, best, circuit_features = load_datasets(full_eval_csv, best_csv, qasm_root)

    ranker, random_eval, random_rank_eval = run_random_group_split(full, cfg)
    lofo_eval, lofo_rank_eval, lofo_summary = run_leave_one_family_out(full, cfg)
    size_eval, size_rank_eval, size_summary = run_size_based_test(full, cfg)

    ranker.save(outdir / "model")

    random_eval.to_csv(outdir / "random_grouped_eval.csv", index=False)
    random_rank_eval.to_csv(outdir / "random_grouped_rank_eval.csv", index=False)

    lofo_eval.to_csv(outdir / "lofo_eval.csv", index=False)
    lofo_rank_eval.to_csv(outdir / "lofo_rank_eval.csv", index=False)
    lofo_summary.to_csv(outdir / "lofo_summary.csv", index=False)

    size_eval.to_csv(outdir / "size_based_eval.csv", index=False)
    size_rank_eval.to_csv(outdir / "size_based_rank_eval.csv", index=False)
    size_summary.to_csv(outdir / "size_based_summary.csv", index=False)

    topk_by_family(lofo_rank_eval, cfg, ks=(1, 3, 5), family_col="heldout_family").to_csv(
        outdir / "lofo_topk_by_family.csv",
        index=False,
    )
    topk_by_family(size_rank_eval, cfg, ks=(1, 3, 5), family_col="heldout_family").to_csv(
        outdir / "size_based_topk_by_family.csv",
        index=False,
    )

    figures_dir = outdir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_summary_heatmap(
        lofo_summary,
        figures_dir / "lofo_summary_heatmap.png",
        title="Leave-one-family-out summary heatmap",
        family_col="heldout_family",
    )
    plot_summary_heatmap(
        size_summary,
        figures_dir / "size_based_summary_heatmap.png",
        title="Size-based summary heatmap",
        family_col="heldout_family",
    )
    plot_runtime_distribution_both_regimes(
        lofo_eval.rename(columns={"runtime_ratio_to_feasible": "runtime_ratio", "heldout_family": "heldout_family"}),
        size_eval.rename(columns={"runtime_ratio_to_feasible": "runtime_ratio", "heldout_family": "heldout_family"}),
        figures_dir / "runtime_distribution_lofo_vs_size_based.png",
        runtime_col="runtime_ratio",
        family_col="heldout_family",
    )

    if default_csv is not None:
        lofo_pred = _prepare_predicted_for_default_compare(lofo_eval, "heldout_family")
        size_pred = _prepare_predicted_for_default_compare(size_eval, "heldout_family")

        lofo_compare_dir = outdir / "compare_default_lofo"
        size_compare_dir = outdir / "compare_default_size_based"

        lofo_pairs = compare_predicted_vs_default(lofo_pred, default_csv, lofo_compare_dir)
        size_pairs = compare_predicted_vs_default(size_pred, default_csv, size_compare_dir)

        plot_runtime_vs_default_distribution_both(
            lofo_pairs,
            size_pairs,
            figures_dir / "predicted_vs_default_runtime_distribution_both.png",
        )
        plot_median_speedup_vs_default_both(
            lofo_pairs,
            size_pairs,
            figures_dir / "predicted_vs_default_median_speedup_both.png",
        )
        plot_ecdf_speedup_vs_default_both(
            lofo_pairs,
            size_pairs,
            figures_dir / "predicted_vs_default_ecdf_both.png",
        )

    _write_metrics_json(
        outdir / "metrics_summary.json",
        cfg,
        random_eval,
        random_rank_eval,
        lofo_eval,
        lofo_rank_eval,
        size_eval,
        size_rank_eval,
    )

    return {
        "ranker": ranker,
        "random_top1": random_eval,
        "lofo_top1": lofo_eval,
        "size_top1": size_eval,
    }
