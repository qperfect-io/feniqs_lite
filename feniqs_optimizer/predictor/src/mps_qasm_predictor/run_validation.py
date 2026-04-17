from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FixedFormatter

from .compare_default import compare_predicted_vs_default
from .config import ValidationConfig
from .data import load_datasets, build_candidate_catalogue
from .model import fit_ranker, FEATURE_SET_PRESETS, MODEL_PROFILES
from .validate import run_validation

FAMILY_ORDER = [
    "ae", "ghz", "graphstate", "qft", "qftentangled", "qnn",
    "qpeexact", "qpeinexact", "qwalk-v-chain", "random",
    "realamprandom", "su2random", "wstate",
]

Q_COLOR = '#4C78A8'
M_COLOR = '#F28E2B'
Y_TICKS = [0.8, 1, 2, 3, 5, 10, 20, 50, 100, 200, 500]


def _load_speedup_pairs(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'family' not in df.columns or 'speedup_vs_default' not in df.columns:
        raise ValueError(f'{path} must contain family and speedup_vs_default columns')
    return df[['family', 'speedup_vs_default']].rename(columns={'speedup_vs_default': 'speedup'}).dropna().copy()


def plot_two_backend_family_boxplot(
    qiskit_pairs_csv: str | Path,
    mimiq_pairs_csv: str | Path,
    outpath: str | Path,
    title: str,
    ylabel: str = 'Speedup (default / predicted)',
) -> None:
    q_df = _load_speedup_pairs(qiskit_pairs_csv)
    m_df = _load_speedup_pairs(mimiq_pairs_csv)

    families = [f for f in FAMILY_ORDER if f in set(q_df['family']).union(set(m_df['family']))]
    q_data = [q_df.loc[q_df['family'] == f, 'speedup'].astype(float).values for f in families]
    m_data = [m_df.loc[m_df['family'] == f, 'speedup'].astype(float).values for f in families]

    positions = np.arange(len(families))
    offset = 0.18

    fig, ax = plt.subplots(figsize=(12.5, 7.5))

    bp1 = ax.boxplot(
        q_data,
        positions=positions - offset,
        widths=0.30,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color='black', linewidth=1.8),
        boxprops=dict(edgecolor=Q_COLOR, linewidth=1.2),
        whiskerprops=dict(color=Q_COLOR, linewidth=1.2),
        capprops=dict(color=Q_COLOR, linewidth=1.2),
    )
    bp2 = ax.boxplot(
        m_data,
        positions=positions + offset,
        widths=0.30,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color='black', linewidth=1.8),
        boxprops=dict(edgecolor=M_COLOR, linewidth=1.2),
        whiskerprops=dict(color=M_COLOR, linewidth=1.2),
        capprops=dict(color=M_COLOR, linewidth=1.2),
    )

    for patch in bp1['boxes']:
        patch.set_facecolor(Q_COLOR)
        patch.set_alpha(0.55)
    for patch in bp2['boxes']:
        patch.set_facecolor(M_COLOR)
        patch.set_alpha(0.55)

    ax.legend(handles=[
        Patch(facecolor=Q_COLOR, edgecolor=Q_COLOR, alpha=0.55, label='Qiskit'),
        Patch(facecolor=M_COLOR, edgecolor=M_COLOR, alpha=0.55, label='MIMIQ'),
    ], loc='upper right')

    ax.set_xticks(positions)
    ax.set_xticklabels(families, rotation=35, ha='right')
    ax.set_yscale('log')
    ax.axhline(1, linestyle='--', color='gray', linewidth=1.0)
    ax.yaxis.set_major_locator(FixedLocator(Y_TICKS))
    ax.yaxis.set_major_formatter(FixedFormatter([str(t) for t in Y_TICKS]))
    ax.set_ylim(0.75, 600)
    ax.grid(True, which='major', axis='y', alpha=0.35)
    ax.grid(True, which='minor', axis='y', alpha=0.15)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _single_backend_train_and_validate(
    *,
    full_eval_csv: str | Path,
    best_csv: str | Path | None,
    qasm_root: str | Path,
    outdir: str | Path,
    backend: str,
    feature_set: str,
    model_profile: str,
    validation_model_profile: str,
    default_csv: str | Path | None,
    best_flag_col: str | None,
    feature_cache_csv: str | Path | None,
    size_split_stride: int,
    max_size_splits_per_family: int | None,
    max_lofo_families: int | None,
    n_bootstrap: int,
) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    full, best, circuit_features = load_datasets(
        full_eval_csv, best_csv, qasm_root,
        backend=backend,
        best_flag_col=best_flag_col,
        feature_cache_csv=feature_cache_csv,
    )

    candidate_catalogue = build_candidate_catalogue(full)
    ranker = fit_ranker(
        full,
        candidate_catalogue,
        feature_set=feature_set,
        tree_n_jobs=1,
        model_profile=model_profile,
    )
    ranker.save(outdir / 'model')
    circuit_features.to_csv(outdir / 'circuit_features.csv', index=False)
    with open(outdir / 'training_summary.json', 'w', encoding='utf-8') as f:
        json.dump({
            'backend': backend,
            'feature_set': feature_set,
            'model_profile': model_profile,
            'validation_model_profile': validation_model_profile,
            'n_rows': int(len(full)),
            'n_unique_qasm': int(full['qasm_path'].nunique()),
            'n_families': int(full['family'].nunique()),
            'n_candidates': int(len(candidate_catalogue)),
        }, f, indent=2)

    vcfg = ValidationConfig(
        n_bootstrap=n_bootstrap,
        validation_tree_n_jobs=1,
        validation_model_profile=validation_model_profile,
        max_k=5,
        save_rank_tables=False,
        save_pdf_figures=False,
        run_random_split=False,
        run_lofo=True,
        run_size_based=True,
        run_backend_transfer=False,
        size_split_stride=max(1, size_split_stride),
        max_size_splits_per_family=max_size_splits_per_family,
        max_lofo_families=max_lofo_families,
    )
    run_validation(
        full_eval_csv, best_csv, qasm_root, outdir / 'validation',
        cfg=vcfg,
        backend=backend,
        feature_set=feature_set,
        compare_feature_sets=False,
        best_flag_col=best_flag_col,
        feature_cache_csv=feature_cache_csv,
    )

    if default_csv is not None:
        lofo_eval = pd.read_csv(outdir / 'validation' / 'lofo_eval.csv')
        size_eval = pd.read_csv(outdir / 'validation' / 'size_based_eval.csv')
        compare_predicted_vs_default(lofo_eval, default_csv, outdir / 'validation' / 'compare_default_lofo', backend=backend)
        compare_predicted_vs_default(size_eval, default_csv, outdir / 'validation' / 'compare_default_size_based', backend=backend)

    del full, best, circuit_features, candidate_catalogue, ranker
    gc.collect()


def main() -> None:
    ap = argparse.ArgumentParser(description='Sequentially train/validate Qiskit and MIMIQ predictors and generate publication-style figures.')
    ap.add_argument('--qiskit-full-eval-csv', required=True)
    ap.add_argument('--mimiq-full-eval-csv', required=True)
    ap.add_argument('--qiskit-best-csv', default=None)
    ap.add_argument('--mimiq-best-csv', default=None)
    ap.add_argument('--qiskit-default-csv', default=None)
    ap.add_argument('--mimiq-default-csv', default=None)
    ap.add_argument('--qasm-root', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--feature-cache-csv', default=None)
    ap.add_argument('--best-flag-col', default='is_best')
    ap.add_argument('--feature-set', default='structural', choices=sorted(FEATURE_SET_PRESETS.keys()), help='Use structural by default to reduce memory pressure while keeping MPS-relevant features.')
    ap.add_argument('--model-profile', default='lite', choices=sorted(MODEL_PROFILES.keys()), help='Final model profile saved for each backend.')
    ap.add_argument('--validation-model-profile', default='ultralite', choices=sorted(MODEL_PROFILES.keys()), help='Validation-only model profile. Ultralite greatly reduces RAM, especially for MIMIQ.')
    ap.add_argument('--size-split-stride', type=int, default=1, help='Use every N-th size bucket in size-based validation. 1 keeps all sizes.')
    ap.add_argument('--max-size-splits-per-family', type=int, default=None, help='Optional cap on held-out sizes per family. Omit to use all available sizes.')
    ap.add_argument('--max-lofo-families', type=int, default=None)
    ap.add_argument('--n-bootstrap', type=int, default=100)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    q_out = outdir / 'qiskit'
    m_out = outdir / 'mimiq'

    _single_backend_train_and_validate(
        full_eval_csv=args.qiskit_full_eval_csv,
        best_csv=args.qiskit_best_csv,
        qasm_root=args.qasm_root,
        outdir=q_out,
        backend='qiskit',
        feature_set=args.feature_set,
        model_profile=args.model_profile,
        validation_model_profile=args.validation_model_profile,
        default_csv=args.qiskit_default_csv,
        best_flag_col=args.best_flag_col,
        feature_cache_csv=args.feature_cache_csv,
        size_split_stride=args.size_split_stride,
        max_size_splits_per_family=args.max_size_splits_per_family,
        max_lofo_families=args.max_lofo_families,
        n_bootstrap=args.n_bootstrap,
    )

    _single_backend_train_and_validate(
        full_eval_csv=args.mimiq_full_eval_csv,
        best_csv=args.mimiq_best_csv,
        qasm_root=args.qasm_root,
        outdir=m_out,
        backend='mimiq',
        feature_set=args.feature_set,
        model_profile=args.model_profile,
        validation_model_profile=args.validation_model_profile,
        default_csv=args.mimiq_default_csv,
        best_flag_col=args.best_flag_col,
        feature_cache_csv=args.feature_cache_csv,
        size_split_stride=args.size_split_stride,
        max_size_splits_per_family=args.max_size_splits_per_family,
        max_lofo_families=args.max_lofo_families,
        n_bootstrap=args.n_bootstrap,
    )

    fig_dir = outdir / 'paper_figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    q_lofo = q_out / 'validation' / 'compare_default_lofo' / 'predicted_vs_default_pairs.csv'
    m_lofo = m_out / 'validation' / 'compare_default_lofo' / 'predicted_vs_default_pairs.csv'
    q_size = q_out / 'validation' / 'compare_default_size_based' / 'predicted_vs_default_pairs.csv'
    m_size = m_out / 'validation' / 'compare_default_size_based' / 'predicted_vs_default_pairs.csv'

    if q_lofo.exists() and m_lofo.exists():
        plot_two_backend_family_boxplot(q_lofo, m_lofo, fig_dir / 'lofo_pred_vs_def.png', 'Prediction speedup by circuit family (family validation)')
    if q_size.exists() and m_size.exists():
        plot_two_backend_family_boxplot(q_size, m_size, fig_dir / 'size_pred_vs_def.png', 'Prediction speedup by circuit family (size-based validation)')


if __name__ == '__main__':
    main()
