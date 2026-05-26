#
# Copyright © 2026 QPerfect. All Rights Reserved.

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .backend import normalize_backend_df, infer_family_from_path
from .config import QASM_PATH_COL, CANDIDATE_COLS


def _styled(ax, title: str, xlabel: str, ylabel: str):
    ax.set_title(title, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _normalize_default_df(df: pd.DataFrame, backend: str = 'auto') -> pd.DataFrame:
    """Normalize a default-results CSV.

    Default baseline files are allowed to be *minimal* and contain only
    qasm_path/runtime/fidelity (plus optional family). They do not need the
    backend hyperparameter columns.

    If candidate columns are present we keep them, but we never require them.
    This is important for Mimiq, where the default CSV often stores only the
    baseline measurements.
    """
    out = df.copy()

    # First try the full backend normalization only when the dataframe appears
    # to actually contain backend parameter columns. Otherwise keep the minimal
    # schema and normalize only the baseline fields.
    candidate_markers = set(CANDIDATE_COLS) | {
        'matrix_product_state_max_bond_dimension',
        'matrix_product_state_truncation_threshold',
        'mps_lapack',
        'mps_sample_measure_algorithm',
        'bond_dimension',
        'entdim',
        'scut',
        'meth',
        'method',
        'sample_algorithm',
    }
    if len(candidate_markers.intersection(out.columns)) > 0:
        out = normalize_backend_df(out, backend=backend)
    else:
        # Minimal baseline CSV path. Keep this deliberately permissive.
        rename_map = {}
        for c in out.columns:
            cl = c.lower().strip()
            if c == 'qasm_path':
                rename_map[c] = 'qasm_path'
            elif cl in {'runtime', 'eval_runtime', 'default_runtime', 'run_time', 'execution_time', 'elapsed_time'}:
                rename_map[c] = 'runtime'
            elif cl in {'fidelity', 'eval_fidelity', 'default_fidelity', 'state_fidelity', 'final_fidelity'}:
                rename_map[c] = 'fidelity'
            elif cl == 'heldout_family':
                rename_map[c] = 'family'
        out = out.rename(columns=rename_map)
        if 'family' not in out.columns and 'qasm_path' in out.columns:
            out['family'] = out['qasm_path'].astype(str).map(infer_family_from_path)

    if 'qasm_path' not in out.columns:
        raise ValueError('default-csv must contain qasm_path')
    if 'runtime_default' not in out.columns:
        if 'runtime' in out.columns:
            out['runtime_default'] = out['runtime']
        else:
            raise ValueError('default-csv must contain runtime or runtime_default')
    if 'fidelity_default' not in out.columns:
        if 'fidelity' in out.columns:
            out['fidelity_default'] = out['fidelity']
        else:
            raise ValueError('default-csv must contain fidelity or fidelity_default')
    if 'family' not in out.columns:
        out['family'] = out['qasm_path'].astype(str).map(infer_family_from_path)

    cols = ['qasm_path', 'family', 'runtime_default', 'fidelity_default']
    for c in CANDIDATE_COLS:
        if c in out.columns:
            cols.append(c)
    out = out[cols].copy()
    out = out.dropna(subset=['qasm_path', 'runtime_default', 'fidelity_default'])
    out = out.drop_duplicates(subset=['qasm_path'], keep='first')
    return out


def _truthy_mask(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    vals = series.astype('string').str.strip().str.lower()
    return vals.isin(['true', '1', 'yes', 'y', 't'])


def _parse_scalar(value: str):
    s = str(value).strip()
    low = s.lower()
    if low in {'true', 'false'}:
        return low == 'true'
    try:
        if any(ch in s for ch in ['.', 'e', 'E']):
            return float(s)
        return int(s)
    except ValueError:
        return s


def _parse_default_filter(default_filter: str | None) -> dict[str, object]:
    if default_filter is None:
        return {}
    filt = {}
    for part in str(default_filter).split(','):
        part = part.strip()
        if not part:
            continue
        if '=' not in part:
            raise ValueError(f"Invalid --default-filter fragment {part!r}. Expected key=value.")
        key, val = part.split('=', 1)
        filt[key.strip()] = _parse_scalar(val)
    return filt


def derive_default_rows(source_csv: str | Path, backend: str = 'auto', default_filter: str | None = None) -> pd.DataFrame:
    full = normalize_backend_df(pd.read_csv(source_csv), backend=backend)

    if 'is_default' in full.columns:
        mask = _truthy_mask(full['is_default'])
        dflt = full.loc[mask].copy()
        if not dflt.empty:
            return _normalize_default_df(dflt, backend=backend)

    filt = _parse_default_filter(default_filter)
    if filt:
        dflt = full.copy()
        for key, expected in filt.items():
            if key not in dflt.columns:
                raise ValueError(f"Default filter references missing column {key!r}. Available columns: {sorted(dflt.columns)}")
            col = dflt[key]
            if pd.api.types.is_numeric_dtype(col):
                dflt = dflt[col == pd.to_numeric(expected, errors='coerce')]
            else:
                dflt = dflt[col.astype('string') == str(expected)]
        if dflt.empty:
            raise ValueError('Default filter matched zero rows in full-eval-csv.')
        # keep best matching row per circuit using fastest runtime, then best fidelity tie-break
        if {'runtime', 'fidelity'}.issubset(dflt.columns):
            dflt = (
                dflt.sort_values([QASM_PATH_COL, 'runtime', 'fidelity'], ascending=[True, True, False])
                .groupby(QASM_PATH_COL, as_index=False)
                .first()
            )
        return _normalize_default_df(dflt, backend=backend)

    raise ValueError(
        'No default baseline could be derived automatically. '
        'Provide --default-csv, or add an is_default column to full-eval-csv, '
        'or pass --default-filter with a canonical tuple such as '
        'bond_dimension=64,entdim=8,opt_level=1,method=zipup or '
        'sample_algorithm=mps_probabilities.'
    )


def compare_predicted_vs_default(
    pred_top1: pd.DataFrame,
    default_csv: str | Path | None,
    outdir: str | Path,
    backend: str = 'auto',
    full_eval_csv: str | Path | None = None,
    default_filter: str | None = None,
):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    pred = pred_top1[[QASM_PATH_COL, 'family', 'eval_runtime', 'eval_fidelity']].copy()
    pred = pred.rename(columns={'eval_runtime': 'runtime_pred', 'eval_fidelity': 'fidelity_pred'})
    pred = pred.groupby([QASM_PATH_COL, 'family'], as_index=False).median(numeric_only=True)

    if default_csv is not None:
        dflt = _normalize_default_df(pd.read_csv(default_csv), backend=backend)
    elif full_eval_csv is not None:
        dflt = derive_default_rows(full_eval_csv, backend=backend, default_filter=default_filter)
    else:
        raise ValueError('Either default_csv or full_eval_csv must be provided for default comparison.')

    dflt = dflt.groupby([QASM_PATH_COL, 'family'], as_index=False).median(numeric_only=True)
    merged = pred.merge(dflt[[QASM_PATH_COL, 'runtime_default', 'fidelity_default']], on=QASM_PATH_COL, how='inner')
    if merged.empty:
        (out / 'summary.json').write_text(json.dumps({'matched_circuits': 0}, indent=2), encoding='utf-8')
        return merged

    merged['speedup_vs_default'] = merged['runtime_default'] / merged['runtime_pred'].clip(lower=1e-12)
    merged['runtime_ratio_vs_default'] = merged['runtime_pred'] / merged['runtime_default'].clip(lower=1e-12)
    merged['fidelity_delta_vs_default'] = merged['fidelity_pred'] - merged['fidelity_default']
    merged.to_csv(out / 'predicted_vs_default_pairs.csv', index=False)

    rr = merged['runtime_ratio_vs_default'].astype(float)
    fd = merged['fidelity_delta_vs_default'].astype(float)
    summary = {
        'matched_circuits': int(len(merged)),
        'median_speedup_vs_default': float(merged['speedup_vs_default'].median()),
        'geomean_speedup_vs_default': float(np.exp(np.log(merged['speedup_vs_default'].clip(lower=1e-12)).mean())),
        'fraction_speedup_gt_1': float((merged['speedup_vs_default'] > 1.0).mean()),
        'fraction_speedup_ge_1p2': float((merged['speedup_vs_default'] >= 1.2).mean()),
        'fraction_speedup_ge_1p5': float((merged['speedup_vs_default'] >= 1.5).mean()),
        'median_runtime_ratio_vs_default': float(rr.median()),
        'p95_runtime_ratio_vs_default': float(rr.quantile(0.95)),
        'median_fidelity_delta_vs_default': float(fd.median()),
        'p95_fidelity_loss_vs_default': float(np.quantile(np.clip(-fd, 0, None), 0.95)),
        'fraction_no_fidelity_drop_tol_1e3': float((fd >= -1e-3).mean()),
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    family = merged.groupby('family', as_index=False).agg(
        n_circuits=(QASM_PATH_COL, 'count'),
        median_speedup_vs_default=('speedup_vs_default', 'median'),
        median_runtime_ratio_vs_default=('runtime_ratio_vs_default', 'median'),
        p95_runtime_ratio_vs_default=('runtime_ratio_vs_default', lambda x: float(np.quantile(x, 0.95))),
        median_fidelity_delta_vs_default=('fidelity_delta_vs_default', 'median'),
        p95_fidelity_loss_vs_default=('fidelity_delta_vs_default', lambda x: float(np.quantile(np.clip(-x, 0, None), 0.95))),
        fraction_speedup_gt_1=('speedup_vs_default', lambda x: float(np.mean(x > 1.0))),
    )
    family.to_csv(out / 'family_summary.csv', index=False)

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.scatter(merged['runtime_default'], merged['runtime_pred'], alpha=0.7)
    lim = max(float(merged['runtime_default'].max()), float(merged['runtime_pred'].max()))
    ax.plot([0, lim], [0, lim], '--', color='black', linewidth=1)
    _styled(ax, 'Predicted top-1 vs default runtime', 'Default runtime', 'Predicted-selected runtime')
    fig.tight_layout(); fig.savefig(out / 'runtime_scatter.png', bbox_inches='tight', dpi=240); plt.close(fig)

    vals = np.sort(merged['speedup_vs_default'].values)
    yy = np.arange(1, len(vals) + 1) / len(vals)
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.step(vals, yy, where='post')
    ax.axvline(1.0, linestyle='--', color='black', linewidth=1)
    _styled(ax, 'ECDF of speedup vs default', 'Speedup vs default', 'ECDF')
    fig.tight_layout(); fig.savefig(out / 'speedup_ecdf.png', bbox_inches='tight', dpi=240); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    ax.scatter(merged['runtime_ratio_vs_default'], merged['fidelity_delta_vs_default'], alpha=0.7)
    ax.axvline(1.0, linestyle='--', color='black', linewidth=1)
    ax.axhline(0.0, linestyle=':', color='black', linewidth=1)
    _styled(ax, 'Predicted vs default trade-off', 'Predicted/default runtime ratio', 'Predicted - default fidelity')
    fig.tight_layout(); fig.savefig(out / 'tradeoff_scatter.png', bbox_inches='tight', dpi=240); plt.close(fig)

    fam = family.sort_values('median_speedup_vs_default', ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.bar(fam['family'], fam['median_speedup_vs_default'])
    ax.axhline(1.0, linestyle='--', color='black', linewidth=1)
    _styled(ax, 'Median speedup vs default by family', 'Family', 'Median speedup')
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout(); fig.savefig(out / 'speedup_by_family.png', bbox_inches='tight', dpi=240); plt.close(fig)

    return merged
