from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

QISKIT = "qiskit"
MIMIQ = "mimiq"
SUPPORTED_BACKENDS = {QISKIT, MIMIQ}


def infer_family_from_path(path: str) -> str:
    s = str(path).replace('\\', '/')
    parts = s.split('/')
    if 'paper_data' in parts:
        idx = parts.index('paper_data')
        if idx + 1 < len(parts):
            return parts[idx + 1]
    stem = Path(s).stem
    parts = stem.split('_')
    if len(parts) >= 3 and parts[-1].isdigit():
        return '_'.join(parts[:-2]).lower()
    if len(parts) >= 2 and parts[-1].isdigit():
        return '_'.join(parts[:-1]).lower()
    return parts[0].lower() if parts else 'unknown'


def detect_backend(df: pd.DataFrame) -> str:
    cols = set(df.columns)
    qiskit_markers = {
        'matrix_product_state_max_bond_dimension',
        'mps_lapack',
        'matrix_product_state_truncation_threshold',
        'mps_sample_measure_algorithm',
    }
    mimiq_markers = {'bond_dimension', 'entdim', 'scut', 'meth'}
    if len(cols & qiskit_markers) >= 2:
        return QISKIT
    if len(cols & mimiq_markers) >= 2:
        return MIMIQ
    raise ValueError(
        'Could not detect backend from columns. '
        f'Columns are: {sorted(df.columns.tolist())}'
    )


def _infer_method_from_series(series: pd.Series) -> pd.Series:
    out = series.astype('string')
    out = out.replace({'<NA>': pd.NA, 'nan': pd.NA, 'None': pd.NA, 'False': pd.NA, 'false': pd.NA, '': pd.NA, '0': pd.NA})
    return out


def normalize_backend_df(df: pd.DataFrame, backend: Optional[str] = None) -> pd.DataFrame:
    out = df.copy()
    if backend is None or backend == 'auto':
        backend = detect_backend(out)
    backend = str(backend).lower()
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f'Unsupported backend: {backend}')

    rename_map: dict[str, str] = {}
    for c in out.columns:
        cl = c.lower()
        if c == 'qasm_path':
            rename_map[c] = 'qasm_path'
        elif cl in {'runtime', 'time_taken', 'run_time', 'execution_time', 'elapsed_time', 'eval_runtime', 'pred_runtime', 'runtime_default', 'default_runtime'}:
            # preserve special names used in downstream compare scripts
            if c in {'eval_runtime', 'pred_runtime', 'runtime_default', 'default_runtime'}:
                rename_map[c] = c
            else:
                rename_map[c] = 'runtime'
        elif cl in {'fidelity', 'state_fidelity', 'final_fidelity', 'eval_fidelity', 'fidelity_default', 'default_fidelity'}:
            if c in {'eval_fidelity', 'fidelity_default', 'default_fidelity'}:
                rename_map[c] = c
            else:
                rename_map[c] = 'fidelity'
    out = out.rename(columns=rename_map)
    out['backend'] = backend

    if backend == QISKIT:
        qmap = {
            'matrix_product_state_max_bond_dimension': 'bond_dimension',
            'matrix_product_state_truncation_threshold': 'entdim',
            'mps_lapack': 'method',
            'mps_sample_measure_algorithm': 'sample_algorithm',
            'opt_level': 'opt_level',
        }
        for old, new in qmap.items():
            if old in out.columns and new not in out.columns:
                out[new] = out[old]
        if 'method' in out.columns:
            out['method'] = pd.to_numeric(out['method'], errors='coerce').astype('Int64').astype('string')
        if 'sample_algorithm' not in out.columns:
            out['sample_algorithm'] = 'unknown'
    else:
        if 'bond_dimension' not in out.columns or 'entdim' not in out.columns:
            raise ValueError('Mimiq data must contain bond_dimension and entdim columns')
        if 'opt_level' not in out.columns:
            out['opt_level'] = 0
        method = pd.Series(pd.NA, index=out.index, dtype='string')
        if 'meth' in out.columns:
            cand = _infer_method_from_series(out['meth'])
            method = method.fillna(cand)
        if 'scut' in out.columns:
            scut = out['scut']
            scut_num = pd.to_numeric(scut, errors='coerce')
            out['scut_value'] = scut_num
            if scut.dtype == object or str(scut.dtype).startswith('string'):
                method = method.fillna(_infer_method_from_series(scut))
        out['method'] = method.fillna('unknown')
        if 'sample_algorithm' not in out.columns:
            out['sample_algorithm'] = 'unknown'

    if 'bond_dimension' in out.columns:
        out['bond_dimension'] = pd.to_numeric(out['bond_dimension'], errors='coerce').astype('Int64')
    if 'entdim' in out.columns:
        out['entdim'] = pd.to_numeric(out['entdim'], errors='coerce')
    if 'opt_level' in out.columns:
        out['opt_level'] = pd.to_numeric(out['opt_level'], errors='coerce').fillna(0).astype('Int64')
    out['method'] = out.get('method', pd.Series('unknown', index=out.index)).astype('string').fillna('unknown')
    out['sample_algorithm'] = out.get('sample_algorithm', pd.Series('unknown', index=out.index)).astype('string').fillna('unknown')

    if 'family' not in out.columns:
        if 'heldout_family' in out.columns:
            out['family'] = out['heldout_family']
        elif 'qasm_path' in out.columns:
            out['family'] = out['qasm_path'].astype(str).map(infer_family_from_path)

    return out
