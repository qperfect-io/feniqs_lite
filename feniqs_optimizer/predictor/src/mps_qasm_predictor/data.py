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
from .backend import normalize_backend_df, infer_family_from_path
from .config import CANDIDATE_COLS, QASM_PATH_COL, RUNTIME_COL, FIDELITY_COL
from .features import extract_qasm_features, read_qasm, qasm_sha256

BEST_FLAG_CANDIDATES = ["is_best", "is-best", "is best", "isbest", "best"]


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


def build_circuit_feature_table(qasm_paths: Iterable[str], qasm_root: str | Path, cache_csv: str | Path | None = None) -> pd.DataFrame:
    requested = sorted(set(map(str, qasm_paths)))
    cached = pd.DataFrame()
    cache_path = Path(cache_csv) if cache_csv else None
    if cache_path is not None and cache_path.exists() and cache_path.stat().st_size > 0:
        try:
            cached = pd.read_csv(cache_path)
        except Exception:
            cached = pd.DataFrame()
    cached_map = {}
    if not cached.empty and QASM_PATH_COL in cached.columns:
        cached_map = {str(qp): row for qp, row in cached.set_index(QASM_PATH_COL).iterrows()}
    rows=[]
    updated = False
    for qp in requested:
        if qp in cached_map:
            rows.append(cached_map[qp].to_dict() | {QASM_PATH_COL: qp})
            continue
        resolved=resolve_qasm_path(qp, qasm_root)
        qasm=read_qasm(resolved)
        feats=extract_qasm_features(qasm)
        feats[QASM_PATH_COL]=qp
        feats["resolved_qasm_path"]=str(resolved)
        feats["family"]=infer_family_from_path(qp)
        feats["qasm_sha256"]=qasm_sha256(qasm)
        rows.append(feats)
        updated = True
    out = pd.DataFrame(rows)
    if cache_path is not None and updated:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(cache_path, index=False)
    return out


def _truthy_mask(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    vals = series.astype("string").str.strip().str.lower()
    return vals.isin(["true", "1", "yes", "y", "t"])


def _find_best_flag_column(columns: Iterable[str], preferred: str | None = None) -> str | None:
    cols = list(columns)
    if preferred and preferred in cols:
        return preferred
    lower_map = {str(c).strip().lower(): c for c in cols}
    for cand in ([preferred] if preferred else []) + BEST_FLAG_CANDIDATES:
        if cand is None:
            continue
        key = str(cand).strip().lower()
        if key in lower_map:
            return lower_map[key]
    return None


def _choose_best_rows(scored: pd.DataFrame, fidelity_tol: float = 1e-3) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()
    tmp = scored.copy()
    best_fidelity = tmp.groupby(QASM_PATH_COL, sort=False)[FIDELITY_COL].transform('max')
    tmp['_best_feasible_runtime'] = (
        tmp.groupby(QASM_PATH_COL, sort=False)[[RUNTIME_COL, FIDELITY_COL]]
        .apply(lambda g: g.loc[g[FIDELITY_COL] >= g[FIDELITY_COL].max() - fidelity_tol, RUNTIME_COL].min())
        .rename('_best_feasible_runtime')
    )
    tmp['_fidelity_gap'] = (best_fidelity - tmp[FIDELITY_COL]).clip(lower=0.0)
    tmp['_runtime_gap'] = (tmp[RUNTIME_COL] - tmp['_best_feasible_runtime']).abs()
    best = (
        tmp.sort_values([QASM_PATH_COL, '_fidelity_gap', '_runtime_gap', RUNTIME_COL], ascending=[True, True, True, True])
        .groupby(QASM_PATH_COL, as_index=False)
        .first()
        .drop(columns=['_best_feasible_runtime', '_fidelity_gap', '_runtime_gap'], errors='ignore')
    )
    return best


def _derive_best_from_full(full: pd.DataFrame, fidelity_tol: float = 1e-3, best_flag_col: str | None = None) -> pd.DataFrame:
    if full.empty:
        return full.copy()

    best_flag_col = _find_best_flag_column(full.columns, preferred=best_flag_col)
    flagged_best = pd.DataFrame(columns=full.columns)
    if best_flag_col is not None:
        mask = _truthy_mask(full[best_flag_col])
        flagged_best = full.loc[mask].copy()
        if not flagged_best.empty:
            flagged_best = _choose_best_rows(flagged_best, fidelity_tol=fidelity_tol)

    flagged_paths = set(flagged_best[QASM_PATH_COL].astype(str)) if not flagged_best.empty else set()
    all_paths = set(full[QASM_PATH_COL].astype(str))
    missing_paths = all_paths - flagged_paths

    if not missing_paths:
        return flagged_best.reset_index(drop=True)

    derived_best = _choose_best_rows(full[full[QASM_PATH_COL].astype(str).isin(missing_paths)].copy(), fidelity_tol=fidelity_tol)

    if flagged_best.empty:
        return derived_best.reset_index(drop=True)

    best = pd.concat([flagged_best, derived_best], ignore_index=True)
    best = _choose_best_rows(best, fidelity_tol=fidelity_tol)
    return best.reset_index(drop=True)


def _merge_with_circuit_features(df: pd.DataFrame, circuit_features: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    merged = df.merge(circuit_features, on=QASM_PATH_COL, how='inner', validate='many_to_one', suffixes=('', '_cf'))
    for col in ['family', 'resolved_qasm_path', 'qasm_sha256']:
        cf_col = f'{col}_cf'
        if cf_col in merged.columns:
            if col in merged.columns:
                merged[col] = merged[col].where(merged[col].notna(), merged[cf_col])
                merged = merged.drop(columns=[cf_col])
            else:
                merged = merged.rename(columns={cf_col: col})
    return merged


def _safe_read_csv(path: str | Path | None, label: str) -> pd.DataFrame | None:
    if path is None:
        return None
    p = Path(path)
    if not str(p).strip():
        return None
    if not p.exists():
        raise FileNotFoundError(f"{label} does not exist: {p}")
    if p.is_dir():
        raise IsADirectoryError(f"{label} points to a directory, expected a CSV file: {p}")
    if p.stat().st_size == 0:
        return None
    return pd.read_csv(p)


def load_datasets(
    full_eval_csv: str | Path,
    best_csv: Optional[str | Path],
    qasm_root: str | Path,
    backend: str = 'auto',
    best_flag_col: str | None = None,
    feature_cache_csv: str | Path | None = None,
):
    full_raw = _safe_read_csv(full_eval_csv, 'full-eval-csv')
    if full_raw is None:
        raise ValueError('full-eval-csv is empty or missing headers; it must contain the full evaluation table.')
    full = normalize_backend_df(full_raw, backend=backend)

    best_raw = _safe_read_csv(best_csv, 'best-csv') if best_csv else None
    if best_raw is not None:
        best = normalize_backend_df(best_raw, backend=backend)
    else:
        best = _derive_best_from_full(full, best_flag_col=best_flag_col)

    circuit_features = build_circuit_feature_table(full[QASM_PATH_COL].unique(), qasm_root, cache_csv=feature_cache_csv)
    full = _merge_with_circuit_features(full, circuit_features)
    best = _merge_with_circuit_features(best, circuit_features)
    return full, best, circuit_features


def build_candidate_catalogue(full_df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in CANDIDATE_COLS if c not in full_df.columns]
    if missing:
        raise ValueError(f"Missing canonical candidate columns: {missing}")
    cat = full_df[CANDIDATE_COLS].drop_duplicates().sort_values(CANDIDATE_COLS).reset_index(drop=True)
    cat["candidate_id"] = range(len(cat))
    return cat
