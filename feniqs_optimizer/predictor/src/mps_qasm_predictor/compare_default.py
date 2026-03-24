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
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import QASM_PATH_COL


def _infer_family_from_path(path: str) -> str:
    s = str(path).replace('\\', '/')
    stem = s.split('/')[-1]
    if stem.lower().endswith('.qasm'):
        stem = stem[:-5]
    parts = stem.split('_')
    if len(parts) >= 3 and parts[-1].isdigit():
        return '_'.join(parts[:-2]).lower()
    if len(parts) >= 2 and parts[-1].isdigit():
        return '_'.join(parts[:-1]).lower()
    return parts[0].lower() if parts else 'unknown'


def _normalize_default_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    colmap = {}
    for c in out.columns:
        cl = c.lower()
        if c == 'qasm_path':
            colmap[c] = 'qasm_path'
        elif cl in {'runtime', 'time_taken', 'run_time', 'execution_time', 'elapsed_time'}:
            colmap[c] = 'runtime_default'
        elif cl in {'fidelity', 'state_fidelity', 'final_fidelity'}:
            colmap[c] = 'fidelity_default'
        elif cl == 'family':
            colmap[c] = 'family'
    out = out.rename(columns=colmap)
    if 'family' not in out.columns:
        out['family'] = out['qasm_path'].astype(str).map(_infer_family_from_path)
    return out[['qasm_path', 'family', 'runtime_default', 'fidelity_default']].copy()


def _styled(ax, title: str, xlabel: str, ylabel: str):
    ax.set_title(title, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def compare_predicted_vs_default(pred_top1: pd.DataFrame, default_csv: str | Path, outdir: str | Path):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    pred = pred_top1[[QASM_PATH_COL, 'family', 'eval_runtime', 'eval_fidelity']].copy()
    pred = pred.rename(columns={'eval_runtime': 'runtime_pred', 'eval_fidelity': 'fidelity_pred'})
    pred = pred.groupby([QASM_PATH_COL, 'family'], as_index=False).median(numeric_only=True)
    dflt = _normalize_default_df(pd.read_csv(default_csv))
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
    fig.tight_layout(); fig.savefig(out / 'family_median_speedup.png', bbox_inches='tight', dpi=240); plt.close(fig)

    families = list(merged.groupby('family')['runtime_ratio_vs_default'].median().sort_values().index)
    data = [merged.loc[merged['family'] == fam, 'runtime_ratio_vs_default'].to_numpy(float) for fam in families]
    fig, ax = plt.subplots(figsize=(11, 5.8))
    parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
    for pc in parts['bodies']:
        pc.set_alpha(0.55)
    ax.boxplot(data, positions=np.arange(1, len(families) + 1), widths=0.18, patch_artist=False, showfliers=True)
    ax.axhline(1.0, linestyle='--', color='black', linewidth=1)
    ax.set_xticks(np.arange(1, len(families) + 1))
    ax.set_xticklabels(families, rotation=45, ha='right')
    ax.set_yscale('log')
    _styled(ax, 'Runtime ratio vs default by family', 'Family', 'Predicted/default runtime ratio')
    fig.tight_layout(); fig.savefig(out / 'family_runtime_ratio_violin.png', bbox_inches='tight', dpi=240); plt.close(fig)

    return merged
