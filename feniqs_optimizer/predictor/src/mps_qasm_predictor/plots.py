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
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

PAPER_DPI = 300

HEATMAP_METRICS = [
    "top_1_near_optimal_rate",
    "top_3_near_optimal_rate",
    "top_5_near_optimal_rate",
    "median_runtime_ratio",
    "median_fidelity_gap",
    "p95_runtime_ratio",
    "p95_fidelity_gap",
]

HIGHER_IS_BETTER = {
    "top_1_near_optimal_rate": True,
    "top_3_near_optimal_rate": True,
    "top_5_near_optimal_rate": True,
    "median_runtime_ratio": False,
    "median_fidelity_gap": False,
    "p95_runtime_ratio": False,
    "p95_fidelity_gap": False,
}

PRETTY_METRICS = {
    "top_1_near_optimal_rate": "Top-1 near-opt.",
    "top_3_near_optimal_rate": "Top-3 near-opt.",
    "top_5_near_optimal_rate": "Top-5 near-opt.",
    "median_runtime_ratio": "Median runtime ratio",
    "median_fidelity_gap": "Median fidelity gap",
    "p95_runtime_ratio": "95th pct runtime ratio",
    "p95_fidelity_gap": "95th pct fidelity gap",
    "runtime_fail_rate_1p2": ">1.2× runtime fail rate",
    "fidelity_fail_rate_tol": "Fidelity fail rate",
}


def _paper_style():
    plt.rcParams.update(
        {
            "figure.dpi": PAPER_DPI,
            "savefig.dpi": PAPER_DPI,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _savefig(fig, outpath: str | Path, save_pdf: bool = True):
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    if save_pdf and outpath.suffix.lower() != ".pdf":
        fig.savefig(outpath.with_suffix('.pdf'), bbox_inches="tight")


def _format_metric_value(metric: str, value: float) -> str:
    if pd.isna(value):
        return ""
    if "rate" in metric:
        return f"{100*value:.1f}%"
    if "fidelity" in metric:
        return f"{value:.2e}" if abs(value) < 1e-3 else f"{value:.3f}"
    return f"{value:.3f}"


def _normalized_heatmap_matrix(df: pd.DataFrame, metrics: list[str]) -> np.ndarray:
    arr = np.full((df.shape[0], len(metrics)), np.nan, dtype=float)
    for j, m in enumerate(metrics):
        vals = pd.to_numeric(df[m], errors='coerce').to_numpy(float)
        mask = np.isfinite(vals)
        if mask.sum() == 0:
            continue
        v = vals[mask]
        if np.isclose(np.nanmin(v), np.nanmax(v)):
            normed = np.ones_like(v) * 0.5
        else:
            normed = (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v))
        if not HIGHER_IS_BETTER.get(m, False):
            normed = 1.0 - normed
        arr[mask, j] = normed
    return arr


def plot_summary_heatmap(summary_df: pd.DataFrame, outpath: str | Path, title: str, family_col: str = "heldout_family", metrics: Iterable[str] = HEATMAP_METRICS, save_pdf: bool = True):
    _paper_style()
    metrics = [m for m in metrics if m in summary_df.columns]
    if family_col not in summary_df.columns or not metrics or summary_df.empty:
        return
    work = summary_df[[family_col] + metrics].copy()
    sort_cols = [c for c in ["top_1_near_optimal_rate", "median_runtime_ratio"] if c in work.columns]
    asc = [False if c == "top_1_near_optimal_rate" else True for c in sort_cols]
    if sort_cols:
        work = work.sort_values(by=sort_cols, ascending=asc).reset_index(drop=True)
    mat = _normalized_heatmap_matrix(work, metrics)
    fig_h = max(4.8, 0.42 * len(work) + 1.6)
    fig_w = max(8.5, 1.5 * len(metrics) + 2.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_title(title, pad=10)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Held-out family")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([PRETTY_METRICS.get(m, m) for m in metrics], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(work)))
    ax.set_yticklabels(work[family_col].astype(str).tolist())
    for i in range(work.shape[0]):
        for j, m in enumerate(metrics):
            val = work.iloc[i][m]
            txt = _format_metric_value(m, val)
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.8, color="black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Normalized desirability")
    fig.tight_layout()
    _savefig(fig, outpath, save_pdf=save_pdf)
    plt.close(fig)


def plot_family_topk_bars(summary_df: pd.DataFrame, outpath: str | Path, family_col: str = "heldout_family", save_pdf: bool = True):
    _paper_style()
    needed = [c for c in ["top_1_near_optimal_rate", "top_3_near_optimal_rate", "top_5_near_optimal_rate"] if c in summary_df.columns]
    if family_col not in summary_df.columns or len(needed) < 2 or summary_df.empty:
        return
    work = summary_df[[family_col] + needed].copy().sort_values(needed[0], ascending=True)
    y = np.arange(len(work))
    height = 0.22
    offsets = np.linspace(-height, height, len(needed))
    fig, ax = plt.subplots(figsize=(8.5, max(4.6, 0.38 * len(work) + 1.4)))
    for off, col in zip(offsets, needed):
        ax.barh(y + off, 100 * pd.to_numeric(work[col], errors='coerce').fillna(0.0), height=height, label=PRETTY_METRICS.get(col, col))
    ax.set_yticks(y)
    ax.set_yticklabels(work[family_col].astype(str))
    ax.set_xlabel("Near-optimal rate (%)")
    ax.set_title("Near-optimal recommendation rate by family")
    ax.set_xlim(0, 100)
    ax.legend(frameon=False, ncol=len(needed), loc="lower right")
    ax.grid(True, axis='x', alpha=0.22)
    fig.tight_layout()
    _savefig(fig, outpath, save_pdf=save_pdf)
    plt.close(fig)


def plot_family_tradeoff_panel(summary_df: pd.DataFrame, outpath: str | Path, family_col: str = "heldout_family", save_pdf: bool = True):
    _paper_style()
    if family_col not in summary_df.columns or summary_df.empty:
        return
    needed = [c for c in ["top_1_near_optimal_rate", "median_runtime_ratio", "median_fidelity_gap"] if c in summary_df.columns]
    if len(needed) < 3:
        return
    work = summary_df[[family_col] + needed].copy().sort_values("top_1_near_optimal_rate", ascending=True)
    y = np.arange(len(work))
    fig, axes = plt.subplots(1, 3, figsize=(12.5, max(4.8, 0.38 * len(work) + 1.2)), sharey=True, gridspec_kw={"width_ratios": [1.0, 1.0, 1.1]})

    axes[0].barh(y, 100 * pd.to_numeric(work["top_1_near_optimal_rate"], errors='coerce').fillna(0.0))
    axes[0].set_title("Top-1 near-opt.")
    axes[0].set_xlabel("Rate (%)")
    axes[0].set_xlim(0, 100)

    axes[1].scatter(pd.to_numeric(work["median_runtime_ratio"], errors='coerce'), y, s=42)
    axes[1].axvline(1.0, linestyle='--', linewidth=1.0, color='black')
    axes[1].set_title("Median runtime ratio")
    axes[1].set_xlabel("ratio")

    axes[2].scatter(pd.to_numeric(work["median_fidelity_gap"], errors='coerce'), y, s=42)
    axes[2].set_title("Median fidelity gap")
    axes[2].set_xlabel("gap")
    axes[2].xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.1e}" if abs(x) < 1e-3 and x != 0 else f"{x:.3f}"))

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(work[family_col].astype(str))
    for ax in axes[1:]:
        ax.tick_params(axis='y', left=False, labelleft=False)
    fig.suptitle("Per-family validation trade-offs", y=0.995)
    fig.tight_layout()
    _savefig(fig, outpath, save_pdf=save_pdf)
    plt.close(fig)


def plot_feature_set_comparison(comp_df: pd.DataFrame, outpath: str | Path, save_pdf: bool = True):
    _paper_style()
    if comp_df.empty or 'feature_set' not in comp_df.columns:
        return
    work = comp_df.copy()
    ok = work.get('status', 'ok').astype(str).eq('ok') if 'status' in work.columns else pd.Series(True, index=work.index)
    work = work[ok].copy()
    if work.empty:
        return
    metrics = [c for c in ['random_near_optimal_rate', 'lofo_near_optimal_rate', 'size_near_optimal_rate'] if c in work.columns]
    if not metrics:
        return
    x = np.arange(len(work))
    width = 0.22
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    offs = np.linspace(-width, width, len(metrics))
    for off, m in zip(offs, metrics):
        ax.bar(x + off, 100 * pd.to_numeric(work[m], errors='coerce').fillna(0.0), width=width, label=PRETTY_METRICS.get(m, m).replace(' rate',''))
    ax.set_xticks(x)
    ax.set_xticklabels(work['feature_set'].astype(str))
    ax.set_ylim(0, 100)
    ax.set_ylabel('Near-optimal rate (%)')
    ax.set_title('Feature-set comparison')
    ax.legend(frameon=False)
    fig.tight_layout()
    _savefig(fig, outpath, save_pdf=save_pdf)
    plt.close(fig)


def _ordered_families_from_union(lofo_df: pd.DataFrame, size_df: pd.DataFrame, family_col: str = "heldout_family") -> list[str]:
    fams = sorted(set(lofo_df[family_col].dropna().astype(str)).union(set(size_df[family_col].dropna().astype(str))))
    return fams


def _make_violin_data(df: pd.DataFrame, value_col: str, regime_name: str, family_col: str = "heldout_family") -> pd.DataFrame:
    work = df[[family_col, value_col]].copy()
    work = work.dropna()
    work = work.rename(columns={family_col: "family", value_col: "value"})
    work["regime"] = regime_name
    return work


def plot_runtime_distribution_both_regimes(lofo_eval: pd.DataFrame, size_eval: pd.DataFrame, outpath: str | Path, runtime_col: str = "runtime_ratio", family_col: str = "heldout_family", save_pdf: bool = True):
    _paper_style()
    if runtime_col not in lofo_eval.columns or runtime_col not in size_eval.columns:
        return
    fam_order = _ordered_families_from_union(lofo_eval, size_eval, family_col=family_col)
    a = _make_violin_data(lofo_eval, runtime_col, "LOFO", family_col=family_col)
    b = _make_violin_data(size_eval, runtime_col, "Size-based", family_col=family_col)
    df = pd.concat([a, b], ignore_index=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for ax, regime in zip(axes, ["LOFO", "Size-based"]):
        sub = df[df["regime"] == regime]
        data = [sub.loc[sub["family"] == fam, "value"].values for fam in fam_order]
        parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_alpha(0.45)
        bp = ax.boxplot(data, widths=0.16, patch_artist=False)
        for median in bp["medians"]:
            median.set_linewidth(1.6)
        ax.axhline(1.0, linestyle="--", linewidth=1.2, color="black")
        ax.set_yscale("log")
        ax.set_ylabel("Runtime ratio")
        ax.set_title(regime)
        ax.set_xticks(np.arange(1, len(fam_order) + 1))
        ax.set_xticklabels(fam_order, rotation=35, ha="right")
    fig.suptitle("Runtime-ratio distribution across validation regimes", y=0.995)
    fig.tight_layout()
    _savefig(fig, outpath, save_pdf=save_pdf)
    plt.close(fig)


def _prepare_compare_default_df(df: pd.DataFrame, regime_name: str) -> pd.DataFrame:
    out = df.copy()
    if "runtime_ratio_vs_default" not in out.columns:
        if {"eval_runtime", "runtime_default"}.issubset(out.columns):
            out["runtime_ratio_vs_default"] = out["eval_runtime"] / out["runtime_default"].clip(lower=1e-12)
        elif {"pred_runtime", "default_runtime"}.issubset(out.columns):
            out["runtime_ratio_vs_default"] = out["pred_runtime"] / out["default_runtime"].clip(lower=1e-12)
        else:
            raise ValueError("Could not infer runtime_ratio_vs_default")
    if "speedup_vs_default" not in out.columns:
        out["speedup_vs_default"] = 1.0 / out["runtime_ratio_vs_default"].clip(lower=1e-12)
    if "family" not in out.columns and "heldout_family" in out.columns:
        out["family"] = out["heldout_family"]
    out["regime"] = regime_name
    return out


def plot_runtime_vs_default_distribution_both(lofo_pairs: pd.DataFrame, size_pairs: pd.DataFrame, outpath: str | Path, save_pdf: bool = True):
    _paper_style()
    lofo = _prepare_compare_default_df(lofo_pairs, "LOFO")
    size = _prepare_compare_default_df(size_pairs, "Size-based")
    df = pd.concat([lofo, size], ignore_index=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, regime in zip(axes, ["LOFO", "Size-based"]):
        sub = df[df["regime"] == regime]["runtime_ratio_vs_default"].dropna().values
        ax.hist(sub, bins=28, alpha=0.9)
        ax.axvline(1.0, linestyle="--", linewidth=1.2, color="black")
        ax.set_xscale("log")
        ax.set_ylabel("Count")
        ax.set_title(regime)
    axes[-1].set_xlabel("Predicted/default runtime ratio")
    fig.suptitle("Predicted vs default: runtime-ratio distribution", y=0.995)
    fig.tight_layout()
    _savefig(fig, outpath, save_pdf=save_pdf)
    plt.close(fig)


def plot_median_speedup_vs_default_both(lofo_pairs: pd.DataFrame, size_pairs: pd.DataFrame, outpath: str | Path, save_pdf: bool = True):
    _paper_style()
    lofo = _prepare_compare_default_df(lofo_pairs, "LOFO")
    size = _prepare_compare_default_df(size_pairs, "Size-based")
    a = lofo.groupby("family", as_index=False)["speedup_vs_default"].median().rename(columns={"speedup_vs_default": "LOFO"})
    b = size.groupby("family", as_index=False)["speedup_vs_default"].median().rename(columns={"speedup_vs_default": "Size-based"})
    merged = a.merge(b, on="family", how="outer").fillna(np.nan)
    merged["sort_key"] = np.nanmean(merged[["LOFO", "Size-based"]].values, axis=1)
    merged = merged.sort_values("sort_key", ascending=False)
    x = np.arange(len(merged))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - width / 2, merged["LOFO"], width=width, label="LOFO")
    ax.bar(x + width / 2, merged["Size-based"], width=width, label="Size-based")
    ax.axhline(1.0, linestyle="--", linewidth=1.2, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels(merged["family"], rotation=35, ha="right")
    ax.set_ylabel("Median speedup vs default")
    ax.set_title("Median speedup vs default across validation regimes")
    ax.legend(frameon=False)
    fig.tight_layout()
    _savefig(fig, outpath, save_pdf=save_pdf)
    plt.close(fig)


def plot_ecdf_speedup_vs_default_both(lofo_pairs: pd.DataFrame, size_pairs: pd.DataFrame, outpath: str | Path, save_pdf: bool = True):
    _paper_style()
    lofo = _prepare_compare_default_df(lofo_pairs, "LOFO")
    size = _prepare_compare_default_df(size_pairs, "Size-based")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for df, label in [(lofo, "LOFO"), (size, "Size-based")]:
        vals = np.sort(df["speedup_vs_default"].dropna().values)
        if len(vals) == 0:
            continue
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.step(vals, y, where="post", linewidth=2.0, label=label)
    ax.axvline(1.0, linestyle="--", linewidth=1.2, color="black")
    ax.set_xlabel("Speedup vs default")
    ax.set_ylabel("ECDF")
    ax.set_title("ECDF of speedup vs default")
    ax.legend(frameon=False)
    fig.tight_layout()
    _savefig(fig, outpath, save_pdf=save_pdf)
    plt.close(fig)
