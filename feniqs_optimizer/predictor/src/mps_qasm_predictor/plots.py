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
from matplotlib.colors import Normalize


PAPER_DPI = 260

HEATMAP_METRICS = [
    "top_1_near_optimal_rate",
    "top_3_near_optimal_rate",
    "top_5_near_optimal_rate",
    "median_runtime_ratio",
    "median_fidelity_gap",
    "p95_runtime_ratio",
    "p95_fidelity_gap",
    "runtime_fail_rate_1p2",
    "fidelity_fail_rate_tol",
]

HIGHER_IS_BETTER = {
    "top_1_near_optimal_rate": True,
    "top_3_near_optimal_rate": True,
    "top_5_near_optimal_rate": True,
    "median_runtime_ratio": False,
    "median_fidelity_gap": False,
    "p95_runtime_ratio": False,
    "p95_fidelity_gap": False,
    "runtime_fail_rate_1p2": False,
    "fidelity_fail_rate_tol": False,
}


def _paper_style():
    plt.rcParams.update(
        {
            "figure.dpi": PAPER_DPI,
            "savefig.dpi": PAPER_DPI,
            "font.size": 12,
            "axes.titlesize": 18,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _format_metric_value(metric: str, value: float) -> str:
    if pd.isna(value):
        return ""
    if "rate" in metric:
        return f"{value:.3f}"
    if "fidelity" in metric:
        return f"{value:.3g}" if abs(value) < 1e-2 else f"{value:.3f}"
    return f"{value:.3f}"


def _normalized_heatmap_matrix(df: pd.DataFrame, metrics: list[str]) -> np.ndarray:
    arr = np.full((df.shape[0], len(metrics)), np.nan, dtype=float)
    for j, m in enumerate(metrics):
        col = df[m].astype(float)
        vals = col.values.astype(float)

        mask = np.isfinite(vals)
        if mask.sum() == 0:
            continue

        v = vals[mask]
        vmin = np.nanmin(v)
        vmax = np.nanmax(v)

        if np.isclose(vmin, vmax):
            normed = np.ones_like(v) * 0.5
        else:
            normed = (v - vmin) / (vmax - vmin)

        if not HIGHER_IS_BETTER.get(m, False):
            normed = 1.0 - normed

        arr[mask, j] = normed
    return arr


def plot_summary_heatmap(
    summary_df: pd.DataFrame,
    outpath: str | Path,
    title: str,
    family_col: str = "heldout_family",
    metrics: Iterable[str] = HEATMAP_METRICS,
):
    _paper_style()
    metrics = [m for m in metrics if m in summary_df.columns]
    if family_col not in summary_df.columns:
        raise ValueError(f"Missing column '{family_col}' in summary_df")

    work = summary_df[[family_col] + metrics].copy()
    work = work.sort_values(
        by=["top_1_near_optimal_rate", "median_runtime_ratio"],
        ascending=[False, True],
    ).reset_index(drop=True)

    mat = _normalized_heatmap_matrix(work, metrics)

    fig_w = max(10.5, 1.15 * len(metrics) + 3.0)
    fig_h = max(5.5, 0.45 * len(work) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = plt.cm.viridis
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)

    ax.set_title(title, pad=12)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Held-out family")

    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(work)))
    ax.set_yticklabels(work[family_col].astype(str).tolist())

    for i in range(work.shape[0]):
        for j, m in enumerate(metrics):
            txt = _format_metric_value(m, work.iloc[i][m])
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Column-wise normalized score")

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
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


def plot_runtime_distribution_both_regimes(
    lofo_eval: pd.DataFrame,
    size_eval: pd.DataFrame,
    outpath: str | Path,
    runtime_col: str = "runtime_ratio",
    family_col: str = "heldout_family",
):
    _paper_style()

    if runtime_col not in lofo_eval.columns or runtime_col not in size_eval.columns:
        raise ValueError(f"Expected '{runtime_col}' in both eval dataframes")

    fam_order = _ordered_families_from_union(lofo_eval, size_eval, family_col=family_col)

    a = _make_violin_data(lofo_eval, runtime_col, "LOFO", family_col=family_col)
    b = _make_violin_data(size_eval, runtime_col, "Size-based", family_col=family_col)
    df = pd.concat([a, b], ignore_index=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

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
        ax.grid(True, alpha=0.22)

    fig.suptitle("Runtime-ratio distribution across validation regimes", y=0.995)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
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


def plot_runtime_vs_default_distribution_both(
    lofo_pairs: pd.DataFrame,
    size_pairs: pd.DataFrame,
    outpath: str | Path,
):
    _paper_style()

    lofo = _prepare_compare_default_df(lofo_pairs, "LOFO")
    size = _prepare_compare_default_df(size_pairs, "Size-based")
    df = pd.concat([lofo, size], ignore_index=True)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    for ax, regime in zip(axes, ["LOFO", "Size-based"]):
        sub = df[df["regime"] == regime]["runtime_ratio_vs_default"].dropna().values
        ax.hist(sub, bins=28, alpha=0.9)
        ax.axvline(1.0, linestyle="--", linewidth=1.2, color="black")
        ax.set_xscale("log")
        ax.set_ylabel("Count")
        ax.set_title(regime)
        ax.grid(True, alpha=0.22)

    axes[-1].set_xlabel("Predicted/default runtime ratio")
    fig.suptitle("Predicted vs default: runtime-ratio distribution", y=0.995)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_median_speedup_vs_default_both(
    lofo_pairs: pd.DataFrame,
    size_pairs: pd.DataFrame,
    outpath: str | Path,
):
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

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - width / 2, merged["LOFO"], width=width, label="LOFO")
    ax.bar(x + width / 2, merged["Size-based"], width=width, label="Size-based")
    ax.axhline(1.0, linestyle="--", linewidth=1.2, color="black")
    ax.set_xticks(x)
    ax.set_xticklabels(merged["family"], rotation=35, ha="right")
    ax.set_ylabel("Median speedup vs default")
    ax.set_title("Median speedup vs default across validation regimes")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.22)

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_ecdf_speedup_vs_default_both(
    lofo_pairs: pd.DataFrame,
    size_pairs: pd.DataFrame,
    outpath: str | Path,
):
    _paper_style()

    lofo = _prepare_compare_default_df(lofo_pairs, "LOFO")
    size = _prepare_compare_default_df(size_pairs, "Size-based")

    fig, ax = plt.subplots(figsize=(10, 6))

    for df, label in [(lofo, "LOFO"), (size, "Size-based")]:
        vals = np.sort(df["speedup_vs_default"].dropna().values)
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.step(vals, y, where="post", linewidth=2.0, label=label)

    ax.axvline(1.0, linestyle="--", linewidth=1.2, color="black")
    ax.set_xlabel("Speedup vs default")
    ax.set_ylabel("ECDF")
    ax.set_title("ECDF of speedup vs default")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.22)

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
