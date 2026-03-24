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

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DPI = 320


def _paper_style():
    plt.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
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


def _load_summary(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "heldout_family" not in df.columns:
        raise ValueError(f"{path} must contain heldout_family column")
    return df.copy()


def _load_eval(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path).copy()


def _aggregate_size_summary_to_family(size_summary: pd.DataFrame) -> pd.DataFrame:
    metric_cols_mean = [
        "top_1_near_optimal_rate",
        "top_3_near_optimal_rate",
        "top_5_near_optimal_rate",
        "runtime_fail_rate_1p2",
        "fidelity_fail_rate_tol",
    ]
    metric_cols_median = [
        "median_runtime_ratio",
        "p95_runtime_ratio",
        "median_fidelity_gap",
        "p95_fidelity_gap",
    ]

    agg_dict = {}
    for c in metric_cols_mean:
        if c in size_summary.columns:
            agg_dict[c] = "mean"
    for c in metric_cols_median:
        if c in size_summary.columns:
            agg_dict[c] = "median"

    return size_summary.groupby("heldout_family", as_index=False).agg(agg_dict)


def _prepare_lofo_family_summary(lofo_summary: pd.DataFrame) -> pd.DataFrame:
    cols = ["heldout_family"]
    for c in [
        "top_1_near_optimal_rate",
        "top_3_near_optimal_rate",
        "top_5_near_optimal_rate",
        "median_runtime_ratio",
        "p95_runtime_ratio",
        "runtime_fail_rate_1p2",
        "median_fidelity_gap",
        "p95_fidelity_gap",
        "fidelity_fail_rate_tol",
    ]:
        if c in lofo_summary.columns:
            cols.append(c)
    return lofo_summary[cols].copy()


def _merge_regimes(lofo_fam: pd.DataFrame, size_fam: pd.DataFrame) -> pd.DataFrame:
    lofo = lofo_fam.rename(columns={c: f"{c}_lofo" for c in lofo_fam.columns if c != "heldout_family"})
    size = size_fam.rename(columns={c: f"{c}_size" for c in size_fam.columns if c != "heldout_family"})
    return lofo.merge(size, on="heldout_family", how="outer")


def _prepare_compare_default(path: str | Path, regime_name: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()

    if "family" not in df.columns and "heldout_family" in df.columns:
        df["family"] = df["heldout_family"]

    if "runtime_ratio_vs_default" not in df.columns:
        if {"eval_runtime", "runtime_default"}.issubset(df.columns):
            df["runtime_ratio_vs_default"] = df["eval_runtime"] / df["runtime_default"].clip(lower=1e-12)
        elif {"pred_runtime", "default_runtime"}.issubset(df.columns):
            df["runtime_ratio_vs_default"] = df["pred_runtime"] / df["default_runtime"].clip(lower=1e-12)
        else:
            raise ValueError(f"Cannot infer runtime_ratio_vs_default from {path}")

    if "speedup_vs_default" not in df.columns:
        df["speedup_vs_default"] = 1.0 / df["runtime_ratio_vs_default"].clip(lower=1e-12)

    df["regime"] = regime_name
    return df


def plot_combined_topk_by_family(merged: pd.DataFrame, outpath: Path):
    _paper_style()

    sort_key = merged["top_1_near_optimal_rate_lofo"].fillna(merged["top_1_near_optimal_rate_size"])
    work = merged.assign(_sort=sort_key).sort_values("_sort", ascending=False)

    fams = work["heldout_family"].astype(str).tolist()
    x = np.arange(len(fams))
    width = 0.18

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), sharex=True)

    metrics = [
        ("top_1_near_optimal_rate", "Top-1 near-optimal rate"),
        ("top_3_near_optimal_rate", "Top-3 near-optimal rate"),
        ("top_5_near_optimal_rate", "Top-5 near-optimal rate"),
    ]

    for ax, (m, title) in zip(axes, metrics):
        ax.bar(x - width / 2, work[f"{m}_lofo"], width=width, label="LOFO")
        ax.bar(x + width / 2, work[f"{m}_size"], width=width, label="Size-based")
        ax.set_title(title)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Rate")
        ax.set_xticks(x)
        ax.set_xticklabels(fams, rotation=40, ha="right")
        ax.grid(True, axis="y", alpha=0.22)

    axes[0].legend(frameon=False, ncols=2, loc="upper left")
    fig.suptitle("Near-optimal top-k performance across validation regimes", y=1.02)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_combined_runtime_metrics_by_family(merged: pd.DataFrame, outpath: Path):
    _paper_style()

    sort_key = merged["median_runtime_ratio_lofo"].fillna(merged["median_runtime_ratio_size"])
    work = merged.assign(_sort=sort_key).sort_values("_sort", ascending=True)

    fams = work["heldout_family"].astype(str).tolist()
    x = np.arange(len(fams))
    width = 0.18

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), sharex=True)

    metrics = [
        ("median_runtime_ratio", "Median runtime ratio", 1.0),
        ("p95_runtime_ratio", "P95 runtime ratio", 1.2),
        ("runtime_fail_rate_1p2", "Runtime fail rate > 1.2", 0.0),
    ]

    for ax, (m, title, ref) in zip(axes, metrics):
        ax.bar(x - width / 2, work[f"{m}_lofo"], width=width, label="LOFO")
        ax.bar(x + width / 2, work[f"{m}_size"], width=width, label="Size-based")
        ax.axhline(ref, linestyle="--", linewidth=1.1, color="black")
        ax.set_title(title)
        ax.set_ylabel(m)
        ax.set_xticks(x)
        ax.set_xticklabels(fams, rotation=40, ha="right")
        ax.grid(True, axis="y", alpha=0.22)

    axes[0].legend(frameon=False, ncols=2, loc="upper left")
    fig.suptitle("Runtime robustness across validation regimes", y=1.02)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_combined_fidelity_metrics_by_family(merged: pd.DataFrame, outpath: Path, fidelity_tol: float = 1e-3):
    _paper_style()

    sort_key = merged["median_fidelity_gap_lofo"].fillna(merged["median_fidelity_gap_size"])
    work = merged.assign(_sort=sort_key).sort_values("_sort", ascending=True)

    fams = work["heldout_family"].astype(str).tolist()
    x = np.arange(len(fams))
    width = 0.18

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), sharex=True)

    metrics = [
        ("median_fidelity_gap", "Median fidelity gap", fidelity_tol),
        ("p95_fidelity_gap", "P95 fidelity gap", fidelity_tol),
        ("fidelity_fail_rate_tol", "Fidelity fail rate > tol", 0.0),
    ]

    for ax, (m, title, ref) in zip(axes, metrics):
        ax.bar(x - width / 2, work[f"{m}_lofo"], width=width, label="LOFO")
        ax.bar(x + width / 2, work[f"{m}_size"], width=width, label="Size-based")
        ax.axhline(ref, linestyle="--", linewidth=1.1, color="black")
        ax.set_title(title)
        ax.set_ylabel(m)
        ax.set_xticks(x)
        ax.set_xticklabels(fams, rotation=40, ha="right")
        ax.grid(True, axis="y", alpha=0.22)

    axes[0].legend(frameon=False, ncols=2, loc="upper left")
    fig.suptitle("Fidelity robustness across validation regimes", y=1.02)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_runtime_ratio_violin_by_family_both(lofo_eval: pd.DataFrame, size_eval: pd.DataFrame, outpath: Path):
    _paper_style()

    def _prep(df: pd.DataFrame, regime: str):
        work = df.copy()
        if "heldout_family" not in work.columns and "family" in work.columns:
            work["heldout_family"] = work["family"]
        work = work.dropna(subset=["heldout_family", "runtime_ratio_to_feasible"])
        work["regime"] = regime
        return work[["heldout_family", "runtime_ratio_to_feasible", "regime"]]

    a = _prep(lofo_eval, "LOFO")
    b = _prep(size_eval, "Size-based")
    df = pd.concat([a, b], ignore_index=True)

    fam_order = (
        df.groupby("heldout_family")["runtime_ratio_to_feasible"]
        .median()
        .sort_values()
        .index
        .tolist()
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    for ax, regime in zip(axes, ["LOFO", "Size-based"]):
        sub = df[df["regime"] == regime]
        data = [sub.loc[sub["heldout_family"] == fam, "runtime_ratio_to_feasible"].values for fam in fam_order]

        parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_alpha(0.45)

        bp = ax.boxplot(data, widths=0.16, patch_artist=False)
        for median in bp["medians"]:
            median.set_linewidth(1.6)

        ax.axhline(1.0, linestyle="--", linewidth=1.1, color="black")
        ax.set_yscale("log")
        ax.set_ylabel("Runtime ratio")
        ax.set_title(regime)
        ax.set_xticks(np.arange(1, len(fam_order) + 1))
        ax.set_xticklabels(fam_order, rotation=40, ha="right")
        ax.grid(True, alpha=0.22)

    fig.suptitle("Runtime-ratio distribution by family across validation regimes", y=0.995)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_speedup_vs_default_by_family(lofo_pairs: pd.DataFrame, size_pairs: pd.DataFrame, outpath: Path):
    _paper_style()

    lofo_default = lofo_pairs.groupby("family", as_index=False)["speedup_vs_default"].median().rename(columns={"speedup_vs_default": "LOFO"})
    size_default = size_pairs.groupby("family", as_index=False)["speedup_vs_default"].median().rename(columns={"speedup_vs_default": "Size-based"})

    merged = lofo_default.merge(size_default, on="family", how="outer")
    merged["sort_key"] = np.nanmean(merged[["LOFO", "Size-based"]].values, axis=1)
    merged = merged.sort_values("sort_key", ascending=False)

    fams = merged["family"].astype(str).tolist()
    x = np.arange(len(fams))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.bar(x - width / 2, merged["LOFO"], width=width, label="LOFO")
    ax.bar(x + width / 2, merged["Size-based"], width=width, label="Size-based")
    ax.axhline(1.0, linestyle="--", linewidth=1.1, color="black")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(fams, rotation=40, ha="right")
    ax.set_ylabel("Median speedup vs default")
    ax.set_title("Median speedup vs default by family")
    ax.legend(frameon=False, ncols=2, loc="upper left")
    ax.grid(True, axis="y", alpha=0.22)

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_speedup_vs_best_by_family(lofo_eval: pd.DataFrame, size_eval: pd.DataFrame, outpath: Path):
    _paper_style()

    a = lofo_eval.groupby("heldout_family", as_index=False)["speedup_vs_best"].median().rename(
        columns={"heldout_family": "family", "speedup_vs_best": "LOFO"}
    )
    b = size_eval.groupby("heldout_family", as_index=False)["speedup_vs_best"].median().rename(
        columns={"heldout_family": "family", "speedup_vs_best": "Size-based"}
    )

    merged = a.merge(b, on="family", how="outer")
    merged["sort_key"] = np.nanmean(merged[["LOFO", "Size-based"]].values, axis=1)
    merged = merged.sort_values("sort_key", ascending=False)

    fams = merged["family"].astype(str).tolist()
    x = np.arange(len(fams))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.bar(x - width / 2, merged["LOFO"], width=width, label="LOFO")
    ax.bar(x + width / 2, merged["Size-based"], width=width, label="Size-based")
    ax.axhline(1.0, linestyle="--", linewidth=1.1, color="black")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(fams, rotation=40, ha="right")
    ax.set_ylabel("Median speedup vs best")
    ax.set_title("Median speedup vs best by family")
    ax.legend(frameon=False, ncols=2, loc="upper left")
    ax.grid(True, axis="y", alpha=0.22)

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_combined_speedup_ecdf(lofo_pairs: pd.DataFrame, size_pairs: pd.DataFrame, outpath: Path):
    _paper_style()

    fig, ax = plt.subplots(figsize=(9, 5.8))

    for df, label in [(lofo_pairs, "LOFO"), (size_pairs, "Size-based")]:
        vals = np.sort(df["speedup_vs_default"].dropna().astype(float).values)
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.step(vals, y, where="post", linewidth=2.0, label=label)

    ax.axvline(1.0, linestyle="--", linewidth=1.1, color="black")
    ax.set_xlabel("Speedup vs default")
    ax.set_ylabel("ECDF")
    ax.set_title("Predicted vs default: ECDF of speedup")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.22)

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lofo-summary", required=True)
    ap.add_argument("--size-summary", required=True)
    ap.add_argument("--lofo-eval", required=True)
    ap.add_argument("--size-eval", required=True)
    ap.add_argument("--lofo-default-pairs", required=True)
    ap.add_argument("--size-default-pairs", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--fidelity-tol", type=float, default=1e-3)
    args = ap.parse_args()

    outdir = _ensure_dir(args.outdir)

    lofo_summary = _load_summary(args.lofo_summary)
    size_summary = _load_summary(args.size_summary)
    lofo_eval = _load_eval(args.lofo_eval)
    size_eval = _load_eval(args.size_eval)

    lofo_family = _prepare_lofo_family_summary(lofo_summary)
    size_family = _aggregate_size_summary_to_family(size_summary)
    merged = _merge_regimes(lofo_family, size_family)

    lofo_pairs = _prepare_compare_default(args.lofo_default_pairs, "LOFO")
    size_pairs = _prepare_compare_default(args.size_default_pairs, "Size-based")

    lofo_eval["speedup_vs_best"] = 1.0 / lofo_eval["runtime_ratio_to_feasible"].clip(lower=1e-12)
    size_eval["speedup_vs_best"] = 1.0 / size_eval["runtime_ratio_to_feasible"].clip(lower=1e-12)

    plot_combined_topk_by_family(
        merged,
        outdir / "combined_topk_by_family.png",
    )
    plot_combined_runtime_metrics_by_family(
        merged,
        outdir / "combined_runtime_metrics_by_family.png",
    )
    plot_combined_fidelity_metrics_by_family(
        merged,
        outdir / "combined_fidelity_metrics_by_family.png",
        fidelity_tol=args.fidelity_tol,
    )
    plot_runtime_ratio_violin_by_family_both(
        lofo_eval,
        size_eval,
        outdir / "runtime_ratio_violin_by_family_both.png",
    )
    plot_speedup_vs_default_by_family(
        lofo_pairs,
        size_pairs,
        outdir / "speedup_vs_default_by_family.png",
    )
    plot_speedup_vs_best_by_family(
        lofo_eval,
        size_eval,
        outdir / "speedup_vs_best_by_family.png",
    )
    plot_combined_speedup_ecdf(
        lofo_pairs,
        size_pairs,
        outdir / "combined_speedup_vs_default_ecdf.png",
    )


if __name__ == "__main__":
    main()
