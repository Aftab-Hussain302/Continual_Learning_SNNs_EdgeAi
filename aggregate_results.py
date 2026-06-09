#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SNN-NeuroEdge — Publication-Quality Aggregation Script (v2)
===========================================================
Reads fl_results_final.json from selected result folders,
computes mean ± std, generates clean EPS+PNG figures for the
paper, and prints the LaTeX table.

Key features:
  - Explicit folder inclusion via --include_folders to avoid
    polluted/duplicate run_ids (e.g. MNIST bad runs).
  - All figures saved as both PNG (300 dpi) and EPS.
  - No plot titles — captions go in the paper, not the figure.
  - Minimal legend text, IEEE/Springer publication style.
  - LaTeX table printed to terminal and saved as .tex file.

Usage:
  # Auto mode (uses best runs per dataset, min 45 rounds)
  python3 aggregate_results_v2.py --results_dir Results --min_rounds 45

  # Explicit folder selection for MNIST (only the good runs)
  python3 aggregate_results_v2.py --results_dir Results \\
    --include_folders \\
      fl_MNIST_5class_run01_2026-03-19_18-44-05 \\
      fl_MNIST_5class_run02_2026-03-19_19-03-10 \\
      fl_MNIST_5class_run03_2026-03-19_19-22-00 \\
    --dataset MNIST_5class

  # Full recommended run (auto-detect all three datasets)
  python3 aggregate_results_v2.py --results_dir Results --min_rounds 45
"""

import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Style constants ────────────────────────────────────────────────────────────
# Color-blind safe, IEEE/Springer compatible
C = {
    "warehouse" : "#D55E00",   # vermillion
    "mnist"     : "#0072B2",   # blue
    "cifar"     : "#009E73",   # green
    "mean"      : "#CC79A7",   # pink
    "gray"      : "#999999",
    "black"     : "#222222",
}
DS_COLORS = {
    "Warehouse"     : C["warehouse"],
    "MNIST_5class"  : C["mnist"],
    "CIFAR10_5class": C["cifar"],
}
DS_LABELS = {
    "Warehouse"     : "Warehouse",
    "MNIST_5class"  : "MNIST-5",
    "CIFAR10_5class": "CIFAR-10-5",
}
CLASS_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]

DATASET_PREFIXES = {
    "Warehouse"     : "fl_Warehouse",
    "MNIST_5class"  : "fl_MNIST_5class",
    "CIFAR10_5class": "fl_CIFAR10_5class",
}


def set_pub_style():
    """IEEE/Springer single-column figure style."""
    plt.rcParams.update({
        "font.family"        : "serif",
        "font.serif"         : ["Times New Roman", "DejaVu Serif"],
        "font.size"          : 8,
        "axes.titlesize"     : 8,       # unused — no titles on figures
        "axes.labelsize"     : 8,
        "xtick.labelsize"    : 7,
        "ytick.labelsize"    : 7,
        "legend.fontsize"    : 7,
        "legend.handlelength": 1.5,
        "legend.borderpad"   : 0.4,
        "lines.linewidth"    : 1.4,
        "lines.markersize"   : 3.0,
        "axes.linewidth"     : 0.7,
        "axes.spines.top"    : False,
        "axes.spines.right"  : False,
        "axes.grid"          : True,
        "axes.grid.axis"     : "y",
        "grid.linewidth"     : 0.35,
        "grid.alpha"         : 0.35,
        "grid.linestyle"     : "--",
        "grid.color"         : "#bbbbbb",
        "xtick.direction"    : "out",
        "ytick.direction"    : "out",
        "xtick.major.width"  : 0.7,
        "ytick.major.width"  : 0.7,
        "legend.frameon"     : True,
        "legend.framealpha"  : 0.92,
        "legend.fancybox"    : False,
        "legend.edgecolor"   : "#cccccc",
        "figure.dpi"         : 300,
        "savefig.dpi"        : 300,
        "savefig.bbox"       : "tight",
        "savefig.pad_inches" : 0.04,
        "figure.facecolor"   : "white",
        "axes.facecolor"     : "white",
        "patch.linewidth"    : 0.5,
    })


def save_fig(fig, out_dir: Path, stem: str):
    """Save PNG + EPS. No title on the figure itself."""
    png = out_dir / f"{stem}.png"
    eps = out_dir / f"{stem}.eps"
    fig.savefig(str(png), dpi=300, bbox_inches="tight", facecolor="white")
    try:
        fig.savefig(str(eps), format="eps", bbox_inches="tight")
        logger.info("  %s  +  %s", png.name, eps.name)
    except Exception:
        logger.info("  %s  (EPS skipped)", png.name)
    plt.close(fig)


# ── Data loading ───────────────────────────────────────────────────────────────
def find_result_folders(results_dir: Path, dataset: str) -> list:
    prefix = DATASET_PREFIXES.get(dataset, f"fl_{dataset}")
    return sorted([
        d for d in results_dir.iterdir()
        if d.is_dir()
        and d.name.startswith(prefix)
        and (d / "fl_results_final.json").exists()
    ])


def load_run(folder: Path, min_rounds: int) -> dict | None:
    json_path = folder / "fl_results_final.json"
    try:
        with open(str(json_path)) as f:
            data = json.load(f)
    except Exception as e:
        logger.warning("Cannot load %s: %s", json_path, e)
        return None

    fl_sum       = data.get("fl_summary", {})
    total_rounds = fl_sum.get("total_rounds", 0)
    if total_rounds < min_rounds:
        logger.warning("  SKIP %s — %d rounds < min %d",
                       folder.name, total_rounds, min_rounds)
        return None

    best_fixed  = fl_sum.get("best_global_fixed_acc",
                  fl_sum.get("best_global_accuracy", 0.0))
    final_fixed = fl_sum.get("final_fixed_accuracy",
                  fl_sum.get("final_global_accuracy", 0.0))
    final_val   = fl_sum.get("final_val_accuracy", 0.0)
    avg_time    = fl_sum.get("avg_round_time", 0.0)

    per_class   = data.get("final_per_class_fixed",
                  data.get("final_per_class_accuracy", {}))

    round_metrics = data.get("round_metrics", [])
    fixed_curve   = [r.get("mean_fixed_accuracy",
                            r.get("mean_accuracy", 0.0))
                     for r in round_metrics]
    val_curve     = [r.get("mean_val_accuracy", 0.0)
                     for r in round_metrics]

    exp_cfg = data.get("experiment_config", {})
    run_id  = exp_cfg.get("run_id", None)
    if run_id is None:
        for part in folder.name.split("_"):
            if part.startswith("run") and part[3:].isdigit():
                run_id = int(part[3:])
                break
        if run_id is None:
            run_id = folder.name

    return {
        "run_id"               : run_id,
        "folder"               : folder.name,
        "total_rounds"         : total_rounds,
        "best_fixed_accuracy"  : float(best_fixed),
        "final_fixed_accuracy" : float(final_fixed),
        "final_val_accuracy"   : float(final_val),
        "avg_round_time"       : float(avg_time),
        "per_class_fixed"      : per_class,
        "fixed_curve"          : fixed_curve,
        "val_curve"            : val_curve,
    }


# ── Aggregation ────────────────────────────────────────────────────────────────
def aggregate(run_data: list, dataset: str) -> dict:
    n    = len(run_data)
    ddof = 1 if n >= 2 else 0

    def stats(values):
        arr = np.array(values, dtype=float)
        return {
            "mean"  : float(np.mean(arr)),
            "std"   : float(np.std(arr, ddof=ddof)),
            "min"   : float(np.min(arr)),
            "max"   : float(np.max(arr)),
            "values": list(arr),
            "n"     : n,
        }

    agg_metrics = {
        "best_fixed_accuracy"  : stats([r["best_fixed_accuracy"]  for r in run_data]),
        "final_fixed_accuracy" : stats([r["final_fixed_accuracy"] for r in run_data]),
        "final_val_accuracy"   : stats([r["final_val_accuracy"]   for r in run_data]),
        "avg_round_time"       : stats([r["avg_round_time"]       for r in run_data]),
    }

    all_classes   = list(run_data[0]["per_class_fixed"].keys())
    per_class_agg = {}
    for cls in all_classes:
        vals = [r["per_class_fixed"].get(cls, 0.0) for r in run_data]
        per_class_agg[cls] = stats(vals)

    # Convergence curves aligned to shortest run
    curves_f = [r["fixed_curve"] for r in run_data if r["fixed_curve"]]
    curves_v = [r["val_curve"]   for r in run_data if r["val_curve"]]
    min_len  = min(len(c) for c in curves_f) if curves_f else 0
    if min_len > 0:
        cf_arr      = np.array([c[:min_len] for c in curves_f])
        cv_arr      = np.array([c[:min_len] for c in curves_v])
        curve_fixed = {"mean": cf_arr.mean(0).tolist(),
                       "std" : cf_arr.std(0, ddof=ddof).tolist(),
                       "individual": [c[:min_len] for c in curves_f]}
        curve_val   = {"mean": cv_arr.mean(0).tolist(),
                       "std" : cv_arr.std(0, ddof=ddof).tolist()}
    else:
        curve_fixed = curve_val = {"mean": [], "std": [], "individual": []}

    return {
        "dataset"       : dataset,
        "n_runs"        : n,
        "run_ids"       : [r["run_id"] for r in run_data],
        "folders"       : [r["folder"] for r in run_data],
        "aggregated"    : agg_metrics,
        "per_class_agg" : per_class_agg,
        "curve_fixed"   : curve_fixed,
        "curve_val"     : curve_val,
        "rounds"        : list(range(1, min_len + 1)),
        "per_run"       : run_data,
        "timestamp"     : datetime.now().isoformat(),
    }


# ── Figure 1: Cross-dataset convergence (3 datasets, mean ± std shading) ───────
def plot_cross_convergence(summaries: dict, out_dir: Path):
    """
    Single panel: mean fixed-test accuracy ± std for all three datasets.
    Recommended for paper: replaces fig:xd_convergence.
    Width: single column (3.5 in). Clean, no title.
    """
    set_pub_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.97, bottom=0.14)

    for ds, s in summaries.items():
        rounds = np.array(s["rounds"])
        mean   = np.array(s["curve_fixed"]["mean"])
        std    = np.array(s["curve_fixed"]["std"])
        col    = DS_COLORS[ds]
        label  = DS_LABELS[ds]
        if len(rounds) == 0:
            continue
        ax.plot(rounds, mean, color=col, lw=1.4, label=label)
        ax.fill_between(rounds, mean - std, mean + std,
                        color=col, alpha=0.15)

    ax.set_xlabel("FL round")
    ax.set_ylabel("Fixed-test accuracy (%)")
    ax.set_xlim(1, max(len(s["rounds"]) for s in summaries.values()))
    ax.set_ylim(0, 105)
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.legend(loc="lower right", fontsize=7)
    save_fig(fig, out_dir, "fig_convergence_cross_dataset")


# ── Figure 2: Per-class bar chart — all datasets side by side ──────────────────
def plot_per_class_all(summaries: dict, out_dir: Path):
    """
    Three-panel bar chart (one per dataset), per-class mean ± std.
    Recommended for paper: replaces fig:xd_heatmap.
    Width: full column (7.2 in).
    """
    set_pub_style()
    datasets = list(summaries.keys())
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6),
                             sharey=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.95,
                        bottom=0.22, wspace=0.12)

    for ax, ds in zip(axes, datasets):
        s       = summaries[ds]
        pc      = s["per_class_agg"]
        classes = list(pc.keys())
        means   = [pc[c]["mean"] for c in classes]
        stds    = [pc[c]["std"]  for c in classes]
        x       = np.arange(len(classes))
        col     = DS_COLORS[ds]

        bars = ax.bar(x, means, width=0.6,
                      color=col, alpha=0.85,
                      edgecolor="white", linewidth=0.4)
        ax.errorbar(x, means, yerr=stds, fmt="none",
                    color="#333333", capsize=3, lw=0.9, capthick=0.9)

        # Short class label
        short = [c.replace("digit_", "d").replace("_", "\n")
                 for c in classes]
        ax.set_xticks(x)
        ax.set_xticklabels(short, fontsize=6.5)
        ax.set_ylim(0, 115)
        ax.set_xlabel(DS_LABELS[ds], fontsize=8, labelpad=3)
        if ax == axes[0]:
            ax.set_ylabel("Accuracy (%)")

        # Mean line
        overall = s["aggregated"]["best_fixed_accuracy"]["mean"]
        ax.axhline(overall, color=col, lw=0.9, ls="--", alpha=0.8)

        # Value labels on bars
        for bar, m, sd in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    m + sd + 2.0,
                    f"{m:.0f}",
                    ha="center", va="bottom", fontsize=5.5)

    axes[0].yaxis.set_minor_locator(MultipleLocator(5))
    save_fig(fig, out_dir, "fig_perclass_all_datasets")


# ── Figure 3: Overall accuracy comparison bar (3 datasets) ────────────────────
def plot_overall_bar(summaries: dict, out_dir: Path):
    """
    Grouped bar: best fixed-test and final fixed-test for each dataset.
    Mean ± std error bars. Good replacement for a summary figure.
    Width: single column (3.5 in).
    """
    set_pub_style()
    datasets = list(summaries.keys())
    labels   = [DS_LABELS[ds] for ds in datasets]
    best_m   = [summaries[ds]["aggregated"]["best_fixed_accuracy"]["mean"]
                for ds in datasets]
    best_s   = [summaries[ds]["aggregated"]["best_fixed_accuracy"]["std"]
                for ds in datasets]
    final_m  = [summaries[ds]["aggregated"]["final_fixed_accuracy"]["mean"]
                for ds in datasets]
    final_s  = [summaries[ds]["aggregated"]["final_fixed_accuracy"]["std"]
                for ds in datasets]

    x = np.arange(len(datasets))
    w = 0.32
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.97, bottom=0.14)

    b1 = ax.bar(x - w/2, best_m,  width=w,
                color=[DS_COLORS[ds] for ds in datasets],
                edgecolor="white", linewidth=0.4, alpha=0.9,
                label="Best")
    b2 = ax.bar(x + w/2, final_m, width=w,
                color=[DS_COLORS[ds] for ds in datasets],
                edgecolor="white", linewidth=0.4, alpha=0.55,
                label="Final")
    ax.errorbar(x - w/2, best_m,  yerr=best_s,
                fmt="none", color="#333333", capsize=3, lw=0.9, capthick=0.9)
    ax.errorbar(x + w/2, final_m, yerr=final_s,
                fmt="none", color="#333333", capsize=3, lw=0.9, capthick=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Fixed-test accuracy (%)")
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.legend(fontsize=7, loc="upper right")

    for bar, m, s in zip(b1, best_m, best_s):
        ax.text(bar.get_x() + bar.get_width() / 2,
                m + s + 1.5,
                f"{m:.1f}",
                ha="center", va="bottom", fontsize=6.5, fontweight="bold")

    save_fig(fig, out_dir, "fig_overall_accuracy_bar")


# ── Figure 4: Heatmap — per-class accuracy across datasets ────────────────────
def plot_heatmap(summaries: dict, out_dir: Path):
    """
    Heatmap: rows = classes, columns = datasets, values = mean accuracy.
    Compact and information-dense — recommended to replace fig:xd_heatmap left.
    Width: single column (3.5 in).
    """
    set_pub_style()
    datasets = list(summaries.keys())

    # Collect all unique classes per dataset (they differ)
    ds_classes = {ds: list(summaries[ds]["per_class_agg"].keys())
                  for ds in datasets}

    # Build matrix — pad missing with NaN
    all_classes_flat = []
    for ds in datasets:
        for c in ds_classes[ds]:
            label = f"{DS_LABELS[ds]}\n{c.replace('_',' ')}"
            all_classes_flat.append((ds, c, label))

    # Rows = class × dataset combinations, col = just one value
    # Better layout: rows=datasets, cols=classes (within-dataset)
    # Use one subplot per dataset row
    n_ds = len(datasets)
    fig, axes = plt.subplots(1, 1, figsize=(3.5, 2.8))
    fig.subplots_adjust(left=0.22, right=0.97, top=0.97, bottom=0.14)

    # Build 2D array: rows=datasets, cols=max 5 classes
    max_cls = max(len(ds_classes[ds]) for ds in datasets)
    matrix  = np.full((n_ds, max_cls), np.nan)
    col_labels = []
    for i, ds in enumerate(datasets):
        for j, cls in enumerate(ds_classes[ds]):
            matrix[i, j] = summaries[ds]["per_class_agg"][cls]["mean"]
        if len(ds_classes[ds]) > len(col_labels):
            col_labels = [c.replace("_", "\n").replace("digit\n", "d")
                          for c in ds_classes[ds]]

    im = axes.imshow(matrix, aspect="auto", cmap="RdYlGn",
                     vmin=20, vmax=100)
    axes.set_xticks(range(max_cls))
    axes.set_xticklabels(col_labels, fontsize=6.5)
    axes.set_yticks(range(n_ds))
    axes.set_yticklabels([DS_LABELS[ds] for ds in datasets], fontsize=7)

    # Annotate cells
    for i in range(n_ds):
        for j in range(max_cls):
            val = matrix[i, j]
            if not np.isnan(val):
                axes.text(j, i, f"{val:.0f}",
                          ha="center", va="center",
                          fontsize=6.5,
                          color="white" if val < 55 else "#222222",
                          fontweight="bold")

    cbar = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label("Accuracy (%)", fontsize=7)

    save_fig(fig, out_dir, "fig_heatmap_perclass")


# ── Figure 5: Per-run scatter/strip plot ──────────────────────────────────────
def plot_per_run_strip(summaries: dict, out_dir: Path):
    """
    Strip plot showing individual run values + mean ± std for each dataset.
    Shows reproducibility at a glance.
    Width: single column (3.5 in).
    """
    set_pub_style()
    datasets = list(summaries.keys())
    fig, ax  = plt.subplots(figsize=(3.5, 2.6))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.97, bottom=0.14)

    for i, ds in enumerate(datasets):
        s      = summaries[ds]
        vals   = s["aggregated"]["best_fixed_accuracy"]["values"]
        mean   = s["aggregated"]["best_fixed_accuracy"]["mean"]
        std    = s["aggregated"]["best_fixed_accuracy"]["std"]
        col    = DS_COLORS[ds]
        n      = len(vals)

        # Jitter x
        jitter = np.linspace(-0.12, 0.12, n)
        for j, v in zip(jitter, vals):
            ax.scatter(i + j, v, color=col, s=18, zorder=3,
                       alpha=0.75, linewidths=0)

        # Mean ± std bar
        ax.plot([i - 0.18, i + 0.18], [mean, mean],
                color=col, lw=2.0, zorder=4)
        ax.errorbar(i, mean, yerr=std, fmt="none",
                    color=col, capsize=5, lw=1.4, capthick=1.4, zorder=4)

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([DS_LABELS[ds] for ds in datasets], fontsize=7.5)
    ax.set_ylabel("Best fixed-test accuracy (%)")
    ax.set_ylim(20, 105)
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.axhline(20, color="none")   # padding

    save_fig(fig, out_dir, "fig_strip_per_run")


# ── LaTeX table ────────────────────────────────────────────────────────────────
def build_latex_table(summaries: dict, out_dir: Path) -> str:
    """
    Generate the cross-dataset LaTeX table with real mean ± std values.
    Replaces the placeholder values in the paper.
    """
    def fmt(s_dict: dict) -> str:
        m = s_dict["mean"]
        s = s_dict["std"]
        return f"{m:.1f} $\\pm$ {s:.1f}"

    def fmt_bold(s_dict: dict) -> str:
        m = s_dict["mean"]
        s = s_dict["std"]
        return f"\\textbf{{{m:.1f} $\\pm$ {s:.1f}}}"

    W  = summaries.get("Warehouse",      {}).get("aggregated", {})
    MN = summaries.get("MNIST_5class",   {}).get("aggregated", {})
    CF = summaries.get("CIFAR10_5class", {}).get("aggregated", {})

    nW  = summaries.get("Warehouse",      {}).get("n_runs", 0)
    nMN = summaries.get("MNIST_5class",   {}).get("n_runs", 0)
    nCF = summaries.get("CIFAR10_5class", {}).get("n_runs", 0)

    Wt  = W.get("avg_round_time",  {})
    MNt = MN.get("avg_round_time", {})
    CFt = CF.get("avg_round_time", {})

    lines = []
    lines.append(r"% ── Auto-generated by aggregate_results_v2.py ──────────────")
    lines.append(r"% Replace the table in your paper with this block.")
    lines.append(r"\begin{table}[!t]")
    lines.append(r"\caption{Cross-dataset FL evaluation: mean $\pm$ std over "
                 f"Warehouse ({nW}), MNIST-5 ({nMN}), CIFAR-10-5 ({nCF}) runs. "
                 r"Fixed held-out test set (30--40 samples/class). "
                 r"All runs: 2 clients, 50 rounds, warehouse-pretrained SNN backbone.}")
    lines.append(r"\label{tab:cross_dataset}")
    lines.append(r"\centering")
    lines.append(r"\setlength{\tabcolsep}{3.2pt}")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{2}{c}{\textbf{Warehouse}}"
                 r" & \multicolumn{2}{c}{\textbf{MNIST-5}}"
                 r" & \multicolumn{2}{c}{\textbf{CIFAR-10-5}} \\")
    lines.append(r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}")
    lines.append(r"\textbf{Metric} & \textbf{n=" + str(nW) + r"}"
                 r" & \textbf{n=" + str(nMN) + r"}"
                 r" & \textbf{n=" + str(nCF) + r"} \\")
    lines.append(r"\midrule")

    # Best accuracy row
    wb = W.get("best_fixed_accuracy",  {})
    mb = MN.get("best_fixed_accuracy", {})
    cb = CF.get("best_fixed_accuracy", {})
    lines.append(
        r"Best Acc.\ (\%)"
        f"\n  & {fmt_bold(wb)}"
        f"\n  & {fmt(mb)}"
        f"\n  & {fmt(cb)} \\\\")

    # Final accuracy row
    wf = W.get("final_fixed_accuracy",  {})
    mf = MN.get("final_fixed_accuracy", {})
    cf = CF.get("final_fixed_accuracy", {})
    lines.append(
        r"Final Acc.\ (\%)"
        f"\n  & {fmt(wf)}"
        f"\n  & {fmt(mf)}"
        f"\n  & {fmt(cf)} \\\\")

    # Round time row
    def fmt_time(s_dict):
        if not s_dict:
            return "---"
        m = s_dict.get("mean", 0)
        s = s_dict.get("std",  0)
        if m < 1.0:
            return f"{m*1000:.0f} $\\pm$ {s*1000:.0f}\\,ms"
        return f"{m:.2f} $\\pm$ {s:.2f}\\,s"

    lines.append(
        r"Avg.\ round time"
        f"\n  & {fmt_time(Wt)}"
        f"\n  & {fmt_time(MNt)}"
        f"\n  & {fmt_time(CFt)} \\\\")

    # Comm row (fixed)
    lines.append(
        r"Comm./round"
        r" & \multicolumn{2}{c}{$\sim$20\,KB}"
        r" & \multicolumn{2}{c}{$\sim$20\,KB}"
        r" & \multicolumn{2}{c}{$\sim$20\,KB} \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    tex_path = out_dir / "table_cross_dataset.tex"
    with open(str(tex_path), "w") as f:
        f.write(tex)
    logger.info("LaTeX table -> %s", tex_path.name)
    return tex


# ── Text summary ───────────────────────────────────────────────────────────────
def write_summary(summaries: dict, out_dir: Path):
    lines = []
    lines.append("=" * 68)
    lines.append("SNN-NeuroEdge  —  Aggregated Results")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 68)

    for ds, s in summaries.items():
        n   = s["n_runs"]
        agg = s["aggregated"]
        lines.append(f"\n── {DS_LABELS[ds]}  (n={n} runs) " + "─" * 30)
        for key, label in [
            ("best_fixed_accuracy",  "Best  fixed-test"),
            ("final_fixed_accuracy", "Final fixed-test"),
            ("final_val_accuracy",   "Final round-val "),
        ]:
            st = agg.get(key, {})
            if not st:
                continue
            vals_str = ", ".join(f"{v:.1f}" for v in st["values"])
            lines.append(
                f"  {label}: {st['mean']:.2f}% ± {st['std']:.2f}%"
                f"  (min {st['min']:.1f}  max {st['max']:.1f})")
            lines.append(f"    runs: [{vals_str}]")

        lines.append(f"  Per-class (final round):")
        for cls, st in s["per_class_agg"].items():
            lines.append(
                f"    {cls.ljust(22)}: "
                f"{st['mean']:.1f}% ± {st['std']:.1f}%")

    lines.append("\n" + "=" * 68)
    lines.append("CROSS-DATASET SUMMARY  (best fixed-test)")
    lines.append(f"  {'Dataset':<22}  {'Mean':>7}  {'±Std':>6}  "
                 f"{'Min':>6}  {'Max':>6}  n")
    lines.append("  " + "-" * 56)
    for ds, s in summaries.items():
        st = s["aggregated"]["best_fixed_accuracy"]
        lines.append(
            f"  {DS_LABELS[ds]:<22}  {st['mean']:>6.1f}%  "
            f"{st['std']:>5.1f}%  {st['min']:>5.1f}%  "
            f"{st['max']:>5.1f}%  {st['n']}")
    lines.append("=" * 68)

    txt = "\n".join(lines)
    print("\n" + txt)
    p = out_dir / "aggregated_summary_all.txt"
    with open(str(p), "w") as f:
        f.write(txt)
    logger.info("Summary -> %s", p.name)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="SNN-NeuroEdge publication aggregation v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect all three datasets (min 45 rounds)
  python3 aggregate_results_v2.py --results_dir Results --min_rounds 45

  # Use only the three good MNIST runs (exclude bad ones)
  python3 aggregate_results_v2.py --results_dir Results \\
    --dataset MNIST_5class \\
    --include_folders \\
      fl_MNIST_5class_run01_2026-03-19_18-44-05 \\
      fl_MNIST_5class_run02_2026-03-19_19-03-10 \\
      fl_MNIST_5class_run03_2026-03-19_19-22-00

  # Full recommended command with MNIST folder selection
  python3 aggregate_results_v2.py --results_dir Results \\
    --mnist_folders \\
      fl_MNIST_5class_run01_2026-03-19_18-44-05 \\
      fl_MNIST_5class_run02_2026-03-19_19-03-10 \\
      fl_MNIST_5class_run03_2026-03-19_19-22-00
        """
    )
    p.add_argument("--results_dir", default="Results")
    p.add_argument("--min_rounds",  type=int, default=45)
    p.add_argument("--dataset",     default=None,
                   choices=list(DATASET_PREFIXES.keys()))
    p.add_argument("--include_folders", nargs="*", default=None,
                   help="Explicit folder names to include (for selected dataset)")
    p.add_argument("--mnist_folders", nargs="*", default=None,
                   help="Explicit MNIST folder names (overrides auto-detect)")
    a = p.parse_args()

    results_dir = Path(a.results_dir)
    out_dir     = results_dir   # write outputs to Results/

    datasets = ([a.dataset] if a.dataset
                else list(DATASET_PREFIXES.keys()))

    summaries = {}

    for ds in datasets:
        print(f"\n{'='*68}")
        print(f"  {ds}")
        print(f"{'='*68}")

        # Determine folders to load
        if ds == "MNIST_5class" and a.mnist_folders:
            folders = [results_dir / fn for fn in a.mnist_folders
                       if (results_dir / fn).exists()]
            logger.info("  Using %d explicit MNIST folders", len(folders))
        elif a.include_folders and a.dataset == ds:
            folders = [results_dir / fn for fn in a.include_folders
                       if (results_dir / fn).exists()]
            logger.info("  Using %d explicit folders", len(folders))
        else:
            folders = find_result_folders(results_dir, ds)

        if not folders:
            logger.warning("  No folders found for %s", ds)
            continue

        run_data = []
        for fld in folders:
            r = load_run(fld, a.min_rounds)
            if r is not None:
                run_data.append(r)
                logger.info("  Loaded %-45s  rounds=%d  best=%.1f%%",
                            fld.name, r["total_rounds"],
                            r["best_fixed_accuracy"])

        if not run_data:
            logger.warning("  No valid runs for %s", ds)
            continue

        s = aggregate(run_data, ds)

        # Save per-dataset JSON
        json_path = out_dir / f"aggregated_v2_{ds}.json"
        with open(str(json_path), "w") as f:
            json.dump(s, f, indent=2, default=str)
        logger.info("  JSON -> %s", json_path.name)

        summaries[ds] = s

    if not summaries:
        logger.error("No data loaded. Check --results_dir and folder names.")
        return

    # ── Generate all publication figures ──────────────────────────────────────
    set_pub_style()
    logger.info("\nGenerating publication figures ...")

    if len(summaries) >= 2:
        plot_cross_convergence(summaries, out_dir)
        plot_overall_bar(summaries, out_dir)
        plot_heatmap(summaries, out_dir)
        plot_per_run_strip(summaries, out_dir)

    plot_per_class_all(summaries, out_dir)

    # ── LaTeX table ───────────────────────────────────────────────────────────
    if len(summaries) >= 2:
        tex = build_latex_table(summaries, out_dir)
        print("\n" + "─" * 68)
        print("LaTeX table (copy into your paper):")
        print("─" * 68)
        print(tex)

    # ── Text summary ──────────────────────────────────────────────────────────
    write_summary(summaries, out_dir)

    print(f"\n{'='*68}")
    print(f"All outputs written to: {out_dir.resolve()}")
    print(f"{'='*68}")


if __name__ == "__main__":
    main()