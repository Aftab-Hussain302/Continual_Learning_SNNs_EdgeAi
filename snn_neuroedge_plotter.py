#!/usr/bin/env python3
"""
SNN-NeuroEdge Results Comparison Plotter.

Generates paper-quality comparison figures:
  1. FedAvg vs Centralized baseline (per dataset)
  2. Cross-dataset comparison (Warehouse vs MNIST vs CIFAR)
  3. Combined summary dashboard

Usage:
    python3 snn_neuroedge_plotter.py \\
        --fedavg_warehouse Results/fl_Warehouse_2026-03-05_... \\
        --centralized_warehouse centralized_Warehouse_... \\
        --fedavg_mnist Results/fl_MNIST_5class_... \\
        --fedavg_cifar Results/fl_CIFAR10_5class_... \\
        --output_dir paper_figures
"""

import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paper-quality plot settings
PLOT_STYLE = {
    "figure.dpi": 200,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
}

PALETTE = {
    "Warehouse": "#4C78A8",
    "MNIST_5class": "#B07AA1",
    "CIFAR10_5class": "#E8A838",
    "centralized": "#E45756",
    "fedavg": "#4C78A8",
}


def load_fl_results(results_dir):
    """Load FL server results JSON."""
    rd = Path(results_dir)
    for name in ["fl_results_final.json",
                 "fl_edge_learning_results_160.json"]:
        p = rd / name
        if p.exists():
            with open(p, 'r') as f:
                return json.load(f)
    raise FileNotFoundError(f"No results JSON in {rd}")


def load_centralized_results(results_dir):
    """Load centralized baseline results JSON."""
    p = Path(results_dir) / "centralized_results.json"
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}")
    with open(p, 'r') as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────
# Plot 1: FedAvg vs Centralized convergence
# ──────────────────────────────────────────────────────────────────
def plot_fedavg_vs_centralized(fl_data, cent_data, dataset_name,
                                output_dir):
    """Side-by-side convergence: FedAvg vs Centralized baseline."""
    plt.rcParams.update(PLOT_STYLE)

    # Extract FL round accuracies
    fl_rounds = fl_data.get('round_metrics', [])
    fl_val = [rm.get('mean_accuracy', 0) for rm in fl_rounds]
    fl_train = [rm.get('mean_train_accuracy', 0) for rm in fl_rounds]
    fl_x = list(range(1, len(fl_val) + 1))

    # Extract centralized round accuracies
    cent_rounds = cent_data.get('round_metrics', [])
    cent_val = [rm.get('round_val_accuracy', 0) for rm in cent_rounds]
    cent_fixed = [rm.get('fixed_test_accuracy', 0) for rm in cent_rounds]
    cent_x = list(range(1, len(cent_val) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    # Left: FedAvg
    ax = axes[0]
    ax.plot(fl_x, fl_val, color=PALETTE['fedavg'], lw=2, label='FedAvg Val')
    ax.plot(fl_x, fl_train, color=PALETTE['fedavg'], lw=1, ls='--',
            alpha=0.6, label='FedAvg Train')
    fl_best = max(fl_val) if fl_val else 0
    ax.axhline(y=fl_best, color='green', ls=':', alpha=0.5,
               label=f'Best: {fl_best:.1f}%')
    ax.set_xlabel('Round')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'FedAvg (2 clients)')
    ax.legend(fontsize=9)

    # Right: Centralized
    ax = axes[1]
    ax.plot(cent_x, cent_fixed, color=PALETTE['centralized'], lw=2,
            label='Centralized Test')
    ax.plot(cent_x, cent_val, color=PALETTE['centralized'], lw=1,
            ls='--', alpha=0.6, label='Centralized Round Val')
    cent_best = max(cent_fixed) if cent_fixed else 0
    ax.axhline(y=cent_best, color='green', ls=':', alpha=0.5,
               label=f'Best: {cent_best:.1f}%')
    ax.set_xlabel('Round')
    ax.set_title(f'Centralized (1 node, all data)')
    ax.legend(fontsize=9)

    fig.suptitle(f'[{dataset_name}] FedAvg vs Centralized Baseline',
                 fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / f"fedavg_vs_centralized_{dataset_name}.png",
                bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: fedavg_vs_centralized_{dataset_name}.png")

    # Per-class comparison bar chart
    fl_pca = fl_data.get('final_per_class_accuracy', {})
    cent_pca = cent_data.get('final_per_class_accuracy', {})
    if fl_pca and cent_pca:
        classes = sorted(fl_pca.keys())
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(classes))
        w = 0.35
        v_fl = [fl_pca.get(c, 0) for c in classes]
        v_cent = [cent_pca.get(c, 0) for c in classes]
        bars1 = ax.bar(x - w/2, v_fl, w, label='FedAvg',
                       color=PALETTE['fedavg'], alpha=0.85)
        bars2 = ax.bar(x + w/2, v_cent, w, label='Centralized',
                       color=PALETTE['centralized'], alpha=0.85)
        ax.set_ylim(0, 110)
        ax.set_ylabel("Accuracy (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=25)
        ax.set_title(f'[{dataset_name}] Per-class: FedAvg vs Centralized')
        ax.legend()
        ax.grid(axis='y', ls='--', alpha=0.25)
        for bars in [bars1, bars2]:
            for b in bars:
                ax.annotate(f"{b.get_height():.0f}",
                            (b.get_x() + b.get_width()/2,
                             b.get_height()),
                            xytext=(0, 5), textcoords="offset points",
                            ha="center", fontsize=8, fontweight="bold")
        fig.tight_layout()
        fig.savefig(output_dir /
                    f"perclass_comparison_{dataset_name}.png")
        plt.close(fig)
        logger.info(f"Saved: perclass_comparison_{dataset_name}.png")


# ──────────────────────────────────────────────────────────────────
# Plot 2: Cross-dataset comparison
# ──────────────────────────────────────────────────────────────────
def plot_cross_dataset(datasets_data, output_dir):
    """Cross-dataset FL comparison: convergence overlay + summary."""
    plt.rcParams.update(PLOT_STYLE)

    # Convergence overlay
    fig, ax = plt.subplots(figsize=(10, 6))
    for ds_name, data in datasets_data.items():
        rounds = data.get('round_metrics', [])
        vals = [rm.get('mean_accuracy', 0) for rm in rounds]
        if vals:
            ax.plot(range(1, len(vals)+1), vals,
                    color=PALETTE.get(ds_name, '#333'),
                    lw=2, label=f'{ds_name} ({max(vals):.0f}% best)')
    ax.set(xlabel='Round', ylabel='Validation Accuracy (%)')
    ax.set_title('Cross-Dataset FL Convergence Comparison')
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / "cross_dataset_convergence.png")
    plt.close(fig)
    logger.info("Saved: cross_dataset_convergence.png")

    # Summary table figure
    summary_rows = []
    for ds_name, data in datasets_data.items():
        fl_sum = data.get('federated_learning_summary', {})
        cfg = data.get('experiment_config', {})
        rounds = data.get('round_metrics', [])
        vals = [rm.get('mean_accuracy', 0) for rm in rounds]
        summary_rows.append({
            'Dataset': ds_name,
            'Best Acc (%)': fl_sum.get('best_global_accuracy',
                                       max(vals) if vals else 0),
            'Final Acc (%)': fl_sum.get('final_global_accuracy',
                                        vals[-1] if vals else 0),
            'Rounds': len(rounds),
            'Avg Time (s)': fl_sum.get('avg_round_time', 0),
            'Total Time (s)': fl_sum.get('total_aggregation_time', 0),
            'Clients': fl_sum.get('total_clients', 2),
        })

    if summary_rows:
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.axis('off')
        cols = list(summary_rows[0].keys())
        cell_text = [[str(round(r[c], 2)) if isinstance(r[c], float)
                       else str(r[c]) for c in cols]
                      for r in summary_rows]
        table = ax.table(cellText=cell_text, colLabels=cols,
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor('#4C78A8')
                cell.set_text_props(color='white', fontweight='bold')
        ax.set_title('Cross-Dataset FL Summary', fontsize=14, pad=20)
        fig.tight_layout()
        fig.savefig(output_dir / "cross_dataset_summary_table.png",
                    bbox_inches='tight')
        plt.close(fig)
        logger.info("Saved: cross_dataset_summary_table.png")

    # Per-class heatmap
    all_classes = []
    all_datasets = []
    for ds_name, data in datasets_data.items():
        pca = data.get('final_per_class_accuracy', {})
        if pca:
            all_datasets.append(ds_name)
            for c in sorted(pca.keys()):
                if c not in all_classes:
                    all_classes.append(c)

    if all_datasets and all_classes:
        matrix = np.zeros((len(all_datasets), len(all_classes)))
        for i, ds in enumerate(all_datasets):
            pca = datasets_data[ds].get('final_per_class_accuracy', {})
            for j, c in enumerate(all_classes):
                matrix[i, j] = pca.get(c, 0)

        fig, ax = plt.subplots(figsize=(max(12, len(all_classes)*0.9), 4))
        im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=100,
                       aspect='auto')
        ax.set_xticks(range(len(all_classes)))
        ax.set_xticklabels(all_classes, rotation=45, ha='right',
                           fontsize=9)
        ax.set_yticks(range(len(all_datasets)))
        ax.set_yticklabels(all_datasets)
        for i in range(len(all_datasets)):
            for j in range(len(all_classes)):
                v = matrix[i, j]
                if v > 0:
                    ax.text(j, i, f"{v:.0f}%", ha='center', va='center',
                            fontsize=8,
                            color='white' if v < 50 else 'black')
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04,
                     label='Accuracy (%)')
        ax.set_title('Per-Class Accuracy Heatmap (All Datasets)')
        fig.tight_layout()
        fig.savefig(output_dir / "cross_dataset_heatmap.png",
                    bbox_inches='tight')
        plt.close(fig)
        logger.info("Saved: cross_dataset_heatmap.png")


# ──────────────────────────────────────────────────────────────────
# Plot 3: Comprehensive baseline comparison table for paper
# ──────────────────────────────────────────────────────────────────
def plot_baseline_summary(fl_data, cent_data, dataset_name, output_dir):
    """Generate a clean comparison summary for the paper."""
    plt.rcParams.update(PLOT_STYLE)

    fl_sum = fl_data.get('federated_learning_summary', {})
    fl_cfg = fl_data.get('experiment_config', {})

    rows = [
        ['Metric', 'FedAvg (Ours)', 'Centralized'],
        ['Best Accuracy (%)',
         f"{fl_sum.get('best_global_accuracy', 0):.1f}",
         f"{cent_data.get('best_fixed_test_accuracy', 0):.1f}"],
        ['Final Accuracy (%)',
         f"{fl_sum.get('final_global_accuracy', 0):.1f}",
         f"{cent_data.get('final_fixed_test_accuracy', 0):.1f}"],
        ['Rounds', str(fl_sum.get('total_rounds', 0)),
         str(cent_data.get('num_rounds', 0))],
        ['Avg Round Time (s)',
         f"{fl_sum.get('avg_round_time', 0):.3f}",
         f"{cent_data.get('avg_round_time', 0):.3f}"],
        ['Clients', str(fl_sum.get('total_clients', 2)), '1 (all data)'],
        ['Data per Client', '100/class', '200/class'],
        ['Privacy', 'Preserved', 'N/A (pooled)'],
        ['Communication', '~20 KB/round', 'N/A'],
    ]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    table = ax.table(
        cellText=[r[1:] for r in rows[1:]],
        colLabels=rows[0][1:],
        rowLabels=[r[0] for r in rows[1:]],
        loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.3, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#4C78A8')
            cell.set_text_props(color='white', fontweight='bold')
        if col == -1:
            cell.set_text_props(fontweight='bold')
    ax.set_title(f'[{dataset_name}] FedAvg vs Centralized Baseline',
                 fontsize=14, pad=20)
    fig.tight_layout()
    fig.savefig(output_dir / f"baseline_table_{dataset_name}.png",
                bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved: baseline_table_{dataset_name}.png")


# ──────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description='SNN-NeuroEdge Results Comparison Plotter')

    p.add_argument('--fedavg_warehouse', type=str, default=None,
                   help='FL results dir for Warehouse')
    p.add_argument('--centralized_warehouse', type=str, default=None,
                   help='Centralized results dir for Warehouse')
    p.add_argument('--fedavg_mnist', type=str, default=None,
                   help='FL results dir for MNIST-5')
    p.add_argument('--centralized_mnist', type=str, default=None,
                   help='Centralized results dir for MNIST-5')
    p.add_argument('--fedavg_cifar', type=str, default=None,
                   help='FL results dir for CIFAR10-5')
    p.add_argument('--centralized_cifar', type=str, default=None,
                   help='Centralized results dir for CIFAR10-5')
    p.add_argument('--output_dir', type=str, default='paper_figures',
                   help='Output directory for figures')

    a = p.parse_args()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cross_data = {}

    # Warehouse
    if a.fedavg_warehouse:
        fl_w = load_fl_results(a.fedavg_warehouse)
        cross_data['Warehouse'] = fl_w
        if a.centralized_warehouse:
            cent_w = load_centralized_results(a.centralized_warehouse)
            plot_fedavg_vs_centralized(fl_w, cent_w, 'Warehouse', out)
            plot_baseline_summary(fl_w, cent_w, 'Warehouse', out)

    # MNIST
    if a.fedavg_mnist:
        fl_m = load_fl_results(a.fedavg_mnist)
        cross_data['MNIST_5class'] = fl_m
        if a.centralized_mnist:
            cent_m = load_centralized_results(a.centralized_mnist)
            plot_fedavg_vs_centralized(fl_m, cent_m, 'MNIST_5class', out)
            plot_baseline_summary(fl_m, cent_m, 'MNIST_5class', out)

    # CIFAR
    if a.fedavg_cifar:
        fl_c = load_fl_results(a.fedavg_cifar)
        cross_data['CIFAR10_5class'] = fl_c
        if a.centralized_cifar:
            cent_c = load_centralized_results(a.centralized_cifar)
            plot_fedavg_vs_centralized(fl_c, cent_c, 'CIFAR10_5class', out)
            plot_baseline_summary(fl_c, cent_c, 'CIFAR10_5class', out)

    # Cross-dataset comparison
    if len(cross_data) >= 2:
        plot_cross_dataset(cross_data, out)

    logger.info(f"\nAll figures saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
