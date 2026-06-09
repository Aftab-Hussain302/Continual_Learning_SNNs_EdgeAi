#!/usr/bin/env python3
"""
SNN-NeuroEdge FL Server v2
Warehouse / MNIST-5class / CIFAR10-5class

Data budget (50-round design):
  200 samples/class total, 2 clients -> 100/class/client
  30/class fixed held-out test pool (never in training)
  70/class training pool -> 35 unique rounds x 2/round
  Rounds 36-50 recycle training pool with epoch-seeded shuffle

Usage:
  python3 snn_neuroedge_server_v2.py --dataset Warehouse --max_rounds 50
  python3 snn_neuroedge_server_v2.py --dataset MNIST_5class --max_rounds 50
  python3 snn_neuroedge_server_v2.py --dataset CIFAR10_5class --max_rounds 50
"""

import json
import time
import logging
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Parameters

import akida
from akida import FullyConnected, AkidaUnsupervised, devices

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ==============================================================
# Dataset configurations
# ==============================================================
DATASET_CONFIGS = {
    "Warehouse": {
        "classes": [
            "boxes_stacked", "empty_aisle", "forklift_present",
            "mixed_objects", "unclear_scene"
        ],
        "reserved_json": "reserved_paths.json",
    },
    "MNIST_5class": {
        "classes": [
            "digit_0", "digit_1", "digit_2", "digit_3", "digit_4"
        ],
        "reserved_json": "/home/sai/datasets/MNIST_5class/reserved_paths.json",
    },
    "CIFAR10_5class": {
        "classes": [
            "airplane", "automobile", "bird", "cat", "deer"
        ],
        "reserved_json": "/home/sai/datasets/CIFAR10_5class/reserved_paths.json",
    },
}

# ==============================================================
# Shared constants  (must match client exactly)
# ==============================================================
IMG_SIZE                 = (160, 160)
SAMPLES_PER_CLASS_TOTAL  = 200
N_CLIENTS                = 2
SAMPLES_PER_CLASS_CLIENT = SAMPLES_PER_CLASS_TOTAL // N_CLIENTS   # 100
FIXED_TEST_PER_CLASS     = 30
TRAIN_POOL_PER_CLASS     = SAMPLES_PER_CLASS_CLIENT - FIXED_TEST_PER_CLASS  # 70
SHOTS_PER_CLASS          = 1
VAL_PER_CLASS            = 1
SAMPLES_PER_ROUND        = SHOTS_PER_CLASS + VAL_PER_CLASS         # 2
MAX_UNIQUE_ROUNDS        = TRAIN_POOL_PER_CLASS // SAMPLES_PER_ROUND  # 35
NEURONS_PER_CLASS        = 100
NUM_WEIGHTS_FRACTION     = 0.50

FBZ_CANDIDATES = [
    "warehouse_edge_backbone_full.fbz",
    "warehouse_edge_backbone_full_v1.fbz",
    "warehouse_edge_backbone_160.fbz",
    "warehouse_edge_backbone.fbz",
]

# Wong (2011) colorblind-safe palette
PALETTE = {
    "train"  : "#0072B2",
    "val"    : "#E69F00",
    "snn"    : "#009E73",
    "accent" : "#D55E00",
    "gray"   : "#999999",
}
CLASS_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]


# ==============================================================
# Publication style
# ==============================================================
def set_pub_style():
    plt.rcParams.update({
        "font.family"        : "serif",
        "font.serif"         : ["Times New Roman", "DejaVu Serif"],
        "font.size"          : 9,
        "axes.titlesize"     : 10,
        "axes.labelsize"     : 9,
        "xtick.labelsize"    : 8,
        "ytick.labelsize"    : 8,
        "legend.fontsize"    : 8,
        "lines.linewidth"    : 1.5,
        "axes.linewidth"     : 0.8,
        "axes.spines.top"    : False,
        "axes.spines.right"  : False,
        "axes.grid"          : True,
        "axes.grid.axis"     : "y",
        "grid.linewidth"     : 0.4,
        "grid.alpha"         : 0.4,
        "grid.linestyle"     : "--",
        "xtick.direction"    : "out",
        "ytick.direction"    : "out",
        "legend.frameon"     : True,
        "legend.framealpha"  : 0.9,
        "legend.fancybox"    : False,
        "figure.dpi"         : 300,
        "savefig.dpi"        : 300,
        "savefig.bbox"       : "tight",
        "savefig.pad_inches" : 0.05,
        "figure.facecolor"   : "white",
        "axes.facecolor"     : "white",
    })


def save_fig(fig, out_dir, name):
    png = out_dir / (name + ".png")
    eps = out_dir / (name + ".eps")
    fig.savefig(str(png), dpi=300, bbox_inches="tight", facecolor="white")
    try:
        fig.savefig(str(eps), format="eps", bbox_inches="tight")
    except Exception:
        pass
    plt.close(fig)
    logger.info("  Saved: " + png.name)


# ==============================================================
# Server class
# ==============================================================
class NeuroEdgeServer:

    def __init__(self, dataset_name="Warehouse",
                 results_dir="Results", max_rounds=50):
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError("Unknown dataset: " + dataset_name +
                             ". Choose from: " +
                             str(list(DATASET_CONFIGS.keys())))

        self.dataset_name = dataset_name
        self.CLASSES      = DATASET_CONFIGS[dataset_name]["classes"]
        self.max_rounds   = max_rounds

        base  = Path(results_dir)
        base.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.results_dir = base / ("fl_" + dataset_name + "_" + stamp)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.global_model  = None
        self.fbz_path      = None
        self.round_metrics = []
        self.best_accuracy = 0.0

        self.round_times             = []
        self.global_val_accs         = []
        self.global_fixed_accs       = []
        self.global_train_accs       = []
        self.per_class_fixed_hist    = {c: [] for c in self.CLASSES}
        self.client_val_history      = defaultdict(list)
        self.client_fixed_history    = defaultdict(list)
        self.client_train_times      = defaultdict(list)
        self.active_clients          = set()

        logger.info("=" * 60)
        logger.info("SNN-NeuroEdge FL Server  [" + dataset_name + "]")
        logger.info("=" * 60)
        logger.info("  max_rounds        : " + str(max_rounds))
        logger.info("  samples/class/cli : " + str(SAMPLES_PER_CLASS_CLIENT))
        logger.info("  fixed test/class  : " + str(FIXED_TEST_PER_CLASS))
        logger.info("  train pool/class  : " + str(TRAIN_POOL_PER_CLASS))
        logger.info("  shots/class/round : " + str(SHOTS_PER_CLASS))
        logger.info("  val/class/round   : " + str(VAL_PER_CLASS))
        logger.info("  unique rounds     : " + str(MAX_UNIQUE_ROUNDS))
        logger.info("  Results -> " + str(self.results_dir))

    def _find_fbz(self):
        for c in FBZ_CANDIDATES:
            if Path(c).exists():
                return c
        raise RuntimeError(
            "No FBZ backbone found. Searched: " + str(FBZ_CANDIDATES))

    def _infer_fan_in(self, model_ak):
        try:
            prev = model_ak.layers[-2]
            if hasattr(prev, "output_dims") and prev.output_dims is not None:
                h, w, c = prev.output_dims
                return int(h * w * c)
        except Exception:
            pass
        return 256

    def create_global_model(self):
        fbz_path = self._find_fbz()
        model_ak = akida.Model(fbz_path)
        devs = devices()
        if devs:
            model_ak.map(devs[0])
            logger.info("Mapped to Akida device: " + str(devs[0]))
        else:
            logger.info("Software simulation (no Akida HW on server)")

        model_ak.pop_layer()
        n_cls         = len(self.CLASSES)
        total_neurons = n_cls * NEURONS_PER_CLASS
        model_ak.add(FullyConnected(
            name="akida_edge_layer",
            units=total_neurons,
            activation=False
        ))
        fan_in = self._infer_fan_in(model_ak)
        nw     = max(32, min(fan_in,
                             int(round(NUM_WEIGHTS_FRACTION * fan_in))))
        model_ak.compile(optimizer=AkidaUnsupervised(
            num_weights=nw, num_classes=n_cls, learning_competition=0.1
        ))
        logger.info(
            "Global model: " + str(n_cls) + " classes x " +
            str(NEURONS_PER_CLASS) + " neurons | fan_in=" +
            str(fan_in) + " nw=" + str(nw)
        )
        self.fbz_path    = fbz_path
        self._fan_in     = fan_in
        self._nw         = nw
        self._n_cls      = n_cls
        return model_ak

    def get_initial_parameters(self):
        """
        Return the actual untrained edge layer weights from the model
        so clients receive correctly shaped parameters from round 1.
        Never send random noise — that breaks Akida binary input requirement.
        """
        if self.global_model is None:
            self.global_model = self.create_global_model()
        try:
            w = self.global_model.layers[-1].get_weights()
            if w:
                logger.info("Initial parameters: " +
                            str(len(w)) + " arrays, shapes: " +
                            str([x.shape for x in w]))
                return fl.common.ndarrays_to_parameters(
                    [np.array(x, dtype=np.float32) for x in w])
        except Exception as e:
            logger.error("get_initial_parameters failed: " + str(e))
        # Should never reach here — raise so we catch misconfiguration early
        raise RuntimeError(
            "Could not extract initial weights from Akida model. "
            "Check FBZ backbone and model compilation.")

    def save_global_model(self, parameters, round_num):
        try:
            if self.global_model is None:
                self.global_model = self.create_global_model()
            pa       = fl.common.parameters_to_ndarrays(parameters)
            fl_layer = self.global_model.layers[-1]
            cur_w    = fl_layer.get_weights()
            # Only set if shapes match
            if len(pa) == len(cur_w) and all(
                    p.shape == w.shape for p, w in zip(pa, cur_w)):
                fl_layer.set_weights(pa)
            else:
                logger.warning(
                    "save_global_model: shape mismatch, skipping set_weights")
            p = self.results_dir / ("global_round_" + str(round_num) + ".fbz")
            self.global_model.save(str(p))
            logger.info("Saved global model -> " + p.name)
        except Exception as e:
            logger.error("save_global_model: " + str(e))


# ==============================================================
# Strategy
# ==============================================================
class NeuroEdgeStrategy(FedAvg):

    def __init__(self, server: NeuroEdgeServer):
        super().__init__(
            min_available_clients=1,
            min_fit_clients=1,
            min_evaluate_clients=1,
        )
        self.server = server

    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        return self.server.get_initial_parameters()

    def aggregate_fit(self, server_round, results, failures):
        t0 = time.time()
        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated is None:
            return aggregated
        parameters_agg, metrics_agg = aggregated

        if server_round % 10 == 0 or server_round == self.server.max_rounds:
            self.server.save_global_model(parameters_agg, server_round)

        rd = self._build_round_data(server_round, results, t0)
        self.server.round_metrics.append(rd)
        self._update_tracking(rd)
        self._save_round_json(server_round, rd)
        self._print_round(server_round, rd)

        return parameters_agg, metrics_agg

    def _safe_mean(self, lst):
        return float(np.mean(lst)) if lst else 0.0

    def _build_round_data(self, rnd, results, t0):
        accs, fixed_accs, taccs, ttimes = [], [], [], []
        pca_val   = {c: [] for c in self.server.CLASSES}
        pca_fixed = {c: [] for c in self.server.CLASSES}
        client_results = []

        for proxy, fit_res in results:
            m   = fit_res.metrics
            cid = m.get("client_id",
                        str(getattr(proxy, "cid", "unknown")))
            self.server.active_clients.add(cid)

            va = float(m.get("mean_accuracy",       0.0))
            fa = float(m.get("fixed_test_accuracy", 0.0))
            ta = float(m.get("train_accuracy",      0.0))
            tt = float(m.get("training_time",       0.0))
            accs.append(va); fixed_accs.append(fa)
            taccs.append(ta); ttimes.append(tt)

            self.server.client_val_history[cid].append(
                {"round": rnd, "accuracy": va})
            self.server.client_fixed_history[cid].append(
                {"round": rnd, "fixed_accuracy": fa})
            self.server.client_train_times[cid].append(
                {"round": rnd, "training_time": tt})

            for c in self.server.CLASSES:
                v = m.get("val_"   + c + "_acc")
                f = m.get("fixed_" + c + "_acc")
                if v is not None: pca_val[c].append(float(v))
                if f is not None: pca_fixed[c].append(float(f))

            client_results.append({
                "client_id"          : cid,
                "num_examples"       : fit_res.num_examples,
                "val_accuracy"       : va,
                "fixed_test_accuracy": fa,
                "train_accuracy"     : ta,
                "training_time"      : tt,
                "data_epoch"         : int(m.get("data_epoch", 1)),
            })

        per_class_val   = {c: self._safe_mean(pca_val[c])
                           for c in self.server.CLASSES}
        per_class_fixed = {c: self._safe_mean(pca_fixed[c])
                           for c in self.server.CLASSES}
        mean_val   = self._safe_mean(accs)
        mean_fixed = self._safe_mean(fixed_accs)
        mean_train = self._safe_mean(taccs)

        if mean_fixed > self.server.best_accuracy:
            self.server.best_accuracy = mean_fixed

        return {
            "round"               : rnd,
            "timestamp"           : datetime.now().isoformat(),
            "num_participants"    : len(results),
            "round_time"          : time.time() - t0,
            "mean_val_accuracy"   : mean_val,
            "mean_fixed_accuracy" : mean_fixed,
            "mean_train_accuracy" : mean_train,
            "mean_training_time"  : self._safe_mean(ttimes),
            "per_class_val"       : per_class_val,
            "per_class_fixed"     : per_class_fixed,
            "client_results"      : client_results,
            "data_recycled"       : rnd > MAX_UNIQUE_ROUNDS,
        }

    def _update_tracking(self, rd):
        self.server.round_times.append(rd["round_time"])
        self.server.global_val_accs.append(rd["mean_val_accuracy"])
        self.server.global_fixed_accs.append(rd["mean_fixed_accuracy"])
        self.server.global_train_accs.append(rd["mean_train_accuracy"])
        for c in self.server.CLASSES:
            self.server.per_class_fixed_hist[c].append(
                rd["per_class_fixed"].get(c, 0.0))

    def _save_round_json(self, rnd, rd):
        try:
            p = self.server.results_dir / ("round_" + str(rnd) + ".json")
            with open(str(p), "w") as f:
                json.dump(rd, f, indent=2, default=str)
        except Exception as e:
            logger.error("save round: " + str(e))

    def _print_round(self, rnd, rd):
        tag = " [RECYCLED]" if rd["data_recycled"] else ""
        print("\n" + "=" * 60)
        print("Round " + str(rnd) + "/" + str(self.server.max_rounds) +
              "  [" + self.server.dataset_name + "]" + tag)
        print("=" * 60)
        print("  Fixed-test : " +
              "{:.1f}".format(rd["mean_fixed_accuracy"]) + "%" +
              "  (best: " +
              "{:.1f}".format(self.server.best_accuracy) + "%)")
        print("  Round-val  : " +
              "{:.1f}".format(rd["mean_val_accuracy"]) + "%" +
              "  Train: " +
              "{:.1f}".format(rd["mean_train_accuracy"]) + "%" +
              "  Time: " +
              "{:.2f}".format(rd["round_time"]) + "s")
        for c, a in rd["per_class_fixed"].items():
            print("    " + c.ljust(22) + ": " + "{:.1f}".format(a) + "%")
        for cr in rd["client_results"]:
            ep = (" [ep" + str(cr["data_epoch"]) + "]"
                  if cr["data_epoch"] > 1 else "")
            print("  " + str(cr["client_id"]) +
                  ": fixed=" + "{:.1f}".format(cr["fixed_test_accuracy"]) +
                  "% val=" + "{:.1f}".format(cr["val_accuracy"]) +
                  "% t=" + "{:.1f}".format(cr["training_time"]) + "s" + ep)

    # -- final save and plots ----------------------------------
    def save_final_results_and_plots(self):
        self._save_final_json()
        self._generate_plots()

    def _save_final_json(self):
        if not self.server.round_metrics:
            return
        fr = self.server.round_metrics[-1]

        client_summary = {}
        for cid in self.server.active_clients:
            vh = self.server.client_val_history[cid]
            fh = self.server.client_fixed_history[cid]
            th = self.server.client_train_times[cid]
            if vh:
                va = [e["accuracy"]       for e in vh]
                fa = [e["fixed_accuracy"] for e in fh]
                tt = [e["training_time"]  for e in th]
                client_summary[cid] = {
                    "rounds_participated"  : len(vh),
                    "best_val_accuracy"    : float(max(va)),
                    "best_fixed_accuracy"  : float(max(fa)) if fa else 0.0,
                    "final_val_accuracy"   : float(va[-1]),
                    "final_fixed_accuracy" : float(fa[-1]) if fa else 0.0,
                    "mean_val_accuracy"    : float(np.mean(va)),
                    "total_training_time"  : float(sum(tt)),
                    "mean_training_time"   : float(np.mean(tt)),
                }

        final = {
            "experiment_config": {
                "dataset"                  : self.server.dataset_name,
                "classes"                  : self.server.CLASSES,
                "img_size"                 : list(IMG_SIZE),
                "n_clients"                : N_CLIENTS,
                "samples_per_class_total"  : SAMPLES_PER_CLASS_TOTAL,
                "samples_per_class_client" : SAMPLES_PER_CLASS_CLIENT,
                "fixed_test_per_class"     : FIXED_TEST_PER_CLASS,
                "train_pool_per_class"     : TRAIN_POOL_PER_CLASS,
                "shots_per_class"          : SHOTS_PER_CLASS,
                "val_per_class"            : VAL_PER_CLASS,
                "max_unique_rounds"        : MAX_UNIQUE_ROUNDS,
                "neurons_per_class"        : NEURONS_PER_CLASS,
                "max_rounds"               : self.server.max_rounds,
                "fbz_loaded"               : self.server.fbz_path,
            },
            "fl_summary": {
                "total_rounds"          : len(self.server.round_metrics),
                "best_global_fixed_acc" : self.server.best_accuracy,
                "final_fixed_accuracy"  : fr.get("mean_fixed_accuracy", 0.0),
                "final_val_accuracy"    : fr.get("mean_val_accuracy",   0.0),
                "total_aggregation_time": float(sum(self.server.round_times)),
                "avg_round_time"        : float(np.mean(self.server.round_times))
                                          if self.server.round_times else 0.0,
                "n_active_clients"      : len(self.server.active_clients),
                "unique_data_rounds"    : MAX_UNIQUE_ROUNDS,
            },
            "final_per_class_fixed" : fr.get("per_class_fixed", {}),
            "final_per_class_val"   : fr.get("per_class_val",   {}),
            "round_metrics"         : self.server.round_metrics,
            "client_summary"        : client_summary,
            "timestamp"             : datetime.now().isoformat(),
        }
        p = self.server.results_dir / "fl_results_final.json"
        with open(str(p), "w") as f:
            json.dump(final, f, indent=2, default=str)
        logger.info("Final results -> " + p.name)

    def _generate_plots(self):
        if not self.server.round_metrics:
            logger.warning("No round metrics to plot.")
            return
        set_pub_style()
        self._plot_convergence()
        self._plot_per_class_final()
        self._plot_client_progression()
        self._plot_round_times()
        self._plot_per_class_evolution()
        self._plot_summary_dashboard()
        logger.info("All plots saved -> " + str(self.server.results_dir))

    def _plot_convergence(self):
        fa = self.server.global_fixed_accs
        va = self.server.global_val_accs
        ta = self.server.global_train_accs
        n  = len(fa)
        if n == 0: return
        r  = np.arange(1, n + 1)

        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        fig.subplots_adjust(left=0.10, right=0.96, top=0.88, bottom=0.14)
        if ta:
            ax.plot(r, ta[:n], color=PALETTE["snn"], lw=1.0,
                    linestyle=":", alpha=0.7, label="Train")
        ax.plot(r, va[:n], color=PALETTE["train"], lw=1.2,
                linestyle="--", alpha=0.85, label="Round-val")
        ax.plot(r, fa[:n], color=PALETTE["accent"], lw=2.0,
                label="Fixed-test")
        ax.axhline(y=self.server.best_accuracy,
                   color=PALETTE["accent"], lw=0.8, linestyle=":",
                   label="Best=" +
                   "{:.1f}".format(self.server.best_accuracy) + "%")
        if self.server.max_rounds > MAX_UNIQUE_ROUNDS:
            ax.axvline(x=MAX_UNIQUE_ROUNDS, color="gray",
                       lw=1.0, linestyle="--", alpha=0.5,
                       label="Data recycle")
        ax.set_xlabel("FL round"); ax.set_ylabel("Accuracy (%)")
        ax.set_xlim(0.5, n + 0.5); ax.set_ylim(0, 108)
        ax.set_title(self.server.dataset_name +
                     " - FL convergence (" + str(n) + " rounds, " +
                     str(len(self.server.active_clients)) + " clients)")
        ax.legend(fontsize=7, ncol=2)
        save_fig(fig, self.server.results_dir, "fig1_fl_convergence")

    def _plot_per_class_final(self):
        if not self.server.round_metrics: return
        rd  = self.server.round_metrics[-1]
        pf  = rd.get("per_class_fixed", {})
        pv  = rd.get("per_class_val",   {})
        if not pf: return
        x     = np.arange(len(self.server.CLASSES))
        short = [c.replace("_", "\n") for c in self.server.CLASSES]

        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
        fig.subplots_adjust(bottom=0.18, top=0.88, wspace=0.40)

        vf   = [pf.get(c, 0.0) for c in self.server.CLASSES]
        bars = axes[0].bar(x, vf,
                           color=CLASS_COLORS[:len(self.server.CLASSES)],
                           edgecolor="white", linewidth=0.5, width=0.55)
        axes[0].axhline(y=float(np.mean(vf)), color=PALETTE["accent"],
                        lw=1.2, linestyle="--",
                        label="Mean=" + "{:.1f}".format(np.mean(vf)) + "%")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(short, fontsize=7.5)
        axes[0].set_ylim(0, 120); axes[0].set_ylabel("Accuracy (%)")
        axes[0].set_title("(a) Fixed-test acc (final round)")
        axes[0].legend(fontsize=7)
        for bar, v in zip(bars, vf):
            axes[0].text(bar.get_x() + bar.get_width() / 2, v + 1.5,
                         "{:.1f}".format(v) + "%",
                         ha="center", va="bottom", fontsize=7.5)

        vv = [pv.get(c, 0.0) for c in self.server.CLASSES]
        axes[1].bar(x, vv,
                    color=CLASS_COLORS[:len(self.server.CLASSES)],
                    edgecolor="white", linewidth=0.5, width=0.55, alpha=0.8)
        axes[1].axhline(y=float(np.mean(vv)), color=PALETTE["accent"],
                        lw=1.2, linestyle="--",
                        label="Mean=" + "{:.1f}".format(np.mean(vv)) + "%")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(short, fontsize=7.5)
        axes[1].set_ylim(0, 120); axes[1].set_ylabel("Accuracy (%)")
        axes[1].set_title("(b) Round-val acc (final round)")
        axes[1].legend(fontsize=7)

        fig.suptitle(self.server.dataset_name +
                     " - Per-class accuracy (final round)", fontsize=10)
        save_fig(fig, self.server.results_dir, "fig2_per_class_final")

    def _plot_client_progression(self):
        if not self.server.client_val_history: return
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
        fig.subplots_adjust(bottom=0.14, top=0.88, wspace=0.38)

        for i, cid in enumerate(sorted(self.server.active_clients)):
            vh = self.server.client_val_history[cid]
            fh = self.server.client_fixed_history[cid]
            if vh:
                axes[0].plot([e["round"]    for e in vh],
                             [e["accuracy"] for e in vh],
                             color=CLASS_COLORS[i % len(CLASS_COLORS)],
                             marker="o", lw=1.5, ms=3, label=str(cid))
            if fh:
                axes[1].plot([e["round"]          for e in fh],
                             [e["fixed_accuracy"] for e in fh],
                             color=CLASS_COLORS[i % len(CLASS_COLORS)],
                             marker="s", lw=1.5, ms=3, label=str(cid))

        n = len(self.server.global_val_accs)
        r = range(1, n + 1)
        if self.server.global_val_accs:
            axes[0].plot(r, self.server.global_val_accs, color="black",
                         lw=2.0, linestyle="--", alpha=0.7, label="Global")
        if self.server.global_fixed_accs:
            axes[1].plot(r, self.server.global_fixed_accs, color="black",
                         lw=2.0, linestyle="--", alpha=0.7, label="Global")

        for ax in axes:
            ax.set_xlabel("FL round"); ax.set_ylabel("Accuracy (%)")
            ax.set_ylim(0, 108); ax.legend(fontsize=7)
            if self.server.max_rounds > MAX_UNIQUE_ROUNDS:
                ax.axvline(x=MAX_UNIQUE_ROUNDS, color="gray",
                           lw=0.8, linestyle="--", alpha=0.5)
        axes[0].set_title("(a) Round-val per client")
        axes[1].set_title("(b) Fixed-test per client")
        fig.suptitle(self.server.dataset_name + " - Client progression",
                     fontsize=10)
        save_fig(fig, self.server.results_dir, "fig3_client_progression")

    def _plot_round_times(self):
        rt = self.server.round_times
        if not rt: return
        n  = len(rt)
        fig, ax = plt.subplots(figsize=(7.2, 3.0))
        fig.subplots_adjust(left=0.10, right=0.96, top=0.88, bottom=0.14)
        ax.bar(np.arange(1, n + 1), rt, color=PALETTE["train"],
               edgecolor="white", linewidth=0.5, width=0.8, alpha=0.8)
        ax.axhline(y=float(np.mean(rt)), color=PALETTE["accent"],
                   lw=1.2, linestyle="--",
                   label="Mean=" + "{:.2f}".format(np.mean(rt)) + "s")
        if self.server.max_rounds > MAX_UNIQUE_ROUNDS:
            ax.axvline(x=MAX_UNIQUE_ROUNDS, color="gray",
                       lw=0.8, linestyle="--", alpha=0.5,
                       label="Data recycle")
        ax.set_xlabel("FL round"); ax.set_ylabel("Time (s)")
        ax.set_xlim(0.5, n + 0.5)
        ax.set_title(self.server.dataset_name + " - Round times")
        ax.legend(fontsize=7)
        save_fig(fig, self.server.results_dir, "fig4_round_times")

    def _plot_per_class_evolution(self):
        n = len(self.server.round_metrics)
        if n == 0: return
        r = np.arange(1, n + 1)
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        fig.subplots_adjust(left=0.10, right=0.96, top=0.88, bottom=0.14)
        for i, cls in enumerate(self.server.CLASSES):
            vals = self.server.per_class_fixed_hist[cls]
            if vals:
                ax.plot(r[:len(vals)], vals,
                        color=CLASS_COLORS[i % len(CLASS_COLORS)],
                        lw=1.5, marker="o", ms=2.5,
                        label=cls.replace("_", " "))
        if self.server.global_fixed_accs:
            m = len(self.server.global_fixed_accs)
            ax.plot(r[:m], self.server.global_fixed_accs,
                    color="black", lw=2.0, linestyle="--",
                    alpha=0.7, label="Overall")
        if self.server.max_rounds > MAX_UNIQUE_ROUNDS:
            ax.axvline(x=MAX_UNIQUE_ROUNDS, color="gray",
                       lw=0.8, linestyle="--", alpha=0.5,
                       label="Data recycle")
        ax.set_xlabel("FL round"); ax.set_ylabel("Fixed-test acc (%)")
        ax.set_xlim(0.5, n + 0.5); ax.set_ylim(0, 108)
        ax.set_title(self.server.dataset_name +
                     " - Per-class accuracy evolution")
        ax.legend(fontsize=7, ncol=2)
        save_fig(fig, self.server.results_dir, "fig5_per_class_evolution")

    def _plot_summary_dashboard(self):
        if not self.server.round_metrics: return
        n   = len(self.server.round_metrics)
        r   = np.arange(1, n + 1)
        fa  = self.server.global_fixed_accs
        va  = self.server.global_val_accs
        ta  = self.server.global_train_accs
        rt  = self.server.round_times
        rd  = self.server.round_metrics[-1]
        pf  = rd.get("per_class_fixed", {})
        x   = np.arange(len(self.server.CLASSES))
        short = [c.replace("_", " ")[:8] for c in self.server.CLASSES]

        fig = plt.figure(figsize=(10.0, 7.5))
        gs  = gridspec.GridSpec(3, 3, figure=fig,
                                hspace=0.55, wspace=0.42,
                                left=0.08, right=0.97,
                                top=0.92,   bottom=0.08)

        # (a) convergence
        ax0 = fig.add_subplot(gs[0, :2])
        if ta:
            ax0.plot(r, ta[:n], color=PALETTE["snn"], lw=1.0,
                     linestyle=":", alpha=0.6, label="Train")
        ax0.plot(r, va[:n], color=PALETTE["train"], lw=1.2,
                 linestyle="--", alpha=0.8, label="Round-val")
        ax0.plot(r, fa[:n], color=PALETTE["accent"], lw=2.0,
                 label="Fixed-test")
        ax0.axhline(y=self.server.best_accuracy,
                    color=PALETTE["accent"], lw=0.8, linestyle=":",
                    label="Best=" +
                    "{:.1f}".format(self.server.best_accuracy) + "%")
        if self.server.max_rounds > MAX_UNIQUE_ROUNDS:
            ax0.axvline(x=MAX_UNIQUE_ROUNDS, color="gray",
                        lw=0.8, linestyle="--", alpha=0.5)
        ax0.set_xlabel("Round"); ax0.set_ylabel("Accuracy (%)")
        ax0.set_xlim(0.5, n + 0.5); ax0.set_ylim(0, 108)
        ax0.set_title("(a) Convergence"); ax0.legend(fontsize=6.5, ncol=2)

        # (b) summary table
        ax1 = fig.add_subplot(gs[0, 2])
        ax1.axis("off")
        avg_rt = float(np.mean(rt)) if rt else 0.0
        rows = [
            ["Best fixed acc",    "{:.1f}".format(self.server.best_accuracy) + "%"],
            ["Final fixed acc",   "{:.1f}".format(fa[-1] if fa else 0) + "%"],
            ["FL rounds",         str(n)],
            ["Unique rounds",     str(MAX_UNIQUE_ROUNDS)],
            ["Active clients",    str(len(self.server.active_clients))],
            ["Avg round time",    "{:.2f}".format(avg_rt) + "s"],
            ["Shots/class/round", str(SHOTS_PER_CLASS)],
            ["Fixed test/class",  str(FIXED_TEST_PER_CLASS)],
        ]
        tbl = ax1.table(cellText=rows, colLabels=["Metric", "Value"],
                        loc="center", cellLoc="left")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.0)
        tbl.scale(1.0, 1.35)
        for j in range(2):
            tbl[(0, j)].set_facecolor("#2C3E50")
            tbl[(0, j)].set_text_props(color="white", fontweight="bold")
        ax1.set_title("(b) Summary", fontsize=9)

        # (c) final per-class
        ax2 = fig.add_subplot(gs[1, :2])
        vf   = [pf.get(c, 0.0) for c in self.server.CLASSES]
        bars = ax2.bar(x, vf,
                       color=CLASS_COLORS[:len(self.server.CLASSES)],
                       edgecolor="white", linewidth=0.5, width=0.55)
        ax2.axhline(y=float(np.mean(vf)), color=PALETTE["accent"],
                    lw=1.2, linestyle="--",
                    label="Mean=" + "{:.1f}".format(np.mean(vf)) + "%")
        ax2.set_xticks(x)
        ax2.set_xticklabels(short, fontsize=8, rotation=20,
                            ha="right", rotation_mode="anchor")
        ax2.set_ylim(0, 120); ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("(c) Final per-class (fixed test)")
        ax2.legend(fontsize=7)
        for bar, v in zip(bars, vf):
            ax2.text(bar.get_x() + bar.get_width() / 2, v + 1.5,
                     "{:.0f}".format(v),
                     ha="center", va="bottom", fontsize=7)

        # (d) round times
        ax3 = fig.add_subplot(gs[1, 2])
        ax3.bar(r, rt[:n], color=PALETTE["train"],
                edgecolor="white", linewidth=0.5, width=0.8, alpha=0.8)
        ax3.axhline(y=avg_rt, color=PALETTE["accent"], lw=1.0,
                    linestyle="--",
                    label="{:.2f}".format(avg_rt) + "s avg")
        ax3.set_xlabel("Round"); ax3.set_ylabel("Time (s)")
        ax3.set_xlim(0.5, n + 0.5)
        ax3.set_title("(d) Round times"); ax3.legend(fontsize=6.5)

        # (e) per-class evolution
        ax4 = fig.add_subplot(gs[2, :])
        for i, cls in enumerate(self.server.CLASSES):
            vals = self.server.per_class_fixed_hist[cls]
            if vals:
                ax4.plot(r[:len(vals)], vals,
                         color=CLASS_COLORS[i % len(CLASS_COLORS)],
                         lw=1.2, marker="o", ms=2.0,
                         label=cls.replace("_", " "))
        if fa:
            m = len(fa)
            ax4.plot(r[:m], fa[:m], color="black", lw=2.0,
                     linestyle="--", alpha=0.7, label="Overall")
        if self.server.max_rounds > MAX_UNIQUE_ROUNDS:
            ax4.axvline(x=MAX_UNIQUE_ROUNDS, color="gray",
                        lw=0.8, linestyle="--", alpha=0.5,
                        label="Recycle")
        ax4.set_xlabel("FL round"); ax4.set_ylabel("Fixed-test acc (%)")
        ax4.set_xlim(0.5, n + 0.5); ax4.set_ylim(0, 108)
        ax4.set_title("(e) Per-class accuracy evolution")
        ax4.legend(fontsize=6.5, ncol=3)

        fig.suptitle(
            self.server.dataset_name +
            "  -  SNN-NeuroEdge Federated Learning  |  " +
            str(n) + " rounds  |  " +
            str(len(self.server.active_clients)) +
            " clients  |  AKD1000",
            fontsize=10, y=0.97
        )
        save_fig(fig, self.server.results_dir, "fig6_summary_dashboard")


# ==============================================================
# Entry point
# ==============================================================
def start_server(dataset_name="Warehouse",
                 server_address="10.0.5.2:8080",
                 max_rounds=50):
    logger.info("Starting FL server: [" + dataset_name + "]  " +
                server_address + "  rounds=" + str(max_rounds))
    srv   = NeuroEdgeServer(dataset_name=dataset_name,
                            max_rounds=max_rounds)
    strat = NeuroEdgeStrategy(srv)
    try:
        fl.server.start_server(
            server_address=server_address,
            config=fl.server.ServerConfig(num_rounds=max_rounds),
            strategy=strat,
        )
        strat.save_final_results_and_plots()
        print("\nAll results -> " + str(srv.results_dir.resolve()))
    except KeyboardInterrupt:
        logger.info("Interrupted - saving partial results")
        strat.save_final_results_and_plots()
    except Exception as e:
        logger.error("Server error: " + str(e))
        try:
            strat.save_final_results_and_plots()
        except Exception:
            pass


def main():
    import argparse
    p = argparse.ArgumentParser(
        description="SNN-NeuroEdge FL Server v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 snn_neuroedge_server_v2.py --dataset Warehouse --max_rounds 50
  python3 snn_neuroedge_server_v2.py --dataset MNIST_5class --max_rounds 50
  python3 snn_neuroedge_server_v2.py --dataset CIFAR10_5class --max_rounds 50
        """
    )
    p.add_argument("--dataset", type=str, default="Warehouse",
                   choices=list(DATASET_CONFIGS.keys()),
                   help="Dataset to use (default: Warehouse)")
    p.add_argument("--server_address", type=str, default="10.0.5.2:8080",
                   help="Server IP:port (default: 10.0.5.2:8080)")
    p.add_argument("--max_rounds", type=int, default=50,
                   help="Number of FL rounds (default: 50)")
    a = p.parse_args()
    start_server(a.dataset, a.server_address, a.max_rounds)


if __name__ == "__main__":
    main()