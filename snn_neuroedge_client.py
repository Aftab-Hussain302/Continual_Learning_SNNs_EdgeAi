#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SNN-NeuroEdge FL Client v2  —  Multi-dataset + Multi-run edition
Supports: Warehouse | MNIST_5class | CIFAR10_5class (FL + centralized baseline)

Changes vs original v2:
  - Full multi-dataset support with per-dataset shot/test/pool configs.
  - --run_id flag for experiment numbering; results dir carries run ID.
  - Centralized baseline also respects run_id for multi-run statistics.
  - Keeps all Akida 2.19.1 variable API fixes (sentinel detection,
    warm-up inference, shape validation, hardware remap).

Usage examples:
  # FL mode — Warehouse (default)
  python3 snn_neuroedge_client_v2.py --client_id SAI_Node_7 --server_address 10.0.5.2:8080

  # FL mode — MNIST-5class
  python3 snn_neuroedge_client_v2.py --client_id SAI_Node_7 \\
      --reserved_paths /home/sai/datasets/MNIST_5class/reserved_paths.json

  # FL mode — CIFAR10-5class, specific run
  python3 snn_neuroedge_client_v2.py --client_id SAI_Node_8 \\
      --reserved_paths /home/sai/datasets/CIFAR10_5class/reserved_paths.json \\
      --run_id 3

  # Centralized baseline — Warehouse, 5 runs
  python3 snn_neuroedge_client_v2.py --mode centralized --n_runs 5

  # Centralized baseline — MNIST-5class
  python3 snn_neuroedge_client_v2.py --mode centralized \\
      --reserved_paths /home/sai/datasets/MNIST_5class/reserved_paths.json

  # Centralized baseline — CIFAR10-5class, run 2
  python3 snn_neuroedge_client_v2.py --mode centralized \\
      --reserved_paths /home/sai/datasets/CIFAR10_5class/reserved_paths.json \\
      --run_id 2 --max_rounds 50
"""

import json
import time
import random
import logging
import socket
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import flwr as fl
from flwr.client import NumPyClient

import akida
from akida import FullyConnected, AkidaUnsupervised, devices

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================
# Dataset configurations (must stay in sync with server)
# ============================================================
DATASET_CONFIGS = {
    "Warehouse": {
        "classes": [
            "boxes_stacked", "empty_aisle", "forklift_present",
            "mixed_objects", "unclear_scene",
        ],
        "reserved_json"   : "reserved_paths.json",
        "shots_per_class" : 1,
        "val_per_class"   : 1,
        "fixed_test"      : 30,
        "samples_total"   : 200,
        "n_clients"       : 2,
    },
    "MNIST_5class": {
        "classes": [
            "digit_0", "digit_1", "digit_2", "digit_3", "digit_4",
        ],
        "reserved_json"   : "/home/sai/datasets/MNIST_5class/reserved_paths.json",
        "shots_per_class" : 2,
        "val_per_class"   : 2,
        "fixed_test"      : 40,
        "samples_total"   : 200,
        "n_clients"       : 2,
    },
    "CIFAR10_5class": {
        "classes": [
            "airplane", "automobile", "bird", "cat", "deer",
        ],
        "reserved_json"   : "/home/sai/datasets/CIFAR10_5class/reserved_paths.json",
        "shots_per_class" : 2,
        "val_per_class"   : 2,
        "fixed_test"      : 40,
        "samples_total"   : 200,
        "n_clients"       : 2,
    },
}

# Shared Akida edge-learning hyperparameters
IMG_SIZE             = (160, 160)
NEURONS_PER_CLASS    = 100
NUM_WEIGHTS_FRACTION = 0.50
SEED                 = 42

FBZ_CANDIDATES = [
    "warehouse_edge_backbone_full.fbz",
    "warehouse_edge_backbone_full_v1.fbz",
    "warehouse_edge_backbone_160.fbz",
    "warehouse_edge_backbone.fbz",
]

PALETTE = {
    "train"  : "#0072B2",
    "val"    : "#E69F00",
    "snn"    : "#009E73",
    "accent" : "#D55E00",
    "gray"   : "#999999",
}
CLASS_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]


# ============================================================
# Derived per-dataset constants helper
# ============================================================
def derive_constants(cfg: dict) -> dict:
    """
    Compute derived split constants from a dataset config dict.
    """
    samples_per_client = cfg["samples_total"] // cfg["n_clients"]
    train_pool         = samples_per_client - cfg["fixed_test"]
    samples_per_round  = cfg["shots_per_class"] + cfg["val_per_class"]
    max_unique_rounds  = train_pool // samples_per_round
    return {
        "samples_per_client" : samples_per_client,
        "train_pool"         : train_pool,
        "samples_per_round"  : samples_per_round,
        "max_unique_rounds"  : max_unique_rounds,
    }


# ============================================================
# Akida 2.19.1 variable helpers
# ============================================================
def akida_get_weights(layer):
    """
    Extract layer variables sorted by name.
    Returns list of (name, np.ndarray) tuples.
    Uses Akida 2.19.1 API: get_variable_names() + get_variable().
    """
    names  = layer.get_variable_names()
    result = []
    for name in sorted(names):
        val = layer.get_variable(name)
        result.append((name, np.array(val, dtype=np.float32)))
    return result


def akida_set_weights(layer, named_arrays):
    """
    Apply variables from list of (name, ndarray) tuples.
    Uses Akida 2.19.1 API: set_variable().
    """
    for name, arr in named_arrays:
        try:
            layer.set_variable(name, arr)
        except Exception as e:
            logger.error("set_variable(%s) failed: %s", name, str(e))


def is_sentinel(parameters) -> bool:
    """
    Detect the sentinel initial parameters sent by the server.
    Sentinel = single array [0.0] — client keeps local model.
    """
    if parameters is None:
        return True
    if len(parameters) == 1:
        arr = np.array(parameters[0])
        if arr.shape == (1,) and arr[0] == 0.0:
            return True
    return False


# ============================================================
# Publication-quality plot style
# ============================================================
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


def save_fig(fig, out_dir: Path, name: str):
    png = out_dir / (name + ".png")
    eps = out_dir / (name + ".eps")
    fig.savefig(str(png), dpi=300, bbox_inches="tight", facecolor="white")
    try:
        fig.savefig(str(eps), format="eps", bbox_inches="tight")
    except Exception:
        pass
    plt.close(fig)
    logger.info("  Saved: %s", png.name)


# ============================================================
# General utilities
# ============================================================
def get_client_id() -> str:
    config_file = Path("fl_client_config.json")
    hostname    = socket.gethostname().lower()
    if config_file.exists():
        try:
            with open(str(config_file), "r") as f:
                cfg = json.load(f)
            if cfg.get("hostname") == hostname and "client_id" in cfg:
                return cfg["client_id"]
        except Exception:
            pass
    import re
    if "node" in hostname:
        m   = re.search(r"node[\s_-]*(\d+)", hostname)
        cid = ("SAI_Node_" + m.group(1)) if m else ("SAI_" + hostname)
    elif "sai" in hostname:
        cid = hostname.replace("-", "_").upper()
    else:
        cid = "Client_" + hostname.replace("-", "_")
    try:
        with open(str(config_file), "w") as f:
            json.dump({"client_id": cid, "hostname": hostname,
                       "created_at": datetime.now().isoformat()}, f, indent=2)
    except Exception:
        pass
    return cid


def load_img_uint8(path) -> np.ndarray | None:
    try:
        img = tf.keras.utils.load_img(str(path), target_size=IMG_SIZE)
        return np.clip(
            tf.keras.utils.img_to_array(img), 0, 255).astype(np.uint8)
    except Exception as e:
        logger.warning("Load fail %s: %s", str(path), str(e))
        return None


def find_fbz() -> str:
    for c in FBZ_CANDIDATES:
        if Path(c).exists():
            return c
    raise RuntimeError(
        "No FBZ backbone found. Searched: " + str(FBZ_CANDIDATES))


def infer_fan_in(model_ak) -> int:
    try:
        prev = model_ak.layers[-2]
        if hasattr(prev, "output_dims") and prev.output_dims is not None:
            h, w, c = prev.output_dims
            return int(h * w * c)
    except Exception:
        pass
    return 256


def build_akida_model(fbz_path: str, devs: list, n_classes: int):
    """
    Build and compile the Akida edge model, then run a warm-up
    inference so the hardware binary-input pipeline is initialised
    before the first call to model.fit().
    """
    model  = akida.Model(fbz_path)
    if devs:
        model.map(devs[0])
    model.pop_layer()
    total  = n_classes * NEURONS_PER_CLASS
    model.add(FullyConnected(
        name="akida_edge_layer", units=total, activation=False))
    fan_in = infer_fan_in(model)
    nw     = max(32, min(fan_in, int(round(NUM_WEIGHTS_FRACTION * fan_in))))
    model.compile(optimizer=AkidaUnsupervised(
        num_weights=nw, num_classes=n_classes, learning_competition=0.1))

    # Warm-up inference to initialise binary input pipeline
    try:
        first = model.layers[0]
        if hasattr(first, "input_dims") and first.input_dims is not None:
            h, w, c = first.input_dims
        else:
            h, w, c = 160, 160, 3
        dummy = np.zeros((1, h, w, c), dtype=np.uint8)
        model.predict(dummy)
        logger.info("build_akida_model: warm-up OK (input %dx%dx%d)", h, w, c)
    except Exception as e:
        logger.warning("build_akida_model: warm-up failed (non-fatal): %s", str(e))

    return model, fan_in, nw


def detect_dataset(classes: list, rp_path) -> str:
    """Infer dataset name from classes or metadata file."""
    meta = Path(rp_path).parent / "dataset_metadata.json"
    if meta.exists():
        try:
            with open(str(meta), "r") as f:
                return json.load(f).get("dataset_name", "Warehouse")
        except Exception:
            pass
    if classes[0].startswith("digit"):
        return "MNIST_5class"
    if classes[0] in ("airplane", "automobile", "bird", "cat", "deer"):
        return "CIFAR10_5class"
    return "Warehouse"


# ============================================================
# Data partitioning
# ============================================================
def partition_client_data(reserved_paths: dict,
                           classes: list,
                           client_index: int,
                           cfg: dict,
                           dc: dict) -> tuple:
    """
    Split samples per client according to dataset config.
      client_index 0 (Node_7): first half
      client_index 1 (Node_8): second half
    Fixed test: last cfg['fixed_test'] of each client's slice.
    Train pool: remainder.
    """
    fixed_test = {}
    train_pool = {}
    spc = dc["samples_per_client"]          # samples per class per client
    tot = cfg["samples_total"]
    ft  = cfg["fixed_test"]

    for cls in classes:
        paths = sorted(reserved_paths[cls])
        client_paths = (paths[:spc]
                        if client_index == 0
                        else paths[spc:tot])
        fixed_test[cls] = client_paths[-ft:]
        train_pool[cls] = client_paths[:-ft]
    return fixed_test, train_pool


def get_round_data(train_pool: dict,
                   classes: list,
                   round_num: int,
                   cfg: dict,
                   dc: dict) -> tuple:
    """
    Return (train_paths, val_paths, epoch) for a given round.
    Rounds 1..max_unique: sequential non-overlapping windows.
    Rounds > max_unique: recycled with epoch-seeded shuffle.
    """
    shots = cfg["shots_per_class"]
    vals  = cfg["val_per_class"]
    spr   = dc["samples_per_round"]
    mu    = dc["max_unique_rounds"]

    epoch = ((round_num - 1) // mu) + 1
    eff   = ((round_num - 1) % mu) + 1
    start = (eff - 1) * spr

    train_paths = {}
    val_paths   = {}
    for cls in classes:
        pool = train_pool[cls]
        if epoch > 1:
            rng      = random.Random(SEED + epoch * 997 + hash(cls))
            shuffled = pool[:]
            rng.shuffle(shuffled)
        else:
            shuffled = pool
        te = start + shots
        ts = te + vals
        if ts <= len(shuffled):
            train_paths[cls] = shuffled[start:te]
            val_paths[cls]   = shuffled[te:ts]
        else:
            train_paths[cls] = shuffled[:shots]
            val_paths[cls]   = shuffled[shots:shots + vals]
    return train_paths, val_paths, epoch


def evaluate_pool(model, data_paths: dict, classes: list) -> dict:
    """
    Evaluate model on a {cls: [paths]} pool.
    Returns per-class accuracy dict (0-100 scale).
    """
    n_cls     = len(classes)
    per_class = {}
    for i, cls in enumerate(classes):
        paths = data_paths.get(cls, [])
        if not paths:
            per_class[cls] = 0.0
            continue
        preds = []
        for p in paths:
            img = load_img_uint8(p)
            if img is not None:
                try:
                    pred = model.predict_classes(
                        np.expand_dims(img, 0), num_classes=n_cls)[0]
                    preds.append(int(pred))
                except Exception:
                    continue
        per_class[cls] = (100.0 * (np.array(preds) == i).mean()
                          if preds else 0.0)
    return per_class


# ============================================================
# FL Client
# ============================================================
class NeuroEdgeClient(NumPyClient):

    def __init__(self, client_id: str = None,
                 reserved_paths_json: str = "reserved_paths.json",
                 run_id: int = 1):
        self.client_id = client_id or get_client_id()
        self.run_id    = run_id

        rp_path = Path(reserved_paths_json)
        if not rp_path.exists():
            raise FileNotFoundError(
                "reserved_paths.json not found: " + str(rp_path))
        with open(str(rp_path), "r") as f:
            self.reserved_paths = json.load(f)

        self.CLASSES      = sorted(list(self.reserved_paths.keys()))
        self.dataset_name = detect_dataset(self.CLASSES, rp_path)

        # Look up dataset config; fall back to Warehouse defaults
        self.cfg = DATASET_CONFIGS.get(self.dataset_name,
                                       DATASET_CONFIGS["Warehouse"])
        self.dc  = derive_constants(self.cfg)

        # Deterministic client index from ID hash
        self.client_index = int(
            hashlib.md5(self.client_id.encode()).hexdigest(), 16
        ) % self.cfg["n_clients"]

        self.fixed_test, self.train_pool = partition_client_data(
            self.reserved_paths, self.CLASSES,
            self.client_index, self.cfg, self.dc)

        run_tag = f"run{run_id:02d}"
        stamp   = datetime.now().strftime("%Y%m%d_%H%M")
        self.results_dir = Path(
            f"edge_{self.client_id}_{self.dataset_name}_{run_tag}_{stamp}")
        self.results_dir.mkdir(exist_ok=True)

        random.seed(SEED)
        np.random.seed(SEED)
        tf.random.set_seed(SEED)

        self.devs     = devices()
        self.fbz_path = find_fbz()
        self.model, self.fan_in, self.num_weights = build_akida_model(
            self.fbz_path, self.devs, len(self.CLASSES))

        # Cache variable metadata for validation in set_parameters
        self._ref_named  = akida_get_weights(self.model.layers[-1])
        self._var_names  = [n for n, _ in self._ref_named]
        self._var_shapes = [a.shape for _, a in self._ref_named]

        self.current_round  = 0
        self.round_history  = []
        self.val_acc_hist   = []
        self.fixed_acc_hist = []
        self.train_acc_hist = []

        self._log_config()
        self._save_partition_info()

    def _log_config(self):
        cfg = self.cfg
        dc  = self.dc
        logger.info("=" * 60)
        logger.info("NeuroEdge FL Client v2  [%s]  run=%02d",
                    self.dataset_name, self.run_id)
        logger.info("=" * 60)
        logger.info("  client_id      : %s", self.client_id)
        logger.info("  client_index   : %d  (0=Node7, 1=Node8)",
                    self.client_index)
        logger.info("  FBZ backbone   : %s", self.fbz_path)
        logger.info("  fan_in         : %d  nw=%d",
                    self.fan_in, self.num_weights)
        logger.info("  Var names      : %s", str(self._var_names))
        logger.info("  Var shapes     : %s", str(self._var_shapes))
        logger.info("  Fixed test     : %d/class", cfg["fixed_test"])
        logger.info("  Train pool     : %d/class", dc["train_pool"])
        logger.info("  Shots/round    : %d/class", cfg["shots_per_class"])
        logger.info("  Val/round      : %d/class", cfg["val_per_class"])
        logger.info("  Unique rounds  : %d",       dc["max_unique_rounds"])
        logger.info("  Hardware       : %s",
                    str(self.devs[0]) if self.devs else "SW backend")
        logger.info("  Results dir    : %s", str(self.results_dir))

    def _save_partition_info(self):
        info = {
            "client_id"         : self.client_id,
            "client_index"      : self.client_index,
            "run_id"            : self.run_id,
            "dataset"           : self.dataset_name,
            "fbz_backbone"      : self.fbz_path,
            "fan_in"            : self.fan_in,
            "num_weights"       : self.num_weights,
            "var_names"         : self._var_names,
            "var_shapes"        : [str(s) for s in self._var_shapes],
            "shots_per_class"   : self.cfg["shots_per_class"],
            "val_per_class"     : self.cfg["val_per_class"],
            "fixed_test_per_cls": self.cfg["fixed_test"],
            "train_pool_per_cls": self.dc["train_pool"],
            "max_unique_rounds" : self.dc["max_unique_rounds"],
            "fixed_test_counts" : {c: len(v)
                                   for c, v in self.fixed_test.items()},
            "train_pool_counts" : {c: len(v)
                                   for c, v in self.train_pool.items()},
        }
        with open(str(self.results_dir / "partition_info.json"), "w") as f:
            json.dump(info, f, indent=2)

    # ----------------------------------------------------------
    # Flower NumPyClient interface
    # ----------------------------------------------------------
    def get_parameters(self, config):
        """Return edge layer variables as flat ndarray list for Flower."""
        try:
            named = akida_get_weights(self.model.layers[-1])
            return [arr for _, arr in named]
        except Exception as e:
            logger.error("get_parameters: %s", str(e))
        return [np.zeros(s, dtype=np.float32) for s in self._var_shapes]

    def set_parameters(self, parameters):
        """
        Apply aggregated weights from server to local edge layer.

        v2 safety logic:
          1. Detect sentinel ([0.0]) from server round 1 → skip.
          2. Validate ndarray count matches variable count.
          3. Validate each shape matches expected variable shape.
          4. Apply via set_variable() (Akida 2.19.1 API).
          5. Remap to hardware + warm-up to reinitialise binary pipeline.
        """
        if self.model is None:
            return

        # Step 1 — sentinel detection
        if is_sentinel(parameters):
            logger.info(
                "set_parameters: sentinel detected, "
                "keeping local model (round 1 init)")
            return

        if not parameters:
            return

        layer = self.model.layers[-1]

        # Step 2 — count check
        if len(parameters) != len(self._var_names):
            logger.warning(
                "set_parameters: count mismatch — got %d, expected %d. "
                "Skipping.", len(parameters), len(self._var_names))
            return

        # Step 3 — shape check
        for i, (arr, expected_shape) in enumerate(
                zip(parameters, self._var_shapes)):
            actual_shape = np.array(arr).shape
            if actual_shape != expected_shape:
                logger.warning(
                    "set_parameters: shape mismatch at index %d (%s)"
                    " — got %s, expected %s. Skipping.",
                    i, self._var_names[i], actual_shape, expected_shape)
                return

        # Step 4 — apply via Akida 2.19.1 variable API
        named = list(zip(self._var_names,
                         [np.array(p, dtype=np.float32)
                          for p in parameters]))
        akida_set_weights(layer, named)

        # Step 5 — remap + warm-up
        if self.devs:
            try:
                self.model.map(self.devs[0])
            except Exception as e:
                logger.warning("set_parameters: remap failed: %s", str(e))
        try:
            first = self.model.layers[0]
            if hasattr(first, "input_dims") and first.input_dims is not None:
                h, w, c = first.input_dims
            else:
                h, w, c = 160, 160, 3
            dummy = np.zeros((1, h, w, c), dtype=np.uint8)
            self.model.predict(dummy)
        except Exception as e:
            logger.warning("set_parameters: warm-up failed (non-fatal): %s",
                           str(e))

    def fit(self, parameters, config):
        self.current_round += 1
        rnd = self.current_round
        t0  = time.time()
        logger.info("\n[%s]  %s  Round %d  run=%02d",
                    self.dataset_name, self.client_id, rnd, self.run_id)

        self.set_parameters(parameters)
        train_p, val_p, epoch = get_round_data(
            self.train_pool, self.CLASSES, rnd, self.cfg, self.dc)

        if epoch > 1:
            logger.info("  Data epoch %d (training pool recycled)", epoch)

        n_cls   = len(self.CLASSES)
        order   = list(range(n_cls))
        random.shuffle(order)
        n_train = 0
        for idx in order:
            cls = self.CLASSES[idx]
            for p in train_p.get(cls, []):
                img = load_img_uint8(p)
                if img is not None:
                    try:
                        self.model.fit(np.expand_dims(img, 0), idx)
                        n_train += 1
                    except Exception as e:
                        logger.error("fit error %s: %s", str(p), str(e))

        total_time = time.time() - t0

        # Evaluate on three sets
        pca_val   = evaluate_pool(self.model, val_p,           self.CLASSES)
        pca_fixed = evaluate_pool(self.model, self.fixed_test, self.CLASSES)
        pca_train = evaluate_pool(self.model, train_p,         self.CLASSES)

        mean_val   = float(np.mean(list(pca_val.values())))
        mean_fixed = float(np.mean(list(pca_fixed.values())))
        mean_train = float(np.mean(list(pca_train.values())))

        self.val_acc_hist.append(mean_val)
        self.fixed_acc_hist.append(mean_fixed)
        self.train_acc_hist.append(mean_train)

        logger.info(
            "  Fixed=%.1f%%  Val=%.1f%%  Train=%.1f%%  "
            "Time=%.2fs  Epoch=%d",
            mean_fixed, mean_val, mean_train, total_time, epoch)

        rd = {
            "round"               : rnd,
            "data_epoch"          : epoch,
            "mean_val_accuracy"   : mean_val,
            "mean_fixed_accuracy" : mean_fixed,
            "mean_train_accuracy" : mean_train,
            "per_class_val"       : pca_val,
            "per_class_fixed"     : pca_fixed,
            "per_class_train"     : pca_train,
            "training_time"       : total_time,
            "n_trained"           : n_train,
            "client_id"           : self.client_id,
            "client_index"        : self.client_index,
            "run_id"              : self.run_id,
        }
        self.round_history.append(rd)
        try:
            with open(str(
                    self.results_dir / f"round_{rnd}.json"), "w") as f:
                json.dump(rd, f, indent=2)
        except Exception:
            pass

        if rnd % 10 == 0:
            try:
                self.model.save(str(
                    self.results_dir / f"model_round_{rnd}.fbz"))
            except Exception:
                pass

        # Metrics dict returned to server
        metrics = {
            "mean_accuracy"       : mean_val,
            "fixed_test_accuracy" : mean_fixed,
            "train_accuracy"      : mean_train,
            "training_time"       : total_time,
            "client_id"           : self.client_id,
            "client_index"        : self.client_index,
            "run_id"              : self.run_id,
            "round"               : rnd,
            "data_epoch"          : epoch,
            "samples_trained"     : n_train,
            "dataset"             : self.dataset_name,
            "hostname"            : socket.gethostname(),
        }
        for cls in self.CLASSES:
            metrics["val_"   + cls + "_acc"] = pca_val.get(cls,   0.0)
            metrics["fixed_" + cls + "_acc"] = pca_fixed.get(cls, 0.0)

        return self.get_parameters(config), n_train, metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        pca = evaluate_pool(self.model, self.fixed_test, self.CLASSES)
        acc = float(np.mean(list(pca.values())))
        return (float(100 - acc),
                sum(len(v) for v in self.fixed_test.values()),
                {"fixed_test_accuracy": acc, "client_id": self.client_id})

    # ----------------------------------------------------------
    # Saving results and plots
    # ----------------------------------------------------------
    def save_client_results_and_plots(self) -> dict:
        self._save_final_json()
        self._generate_plots()
        fa = self.fixed_acc_hist
        return {
            "run_id"               : self.run_id,
            "best_fixed_accuracy"  : float(max(fa)) if fa else 0.0,
            "final_fixed_accuracy" : float(fa[-1])  if fa else 0.0,
            "final_val_accuracy"   : (float(self.val_acc_hist[-1])
                                      if self.val_acc_hist else 0.0),
            "per_class_fixed"      : (
                self.round_history[-1].get("per_class_fixed", {})
                if self.round_history else {}),
            "results_dir"          : str(self.results_dir),
        }

    def _save_final_json(self):
        final = {
            "client_id"           : self.client_id,
            "client_index"        : self.client_index,
            "run_id"              : self.run_id,
            "dataset"             : self.dataset_name,
            "fbz_backbone"        : self.fbz_path,
            "total_rounds"        : self.current_round,
            "max_unique_rounds"   : self.dc["max_unique_rounds"],
            "shots_per_class"     : self.cfg["shots_per_class"],
            "val_per_class"       : self.cfg["val_per_class"],
            "fixed_test_per_class": self.cfg["fixed_test"],
            "best_fixed_accuracy" : (float(max(self.fixed_acc_hist))
                                     if self.fixed_acc_hist else 0.0),
            "final_fixed_accuracy": (float(self.fixed_acc_hist[-1])
                                     if self.fixed_acc_hist else 0.0),
            "val_acc_history"     : self.val_acc_hist,
            "fixed_acc_history"   : self.fixed_acc_hist,
            "train_acc_history"   : self.train_acc_hist,
            "round_history"       : self.round_history,
            "timestamp"           : datetime.now().isoformat(),
        }
        p = self.results_dir / "client_results_final.json"
        with open(str(p), "w") as f:
            json.dump(final, f, indent=2)
        logger.info("Client final results -> %s", p.name)

    def _generate_plots(self):
        if not self.round_history:
            return
        set_pub_style()
        n      = len(self.round_history)
        rounds = np.arange(1, n + 1)
        fa     = self.fixed_acc_hist
        va     = self.val_acc_hist
        ta     = self.train_acc_hist
        mu     = self.dc["max_unique_rounds"]

        # Fig 1: accuracy convergence
        fig, ax = plt.subplots(figsize=(7.2, 3.5))
        fig.subplots_adjust(left=0.10, right=0.96, top=0.88, bottom=0.14)
        if ta:
            ax.plot(rounds, ta, color=PALETTE["snn"], lw=1.0,
                    linestyle=":", alpha=0.7, label="Train")
        if va:
            ax.plot(rounds, va, color=PALETTE["train"], lw=1.2,
                    linestyle="--", alpha=0.85, label="Round-val")
        if fa:
            ax.plot(rounds, fa, color=PALETTE["accent"], lw=2.0,
                    label="Fixed-test")
            ax.axhline(y=max(fa), color=PALETTE["accent"], lw=0.8,
                       linestyle=":",
                       label=f"Best={max(fa):.1f}%")
        if n > mu:
            ax.axvline(x=mu, color="gray", lw=1.0, linestyle="--",
                       alpha=0.6, label="Data recycle")
        ax.set_xlabel("FL round"); ax.set_ylabel("Accuracy (%)")
        ax.set_xlim(0.5, n + 0.5); ax.set_ylim(0, 108)
        ax.set_title(
            f"{self.client_id}  [{self.dataset_name}]  run={self.run_id:02d}")
        ax.legend(fontsize=7, ncol=2)
        save_fig(fig, self.results_dir, "fig1_client_convergence")

        # Fig 2: final per-class fixed-test bar chart
        last_fixed = self.round_history[-1].get("per_class_fixed", {})
        if last_fixed:
            x     = np.arange(len(self.CLASSES))
            short = [c.replace("_", "\n") for c in self.CLASSES]
            fig, ax = plt.subplots(figsize=(5.5, 3.5))
            fig.subplots_adjust(bottom=0.20, top=0.88)
            vals = [last_fixed.get(c, 0.0) for c in self.CLASSES]
            bars = ax.bar(x, vals,
                          color=CLASS_COLORS[:len(self.CLASSES)],
                          edgecolor="white", linewidth=0.5, width=0.55)
            ax.axhline(y=float(np.mean(vals)), color=PALETTE["accent"],
                       lw=1.2, linestyle="--",
                       label=f"Mean={np.mean(vals):.1f}%")
            ax.set_xticks(x); ax.set_xticklabels(short, fontsize=7.5)
            ax.set_ylim(0, 120); ax.set_ylabel("Accuracy (%)")
            ax.set_title(
                f"{self.client_id} — Fixed-test acc (final round)")
            ax.legend(fontsize=7)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5,
                        f"{v:.1f}%", ha="center", va="bottom", fontsize=7.5)
            save_fig(fig, self.results_dir, "fig2_client_perclass")


# ============================================================
# Centralized baseline (supports multi-run + all datasets)
# ============================================================
def run_centralized_baseline(reserved_paths_json: str = "reserved_paths.json",
                              num_rounds: int = 50,
                              run_id: int = 1) -> dict:
    """
    Single-node centralized SNN baseline.
    Uses ALL samples/class (no client split).
    Per-dataset config determines shot counts and fixed-test size.
    Fair comparison to FL: same round count.
    """
    logger.info("=" * 65)
    logger.info("CENTRALIZED BASELINE (no FL)  run=%02d", run_id)
    logger.info("=" * 65)

    rp_path = Path(reserved_paths_json)
    if not rp_path.exists():
        raise FileNotFoundError("Not found: " + str(rp_path))
    with open(str(rp_path), "r") as f:
        reserved = json.load(f)
    classes      = sorted(list(reserved.keys()))
    dataset_name = detect_dataset(classes, rp_path)

    # Centralized uses the full sample pool (no client split)
    # but scales shots to match dataset config for fair comparison.
    base_cfg  = DATASET_CONFIGS.get(dataset_name, DATASET_CONFIGS["Warehouse"])
    C_FIXED   = base_cfg["fixed_test"] * 2          # double since no split
    C_TOTAL   = base_cfg["samples_total"]
    C_SHOTS   = base_cfg["shots_per_class"] * 2     # proportionally more shots
    C_TRAIN   = C_TOTAL - C_FIXED
    C_UNIQUE  = max(1, C_TRAIN // C_SHOTS)

    run_tag     = f"run{run_id:02d}"
    stamp       = datetime.now().strftime("%Y%m%d_%H%M")
    results_dir = Path(f"centralized_{dataset_name}_{run_tag}_{stamp}")
    results_dir.mkdir(exist_ok=True)

    random.seed(SEED + run_id)           # different seed per run
    np.random.seed(SEED + run_id)
    tf.random.set_seed(SEED + run_id)

    all_data   = {cls: sorted(reserved[cls])[:C_TOTAL] for cls in classes}
    fixed_test = {cls: all_data[cls][-C_FIXED:] for cls in classes}
    train_pool = {cls: all_data[cls][:-C_FIXED]  for cls in classes}

    logger.info("  Dataset     : %s", dataset_name)
    logger.info("  Fixed test  : %d/class", C_FIXED)
    logger.info("  Train pool  : %d/class", C_TRAIN)
    logger.info("  Shots/round : %d/class", C_SHOTS)
    logger.info("  Unique rds  : %d",       C_UNIQUE)
    logger.info("  Total rds   : %d",       num_rounds)

    devs = devices()
    fbz  = find_fbz()
    model, fan_in, nw = build_akida_model(fbz, devs, len(classes))
    logger.info("Model: fan_in=%d nw=%d", fan_in, nw)

    n_cls         = len(classes)
    round_metrics = []
    fixed_hist    = []

    for rnd in range(1, num_rounds + 1):
        t0    = time.time()
        epoch = ((rnd - 1) // C_UNIQUE) + 1
        eff   = ((rnd - 1) % C_UNIQUE) + 1
        start = (eff - 1) * C_SHOTS

        round_train = {}
        for cls in classes:
            pool = train_pool[cls]
            if epoch > 1:
                rng  = random.Random(SEED + run_id + epoch * 997 + hash(cls))
                pool = pool[:]
                rng.shuffle(pool)
            te = start + C_SHOTS
            round_train[cls] = (pool[start:te]
                                if te <= len(pool) else pool[:C_SHOTS])

        order = list(range(n_cls))
        random.shuffle(order)
        for idx in order:
            cls = classes[idx]
            for p in round_train[cls]:
                img = load_img_uint8(p)
                if img is not None:
                    try:
                        model.fit(np.expand_dims(img, 0), idx)
                    except Exception as e:
                        logger.error("centralized fit: %s", str(e))

        rt        = time.time() - t0
        pca_fixed = evaluate_pool(model, fixed_test, classes)
        mean_f    = float(np.mean(list(pca_fixed.values())))
        fixed_hist.append(mean_f)

        rd = {
            "round"          : rnd,
            "epoch"          : epoch,
            "round_time"     : rt,
            "fixed_test_acc" : mean_f,
            "per_class_fixed": pca_fixed,
        }
        round_metrics.append(rd)

        tag = f" [recycle ep{epoch}]" if epoch > 1 else ""
        print(f"  Round {rnd:3d}/{num_rounds} | "
              f"Fixed={mean_f:.1f}% | Time={rt:.2f}s{tag}")

    best_f  = max(rm["fixed_test_acc"] for rm in round_metrics)
    final_f = round_metrics[-1]["fixed_test_acc"]
    final_p = round_metrics[-1]["per_class_fixed"]

    results = {
        "experiment"               : "centralized_baseline",
        "run_id"                   : run_id,
        "dataset"                  : dataset_name,
        "classes"                  : classes,
        "num_rounds"               : num_rounds,
        "unique_rounds"            : C_UNIQUE,
        "shots_per_class"          : C_SHOTS,
        "fixed_test_per_class"     : C_FIXED,
        "best_fixed_test_accuracy" : best_f,
        "final_fixed_test_accuracy": final_f,
        "final_per_class_accuracy" : final_p,
        "total_time"               : float(
            sum(rm["round_time"] for rm in round_metrics)),
        "round_metrics"            : round_metrics,
        "fbz_loaded"               : fbz,
        "timestamp"                : datetime.now().isoformat(),
    }
    with open(str(results_dir / "centralized_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    try:
        model.save(str(results_dir / "centralized_final.fbz"))
    except Exception:
        pass

    # Plot
    set_pub_style()
    r  = np.arange(1, num_rounds + 1)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    fig.subplots_adjust(bottom=0.14, top=0.88, wspace=0.40)

    axes[0].plot(r, fixed_hist, color=PALETTE["accent"], lw=2.0,
                 label="Fixed-test acc")
    axes[0].axhline(y=best_f, color=PALETTE["accent"], lw=0.8,
                    linestyle=":", label=f"Best={best_f:.1f}%")
    if num_rounds > C_UNIQUE:
        axes[0].axvline(x=C_UNIQUE, color="gray", lw=1.0,
                        linestyle="--", alpha=0.6, label="Recycle")
    axes[0].set_xlabel("Round"); axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_xlim(0.5, num_rounds + 0.5); axes[0].set_ylim(0, 108)
    axes[0].set_title(f"(a) Convergence ({num_rounds} rounds)")
    axes[0].legend(fontsize=7)

    vals  = [final_p.get(c, 0.0) for c in classes]
    x     = np.arange(len(classes))
    short = [c.replace("_", "\n") for c in classes]
    bars  = axes[1].bar(x, vals,
                        color=CLASS_COLORS[:len(classes)],
                        edgecolor="white", linewidth=0.5, width=0.55)
    axes[1].axhline(y=float(np.mean(vals)), color=PALETTE["accent"],
                    lw=1.2, linestyle="--",
                    label=f"Mean={np.mean(vals):.1f}%")
    axes[1].set_xticks(x); axes[1].set_xticklabels(short, fontsize=7.5)
    axes[1].set_ylim(0, 120); axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("(b) Final per-class accuracy")
    axes[1].legend(fontsize=7)
    for bar, v in zip(bars, vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v + 1.5,
                     f"{v:.1f}%", ha="center", va="bottom", fontsize=7.5)

    fig.suptitle(
        f"Centralized baseline  [{dataset_name}]  run={run_id:02d}",
        fontsize=10)
    save_fig(fig, results_dir, "fig1_centralized_convergence")

    logger.info("Centralized run=%02d complete!", run_id)
    logger.info("  Best  fixed-test: %.1f%%", best_f)
    logger.info("  Final fixed-test: %.1f%%", final_f)
    logger.info("  Results -> %s", str(results_dir.resolve()))

    return {
        "run_id"               : run_id,
        "best_fixed_accuracy"  : best_f,
        "final_fixed_accuracy" : final_f,
        "final_val_accuracy"   : final_f,  # centralized has no round-val
        "per_class_fixed"      : final_p,
        "results_dir"          : str(results_dir),
    }


def aggregate_centralized_runs(run_results: list,
                                dataset_name: str,
                                results_base: Path) -> dict:
    """
    Compute mean ± std for centralized baseline across multiple runs.
    Mirrors the server-side aggregate_runs() function.
    """
    if not run_results:
        return {}

    keys = ["best_fixed_accuracy", "final_fixed_accuracy"]
    agg  = {}
    for k in keys:
        vals   = [r[k] for r in run_results if k in r]
        agg[k] = {
            "mean"  : float(np.mean(vals)),
            "std"   : float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0),
            "min"   : float(np.min(vals)),
            "max"   : float(np.max(vals)),
            "values": vals,
        }

    all_classes = list(run_results[0].get("per_class_fixed", {}).keys())
    per_class_agg = {}
    for c in all_classes:
        vals = [r["per_class_fixed"].get(c, 0.0)
                for r in run_results if "per_class_fixed" in r]
        per_class_agg[c] = {
            "mean"  : float(np.mean(vals)),
            "std"   : float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0),
            "values": vals,
        }

    summary = {
        "experiment"    : "centralized_baseline_aggregate",
        "dataset"       : dataset_name,
        "n_runs"        : len(run_results),
        "run_ids"       : [r["run_id"] for r in run_results],
        "aggregated"    : agg,
        "per_class_agg" : per_class_agg,
        "per_run"       : run_results,
        "timestamp"     : datetime.now().isoformat(),
    }

    out_json = results_base / f"aggregated_centralized_{dataset_name}.json"
    with open(str(out_json), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Centralized aggregated stats -> %s", out_json.name)
    return summary


# ============================================================
# FL client entry point
# ============================================================
def start_fl_client(client_id: str = None,
                    server_address: str = "10.0.5.2:8080",
                    reserved_paths: str = "reserved_paths.json",
                    run_id: int = 1) -> dict:
    cid = client_id or get_client_id()
    logger.info("Starting FL client %s -> %s  run=%02d",
                cid, server_address, run_id)
    try:
        client = NeuroEdgeClient(client_id=cid,
                                 reserved_paths_json=reserved_paths,
                                 run_id=run_id)
        fl.client.start_numpy_client(
            server_address=server_address, client=client)
        summary = client.save_client_results_and_plots()
        logger.info("Done! Results -> %s", str(client.results_dir.resolve()))
        return summary
    except Exception as e:
        logger.error("Client error: %s", str(e))
        return {"run_id": run_id, "best_fixed_accuracy": 0.0,
                "final_fixed_accuracy": 0.0}


# ============================================================
# Entry point
# ============================================================
def main():
    import argparse
    p = argparse.ArgumentParser(
        description="SNN-NeuroEdge FL Client v2 — Multi-dataset & Multi-run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # FL mode — Warehouse (auto-detects reserved_paths.json in CWD)
  python3 snn_neuroedge_client_v2.py --client_id SAI_Node_7 --server_address 10.0.5.2:8080

  # FL mode — MNIST-5class
  python3 snn_neuroedge_client_v2.py --client_id SAI_Node_7 \\
      --reserved_paths /home/sai/datasets/MNIST_5class/reserved_paths.json

  # FL mode — CIFAR10-5class, run 3
  python3 snn_neuroedge_client_v2.py --client_id SAI_Node_8 \\
      --reserved_paths /home/sai/datasets/CIFAR10_5class/reserved_paths.json \\
      --run_id 3

  # Centralized baseline — Warehouse, 50 rounds, 5 runs for statistics
  python3 snn_neuroedge_client_v2.py --mode centralized --max_rounds 50 --n_runs 5

  # Centralized baseline — MNIST-5class, single run
  python3 snn_neuroedge_client_v2.py --mode centralized \\
      --reserved_paths /home/sai/datasets/MNIST_5class/reserved_paths.json

  # Centralized baseline — CIFAR10-5class, run 2
  python3 snn_neuroedge_client_v2.py --mode centralized \\
      --reserved_paths /home/sai/datasets/CIFAR10_5class/reserved_paths.json \\
      --run_id 2 --max_rounds 50

  # All three datasets centralized, 3 runs each
  for ds_path in reserved_paths.json \\
      /home/sai/datasets/MNIST_5class/reserved_paths.json \\
      /home/sai/datasets/CIFAR10_5class/reserved_paths.json; do
    python3 snn_neuroedge_client_v2.py --mode centralized \\
        --reserved_paths $ds_path --max_rounds 50 --n_runs 3
  done
        """
    )
    p.add_argument("--mode", type=str, default="federated",
                   choices=["federated", "centralized"],
                   help="federated (FL) or centralized baseline (default: federated)")
    p.add_argument("--client_id", type=str, default=None,
                   help="Client ID (default: auto-detect from hostname)")
    p.add_argument("--server_address", type=str, default="10.0.5.2:8080",
                   help="Server IP:port (default: 10.0.5.2:8080)")
    p.add_argument("--reserved_paths", type=str,
                   default="reserved_paths.json",
                   help="Path to reserved_paths.json (default: reserved_paths.json)")
    p.add_argument("--max_rounds", type=int, default=50,
                   help="Rounds for centralized baseline (default: 50)")
    p.add_argument("--n_runs", type=int, default=1,
                   help="Number of repeated runs (centralized mode, default: 1)")
    p.add_argument("--run_id", type=int, default=None,
                   help="Run a specific run number (overrides --n_runs, default: 1)")
    a = p.parse_args()

    if a.mode == "centralized":
        results_base = Path("Results")
        results_base.mkdir(exist_ok=True)

        # --run_id: just one specific run
        if a.run_id is not None:
            r = run_centralized_baseline(
                a.reserved_paths, a.max_rounds, run_id=a.run_id)
            print(f"\nRun {a.run_id} complete.")
            print(f"  Best fixed-test  : {r['best_fixed_accuracy']:.1f}%")
            print(f"  Final fixed-test : {r['final_fixed_accuracy']:.1f}%")
            print(f"  Results          : {r['results_dir']}")
            return

        # Multiple runs
        n_runs      = max(1, a.n_runs)
        run_results = []
        for rid in range(1, n_runs + 1):
            print(f"\n{'#'*65}")
            print(f"#  Centralized run {rid}/{n_runs}")
            print(f"{'#'*65}")
            r = run_centralized_baseline(
                a.reserved_paths, a.max_rounds, run_id=rid)
            run_results.append(r)
            print(f"\n  Run {rid} best fixed-test: "
                  f"{r['best_fixed_accuracy']:.1f}%")

        if n_runs > 1:
            # Detect dataset name for aggregation label
            from pathlib import Path as _P
            try:
                with open(a.reserved_paths) as fp:
                    keys   = sorted(json.load(fp).keys())
                ds_name = detect_dataset(keys, a.reserved_paths)
            except Exception:
                ds_name = "Unknown"

            agg = aggregate_centralized_runs(run_results, ds_name, results_base)
            print(f"\n{'='*65}")
            print(f"  Centralized [{ds_name}]  —  {n_runs} runs")
            ba = agg["aggregated"]["best_fixed_accuracy"]
            fa = agg["aggregated"]["final_fixed_accuracy"]
            print(f"  Best fixed-test  : {ba['mean']:.1f}% ± {ba['std']:.1f}%")
            print(f"  Final fixed-test : {fa['mean']:.1f}% ± {fa['std']:.1f}%")
            print("  Per-class (final round mean ± std):")
            for c, s in agg["per_class_agg"].items():
                print(f"    {c.ljust(22)}: {s['mean']:.1f}% ± {s['std']:.1f}%")
            print(f"{'='*65}")

    else:  # federated
        rid = a.run_id if a.run_id is not None else 1
        summary = start_fl_client(
            client_id      = a.client_id,
            server_address = a.server_address,
            reserved_paths = a.reserved_paths,
            run_id         = rid,
        )
        print(f"\nFL client run={rid} complete.")
        print(f"  Best fixed-test  : {summary.get('best_fixed_accuracy', 0):.1f}%")
        print(f"  Results          : {summary.get('results_dir', 'N/A')}")


if __name__ == "__main__":
    main()
