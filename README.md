# SNN-NeuroEdge

**Federated Continual Learning on Neuromorphic Edge**
Aftab Hussain, Alois Ferscha — Institute of Pervasive Computing, JKU Linz

This repository contains the complete code, the converted spiking model, and the
result folders behind the paper *SNN-NeuroEdge: Federated Continual Learning on
Neuromorphic Edge* (ICANN 2026). It reproduces the full pipeline: CNN training →
8/4/4-bit quantization-aware compression → CNN-to-SNN conversion → federated
incremental adaptation on BrainChip Akida hardware.

> **Reproducibility at a glance**
> - **Phase 1** (train / quantize / convert) runs **end-to-end on a free GPU** —
>   one click via the Colab badge below. No special hardware needed.
> - **Phase 2** (federated edge learning) runs on **physical Akida AKD1000 nodes**.
>   If you do not have Akida hardware, the same `.fbz` model runs on the **Akida
>   software backend** (CPU), so the federated pipeline is still reproducible —
>   only the on-chip power/timing numbers require the physical device.

[![Open In Colab] https://colab.research.google.com/drive/1f3sjRuNkAcwQFdbFbyrPISaOX5_qdqep?usp=sharing

---

## Repository layout

```
snn-neuroedge/
├── notebooks/
│   └── phase1_train_quantize_convert.ipynb   # one-click Phase 1 (Colab-ready)
├── phase2_federated/
│   ├── snn_neuroedge_server_v1.py            # Flower FedAvg server
│   ├── snn_neuroedge_client_v2.py            # Akida edge-learning client (x2)
│   ├── aggregate_results.py                  # builds paper figures + LaTeX table
│   └── snn_neuroedge_plotter.py
├── models/
│   └── warehouse_edge_backbone_full_v1.fbz   # converted SNN backbone (1.91 MB)
├── results/                                  # exact run folders behind the paper
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/USER/snn-neuroedge.git
cd snn-neuroedge
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested with **Python 3.12**. Phase 1 uses TensorFlow/Keras and BrainChip's
`quantizeml` / `cnn2snn` / `akida` packages; Phase 2 adds `flwr` (Flower).

---

## Dataset

| Dataset | How to get it | Notes |
|---|---|---|
| **Fraunhofer IIS Warehouse** | See https://www.kaggle.com/datasets/aftabhussaincui/warehouse-objects-dataset [original source](#) (Löffler et al., 2018) | Subset of 13,187 images, 5 scene classes. Resized to 160×160 RGB. **Not redistributed here** — obtain from the source per its license. |
| **MNIST-5 / CIFAR-10-5** | Downloaded automatically | First 5 classes only; `tensorflow.keras.datasets` fetches them. |

The Phase 1 notebook downloads the warehouse subset from Kaggle automatically
(set your Kaggle API token as described in the notebook's first cell), or you can
point it at a local copy.

---

## Phase 1 — Train, Quantize, Convert  (no special hardware)

**One click:** open the Colab badge above and run all cells. It will:

1. Download the warehouse dataset (Kaggle) and the MNIST/CIFAR benchmarks.
2. Train the 20-layer lightweight CNN (160×160 input, depthwise-separable blocks).
3. Apply 8/4/4-bit quantization-aware compression (`cnn2snn`):
   8-bit input, 4-bit weights/activations, 1-bit spike layer, 2-bit logits.
4. Convert to an Akida-compatible SNN and export `warehouse_edge_backbone_full_v1.fbz`.
5. Print the pipeline summary (Float 91.1% → Quant 90.4% → Akida SNN 92.0%;
   19.61 MB → 6.53 MB → 1.91 MB, 10.3× compression).

Locally instead of Colab:

```bash
jupyter notebook notebooks/phase1_train_quantize_convert.ipynb
```

The exported `.fbz` is the input to Phase 2. A pre-converted copy is in `models/`
so Phase 2 can be run without redoing Phase 1.

---

## Phase 2 — Federated Neuromorphic Edge Learning

Two clients + one server, FedAvg over 50 rounds. Each client replaces the
backbone's final layer with an Akida edge-learning layer and adapts on-device
using the chip's native **unsupervised** rule (one labeled sample at a time, no
backpropagation).

### With Akida hardware (the paper's testbed)

On the **server** node:

```bash
cd phase2_federated
python snn_neuroedge_server_v1.py --rounds 50 --dataset Warehouse
```

On **each client** node (run identically on both):

```bash
cd phase2_federated
python snn_neuroedge_client_v2.py \
    --server <SERVER_IP>:8080 \
    --model ../models/warehouse_edge_backbone_full_v1.fbz \
    --dataset Warehouse \
    --node SAI_Node_7        # use SAI_Node_8 on the second client
```

Repeat with `--dataset MNIST_5class` and `--dataset CIFAR10_5class` for the
cross-dataset results.

### Without Akida hardware (software backend)

The client auto-detects hardware: if `akida.devices()` returns nothing, the same
model executes on the Akida software backend (CPU). The accuracy results are
reproducible this way; only the measured **power** and **per-round latency**
numbers require the physical AKD1000.

---

## Reproducing the paper figures and table

The exact run folders behind the reported numbers are in `results/`. To rebuild
the convergence figure, per-class figure, and the LaTeX table:

```bash
cd phase2_federated
python aggregate_results.py --results_dir ../results --min_rounds 45
```

This regenerates `fig_convergence_cross_dataset`, `fig_perclass_all_datasets`
(both EPS + PNG, uniform fonts), and prints the cross-dataset LaTeX table.

---

## Citation

If you use this code or build on it, please cite:

```bibtex
@inproceedings{hussain2026snnneuroedge,
  title     = {SNN-NeuroEdge: Federated Continual Learning on Neuromorphic Edge},
  author    = {Hussain, Aftab and Ferscha, Alois},
  booktitle = {Proc. International Conference on Artificial Neural Networks (ICANN)},
  year      = {2026}
}
```

## License

Released under the MIT License (see `LICENSE`). The warehouse dataset is **not**
covered by this license; refer to its original terms.
