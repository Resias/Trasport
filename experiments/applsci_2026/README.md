# Appl. Sci. 2026 Experiment

This folder is the canonical map for reproducing the Appl. Sci. 2026 metro OD forecasting experiment.
It does not replace the existing training code yet. Instead, it gathers the paper-specific settings,
preservation manifests, and wrapper scripts in one place.

## Model Mapping

The paper's `Metro-GATF` corresponds to the current repository implementation:

- Model: `models/GATTransformerdecoder.py::GATTransformerODWeek`
- Trainer: `train/train_graph.py`
- Lightning module: `train/trainer.py::MetroGraphWeekLM`
- Local run family: `metro-GATrasformer-od-week-latlon/`

The run directory has a historical spelling mismatch. Do not rename it until checkpoint references
and manifests are migrated.

## Paper Settings

- Data period: 2022-05-01 through 2022-05-31
- Network: 637 Seoul metropolitan subway stations
- Time resolution: 1 minute
- Operating hours: 05:30 to 24:00
- Input window: 60 minutes
- Prediction horizon: 30 minutes
- Hop size: 10 minutes
- Train split: 2022-05-01 through 2022-05-19
- Validation split: 2022-05-20 through 2022-05-24
- Test split: 2022-05-25 through 2022-05-31
- Transform: `log1p(x)` during training, inverse transform for evaluation
- Metrics: MAE, RMSE, sMAPE
- sMAPE epsilon: `1e-3`
- Final Metro-GATF gate threshold: `0.9`

## Preserved Models

Main full-matrix models:

- HA
- ARIMA
- GCN-LSTM
- Autoformer
- ODFormer
- MPGCN
- Metro-GATF

Supplementary pair-level model:

- ST-LSTM for OD pairs `(16, 523)` and `(10, 25)`

Not in the main paper comparison:

- ST-DAMHGN
- ODMixer
- HIAM
- DT-HGN

## Local Assets

Large datasets, checkpoints, and generated results remain local-only and ignored by git. The manifests
under `manifests/` record what should be preserved before any cleanup.

Important manifests:

- `manifests/data_manifest.md`
- `manifests/checkpoint_manifest.md`
- `manifests/result_manifest.md`
- `manifests/cleanup_classification.md`

## Wrapper Scripts

All scripts are environment-variable based and avoid hardcoded `/root/tmp/Trasport` paths.

Train Metro-GATF:

```bash
experiments/applsci_2026/scripts/train_metro_gatf.sh
```

Train one baseline:

```bash
MODEL=mpgcn experiments/applsci_2026/scripts/train_baseline.sh
MODEL=gcn_lstm experiments/applsci_2026/scripts/train_baseline.sh
MODEL=autoformer experiments/applsci_2026/scripts/train_baseline.sh
MODEL=odformer experiments/applsci_2026/scripts/train_baseline.sh
MODEL=st_lstm TARGET_S=16 TARGET_E=523 experiments/applsci_2026/scripts/train_baseline.sh
```

Run statistical baselines:

```bash
experiments/applsci_2026/scripts/evaluate_statistical_baselines.sh
```

Run full-matrix checkpoint evaluation:

```bash
GNN_CKPT=/path/to/metro_gatf.ckpt \
MPGCN_CKPT=/path/to/mpgcn.ckpt \
experiments/applsci_2026/scripts/evaluate_full_matrix.sh
```

Run gate-threshold sensitivity:

```bash
CKPT_GLOB='ablation_runs/progressive_core_v1/S8_gate/seed_*/checkpoints/best-*.ckpt' \
experiments/applsci_2026/scripts/evaluate_gate_sensitivity.sh
```

Run efficiency evaluation:

```bash
experiments/applsci_2026/scripts/evaluate_efficiency.sh
```

Run progressive ablation:

```bash
experiments/applsci_2026/scripts/run_progressive_ablation.sh
```

## Cleanup Rule

Do not delete local outputs until the relevant item is classified in `manifests/cleanup_classification.md`.
The first refactoring objective is discoverability and preservation, not destructive cleanup.
