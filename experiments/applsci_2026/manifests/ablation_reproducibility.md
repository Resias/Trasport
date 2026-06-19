# Ablation Reproducibility Review

Date: 2026-06-19

Purpose:

- Confirm whether the existing ablation outputs can remain reproducible before renaming
  `train/train_abligation.py` to `train/train_ablation.py`.

## Reviewed Outputs

Reviewed directory:

- `ablation_runs/progressive_core_v1/`

Observed progressive stages:

- `S0_minimal`
- `S1_factorization`
- `S2_multiscale_static`
- `S3_dynamic`
- `S4_transformer`
- `S5_time`
- `S6_weekday`
- `S7_geo`
- `S8_gate`

Observed seeds:

- `0`
- `1`
- `2`
- `3`
- `4`

Summary:

- Expected run count from observed stage/seed grid: 45
- Runs with `config.json`: 45
- Runs with `run_result.json`: 45
- Runs whose `best_checkpoint` exists on disk: 45
- Discovered checkpoint files under `progressive_core_v1`: 92

Representative full model result:

- Stage: `S8_gate`
- Seed: `0`
- Best checkpoint: `ablation_runs/progressive_core_v1/S8_gate/seed_0/checkpoints/best-029.ckpt`
- MAE: `0.01693837502993043`
- RMSE: `0.1637158778282071`
- sMAPE: `2.662081347323203`

## Compatibility Decision

The ablation entrypoint can be renamed safely if the historical path remains available as a wrapper.

Applied policy:

- Canonical entrypoint: `train/train_ablation.py`
- Compatibility wrapper: `train/train_abligation.py`
- Historical W&B project in saved configs: `transport-abligation`
- New default W&B project: `transport-ablation`
- Existing result files are not rewritten.
- Existing shell scripts may use either path.

## Reproduction Notes

The saved configs include absolute paths from the original machine state:

- `/home/data/od_minute`
- `/root/tmp/Trasport/AD_matrix_trimmed_common.csv`
- `/root/tmp/Trasport/ad_station_latlon.csv`
- `/root/tmp/Trasport/ablation_runs`

The wrapper scripts under `experiments/applsci_2026/scripts/` prefer repository-relative paths,
but exact reproduction of historical runs should preserve the original data source or recreate
equivalent symlinks under `data_splits/od_minute_review_3way/`.

## Minimal Verification Commands

```bash
python train/train_ablation.py --help
python train/train_abligation.py --help
bash -n experiments/applsci_2026/scripts/run_progressive_ablation.sh
```
