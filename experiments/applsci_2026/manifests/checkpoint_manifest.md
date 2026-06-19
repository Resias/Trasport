# Checkpoint Preservation Manifest

Date: 2026-06-19

Policy:

- Keep these checkpoints local and ignored by git.
- Do not delete or rename them until final paper artifact selection is confirmed.
- Final paper checkpoint basis: `review_runs/review_3way_50ep_20260512/`.
- Standalone `metro-*` checkpoints are legacy preserve candidates and should be archived only after
  the review-run checkpoints and summaries are verified.

## Legacy Standalone Metro-GATF Candidates

| Size bytes | SHA-256 | Path | Status |
| ---: | --- | --- | --- |
| 17661743 | `0cedf1a74c34b3489b73a3871b5443ce8ebb0e6f008d713dfbc5568dfe9f969a` | `metro-GATrasformer-od-week-latlon/3zcbyb9y/checkpoints/epoch=16-step=20808.ckpt` | archive-after-final-verification |
| 17661743 | `2d571f0fea0ecd122784e896d809f393453a90522685727e911bb90d901a6170` | `metro-GATrasformer-od-week-latlon/5l2xoh3u/checkpoints/epoch=87-step=107712.ckpt` | archive-after-final-verification |
| 17661743 | `cd3f209ee54de1676d55420c762be8c34006be70d40f35d4be7ff33576833353` | `metro-GATrasformer-od-week-latlon/cvzrix56/checkpoints/epoch=172-step=211752.ckpt` | archive-after-final-verification |
| 5241583 | `b5004b83ea5056b7aa801c300d5b35668c87597f86ec2a092b52ada10b483d88` | `metro-GATrasformer-od-week-latlon/e5c1rocs/checkpoints/epoch=99-step=61200.ckpt` | archive-after-final-verification |
| 17661743 | `14d29890568562a18ee9150865e24eb6fc7e904c5f5ea36add53aedc8b9505d1` | `metro-GATrasformer-od-week-latlon/rnc2c6pf/checkpoints/epoch=37-step=46512.ckpt` | archive-after-final-verification |
| 17661743 | `8eddfe9b8df946eb8285bcd21f7f3b1e083f34a331eda203de91ec0ef6841945` | `metro-GATrasformer-od-week-latlon/waghxfkn/checkpoints/epoch=13-step=17136.ckpt` | archive-after-final-verification |

## Legacy Standalone Full-Matrix Baselines

| Model | Size bytes | SHA-256 | Path | Status |
| --- | ---: | --- | --- | --- |
| Autoformer | 692512293 | `a894d8b997a692a0b8df508d769e845b483ba3c4e4bc34abdc6ec8ee7e414ae4` | `metro-AutoformerOD/1z58kfps/checkpoints/epoch=199-step=15400.ckpt` | archive-after-final-verification |
| Autoformer | 692512293 | `121be1e3b6f1eebf3fa908c983cdfa7b6bf171bf1d2f278f66ba8ef088d415f2` | `metro-AutoformerOD/zatq32lv/checkpoints/epoch=199-step=15400.ckpt` | archive-after-final-verification |
| GCN-LSTM | 577622 | `b1d8767b31154617d20f74ef18ace38d2c12599fa67fd2ffcf17e96a6284fa09` | `metro-GCN-LSTM/b5yp9uky/checkpoints/epoch=199-step=15400.ckpt` | archive-after-final-verification |
| GCN-LSTM | 577622 | `62f9a96b3a66895f36ba1b4b1d525a12070b818431fa3d727d342f8f24992162` | `metro-GCN-LSTM/rr0b4isj/checkpoints/epoch=199-step=15400.ckpt` | archive-after-final-verification |
| MPGCN | 105346 | `992903a8d8aa6d1817e48ebe2d6f9294a4a981fc94ea4d8e8f63c4771f0d065f` | `metro-MPGCN/szxzk5b4/checkpoints/epoch=199-step=489600.ckpt` | archive-after-final-verification |
| MPGCN | 105346 | `987ccf39277adec0439557cd211b3eb17edb0bb733e7acf9e6e8ba8eed622380` | `metro-MPGCN/ue07bxgd/checkpoints/epoch=176-step=433296.ckpt` | archive-after-final-verification |

## Pair-Level ST-LSTM

| Size bytes | SHA-256 | Path | Status |
| ---: | --- | --- | --- |
| 676069 | `3866e282984ba54848f4a80931765f7b3a0b15e8b7543e98377d8105ff155f8a` | `metro-st-lstm/jl1qx95r/checkpoints/epoch=299-step=57000.ckpt` | preserve-candidate |
| 673424 | `e4550031c0eb9a3adc63a0de29f4a984c20da8e13e9d8fd8362e81b9b6f9bed1` | `metro-st-lstm/qac9npmp/checkpoints/epoch=299-step=28500.ckpt` | preserve-candidate |

## Review-Run Best Checkpoints

These appear to be the curated reviewer rerun checkpoints used by later summary/evaluation scripts.
They are preserved by directory classification. SHA-256 values are intentionally deferred for the
largest ODFormer files and should be computed before any deletion or external archival.

| Model | Seed | Size bytes | Path | Status |
| --- | ---: | ---: | --- | --- |
| S8_gate / Metro-GATF | 0 | 17661935 | `review_runs/review_3way_50ep_20260512/S8_gate/seed_0/checkpoints/best-025.ckpt` | preserve-final-candidate |
| S8_gate / Metro-GATF | 1 | 17661935 | `review_runs/review_3way_50ep_20260512/S8_gate/seed_1/checkpoints/best-046.ckpt` | preserve-final-candidate |
| S8_gate / Metro-GATF | 2 | 17661935 | `review_runs/review_3way_50ep_20260512/S8_gate/seed_2/checkpoints/best-048.ckpt` | preserve-final-candidate |
| Autoformer | 0 | 692512485 | `review_runs/review_3way_50ep_20260512/Autoformer/seed_0/checkpoints/best-001.ckpt` | preserve-final-candidate |
| Autoformer | 1 | 692512485 | `review_runs/review_3way_50ep_20260512/Autoformer/seed_1/checkpoints/best-001.ckpt` | preserve-final-candidate |
| Autoformer | 2 | 692512485 | `review_runs/review_3way_50ep_20260512/Autoformer/seed_2/checkpoints/best-001.ckpt` | preserve-final-candidate |
| GCN-LSTM | 0 | 577878 | `review_runs/review_3way_50ep_20260512/GCN_LSTM/seed_0/checkpoints/best-048.ckpt` | preserve-final-candidate |
| GCN-LSTM | 1 | 577878 | `review_runs/review_3way_50ep_20260512/GCN_LSTM/seed_1/checkpoints/best-044.ckpt` | preserve-final-candidate |
| GCN-LSTM | 2 | 577878 | `review_runs/review_3way_50ep_20260512/GCN_LSTM/seed_2/checkpoints/best-043.ckpt` | preserve-final-candidate |
| MPGCN | 0 | 105474 | `review_runs/review_3way_50ep_20260512/MPGCN/seed_0/checkpoints/best-000.ckpt` | preserve-final-candidate |
| MPGCN | 1 | 105602 | `review_runs/review_3way_50ep_20260512/MPGCN/seed_1/checkpoints/best-030.ckpt` | preserve-final-candidate |
| MPGCN | 2 | 105602 | `review_runs/review_3way_50ep_20260512/MPGCN/seed_2/checkpoints/best-007.ckpt` | preserve-final-candidate |
| ODFormer | 0 | 3169203223 | `review_runs/review_3way_50ep_20260512/ODFormer/seed_0/checkpoints/best-003.ckpt` | preserve-final-candidate |
| ODFormer | 1 | 3169203223 | `review_runs/review_3way_50ep_20260512/ODFormer/seed_1/checkpoints/best-005.ckpt` | preserve-final-candidate |
| ODFormer | 2 | 3169203223 | `review_runs/review_3way_50ep_20260512/ODFormer/seed_2/checkpoints/best-003.ckpt` | preserve-final-candidate |

## Selection Needed

The next cleanup pass should map each checkpoint to:

- paper table or supplementary experiment
- seed
- exact run config
- best validation metric
- final test metric

Only then should duplicate or superseded checkpoints be archived or deleted.
