# RR 1s FullBuffer Reproduction Code

This folder collects the original code files used to generate:

- `/4T/xty/Cross_Stitch/paper/rr_1s_fullbuffer_horizon`
- `/4T/xty/Cross_Stitch/paper/rr_1s_fullbuffer_sinr`

The dataset itself is not copied here. The experiment reads:

`/8T2/xty/code/cross_stitch_1s_datasets/ul_rr_1s_cdlc_10000s`

This is the 1-second-sampling RR FullBuffer dataset with:

- `train_9000_HDF5.pkl`
- `test_1000_HDF5.pkl`
- `metadata.json`

## Included Files

Code:

- `code/Train/run_single_dataset_experiment.py`: PRNet training entry.
- `code/Train/mtl_model.py`: Cross-Stitch PRNet model definition.
- `code/Train/mtl_dataset.py`: RB/SINR sequence dataset.
- `code/Train/train_direct_rate_gru.py`: protocol throughput helper used by horizon evaluation.
- `code/evaluate_horizon_throughput.py`: throughput MAE/MSE horizon evaluation and plots.
- `code/analyze_sinr_horizon.py`: SINR MAE/MSE horizon evaluation and plots.

Model:

- `model/best_mtl_model_decoder.pth`: trained RR 1s PRNet checkpoint.
- `model/rf_rate_model_rate_only_multi_horizon_raw_h16_p10.joblib`: trained RF-Rate baseline model.
- `model/metrics.json`: training/evaluation metrics.
- `model/history.json`: training history.

## Commands

Run from `/4T/xty/Cross_Stitch`.

Train PRNet:

```bash
/4T/xty/miniconda3/envs/bel-gpu/bin/python Train/run_single_dataset_experiment.py \
  --dataset_dir /8T2/xty/code/cross_stitch_1s_datasets/ul_rr_1s_cdlc_10000s \
  --output_root /4T/xty/Cross_Stitch/experiments_1s_rr_fullbuffer \
  --device cuda:0 \
  --epochs 80 --patience 10 \
  --batch_size 128 --eval_batch_size 128 \
  --hist_window 16 --pre_window 10 --skip_frames 0 \
  --lr 1e-4 \
  --sinr_loss huber --huber_beta 0.05 \
  --gamma_rb_dist 0.1 --gamma_rate_proxy 0.2 \
  --tf_ratio_start 0.5 --tf_ratio_end 0.05 --tf_decay_epochs 30 \
  --selection_metric throughput_mae \
  --force
```

Generate throughput horizon results:

```bash
/4T/xty/miniconda3/envs/bel-gpu/bin/python evaluate_horizon_throughput.py \
  --dataset_dir /8T2/xty/code/cross_stitch_1s_datasets/ul_rr_1s_cdlc_10000s \
  --prnet_model_path /4T/xty/Cross_Stitch/experiments_1s_rr_fullbuffer/ul_rr_1s_cdlc_10000s/best_mtl_model_decoder.pth \
  --output_dir /4T/xty/Cross_Stitch/paper/rr_1s_fullbuffer_horizon \
  --hist_window 16 --max_horizon 10 --skip_frames 0 \
  --target_mode raw \
  --last_value_mode last \
  --rf_feature_mode rate_only \
  --rf_output_mode multi_horizon \
  --batch_size 256 \
  --device cuda:0 \
  --force_rf
```

Generate SINR horizon results:

```bash
/4T/xty/miniconda3/envs/bel-gpu/bin/python analyze_sinr_horizon.py \
  --dataset_dir /8T2/xty/code/cross_stitch_1s_datasets/ul_rr_1s_cdlc_10000s \
  --model_path /4T/xty/Cross_Stitch/experiments_1s_rr_fullbuffer/ul_rr_1s_cdlc_10000s/best_mtl_model_decoder.pth \
  --output_dir /4T/xty/Cross_Stitch/paper/rr_1s_fullbuffer_sinr \
  --hist_window 16 --max_horizon 10 \
  --batch_size 256 \
  --device cuda:0
```

## Notes

- The prediction setting is `past 16 seconds -> future 1..10 seconds`, because this dataset is already sampled at 1 Hz.
- The original MATLAB generator used RR scheduling through `getUserPRBSet.m`.
- In this RR dataset, user-2 RB is constant at 11, so the RB task is trivial and most of the diagnostic value comes from SINR/throughput behavior.
