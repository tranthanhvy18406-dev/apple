# Cross-Stitch v2 Experiment Summary

This directory contains the cleaned code and result bundle for the RR 1-second FullBuffer experiments.

## Dataset

The dataset is not included in this repository. The local experiment used:

```text
/8T2/xty/code/cross_stitch_1s_datasets/ul_rr_1s_cdlc_10000s
```

Metadata:

- Sampling interval: 1 second
- Train length: 9000 samples
- Test length: 1000 samples
- Scheduling: RR
- Traffic: FullBuffer
- Channel: CDL-C

## Main Code

- `code/Train/run_single_dataset_experiment.py`: PRNet training entry.
- `code/Train/mtl_model.py`: Cross-Stitch PRNet model.
- `code/Train/mtl_dataset.py`: RB/SINR sequence dataset.
- `code/evaluate_horizon_throughput.py`: throughput horizon evaluation with Last Value, RF-Rate, and PRNet.
- `code/analyze_sinr_horizon.py`: PRNet SINR horizon diagnostics.
- `code/train_lstm_sinr_baseline.py`: standalone SINR-only LSTM baseline and hyperparameter sweep.

## Result Directories

- `rr_1s_fullbuffer_horizon/`: throughput horizon MAE/MSE results and figures.
- `rr_1s_fullbuffer_sinr/`: PRNet SINR horizon diagnostics.
- `lstm_sinr_baseline/`: quick SINR-only LSTM, past 16s to future 1..10s.
- `lstm_sinr_baseline_past5_future3/`: quick SINR-only LSTM, past 5s to future 1..3s.
- `lstm_sinr_wide_past5_future3/`: wide LSTM sweep, past 5s to future 1..3s.
- `lstm_sinr_wide_past16_future10/`: wide LSTM sweep, past 16s to future 1..10s.

## Throughput Horizon Results

The throughput horizon experiment uses:

```text
past 16 seconds -> future 1..10 seconds
```

Important output files:

- `rr_1s_fullbuffer_horizon/horizon_throughput_metrics.csv`
- `rr_1s_fullbuffer_horizon/horizon_throughput_metrics.json`
- `rr_1s_fullbuffer_horizon/throughput_mae_by_horizon.png`
- `rr_1s_fullbuffer_horizon/throughput_mse_by_horizon.png`

## SINR LSTM Diagnostics

The SINR-only LSTM experiments were added to test whether the 1-second SINR sequence itself contains learnable temporal information.

### Wide sweep: past 5s -> future 1..3s

Output directory:

```text
lstm_sinr_wide_past5_future3/
```

Best configuration:

- hidden dimension: 128
- LSTM layers: 3
- dropout: 0.1
- learning rate: 0.001
- weight decay: 1e-5
- residual: false
- loss: SmoothL1

Average SINR metrics over h=1..3:

| Method | MAE | MSE | Correlation | Pred. Std. | True Std. |
|---|---:|---:|---:|---:|---:|
| Last SINR | 1.7854 | 5.7019 | -0.0053 | 1.6839 | 1.6841 |
| History Mean | 1.3697 | 3.3235 | 0.0235 | 0.7388 | 1.6841 |
| Train Mean | 1.2426 | 2.8371 | N/A | 0.0000 | 1.6841 |
| LSTM | 1.2167 | 2.8826 | 0.0085 | 0.1018 | 1.6841 |

### Wide sweep: past 16s -> future 1..10s

Output directory:

```text
lstm_sinr_wide_past16_future10/
```

Best configuration:

- hidden dimension: 64
- LSTM layers: 2
- dropout: 0.1
- learning rate: 0.001
- weight decay: 1e-4
- residual: false
- loss: SmoothL1

Average SINR metrics over h=1..10:

| Method | MAE | MSE | Correlation | Pred. Std. | True Std. |
|---|---:|---:|---:|---:|---:|
| Last SINR | 1.7714 | 5.6221 | 0.0004 | 1.6701 | 1.6838 |
| History Mean | 1.2994 | 3.0644 | -0.0387 | 0.4178 | 1.6838 |
| Train Mean | 1.2431 | 2.8367 | N/A | 0.0000 | 1.6838 |
| LSTM | 1.2137 | 2.8643 | 0.0262 | 0.0490 | 1.6838 |

## Interpretation

The tuned LSTM slightly improves SINR MAE compared with a train-set mean, but the predicted standard deviation remains much smaller than the true standard deviation. This indicates that the LSTM mostly learns a near-mean predictor and does not reliably track 1-second SINR fluctuations in this RR FullBuffer dataset.

This supports the diagnosis that the 1-second SINR prediction problem has weak temporal predictability under the current generated data setting, rather than being only a PRNet/Cross-Stitch implementation issue.

## Large Artifacts

The RF-Rate `.joblib` files are larger than GitHub's normal 100 MB file limit and are tracked with Git LFS:

- `model/rf_rate_model_rate_only_multi_horizon_raw_h16_p10.joblib`
- `rr_1s_fullbuffer_horizon/rf_rate_model_rate_only_multi_horizon_raw_h16_p10.joblib`

