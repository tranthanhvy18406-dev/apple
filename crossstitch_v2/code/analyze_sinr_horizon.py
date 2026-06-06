import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Train"))

from mtl_dataset import MTL_Dataset
from mtl_model import CrossStitch_MTL_Model


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze horizon-wise SINR prediction for PRNet.")
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--hist_window", type=int, default=16)
    parser.add_argument("--max_horizon", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def flatten_series(payload, key):
    aliases = {
        "SINR": ["SINR", "sinr", "Sinr", "SNR", "snr", "eff_SINR", "eff_sinr"],
        "RB": ["RB", "rb", "Rb", "rB"],
    }
    if isinstance(payload, dict):
        data = None
        for candidate in aliases.get(key, [key]):
            if candidate in payload:
                data = payload[candidate]
                break
        if data is None:
            raise KeyError(f"Missing {key} in {list(payload.keys())}")
    else:
        data = payload

    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    return arr.reshape(-1)


def make_windows(series, hist_window, horizon):
    samples = len(series) - hist_window - horizon + 1
    if samples <= 0:
        raise ValueError("Not enough samples for the requested windows.")
    starts = np.arange(samples)
    hist_idx = starts[:, None] + np.arange(hist_window)[None, :]
    target_idx = starts[:, None] + hist_window + np.arange(horizon)[None, :]
    return series[hist_idx], series[target_idx]


def build_model(horizon, device):
    return CrossStitch_MTL_Model(
        rb_input_size=1,
        sinr_input_size=1,
        rb_hidden_size=256,
        sinr_hidden_size=64,
        rb_num_layers=3,
        sinr_num_layers=1,
        rb_output_size=106,
        sinr_output_size=1,
        pre_window=horizon,
        device=device,
        cross_stitch_mode="learn",
    ).to(device)


@torch.no_grad()
def predict_prnet(args, train_dataset):
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    test_dataset = MTL_Dataset(
        str(args.dataset_dir / "test_1000_HDF5.pkl"),
        str(args.dataset_dir / "test_1000_HDF5.pkl"),
        obs_window=args.hist_window,
        pre_window=args.max_horizon,
        sinr_mean=train_dataset.sinr_mean,
        sinr_std=train_dataset.sinr_std,
    )
    loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = build_model(args.max_horizon, device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.eval()

    sinr_chunks = []
    rb_chunks = []
    for rb_seq, sinr_seq, _, _ in loader:
        rb_seq = rb_seq.to(device)
        sinr_seq = sinr_seq.to(device)
        rb_logits, sinr_pred = model(rb_seq, sinr_seq, tf_ratio=0.0)
        sinr = sinr_pred.squeeze(-1).cpu().numpy()
        sinr = sinr * float(train_dataset.sinr_std) + float(train_dataset.sinr_mean)
        sinr_chunks.append(sinr.astype(np.float32, copy=False))
        rb_chunks.append((rb_logits.argmax(dim=2).cpu().numpy() + 1).astype(np.float32, copy=False))

    return np.concatenate(sinr_chunks, axis=0), np.concatenate(rb_chunks, axis=0)


def horizon_metrics(pred, target):
    err = pred - target
    mae = np.mean(np.abs(err), axis=0)
    mse = np.mean(err * err, axis=0)
    corr = []
    for h in range(target.shape[1]):
        if np.std(pred[:, h]) < 1e-8 or np.std(target[:, h]) < 1e-8:
            corr.append(np.nan)
        else:
            corr.append(np.corrcoef(pred[:, h], target[:, h])[0, 1])
    return mae, mse, np.asarray(corr, dtype=np.float32)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_payload = pd.read_pickle(args.dataset_dir / "train_9000_HDF5.pkl")
    test_payload = pd.read_pickle(args.dataset_dir / "test_1000_HDF5.pkl")
    train_sinr = flatten_series(train_payload, "SINR")
    test_sinr = flatten_series(test_payload, "SINR")

    train_dataset = MTL_Dataset(
        str(args.dataset_dir / "train_9000_HDF5.pkl"),
        str(args.dataset_dir / "train_9000_HDF5.pkl"),
        obs_window=args.hist_window,
        pre_window=args.max_horizon,
    )

    hist_sinr, target = make_windows(test_sinr, args.hist_window, args.max_horizon)
    prnet_pred, rb_pred = predict_prnet(args, train_dataset)
    n = min(len(target), len(prnet_pred))
    target = target[:n]
    hist_sinr = hist_sinr[:n]
    prnet_pred = prnet_pred[:n]
    rb_pred = rb_pred[:n]

    methods = {
        "Last SINR": np.repeat(hist_sinr[:, -1:], args.max_horizon, axis=1),
        "History Mean": np.repeat(hist_sinr.mean(axis=1, keepdims=True), args.max_horizon, axis=1),
        "Train Mean": np.full_like(target, float(train_sinr.mean())),
        "PRNet": prnet_pred,
    }

    rows = []
    summary = {}
    for name, pred in methods.items():
        mae, mse, corr = horizon_metrics(pred, target)
        summary[name] = {
            "avg_mae_1_10": float(mae.mean()),
            "avg_mse_1_10": float(mse.mean()),
            "avg_corr_1_10": float(np.nanmean(corr)),
            "mae_by_horizon": [float(x) for x in mae],
            "mse_by_horizon": [float(x) for x in mse],
            "corr_by_horizon": [float(x) for x in corr],
        }
        for h in range(args.max_horizon):
            rows.append(
                {
                    "h": h + 1,
                    "method": name,
                    "sinr_mae": float(mae[h]),
                    "sinr_mse": float(mse[h]),
                    "corr": float(corr[h]),
                    "pred_std": float(np.std(pred[:, h])),
                    "true_std": float(np.std(target[:, h])),
                    "pred_mean": float(np.mean(pred[:, h])),
                    "true_mean": float(np.mean(target[:, h])),
                }
            )

    metrics_path = args.output_dir / "sinr_horizon_metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)

    summary_payload = {
        "dataset_dir": str(args.dataset_dir),
        "model_path": str(args.model_path),
        "train_sinr_mean": float(train_sinr.mean()),
        "train_sinr_std": float(train_sinr.std(ddof=1)),
        "test_sinr_mean": float(test_sinr.mean()),
        "test_sinr_std": float(test_sinr.std(ddof=1)),
        "num_test_windows": int(n),
        "rb_pred_minmax": [float(rb_pred.min()), float(rb_pred.max())],
        "summary": summary,
    }
    summary_path = args.output_dir / "sinr_analysis_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2))

    horizons = np.arange(1, args.max_horizon + 1)
    plt.figure(figsize=(6.4, 4.2))
    for name in ["Last SINR", "History Mean", "Train Mean", "PRNet"]:
        plt.plot(horizons, summary[name]["mae_by_horizon"], marker="o", linewidth=2.0, label=name)
    plt.xlabel("Prediction horizon h (s)")
    plt.ylabel("SINR MAE (dB)")
    plt.xticks(horizons)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(args.output_dir / "sinr_mae_by_horizon.png", dpi=220, bbox_inches="tight")
    plt.close()

    idx = np.arange(min(180, n))
    plt.figure(figsize=(8.0, 5.2))
    for subplot_idx, horizon in enumerate([1, 2, 3], start=1):
        ax = plt.subplot(3, 1, subplot_idx)
        ax.plot(idx, target[idx, horizon - 1], color="black", linewidth=1.8, label="True" if subplot_idx == 1 else None)
        ax.plot(idx, prnet_pred[idx, horizon - 1], color="#54A24B", linewidth=1.4, label="PRNet" if subplot_idx == 1 else None)
        ax.plot(idx, methods["Last SINR"][idx, horizon - 1], color="#4C78A8", linewidth=1.0, alpha=0.7, label="Last SINR" if subplot_idx == 1 else None)
        ax.set_ylabel(f"h={horizon}s")
        ax.grid(True, linestyle="--", alpha=0.25)
        if subplot_idx == 1:
            ax.legend(frameon=False, loc="upper right")
    plt.xlabel("Test-window index (1-second sliding windows)")
    plt.tight_layout()
    plt.savefig(args.output_dir / "sinr_true_vs_pred_h1_h2_h3.png", dpi=220, bbox_inches="tight")
    plt.close()

    print(json.dumps({"metrics": str(metrics_path), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
