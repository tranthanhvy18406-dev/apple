import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Basic LSTM baseline for SINR multi-step prediction.")
    parser.add_argument("--dataset_dir", type=Path, default=Path("/8T2/xty/code/cross_stitch_1s_datasets/ul_rr_1s_cdlc_10000s"))
    parser.add_argument("--output_dir", type=Path, default=Path("/4T/xty/crossstitch_v2/lstm_sinr_baseline"))
    parser.add_argument("--hist_window", type=int, default=16)
    parser.add_argument("--max_horizon", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--plot_windows", type=int, default=180)
    parser.add_argument("--sweep_mode", choices=["quick", "wide"], default="quick")
    parser.add_argument("--max_configs", type=int, default=None)
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def flatten_sinr(path):
    payload = pd.read_pickle(path)
    aliases = ["SINR", "sinr", "Sinr", "SNR", "snr", "eff_SINR", "eff_sinr"]
    data = None
    if isinstance(payload, dict):
        for key in aliases:
            if key in payload:
                data = payload[key]
                break
    else:
        data = payload
    if data is None:
        raise KeyError(f"Cannot find SINR key in {path}")
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    return arr.reshape(-1)


def make_windows(series, hist_window, horizon):
    samples = len(series) - hist_window - horizon + 1
    if samples <= 0:
        raise ValueError("Not enough samples for requested history/horizon.")
    starts = np.arange(samples)
    hist_idx = starts[:, None] + np.arange(hist_window)[None, :]
    target_idx = starts[:, None] + hist_window + np.arange(horizon)[None, :]
    return series[hist_idx].astype(np.float32), series[target_idx].astype(np.float32)


class BasicSinrLSTM(nn.Module):
    def __init__(self, hidden_dim, num_layers, horizon, dropout=0.0, residual=False):
        super().__init__()
        self.residual = residual
        self.horizon = horizon
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        pred = self.head(h[-1])
        if self.residual:
            pred = x[:, -1, 0:1] + pred
        return pred


def denorm(x, mean, std):
    return x * std + mean


def score(pred, target):
    err = pred - target
    mae = np.mean(np.abs(err), axis=0)
    mse = np.mean(err * err, axis=0)
    corr = []
    for h in range(target.shape[1]):
        if np.std(pred[:, h]) < 1e-8 or np.std(target[:, h]) < 1e-8:
            corr.append(np.nan)
        else:
            corr.append(float(np.corrcoef(pred[:, h], target[:, h])[0, 1]))
    return mae, mse, np.asarray(corr, dtype=np.float32)


def build_criterion(config):
    loss_name = config.get("loss", "smooth_l1")
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "l1":
        return nn.L1Loss()
    if loss_name == "smooth_l1":
        return nn.SmoothL1Loss(beta=config.get("beta", 0.25))
    raise ValueError(f"Unsupported loss: {loss_name}")


def train_one(config, train_x, train_y, val_x, val_y, mean, std, args, device):
    model = BasicSinrLSTM(
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        horizon=args.max_horizon,
        dropout=config["dropout"],
        residual=config["residual"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = build_criterion(config)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x[:, :, None]), torch.from_numpy(train_y)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_x_t = torch.from_numpy(val_x[:, :, None]).to(device)
    val_y_t = torch.from_numpy(val_y).to(device)

    best_state = None
    best_val_mae = float("inf")
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        seen = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
            seen += len(xb)

        model.eval()
        with torch.no_grad():
            val_pred_norm = model(val_x_t).cpu().numpy()
        val_pred = denorm(val_pred_norm, mean, std)
        val_true = denorm(val_y, mean, std)
        val_mae = float(np.mean(np.abs(val_pred - val_true)))
        history.append({"epoch": epoch, "train_loss": train_loss / max(seen, 1), "val_mae": val_mae})

        if val_mae < best_val_mae - 1e-5:
            best_val_mae = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    model.load_state_dict(best_state)
    return model, best_val_mae, history


def build_configs(mode):
    quick = [
        {"hidden_dim": 16, "num_layers": 1, "dropout": 0.0, "lr": 1e-3, "weight_decay": 1e-4, "residual": False, "loss": "smooth_l1", "beta": 0.25},
        {"hidden_dim": 32, "num_layers": 1, "dropout": 0.0, "lr": 1e-3, "weight_decay": 1e-4, "residual": False, "loss": "smooth_l1", "beta": 0.25},
        {"hidden_dim": 64, "num_layers": 1, "dropout": 0.0, "lr": 1e-3, "weight_decay": 1e-4, "residual": False, "loss": "smooth_l1", "beta": 0.25},
        {"hidden_dim": 32, "num_layers": 2, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4, "residual": False, "loss": "smooth_l1", "beta": 0.25},
        {"hidden_dim": 64, "num_layers": 2, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4, "residual": False, "loss": "smooth_l1", "beta": 0.25},
        {"hidden_dim": 32, "num_layers": 1, "dropout": 0.0, "lr": 3e-4, "weight_decay": 1e-4, "residual": False, "loss": "smooth_l1", "beta": 0.25},
        {"hidden_dim": 64, "num_layers": 1, "dropout": 0.0, "lr": 3e-4, "weight_decay": 1e-4, "residual": False, "loss": "smooth_l1", "beta": 0.25},
        {"hidden_dim": 32, "num_layers": 1, "dropout": 0.0, "lr": 1e-3, "weight_decay": 1e-4, "residual": True, "loss": "smooth_l1", "beta": 0.25},
        {"hidden_dim": 64, "num_layers": 1, "dropout": 0.0, "lr": 1e-3, "weight_decay": 1e-4, "residual": True, "loss": "smooth_l1", "beta": 0.25},
        {"hidden_dim": 32, "num_layers": 2, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4, "residual": True, "loss": "smooth_l1", "beta": 0.25},
        {"hidden_dim": 64, "num_layers": 2, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4, "residual": True, "loss": "smooth_l1", "beta": 0.25},
        {"hidden_dim": 64, "num_layers": 1, "dropout": 0.0, "lr": 3e-4, "weight_decay": 1e-4, "residual": True, "loss": "smooth_l1", "beta": 0.25},
    ]
    if mode == "quick":
        return quick

    wide_extra = [
        {"hidden_dim": 8, "num_layers": 1, "dropout": 0.0, "lr": 1e-3, "weight_decay": 0.0, "residual": False, "loss": "smooth_l1", "beta": 0.25},
        {"hidden_dim": 128, "num_layers": 1, "dropout": 0.0, "lr": 1e-3, "weight_decay": 1e-5, "residual": False, "loss": "smooth_l1", "beta": 0.25},
        {"hidden_dim": 128, "num_layers": 1, "dropout": 0.0, "lr": 3e-4, "weight_decay": 1e-5, "residual": False, "loss": "smooth_l1", "beta": 0.25},
        {"hidden_dim": 128, "num_layers": 1, "dropout": 0.0, "lr": 1e-3, "weight_decay": 1e-5, "residual": True, "loss": "smooth_l1", "beta": 0.25},
        {"hidden_dim": 128, "num_layers": 2, "dropout": 0.0, "lr": 1e-3, "weight_decay": 1e-5, "residual": False, "loss": "smooth_l1", "beta": 0.25},
        {"hidden_dim": 128, "num_layers": 2, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-5, "residual": True, "loss": "smooth_l1", "beta": 0.25},
        {"hidden_dim": 128, "num_layers": 3, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-5, "residual": False, "loss": "smooth_l1", "beta": 0.25},
        {"hidden_dim": 128, "num_layers": 3, "dropout": 0.1, "lr": 3e-4, "weight_decay": 1e-5, "residual": True, "loss": "smooth_l1", "beta": 0.25},
        {"hidden_dim": 256, "num_layers": 1, "dropout": 0.0, "lr": 3e-4, "weight_decay": 1e-5, "residual": False, "loss": "smooth_l1", "beta": 0.25},
        {"hidden_dim": 256, "num_layers": 2, "dropout": 0.1, "lr": 3e-4, "weight_decay": 1e-5, "residual": False, "loss": "smooth_l1", "beta": 0.25},
        {"hidden_dim": 256, "num_layers": 2, "dropout": 0.2, "lr": 3e-4, "weight_decay": 1e-4, "residual": True, "loss": "smooth_l1", "beta": 0.25},
        {"hidden_dim": 64, "num_layers": 3, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4, "residual": False, "loss": "smooth_l1", "beta": 0.25},
        {"hidden_dim": 64, "num_layers": 3, "dropout": 0.2, "lr": 3e-4, "weight_decay": 1e-4, "residual": True, "loss": "smooth_l1", "beta": 0.25},
        {"hidden_dim": 32, "num_layers": 2, "dropout": 0.2, "lr": 1e-3, "weight_decay": 1e-3, "residual": False, "loss": "smooth_l1", "beta": 0.25},
        {"hidden_dim": 64, "num_layers": 2, "dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-3, "residual": False, "loss": "smooth_l1", "beta": 0.25},
        {"hidden_dim": 64, "num_layers": 1, "dropout": 0.0, "lr": 1e-3, "weight_decay": 0.0, "residual": False, "loss": "mse"},
        {"hidden_dim": 128, "num_layers": 1, "dropout": 0.0, "lr": 1e-3, "weight_decay": 1e-5, "residual": False, "loss": "mse"},
        {"hidden_dim": 128, "num_layers": 2, "dropout": 0.1, "lr": 3e-4, "weight_decay": 1e-5, "residual": False, "loss": "mse"},
        {"hidden_dim": 64, "num_layers": 1, "dropout": 0.0, "lr": 1e-3, "weight_decay": 0.0, "residual": False, "loss": "l1"},
        {"hidden_dim": 128, "num_layers": 2, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-5, "residual": True, "loss": "l1"},
    ]
    return quick + wide_extra


def plot_mae(out_path, horizons, curves):
    plt.figure(figsize=(6.4, 4.2))
    for name, values in curves.items():
        plt.plot(horizons, values, marker="o", linewidth=2.0, label=name)
    plt.xlabel("Prediction horizon h (s)")
    plt.ylabel("SINR MAE (dB)")
    plt.xticks(horizons)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_trace(out_path, target, predictions, last_pred, plot_windows):
    n = min(plot_windows, len(target))
    idx = np.arange(n)
    plt.figure(figsize=(8.2, 5.4))
    shown_horizons = list(range(1, min(3, target.shape[1]) + 1))
    for subplot_idx, horizon in enumerate(shown_horizons, start=1):
        ax = plt.subplot(len(shown_horizons), 1, subplot_idx)
        h = horizon - 1
        ax.plot(idx, target[:n, h], color="black", linewidth=1.8, label="True" if subplot_idx == 1 else None)
        ax.plot(idx, predictions[:n, h], color="#54A24B", linewidth=1.4, label="LSTM" if subplot_idx == 1 else None)
        ax.plot(idx, last_pred[:n, h], color="#88A9C9", linewidth=1.0, alpha=0.8, label="Last SINR" if subplot_idx == 1 else None)
        ax.set_ylabel(f"h={horizon}s")
        ax.grid(True, linestyle="--", alpha=0.28)
        if subplot_idx == 1:
            ax.legend(frameon=False, loc="upper right")
    plt.xlabel("Test-window index (1-second sliding windows)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    train_sinr_raw = flatten_sinr(args.dataset_dir / "train_9000_HDF5.pkl")
    test_sinr_raw = flatten_sinr(args.dataset_dir / "test_1000_HDF5.pkl")
    mean = float(train_sinr_raw.mean())
    std = float(train_sinr_raw.std() + 1e-8)

    train_sinr = (train_sinr_raw - mean) / std
    test_sinr = (test_sinr_raw - mean) / std
    train_x_all, train_y_all = make_windows(train_sinr, args.hist_window, args.max_horizon)
    test_x, test_y_norm = make_windows(test_sinr, args.hist_window, args.max_horizon)
    test_hist_raw, test_y_raw = make_windows(test_sinr_raw, args.hist_window, args.max_horizon)

    val_count = max(64, int(round(len(train_x_all) * args.val_fraction)))
    train_x, val_x = train_x_all[:-val_count], train_x_all[-val_count:]
    train_y, val_y = train_y_all[:-val_count], train_y_all[-val_count:]

    configs = build_configs(args.sweep_mode)
    if args.max_configs is not None:
        configs = configs[: args.max_configs]

    sweep_rows = []
    best = None
    for idx, config in enumerate(configs):
        model, val_mae, history = train_one(config, train_x, train_y, val_x, val_y, mean, std, args, device)
        row = {"config_id": idx, "val_mae": val_mae, "epochs_ran": len(history), **config}
        sweep_rows.append(row)
        (args.output_dir / f"history_config_{idx}.json").write_text(json.dumps(history, indent=2))
        if best is None or val_mae < best["val_mae"]:
            best = {"config_id": idx, "config": config, "val_mae": val_mae, "model": model, "history": history}
        print(json.dumps(row))

    best_model = best["model"]
    best_model.eval()
    with torch.no_grad():
        pred_norm = best_model(torch.from_numpy(test_x[:, :, None]).to(device)).cpu().numpy()
    pred = denorm(pred_norm, mean, std)
    target = test_y_raw
    last_pred = np.repeat(test_hist_raw[:, -1:], args.max_horizon, axis=1)
    hist_mean_pred = np.repeat(test_hist_raw.mean(axis=1, keepdims=True), args.max_horizon, axis=1)
    train_mean_pred = np.full_like(target, mean)

    methods = {
        "Last SINR": last_pred,
        "History Mean": hist_mean_pred,
        "Train Mean": train_mean_pred,
        "LSTM": pred,
    }

    rows = []
    summary = {}
    avg_suffix = f"1_{args.max_horizon}"
    for name, method_pred in methods.items():
        mae, mse, corr = score(method_pred, target)
        summary[name] = {
            f"avg_mae_{avg_suffix}": float(mae.mean()),
            f"avg_mse_{avg_suffix}": float(mse.mean()),
            f"avg_corr_{avg_suffix}": float(np.nanmean(corr)),
            "mae_by_horizon": [float(x) for x in mae],
            "mse_by_horizon": [float(x) for x in mse],
            "corr_by_horizon": [float(x) for x in corr],
            "pred_std_mean": float(np.mean([np.std(method_pred[:, h]) for h in range(args.max_horizon)])),
            "true_std_mean": float(np.mean([np.std(target[:, h]) for h in range(args.max_horizon)])),
        }
        for h in range(args.max_horizon):
            rows.append(
                {
                    "h": h + 1,
                    "method": name,
                    "sinr_mae": float(mae[h]),
                    "sinr_mse": float(mse[h]),
                    "corr": float(corr[h]),
                    "pred_std": float(np.std(method_pred[:, h])),
                    "true_std": float(np.std(target[:, h])),
                    "pred_mean": float(np.mean(method_pred[:, h])),
                    "true_mean": float(np.mean(target[:, h])),
                }
            )

    pd.DataFrame(sweep_rows).to_csv(args.output_dir / "lstm_sinr_sweep.csv", index=False)
    pd.DataFrame(rows).to_csv(args.output_dir / "lstm_sinr_horizon_metrics.csv", index=False)
    torch.save(
        {
            "model_state": best_model.state_dict(),
            "config": best["config"],
            "sinr_mean": mean,
            "sinr_std": std,
            "hist_window": args.hist_window,
            "max_horizon": args.max_horizon,
            "sweep_mode": args.sweep_mode,
        },
        args.output_dir / "best_lstm_sinr.pth",
    )

    payload = {
        "dataset_dir": str(args.dataset_dir),
        "sweep_mode": args.sweep_mode,
        "config_count": len(configs),
        "hist_window": args.hist_window,
        "max_horizon": args.max_horizon,
        "train_windows": int(len(train_x_all)),
        "test_windows": int(len(test_x)),
        "sinr_mean": mean,
        "sinr_std": std,
        "best_config_id": int(best["config_id"]),
        "best_config": best["config"],
        "best_val_mae": float(best["val_mae"]),
        "summary": summary,
    }
    (args.output_dir / "lstm_sinr_summary.json").write_text(json.dumps(payload, indent=2))

    horizons = np.arange(1, args.max_horizon + 1)
    plot_mae(
        args.output_dir / "lstm_sinr_mae_by_horizon.png",
        horizons,
        {name: summary[name]["mae_by_horizon"] for name in methods},
    )
    plot_trace(args.output_dir / "lstm_sinr_true_vs_pred_h1_h2_h3.png", target, pred, last_pred, args.plot_windows)
    pd.DataFrame(
        {
            "window": np.repeat(np.arange(len(target)), min(3, args.max_horizon)),
            "h": np.tile(np.arange(1, min(3, args.max_horizon) + 1), len(target)),
            "true": target[:, : min(3, args.max_horizon)].reshape(-1),
            "lstm": pred[:, : min(3, args.max_horizon)].reshape(-1),
            "last_sinr": last_pred[:, : min(3, args.max_horizon)].reshape(-1),
        }
    ).to_csv(args.output_dir / "lstm_sinr_trace_values_h1_h2_h3.csv", index=False)

    print(json.dumps({"output_dir": str(args.output_dir), "best_config": best["config"], "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
