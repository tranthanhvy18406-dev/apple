import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Train a direct GRU throughput baseline.")
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--hist_window", type=int, default=16)
    parser.add_argument("--pre_window", type=int, default=100)
    parser.add_argument("--skip_frames", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_mode", type=str, default="raw", choices=["raw", "v2"])
    parser.add_argument("--input_mode", type=str, default="rb_sinr", choices=["rb_sinr", "sinr", "rb"])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def flatten_series(obj, key: str) -> np.ndarray:
    if isinstance(obj, dict):
        if key not in obj:
            raise KeyError(f"Missing key {key} in dataset.")
        data = obj[key]
    else:
        data = obj
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    return arr.reshape(-1)


_MCS_TABLE = torch.tensor(
    [
        [1, 4, 0.10, -6.4131],
        [2, 4, 0.14, -4.7292],
        [3, 4, 0.18, -3.6312],
        [4, 4, 0.23, -2.9651],
        [5, 4, 0.26, -2.0542],
        [6, 4, 0.34, -1.2451],
        [7, 4, 0.40, -0.3223],
        [8, 4, 0.45, 0.8364],
        [9, 4, 0.52, 1.5261],
        [10, 4, 0.59, 2.5709],
        [11, 16, 0.31, 3.1961],
        [12, 16, 0.32, 3.7165],
        [13, 16, 0.37, 4.9642],
        [14, 16, 0.45, 5.4653],
        [15, 16, 0.47, 6.9965],
        [16, 16, 0.54, 7.1562],
        [18, 16, 0.57, 7.2301],
        [17, 16, 0.59, 7.9862],
        [19, 64, 0.35, 8.0651],
        [20, 64, 0.38, 8.6985],
        [21, 64, 0.41, 9.0224],
        [22, 64, 0.43, 9.3017],
        [23, 64, 0.45, 9.9628],
        [24, 64, 0.47, 10.3957],
        [25, 64, 0.49, 10.7214],
        [26, 64, 0.55, 12.0541],
        [27, 64, 0.61, 12.8769],
        [28, 64, 0.63, 13.5547],
        [29, 64, 0.65, 14.6139],
    ],
    dtype=torch.float32,
)

_TBS_TABLE = torch.tensor(
    [
        24, 32, 40, 48, 56, 64, 72, 80, 88, 96,
        104, 112, 120, 128, 136, 144, 152, 160, 168, 176,
        184, 192, 208, 224, 240, 256, 272, 288, 304, 320,
        336, 352, 368, 384, 408, 432, 456, 480, 504, 528,
        552, 576, 608, 640, 672, 704, 736, 768, 808, 848,
        888, 928, 984, 1032, 1064, 1128, 1160, 1192, 1224, 1256,
        1288, 1320, 1352, 1416, 1480, 1544, 1608, 1672, 1736, 1800,
        1864, 1928, 2024, 2088, 2152, 2216, 2280, 2408, 2472, 2536,
        2600, 2664, 2728, 2792, 2856, 2976, 3104, 3240, 3368, 3496,
        3624, 3752, 3824,
    ],
    dtype=torch.float32,
)


def calculate_v2_throughput_series(rb: np.ndarray, sinr: np.ndarray) -> np.ndarray:
    rb_t = torch.as_tensor(rb, dtype=torch.float32)
    sinr_t = torch.as_tensor(sinr, dtype=torch.float32)
    table = _MCS_TABLE
    idx = torch.bucketize(sinr_t, table[:, 3].contiguous(), right=True) - 1
    idx = idx.clamp(min=0, max=table.size(0) - 1)

    code_rate = table[:, 2][idx]
    mod_order = torch.log2(table[:, 1][idx]).to(torch.float32)
    n0 = mod_order * rb_t * 136.0 * code_rate
    tbs = torch.empty_like(n0, dtype=torch.float32)
    mask_hi = n0 > 3824.0

    if mask_hi.any():
        n0_hi = n0[mask_hi]
        code_rate_hi = code_rate[mask_hi]
        n_hi = torch.floor(torch.log2(n0_hi - 24.0)) - 5.0
        two_n_hi = torch.pow(torch.tensor(2.0), n_hi)
        n_hi_rounded = torch.maximum(
            torch.tensor(3840.0),
            torch.round((n0_hi - 24.0) / two_n_hi) * two_n_hi,
        )

        tbs_hi = torch.empty_like(n_hi_rounded)
        mask_lowrate = code_rate_hi <= 0.25
        mask_large = (~mask_lowrate) & (n_hi_rounded > 8424.0)
        mask_other = (~mask_lowrate) & (~mask_large)

        if mask_lowrate.any():
            c = torch.ceil((n_hi_rounded[mask_lowrate] + 24.0) / 3816.0)
            tbs_hi[mask_lowrate] = 8.0 * c * torch.ceil((n_hi_rounded[mask_lowrate] + 24.0) / 8.0 / c) - 24.0
        if mask_large.any():
            c = torch.ceil((n_hi_rounded[mask_large] + 24.0) / 8424.0)
            tbs_hi[mask_large] = 8.0 * c * torch.ceil((n_hi_rounded[mask_large] + 24.0) / 8.0 / c) - 24.0
        if mask_other.any():
            tbs_hi[mask_other] = 8.0 * torch.ceil((n_hi_rounded[mask_other] + 24.0) / 8.0) - 24.0
        tbs[mask_hi] = tbs_hi

    if (~mask_hi).any():
        n0_low = n0[~mask_hi]
        n_low = torch.floor(torch.log2(n0_low.clamp_min(1e-6))) - 6.0
        n_low = torch.maximum(torch.tensor(3.0), n_low)
        two_n_low = torch.pow(torch.tensor(2.0), n_low)
        n_low_rounded = torch.maximum(
            torch.tensor(24.0),
            torch.floor(n0_low / two_n_low) * two_n_low,
        )
        jdx = torch.bucketize(n_low_rounded, _TBS_TABLE, right=False).clamp(max=_TBS_TABLE.numel() - 1)
        tbs[~mask_hi] = _TBS_TABLE[jdx]

    throughput = tbs * 1e-6 / (5e-4)
    return throughput.cpu().numpy().astype(np.float32)


class ThroughputDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        obs_window: int,
        pre_window: int,
        skip_initial_frames: int = 0,
        stats: dict | None = None,
        target_mode: str = "raw",
        input_mode: str = "rb_sinr",
    ):
        raw = pd.read_pickle(data_path)
        rb = flatten_series(raw, "RB")
        sinr = flatten_series(raw, "SINR")
        throughput_raw = flatten_series(raw, "Throughput")
        if target_mode == "v2":
            throughput = calculate_v2_throughput_series(rb, sinr)
        else:
            throughput = throughput_raw

        if skip_initial_frames > 0:
            rb = rb[skip_initial_frames:]
            sinr = sinr[skip_initial_frames:]
            throughput = throughput[skip_initial_frames:]
            throughput_raw = throughput_raw[skip_initial_frames:]

        if stats is None:
            stats = {
                "rb_mean": float(rb.mean()),
                "rb_std": float(rb.std() + 1e-8),
                "sinr_mean": float(sinr.mean()),
                "sinr_std": float(sinr.std() + 1e-8),
                "thr_mean": float(throughput.mean()),
                "thr_std": float(throughput.std() + 1e-8),
            }

        self.stats = stats
        rb_norm = (rb - stats["rb_mean"]) / stats["rb_std"]
        sinr_norm = (sinr - stats["sinr_mean"]) / stats["sinr_std"]
        thr_norm = (throughput - stats["thr_mean"]) / stats["thr_std"]

        if input_mode == "rb_sinr":
            inputs = np.stack([rb_norm, sinr_norm], axis=-1)
        elif input_mode == "sinr":
            inputs = sinr_norm[:, None]
        elif input_mode == "rb":
            inputs = rb_norm[:, None]
        else:
            raise ValueError(f"Unsupported input_mode={input_mode!r}")

        self.input_mode = input_mode
        self.inputs = inputs.astype(np.float32)
        self.targets = thr_norm.astype(np.float32)
        self.targets_raw = throughput.astype(np.float32)
        self.dataset_throughput_raw = throughput_raw.astype(np.float32)
        self.obs_window = int(obs_window)
        self.pre_window = int(pre_window)
        self.num_samples = len(self.inputs) - self.obs_window - self.pre_window + 1
        if self.num_samples <= 0:
            raise ValueError("Not enough data to create samples.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        src_start = idx
        src_end = src_start + self.obs_window
        tgt_start = src_end
        tgt_end = tgt_start + self.pre_window
        return (
            torch.from_numpy(self.inputs[src_start:src_end]),
            torch.from_numpy(self.targets[tgt_start:tgt_end]),
            torch.from_numpy(self.targets_raw[tgt_start:tgt_end]),
        )


class DirectRateGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, pre_window: int, dropout: float):
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=gru_dropout,
            batch_first=True,
        )
        # Keep the direct-rate baseline intentionally lightweight.
        self.head = nn.Linear(hidden_dim, pre_window)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.encoder(x)
        last_hidden = h_n[-1]
        return self.head(last_hidden)


@torch.no_grad()
def evaluate_model(model, loader, thr_mean: float, thr_std: float, device: torch.device) -> dict:
    model.eval()
    total_points = 0
    total_mae = 0.0
    total_mse = 0.0

    for features, target_norm, _ in loader:
        features = features.to(device)
        target_norm = target_norm.to(device)

        pred_norm = model(features)
        pred = pred_norm * thr_std + thr_mean
        target = target_norm * thr_std + thr_mean

        total_points += target.numel()
        total_mae += torch.abs(pred - target).sum().item()
        total_mse += ((pred - target) ** 2).sum().item()

    return {
        "throughput_mae": total_mae / total_points,
        "throughput_mse": total_mse / total_points,
    }


def plot_history(history: list[dict], out_path: Path) -> None:
    epochs = [item["epoch"] for item in history]
    train_mse = [item["train_throughput_mse"] for item in history]
    test_mse = [item["test_throughput_mse"] for item in history]
    train_mae = [item["train_throughput_mae"] for item in history]
    test_mae = [item["test_throughput_mae"] for item in history]

    plt.figure(figsize=(10, 4.2))

    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(epochs, train_mse, label="Train", linewidth=2)
    ax1.plot(epochs, test_mse, label="Test", linewidth=2)
    ax1.set_title("MSE over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE")
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend()

    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(epochs, train_mae, label="Train", linewidth=2)
    ax2.plot(epochs, test_mae, label="Test", linewidth=2)
    ax2.set_title("MAE over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MAE")
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    seed_everything(args.seed)

    dataset_dir = args.dataset_dir.resolve()
    output_dir = (args.output_root / dataset_dir.name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    history_path = output_dir / "history.json"
    model_path = output_dir / "best_direct_rate_gru.pth"
    curve_path = output_dir / "convergence_curves.png"

    if metrics_path.exists() and not args.force:
        print(metrics_path.read_text(encoding="utf-8"))
        return

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    train_path = dataset_dir / "train_9000_HDF5.pkl"
    test_path = dataset_dir / "test_1000_HDF5.pkl"
    train_dataset = ThroughputDataset(
        train_path,
        obs_window=args.hist_window,
        pre_window=args.pre_window,
        skip_initial_frames=args.skip_frames,
        target_mode=args.target_mode,
        input_mode=args.input_mode,
    )
    test_dataset = ThroughputDataset(
        test_path,
        obs_window=args.hist_window,
        pre_window=args.pre_window,
        stats=train_dataset.stats,
        target_mode=args.target_mode,
        input_mode=args.input_mode,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    train_eval_loader = DataLoader(train_dataset, batch_size=args.eval_batch_size, shuffle=False)
    eval_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)

    model = DirectRateGRU(
        input_dim=train_dataset.inputs.shape[-1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        pre_window=args.pre_window,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    best_mae = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []
    thr_mean = train_dataset.stats["thr_mean"]
    thr_std = train_dataset.stats["thr_std"]

    print(f"Starting GRU direct-rate baseline for {dataset_dir.name} on {device}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"{dataset_dir.name} gru epoch {epoch}/{args.epochs}", ncols=120)
        for features, target_norm, _ in pbar:
            features = features.to(device)
            target_norm = target_norm.to(device)

            optimizer.zero_grad()
            pred_norm = model(features)
            loss = criterion(pred_norm, target_norm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        train_metrics = evaluate_model(model, train_eval_loader, thr_mean, thr_std, device)
        eval_metrics = evaluate_model(model, eval_loader, thr_mean, thr_std, device)
        history_item = {
            "epoch": epoch,
            "train_throughput_mae": float(train_metrics["throughput_mae"]),
            "train_throughput_mse": float(train_metrics["throughput_mse"]),
            "test_throughput_mae": float(eval_metrics["throughput_mae"]),
            "test_throughput_mse": float(eval_metrics["throughput_mse"]),
        }
        history.append(history_item)
        print(
            f"{dataset_dir.name} epoch {epoch}: "
            f"Train MAE={train_metrics['throughput_mae']:.4f} | "
            f"Test MAE={eval_metrics['throughput_mae']:.4f} | "
            f"Train MSE={train_metrics['throughput_mse']:.4f} | "
            f"Test MSE={eval_metrics['throughput_mse']:.4f}"
        )

        if eval_metrics["throughput_mae"] < best_mae:
            best_mae = eval_metrics["throughput_mae"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), model_path)
            print(f"{dataset_dir.name} saved new best checkpoint at epoch {epoch}")
        else:
            stale_epochs += 1

        if args.patience > 0 and stale_epochs >= args.patience:
            print(f"{dataset_dir.name} early-stopped after {stale_epochs} stale epochs")
            break

    best_model = DirectRateGRU(
        input_dim=train_dataset.inputs.shape[-1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        pre_window=args.pre_window,
        dropout=args.dropout,
    ).to(device)
    best_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    final_metrics = evaluate_model(best_model, eval_loader, thr_mean, thr_std, device)

    result = {
        "dataset": dataset_dir.name,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "throughput_mae": float(final_metrics["throughput_mae"]),
        "throughput_mse": float(final_metrics["throughput_mse"]),
        "model_path": str(model_path),
        "curve_path": str(curve_path),
        "stats": train_dataset.stats,
        "config": {
            "hist_window": args.hist_window,
            "pre_window": args.pre_window,
            "skip_frames": args.skip_frames,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "target_mode": args.target_mode,
            "input_mode": args.input_mode,
        },
    }

    metrics_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_history(history, curve_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
