import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mtl_dataset import MTL_Dataset
from mtl_model import CrossStitch_MTL_Model
from train_direct_rate_gru import calculate_v2_throughput_series, plot_history, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a GRU that replaces the RB/SINR -> throughput protocol block."
    )
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument(
        "--base_model_path",
        type=Path,
        required=True,
        help="Cross-Stitch checkpoint used to generate RB/SINR predictions.",
    )
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--hist_window", type=int, default=16)
    parser.add_argument("--pre_window", type=int, default=100)
    parser.add_argument("--skip_frames", type=int, default=10)
    parser.add_argument("--rb_hidden_dim", type=int, default=256)
    parser.add_argument("--sinr_hidden_dim", type=int, default=64)
    parser.add_argument("--rb_layers", type=int, default=3)
    parser.add_argument("--sinr_layers", type=int, default=1)
    parser.add_argument("--cross_stitch_mode", type=str, default="learn")
    parser.add_argument("--mapper_hidden_dim", type=int, default=8)
    parser.add_argument("--mapper_layers", type=int, default=1)
    parser.add_argument("--mapper_dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


class ProtocolReplacementDataset(Dataset):
    def __init__(self, features: np.ndarray, throughput: np.ndarray, stats: dict | None = None):
        features = np.asarray(features, dtype=np.float32)
        throughput = np.asarray(throughput, dtype=np.float32)
        if features.ndim != 3 or features.shape[-1] != 2:
            raise ValueError(f"Expected features with shape [N, T, 2], got {features.shape}")
        if throughput.ndim != 2:
            raise ValueError(f"Expected throughput with shape [N, T], got {throughput.shape}")

        if stats is None:
            feat_mean = features.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
            feat_std = features.std(axis=(0, 1), dtype=np.float64).astype(np.float32) + 1e-8
            thr_mean = float(throughput.mean())
            thr_std = float(throughput.std() + 1e-8)
            stats = {
                "feature_mean": feat_mean.tolist(),
                "feature_std": feat_std.tolist(),
                "thr_mean": thr_mean,
                "thr_std": thr_std,
            }

        feat_mean = np.asarray(stats["feature_mean"], dtype=np.float32).reshape(1, 1, 2)
        feat_std = np.asarray(stats["feature_std"], dtype=np.float32).reshape(1, 1, 2)
        thr_mean = float(stats["thr_mean"])
        thr_std = float(stats["thr_std"])

        self.features_raw = features
        self.targets_raw = throughput
        self.features = ((features - feat_mean) / feat_std).astype(np.float32)
        self.targets = ((throughput - thr_mean) / thr_std).astype(np.float32)
        self.stats = stats

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.features[idx]),
            torch.from_numpy(self.targets[idx]),
            torch.from_numpy(self.targets_raw[idx]),
        )


class ProtocolReplacementGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=gru_dropout,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.encoder(features)
        return self.head(hidden).squeeze(-1)


def build_cross_stitch_model(args, device: torch.device):
    return CrossStitch_MTL_Model(
        rb_input_size=1,
        sinr_input_size=1,
        rb_hidden_size=args.rb_hidden_dim,
        sinr_hidden_size=args.sinr_hidden_dim,
        rb_num_layers=args.rb_layers,
        sinr_num_layers=args.sinr_layers,
        rb_output_size=106,
        sinr_output_size=1,
        pre_window=args.pre_window,
        device=device,
        cross_stitch_mode=args.cross_stitch_mode,
    ).to(device)


@torch.no_grad()
def collect_cross_stitch_predictions(model, loader, sinr_mean: float, sinr_std: float, device: torch.device):
    model.eval()
    feature_chunks = []
    throughput_chunks = []

    for rb_seq, sinr_seq, rb_label, sinr_label in tqdm(loader, desc="Collecting Cross-Stitch predictions", ncols=120):
        rb_seq = rb_seq.to(device)
        sinr_seq = sinr_seq.to(device)
        rb_label = rb_label.to(device).squeeze(-1)
        sinr_label = sinr_label.to(device).squeeze(-1)

        rb_pred, sinr_pred = model(rb_seq, sinr_seq, tf_ratio=0.0)
        rb_pred = torch.argmax(rb_pred, dim=2).to(torch.float32) + 1.0
        sinr_pred = sinr_pred.squeeze(-1) * sinr_std + sinr_mean
        sinr_label = sinr_label * sinr_std + sinr_mean

        features = torch.stack([rb_pred, sinr_pred], dim=-1).cpu().numpy().astype(np.float32)
        throughput = calculate_v2_throughput_series(
            rb_label.cpu().numpy().reshape(-1),
            sinr_label.cpu().numpy().reshape(-1),
        ).reshape(rb_label.shape[0], rb_label.shape[1])

        feature_chunks.append(features)
        throughput_chunks.append(throughput.astype(np.float32))

    return np.concatenate(feature_chunks, axis=0), np.concatenate(throughput_chunks, axis=0)


@torch.no_grad()
def evaluate_model(model, loader, thr_mean: float, thr_std: float, device: torch.device):
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


def main():
    args = parse_args()
    seed_everything(args.seed)

    dataset_dir = args.dataset_dir.resolve()
    output_dir = (args.output_root / dataset_dir.name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    history_path = output_dir / "history.json"
    model_path = output_dir / "best_protocol_replacement_gru.pth"
    curve_path = output_dir / "convergence_curves.png"
    cache_path = output_dir / "predicted_rb_sinr_cache.npz"

    if metrics_path.exists() and not args.force:
        print(metrics_path.read_text(encoding="utf-8"))
        return

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    train_path = dataset_dir / "train_9000_HDF5.pkl"
    test_path = dataset_dir / "test_1000_HDF5.pkl"
    train_base_dataset = MTL_Dataset(
        str(train_path),
        str(train_path),
        obs_window=args.hist_window,
        pre_window=args.pre_window,
        skip_initial_frames=args.skip_frames,
    )
    sinr_mean = float(train_base_dataset.sinr_mean)
    sinr_std = float(train_base_dataset.sinr_std)
    test_base_dataset = MTL_Dataset(
        str(test_path),
        str(test_path),
        obs_window=args.hist_window,
        pre_window=args.pre_window,
        sinr_mean=sinr_mean,
        sinr_std=sinr_std,
    )

    train_base_loader = DataLoader(train_base_dataset, batch_size=args.eval_batch_size, shuffle=False)
    test_base_loader = DataLoader(test_base_dataset, batch_size=args.eval_batch_size, shuffle=False)

    if cache_path.exists() and not args.force:
        cache = np.load(cache_path)
        train_features = cache["train_features"]
        train_targets = cache["train_targets"]
        test_features = cache["test_features"]
        test_targets = cache["test_targets"]
    else:
        base_model = build_cross_stitch_model(args, device)
        base_model.load_state_dict(torch.load(args.base_model_path, map_location=device, weights_only=True))

        train_features, train_targets = collect_cross_stitch_predictions(
            base_model,
            train_base_loader,
            sinr_mean=sinr_mean,
            sinr_std=sinr_std,
            device=device,
        )
        test_features, test_targets = collect_cross_stitch_predictions(
            base_model,
            test_base_loader,
            sinr_mean=sinr_mean,
            sinr_std=sinr_std,
            device=device,
        )
        np.savez_compressed(
            cache_path,
            train_features=train_features,
            train_targets=train_targets,
            test_features=test_features,
            test_targets=test_targets,
        )

    train_dataset = ProtocolReplacementDataset(train_features, train_targets)
    test_dataset = ProtocolReplacementDataset(test_features, test_targets, stats=train_dataset.stats)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    train_eval_loader = DataLoader(train_dataset, batch_size=args.eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)

    model = ProtocolReplacementGRU(
        input_dim=2,
        hidden_dim=args.mapper_hidden_dim,
        num_layers=args.mapper_layers,
        dropout=args.mapper_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    best_mae = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []
    thr_mean = float(train_dataset.stats["thr_mean"])
    thr_std = float(train_dataset.stats["thr_std"])

    print(f"Starting protocol-replacement GRU baseline for {dataset_dir.name} on {device}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"{dataset_dir.name} protocol-gru epoch {epoch}/{args.epochs}", ncols=120)
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
        test_metrics = evaluate_model(model, test_loader, thr_mean, thr_std, device)
        history_item = {
            "epoch": epoch,
            "train_throughput_mae": float(train_metrics["throughput_mae"]),
            "train_throughput_mse": float(train_metrics["throughput_mse"]),
            "test_throughput_mae": float(test_metrics["throughput_mae"]),
            "test_throughput_mse": float(test_metrics["throughput_mse"]),
        }
        history.append(history_item)
        print(
            f"{dataset_dir.name} epoch {epoch}: "
            f"Train MAE={train_metrics['throughput_mae']:.4f} | "
            f"Test MAE={test_metrics['throughput_mae']:.4f} | "
            f"Train MSE={train_metrics['throughput_mse']:.4f} | "
            f"Test MSE={test_metrics['throughput_mse']:.4f}"
        )

        if test_metrics["throughput_mae"] < best_mae:
            best_mae = test_metrics["throughput_mae"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), model_path)
            print(f"{dataset_dir.name} saved new best protocol checkpoint at epoch {epoch}")
        else:
            stale_epochs += 1

        if args.patience > 0 and stale_epochs >= args.patience:
            print(f"{dataset_dir.name} early-stopped after {stale_epochs} stale epochs")
            break

    best_model = ProtocolReplacementGRU(
        input_dim=2,
        hidden_dim=args.mapper_hidden_dim,
        num_layers=args.mapper_layers,
        dropout=args.mapper_dropout,
    ).to(device)
    best_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    final_metrics = evaluate_model(best_model, test_loader, thr_mean, thr_std, device)

    result = {
        "dataset": dataset_dir.name,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "throughput_mae": float(final_metrics["throughput_mae"]),
        "throughput_mse": float(final_metrics["throughput_mse"]),
        "base_model_path": str(args.base_model_path),
        "model_path": str(model_path),
        "cache_path": str(cache_path),
        "curve_path": str(curve_path),
        "stats": train_dataset.stats,
        "config": {
            "hist_window": args.hist_window,
            "pre_window": args.pre_window,
            "skip_frames": args.skip_frames,
            "mapper_hidden_dim": args.mapper_hidden_dim,
            "mapper_layers": args.mapper_layers,
            "mapper_dropout": args.mapper_dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
    }

    metrics_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_history(history, curve_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
