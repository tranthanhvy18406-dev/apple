import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm

from run_single_dataset_experiment import calculate_throughput_from_predictions_parallel
from train_direct_horizon_crossstitch import (
    DirectHorizonCrossStitch,
    build_sinr_criterion,
    rate_proxy,
    val_to_index,
)


DOPPLERS = [5, 20, 50, 100, 200, 300, 400, 500]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Doppler split LEAF/Cross-Stitch experiment for 10 ms throughput forecasting."
    )
    parser.add_argument("--dataset_root", type=Path, default=Path("/4T/xty/new_mcs_dataset"))
    parser.add_argument("--output_dir", type=Path, default=Path("leaf_doppler_leave_one_out"))
    parser.add_argument("--dopplers", type=str, default="5,20,50,100,200,300,400,500")
    parser.add_argument(
        "--target_dopplers",
        type=str,
        default="all",
        help="Comma-separated target Dopplers or 'all'. Defaults to leave-one-out targets.",
    )
    parser.add_argument(
        "--source_dopplers",
        type=str,
        default=None,
        help="Optional comma-separated fixed source Dopplers. If omitted, each run uses leave-one-out sources.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="pooled,finetune,leaf,oracle",
        help="Comma-separated subset of pooled,finetune,leaf,oracle.",
    )

    parser.add_argument("--hist_window", type=int, default=16)
    parser.add_argument("--horizons", type=str, default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--support_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")

    # Cross-Stitch architecture defaults match the current best split-LSTM setup.
    parser.add_argument("--rb_hidden_dim", type=int, default=256)
    parser.add_argument("--sinr_hidden_dim", type=int, default=64)
    parser.add_argument("--rb_layers", type=int, default=2)
    parser.add_argument("--sinr_layers", type=int, default=2)
    parser.add_argument("--cross_stitch_mode", choices=["learn", "identity", "zeros", "none"], default="learn")
    parser.add_argument("--head_type", choices=["linear", "lstm", "split_lstm"], default="split_lstm")
    parser.add_argument("--head_hidden_dim", type=int, default=256)
    parser.add_argument("--rb_head_hidden_dim", type=int, default=256)
    parser.add_argument("--sinr_head_hidden_dim", type=int, default=256)
    parser.add_argument("--head_layers", type=int, default=1)
    parser.add_argument("--rb_head_layers", type=int, default=1)
    parser.add_argument("--sinr_head_layers", type=int, default=1)
    parser.add_argument("--horizon_embed_dim", type=int, default=32)
    parser.add_argument("--head_dropout", type=float, default=0.0)

    # Supervised training.
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--sinr_loss", choices=["mse", "mae", "huber"], default="huber")
    parser.add_argument("--huber_beta", type=float, default=0.1)
    parser.add_argument("--gamma_rb_dist", type=float, default=0.1)
    parser.add_argument("--gamma_rate_proxy", type=float, default=0.1)
    parser.add_argument(
        "--max_source_windows",
        type=int,
        default=0,
        help="Optional cap per source train split for quick debugging. 0 uses all source windows.",
    )

    # Fine-tuning baseline.
    parser.add_argument("--finetune_epochs", type=int, default=30)
    parser.add_argument("--finetune_lr", type=float, default=1e-4)
    parser.add_argument("--finetune_weight_decay", type=float, default=0.0)
    parser.add_argument("--finetune_scope", choices=["all", "head"], default="all")

    # Oracle baseline.
    parser.add_argument("--oracle_val_fraction", type=float, default=0.1)

    # LEAF-style latent extrapolation and adjustment.
    parser.add_argument("--leaf_epochs", type=int, default=80)
    parser.add_argument("--leaf_patience", type=int, default=15)
    parser.add_argument("--leaf_lr", type=float, default=5e-4)
    parser.add_argument("--leaf_weight_decay", type=float, default=1e-4)
    parser.add_argument("--leaf_latent_dim", type=int, default=64)
    parser.add_argument("--leaf_hidden_dim", type=int, default=128)
    parser.add_argument("--leaf_inner_steps", type=int, default=5)
    parser.add_argument("--leaf_target_inner_steps", type=int, default=20)
    parser.add_argument("--leaf_inner_lr", type=float, default=0.05)
    parser.add_argument("--leaf_support_size", type=int, default=128)
    parser.add_argument("--leaf_query_size", type=int, default=512)
    parser.add_argument("--leaf_num_task_segments", type=int, default=4)
    parser.add_argument("--leaf_rb_bias_scale", type=float, default=0.5)
    parser.add_argument("--leaf_sinr_shift_scale", type=float, default=0.25)
    parser.add_argument("--leaf_freeze_base", action="store_true", default=True)
    parser.add_argument("--leaf_train_base", action="store_false", dest="leaf_freeze_base")
    parser.add_argument("--leaf_disable_adjustment", action="store_true")
    return parser.parse_args()


def parse_int_list(value):
    if value == "all":
        return DOPPLERS[:]
    return [int(item) for item in value.split(",") if item.strip()]


def parse_methods(value):
    methods = [item.strip().lower() for item in value.split(",") if item.strip()]
    valid = {"pooled", "finetune", "leaf", "oracle"}
    bad = sorted(set(methods) - valid)
    if bad:
        raise ValueError(f"Unknown methods: {bad}")
    return methods


def resolve_source_hzs(args, target_hz, all_hzs):
    if args.source_dopplers is None:
        return [hz for hz in all_hzs if hz != target_hz]
    source_hzs = parse_int_list(args.source_dopplers)
    missing = [hz for hz in source_hzs if hz not in all_hzs]
    if missing:
        raise ValueError(f"Source Dopplers {missing} are not in doppler set {all_hzs}")
    if target_hz in source_hzs:
        raise ValueError(f"Target {target_hz}Hz is also present in source_dopplers={source_hzs}")
    if not source_hzs:
        raise ValueError("source_dopplers cannot be empty")
    return source_hzs


def fixed_source_cache_dir(args, source_hzs):
    if args.source_dopplers is None:
        return None
    tag = "_".join(str(hz) for hz in source_hzs)
    return args.output_dir / f"source_{tag}Hz"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _to_numpy(x):
    if isinstance(x, pd.DataFrame):
        return x.values
    if isinstance(x, pd.Series):
        return x.to_numpy()
    return np.asarray(x)


def flatten_series(payload, aliases):
    data = None
    if isinstance(payload, dict):
        for key in aliases:
            if key in payload:
                data = payload[key]
                break
    else:
        data = payload
    if data is None:
        raise KeyError(f"None of {aliases} found in payload.")
    arr = _to_numpy(data).astype(np.float32)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    return arr.reshape(-1)


def load_trace(path):
    payload = pd.read_pickle(path)
    return {
        "rb": flatten_series(payload, ["RB", "rb", "Rb", "rB"]),
        "sinr": flatten_series(payload, ["SINR", "sinr", "Sinr", "SNR", "snr", "eff_SINR", "eff_sinr"]),
    }


def dataset_dir(root, doppler_hz):
    return root / f"single_user_pf_{int(doppler_hz)}Hz"


def compute_sinr_stats(root, dopplers, split="train"):
    values = []
    filename = "train_9000_HDF5.pkl" if split == "train" else "test_1000_HDF5.pkl"
    for hz in dopplers:
        trace = load_trace(dataset_dir(root, hz) / filename)
        values.append(trace["sinr"])
    concat = np.concatenate(values).astype(np.float32)
    return float(concat.mean()), float(concat.std() + 1e-8)


class DopplerWindowDataset(Dataset):
    def __init__(
        self,
        path,
        hist_window,
        horizons,
        sinr_mean,
        sinr_std,
        start_window=0,
        end_window=None,
        doppler_hz=None,
    ):
        trace = load_trace(path)
        self.rb = torch.as_tensor(trace["rb"][:, None], dtype=torch.float32)
        self.sinr_raw = torch.as_tensor(trace["sinr"][:, None], dtype=torch.float32)
        self.sinr_mean = float(sinr_mean)
        self.sinr_std = float(sinr_std)
        self.sinr = (self.sinr_raw - self.sinr_mean) / self.sinr_std
        self.hist_window = int(hist_window)
        self.horizons = torch.as_tensor(horizons, dtype=torch.long)
        self.max_horizon = int(max(horizons))
        self.total_windows = len(self.rb) - self.hist_window - self.max_horizon + 1
        if self.total_windows <= 0:
            raise ValueError(f"Not enough frames in {path}")
        self.start_window = int(start_window)
        self.end_window = self.total_windows if end_window is None else int(end_window)
        self.start_window = max(0, self.start_window)
        self.end_window = min(self.total_windows, self.end_window)
        if self.end_window <= self.start_window:
            raise ValueError(f"Empty window slice [{start_window}, {end_window}) for {path}")
        self.doppler_hz = None if doppler_hz is None else int(doppler_hz)

    def __len__(self):
        return self.end_window - self.start_window

    def __getitem__(self, idx):
        idx = self.start_window + int(idx)
        input_end = idx + self.hist_window
        target_idx = input_end + self.horizons - 1
        rb_seq = self.rb[idx:input_end]
        sinr_seq = self.sinr[idx:input_end]
        rb_label = self.rb[target_idx]
        sinr_label = self.sinr[target_idx]
        if self.doppler_hz is None:
            return rb_seq, sinr_seq, rb_label, sinr_label
        return rb_seq, sinr_seq, rb_label, sinr_label, torch.tensor(self.doppler_hz, dtype=torch.float32)


def make_dataset(root, hz, split, hist_window, horizons, sinr_mean, sinr_std, start=0, end=None, with_doppler=False):
    filename = "train_9000_HDF5.pkl" if split == "train" else "test_1000_HDF5.pkl"
    return DopplerWindowDataset(
        dataset_dir(root, hz) / filename,
        hist_window=hist_window,
        horizons=horizons,
        sinr_mean=sinr_mean,
        sinr_std=sinr_std,
        start_window=start,
        end_window=end,
        doppler_hz=hz if with_doppler else None,
    )


def build_direct_model(args, horizons, device):
    return DirectHorizonCrossStitch(
        rb_hidden_size=args.rb_hidden_dim,
        sinr_hidden_size=args.sinr_hidden_dim,
        rb_num_layers=args.rb_layers,
        sinr_num_layers=args.sinr_layers,
        horizons=horizons,
        cross_stitch_mode=args.cross_stitch_mode,
        head_type=args.head_type,
        head_hidden_dim=args.head_hidden_dim,
        rb_head_hidden_dim=args.rb_head_hidden_dim,
        sinr_head_hidden_dim=args.sinr_head_hidden_dim,
        head_layers=args.head_layers,
        rb_head_layers=args.rb_head_layers,
        sinr_head_layers=args.sinr_head_layers,
        horizon_embed_dim=args.horizon_embed_dim,
        head_dropout=args.head_dropout,
    ).to(device)


class SupervisedObjective:
    def __init__(self, args, sinr_mean, sinr_std, device):
        self.args = args
        self.sinr_mean = float(sinr_mean)
        self.sinr_std = float(sinr_std)
        self.device = device
        self.criterion_rb = nn.CrossEntropyLoss()
        self.criterion_sinr = build_sinr_criterion(args.sinr_loss, args.huber_beta)
        self.classes = torch.arange(1, 107, device=device).float()

    def __call__(self, rb_logits, sinr_pred, rb_label, sinr_label):
        loss_rb = self.criterion_rb(rb_logits.reshape(-1, 106), val_to_index(rb_label.reshape(-1)))
        loss_sinr = self.criterion_sinr(sinr_pred, sinr_label)
        loss = loss_rb + loss_sinr

        if self.args.gamma_rb_dist > 0:
            probs = F.softmax(rb_logits, dim=2)
            rb_expect = (probs * self.classes).sum(dim=2)
            loss = loss + self.args.gamma_rb_dist * F.l1_loss(rb_expect, rb_label.squeeze(-1).float())

        if self.args.gamma_rate_proxy > 0:
            probs = F.softmax(rb_logits, dim=2)
            rb_expect = (probs * self.classes).sum(dim=2)
            pred_proxy = rate_proxy(rb_expect, sinr_pred.squeeze(-1), self.sinr_mean, self.sinr_std)
            true_proxy = rate_proxy(
                rb_label.squeeze(-1).float(),
                sinr_label.squeeze(-1),
                self.sinr_mean,
                self.sinr_std,
            )
            loss = loss + self.args.gamma_rate_proxy * F.smooth_l1_loss(
                pred_proxy / 50.0,
                true_proxy / 50.0,
                beta=0.05,
            )
        return loss


def unpack_batch(batch, device):
    if len(batch) == 5:
        rb_seq, sinr_seq, rb_label, sinr_label, doppler = batch
        doppler = doppler.to(device)
    else:
        rb_seq, sinr_seq, rb_label, sinr_label = batch
        doppler = None
    return (
        rb_seq.to(device),
        sinr_seq.to(device),
        rb_label.to(device),
        sinr_label.to(device),
        doppler,
    )


@torch.no_grad()
def evaluate_predictions(predict_fn, loader, sinr_mean, sinr_std, device):
    total = 0
    rb_correct = 0
    sums = {
        "rb_abs": 0.0,
        "sinr_abs": 0.0,
        "sinr_sq": 0.0,
        "thr_abs": 0.0,
        "thr_sq": 0.0,
    }
    by_h = None
    for batch in loader:
        rb_seq, sinr_seq, rb_label, sinr_label, doppler = unpack_batch(batch, device)
        rb_label_flat = rb_label.squeeze(-1)
        rb_logits, sinr_pred = predict_fn(rb_seq, sinr_seq, doppler)
        rb_class = torch.argmax(rb_logits, dim=2) + 1
        sinr_pred_denorm = (sinr_pred * sinr_std + sinr_mean).squeeze(-1)
        sinr_label_denorm = (sinr_label * sinr_std + sinr_mean).squeeze(-1)

        err_sinr = sinr_pred_denorm - sinr_label_denorm
        thr_pred = calculate_throughput_from_predictions_parallel(
            rb_class.reshape(-1),
            sinr_pred_denorm.reshape(-1),
            device,
        ).view_as(sinr_pred_denorm)
        thr_true = calculate_throughput_from_predictions_parallel(
            rb_label_flat.reshape(-1),
            sinr_label_denorm.reshape(-1),
            device,
        ).view_as(sinr_pred_denorm)
        err_thr = thr_pred - thr_true
        rb_abs = torch.abs(rb_class - rb_label_flat)

        total += rb_label_flat.numel()
        rb_correct += (rb_class == rb_label_flat).sum().item()
        sums["rb_abs"] += rb_abs.sum().item()
        sums["sinr_abs"] += torch.abs(err_sinr).sum().item()
        sums["sinr_sq"] += (err_sinr ** 2).sum().item()
        sums["thr_abs"] += torch.abs(err_thr).sum().item()
        sums["thr_sq"] += (err_thr ** 2).sum().item()

        batch_by_h = {
            "count": torch.full((rb_label_flat.shape[1],), rb_label_flat.shape[0], device=device, dtype=torch.float32),
            "rb_abs": rb_abs.float().sum(dim=0),
            "sinr_abs": torch.abs(err_sinr).sum(dim=0),
            "sinr_sq": (err_sinr ** 2).sum(dim=0),
            "thr_abs": torch.abs(err_thr).sum(dim=0),
            "thr_sq": (err_thr ** 2).sum(dim=0),
        }
        if by_h is None:
            by_h = {k: v.detach().cpu() for k, v in batch_by_h.items()}
        else:
            for k, v in batch_by_h.items():
                by_h[k] += v.detach().cpu()

    metrics = {
        "rb_acc": rb_correct / total,
        "rb_mae": sums["rb_abs"] / total,
        "sinr_mae": sums["sinr_abs"] / total,
        "sinr_mse": sums["sinr_sq"] / total,
        "throughput_mae": sums["thr_abs"] / total,
        "throughput_mse": sums["thr_sq"] / total,
    }
    horizon_metrics = []
    for i in range(len(by_h["count"])):
        denom = float(by_h["count"][i])
        horizon_metrics.append(
            {
                "output_index": i,
                "rb_mae": float(by_h["rb_abs"][i] / denom),
                "sinr_mae": float(by_h["sinr_abs"][i] / denom),
                "sinr_mse": float(by_h["sinr_sq"][i] / denom),
                "throughput_mae": float(by_h["thr_abs"][i] / denom),
                "throughput_mse": float(by_h["thr_sq"][i] / denom),
            }
        )
    return metrics, horizon_metrics


def direct_predict_fn(model):
    model.eval()

    def predict(rb_seq, sinr_seq, doppler=None):
        return model(rb_seq, sinr_seq)

    return predict


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_horizon_csv(path, target_hz, method, horizons, rows):
    fieldnames = [
        "target_hz",
        "method",
        "h",
        "time_ms",
        "rb_mae",
        "sinr_mae",
        "sinr_mse",
        "throughput_mae",
        "throughput_mse",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for h, row in zip(horizons, rows):
            writer.writerow(
                {
                    "target_hz": target_hz,
                    "method": method,
                    "h": h,
                    "time_ms": h * 10,
                    "rb_mae": row["rb_mae"],
                    "sinr_mae": row["sinr_mae"],
                    "sinr_mse": row["sinr_mse"],
                    "throughput_mae": row["throughput_mae"],
                    "throughput_mse": row["throughput_mse"],
                }
            )


def train_direct_supervised(
    args,
    model,
    train_loader,
    val_loader,
    objective,
    output_dir,
    sinr_mean,
    sinr_std,
    max_epochs=None,
    patience=None,
    lr=None,
    weight_decay=None,
    tag="model",
):
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"best_{tag}.pth"
    history_path = output_dir / "training_history.csv"
    max_epochs = args.epochs if max_epochs is None else int(max_epochs)
    patience = args.patience if patience is None else int(patience)
    lr = args.lr if lr is None else float(lr)
    weight_decay = args.weight_decay if weight_decay is None else float(weight_decay)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    device = objective.device
    best_value = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        pbar = tqdm(train_loader, desc=f"{tag} epoch {epoch}/{max_epochs}", ncols=110)
        for batch in pbar:
            rb_seq, sinr_seq, rb_label, sinr_label, _ = unpack_batch(batch, device)
            optimizer.zero_grad()
            rb_logits, sinr_pred = model(rb_seq, sinr_seq)
            loss = objective(rb_logits, sinr_pred, rb_label, sinr_label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(rb_seq)
            seen += len(rb_seq)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()

        val_metrics, _ = evaluate_predictions(direct_predict_fn(model), val_loader, sinr_mean, sinr_std, device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(seen, 1),
            **{k: float(v) for k, v in val_metrics.items()},
        }
        history.append(row)
        print(
            f"{tag} epoch {epoch}: train_loss={row['train_loss']:.4f} "
            f"val_throughput_mae={val_metrics['throughput_mae']:.4f}"
        )
        if val_metrics["throughput_mae"] < best_value:
            best_value = float(val_metrics["throughput_mae"])
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), model_path)
            print(f"{tag}: saved best checkpoint at epoch {epoch}")
        else:
            stale += 1
            if patience > 0 and stale >= patience:
                print(f"{tag}: early stopped at epoch {epoch}")
                break

    pd.DataFrame(history).to_csv(history_path, index=False)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    return {"best_epoch": best_epoch, "best_val_throughput_mae": best_value, "model_path": str(model_path)}


def configure_finetune_scope(model, scope):
    if scope == "all":
        for param in model.parameters():
            param.requires_grad = True
        return list(model.parameters())
    if scope != "head":
        raise ValueError(scope)

    for param in model.parameters():
        param.requires_grad = False
    head_prefixes = (
        "fusion",
        "horizon_embedding",
        "head_lstm",
        "head_norm",
        "rb_context",
        "sinr_context",
        "rb_head_lstms",
        "sinr_head_lstms",
        "head_cross_stitch_units",
        "rb_head_projections_up",
        "sinr_head_projections_up",
        "rb_head_projections_down",
        "sinr_head_projections_down",
        "rb_head_norm",
        "sinr_head_norm",
        "rb_head",
        "sinr_head",
    )
    trainable = []
    for name, param in model.named_parameters():
        if name.startswith(head_prefixes):
            param.requires_grad = True
            trainable.append(param)
    if not trainable:
        raise RuntimeError("No trainable head parameters were selected for fine-tuning.")
    return trainable


def finetune_direct_model(args, base_state, model, support_loader, objective):
    device = objective.device
    model.load_state_dict(base_state)
    trainable = configure_finetune_scope(model, args.finetune_scope)
    optimizer = torch.optim.AdamW(trainable, lr=args.finetune_lr, weight_decay=args.finetune_weight_decay)
    history = []
    for epoch in range(1, args.finetune_epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch in support_loader:
            rb_seq, sinr_seq, rb_label, sinr_label, _ = unpack_batch(batch, device)
            optimizer.zero_grad()
            rb_logits, sinr_pred = model(rb_seq, sinr_seq)
            loss = objective(rb_logits, sinr_pred, rb_label, sinr_label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            total_loss += loss.item() * len(rb_seq)
            seen += len(rb_seq)
        history.append(
            {
                "epoch": epoch,
                "support_loss": total_loss / max(seen, 1),
                "finetune_scope": args.finetune_scope,
                "trainable_params": int(sum(param.numel() for param in trainable)),
            }
        )
    return history


class LeafCrossStitch(nn.Module):
    def __init__(
        self,
        base_model,
        horizons,
        latent_dim=64,
        hidden_dim=128,
        rb_bias_scale=0.5,
        sinr_shift_scale=0.25,
        use_adjustment=True,
    ):
        super().__init__()
        self.base_model = base_model
        self.horizons = tuple(horizons)
        self.num_horizons = len(horizons)
        self.latent_dim = int(latent_dim)
        self.rb_bias_scale = float(rb_bias_scale)
        self.sinr_shift_scale = float(sinr_shift_scale)
        self.use_adjustment = bool(use_adjustment)

        self.extrapolator = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        decoder_out = self.num_horizons * 106 + self.num_horizons
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, decoder_out),
        )
        # The zero init makes LEAF start exactly from the pooled Cross-Stitch model.
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

        feature_dim = 9 + latent_dim
        self.adjustment = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        nn.init.zeros_(self.adjustment[-1].weight)
        nn.init.zeros_(self.adjustment[-1].bias)

    def encode_doppler(self, doppler):
        doppler = doppler.to(dtype=torch.float32)
        log_hz = torch.log(doppler.clamp_min(1.0))
        norm = (log_hz - math.log(50.0)) / math.log(100.0)
        return torch.stack([norm, norm.square()], dim=-1)

    def initial_latent(self, doppler):
        if not torch.is_tensor(doppler):
            doppler = torch.tensor([float(doppler)], device=next(self.parameters()).device)
        if doppler.ndim == 0:
            doppler = doppler[None]
        return self.extrapolator(self.encode_doppler(doppler))

    def sample_features(self, rb_seq, sinr_seq, rb_logits, sinr_pred, latent):
        rb_scaled = rb_seq[..., 0] / 106.0
        rb_prob = F.softmax(rb_logits.detach(), dim=2)
        classes = torch.arange(1, 107, device=rb_seq.device, dtype=rb_seq.dtype)
        rb_expect = (rb_prob * classes).sum(dim=2) / 106.0
        pred_sinr = sinr_pred.detach().squeeze(-1)
        features = torch.stack(
            [
                rb_scaled[:, -1],
                rb_scaled.mean(dim=1),
                rb_scaled.std(dim=1, unbiased=False),
                sinr_seq[:, -1, 0],
                sinr_seq[:, :, 0].mean(dim=1),
                sinr_seq[:, :, 0].std(dim=1, unbiased=False),
                rb_expect.mean(dim=1),
                pred_sinr.mean(dim=1),
                pred_sinr.std(dim=1, unbiased=False),
            ],
            dim=1,
        )
        return torch.cat([features, latent], dim=1)

    def forward(self, rb_seq, sinr_seq, doppler=None, latent=None, sample_adjust=True):
        rb_logits, sinr_pred = self.base_model(rb_seq, sinr_seq)
        batch_size = rb_seq.shape[0]
        if latent is None:
            if doppler is None:
                raise ValueError("Either doppler or latent must be provided.")
            latent = self.initial_latent(doppler)
        if latent.ndim == 1:
            latent = latent.unsqueeze(0)
        if latent.shape[0] == 1 and batch_size != 1:
            latent = latent.expand(batch_size, -1)
        elif latent.shape[0] != batch_size:
            raise ValueError(f"Latent batch {latent.shape[0]} does not match input batch {batch_size}.")

        sample_latent = latent
        if self.use_adjustment and sample_adjust:
            sample_latent = latent + self.adjustment(self.sample_features(rb_seq, sinr_seq, rb_logits, sinr_pred, latent))

        params = self.decoder(sample_latent)
        rb_bias = params[:, : self.num_horizons * 106].view(batch_size, self.num_horizons, 106)
        sinr_shift = params[:, self.num_horizons * 106 :].view(batch_size, self.num_horizons, 1)
        return (
            rb_logits + self.rb_bias_scale * rb_bias,
            sinr_pred + self.sinr_shift_scale * sinr_shift,
        )


def make_leaf_tasks(args, root, source_hzs, horizons, sinr_mean, sinr_std):
    tasks = []
    support_size = int(args.leaf_support_size)
    query_size = int(args.leaf_query_size)
    gap = int(max(horizons))
    segment_count = max(1, int(args.leaf_num_task_segments))
    for hz in source_hzs:
        full = make_dataset(root, hz, "train", args.hist_window, horizons, sinr_mean, sinr_std, with_doppler=True)
        span = support_size + gap + query_size
        max_start = max(0, full.total_windows - span)
        if segment_count == 1:
            starts = [0]
        else:
            starts = np.linspace(0, max_start, num=segment_count, dtype=int).tolist()
        for segment_id, start in enumerate(starts):
            support = make_dataset(
                root,
                hz,
                "train",
                args.hist_window,
                horizons,
                sinr_mean,
                sinr_std,
                start=start,
                end=start + support_size,
                with_doppler=True,
            )
            query_start = start + support_size + gap
            query = make_dataset(
                root,
                hz,
                "train",
                args.hist_window,
                horizons,
                sinr_mean,
                sinr_std,
                start=query_start,
                end=query_start + query_size,
                with_doppler=True,
            )
            tasks.append(
                {
                    "hz": hz,
                    "segment_id": segment_id,
                    "support": support,
                    "query": query,
                    "gap_windows": gap,
                    "support_start": start,
                    "query_start": query_start,
                }
            )
    return tasks


def one_batch_loader(dataset, batch_size, shuffle):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
    return next(iter(loader))


def adapt_leaf_latent(leaf_model, support_batch, objective, doppler_hz, steps, inner_lr, first_order=False):
    device = objective.device
    rb_seq, sinr_seq, rb_label, sinr_label, _ = unpack_batch(support_batch, device)
    latent = leaf_model.initial_latent(torch.tensor([float(doppler_hz)], device=device)).squeeze(0)
    for _ in range(int(steps)):
        rb_logits, sinr_pred = leaf_model(rb_seq, sinr_seq, latent=latent, sample_adjust=True)
        loss = objective(rb_logits, sinr_pred, rb_label, sinr_label)
        grad = torch.autograd.grad(loss, latent, create_graph=not first_order, retain_graph=True)[0]
        if first_order:
            grad = grad.detach()
        latent = latent - float(inner_lr) * grad
    return latent


def train_leaf_model(args, leaf_model, tasks, objective, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    device = objective.device
    if args.leaf_freeze_base:
        for param in leaf_model.base_model.parameters():
            param.requires_grad = False
    trainable = [param for param in leaf_model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.leaf_lr, weight_decay=args.leaf_weight_decay)
    model_path = output_dir / "best_leaf_crossstitch.pth"
    history = []
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    for epoch in range(1, args.leaf_epochs + 1):
        random.shuffle(tasks)
        leaf_model.train()
        total_loss = 0.0
        total_tasks = 0
        for task in tqdm(tasks, desc=f"leaf epoch {epoch}/{args.leaf_epochs}", ncols=110):
            support_batch = one_batch_loader(task["support"], args.leaf_support_size, shuffle=True)
            query_loader = DataLoader(task["query"], batch_size=args.batch_size, shuffle=True, drop_last=False)
            query_batch = next(iter(query_loader))
            rb_seq, sinr_seq, rb_label, sinr_label, _ = unpack_batch(query_batch, device)

            optimizer.zero_grad()
            adapted_latent = adapt_leaf_latent(
                leaf_model,
                support_batch,
                objective,
                task["hz"],
                args.leaf_inner_steps,
                args.leaf_inner_lr,
                first_order=True,
            )
            rb_logits, sinr_pred = leaf_model(rb_seq, sinr_seq, latent=adapted_latent, sample_adjust=True)
            query_loss = objective(rb_logits, sinr_pred, rb_label, sinr_label)
            query_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            total_loss += query_loss.item()
            total_tasks += 1

        avg_loss = total_loss / max(total_tasks, 1)
        history.append({"epoch": epoch, "source_task_query_loss": avg_loss})
        print(f"leaf epoch {epoch}: source_task_query_loss={avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            stale = 0
            torch.save(leaf_model.state_dict(), model_path)
            print(f"leaf: saved best checkpoint at epoch {epoch}")
        else:
            stale += 1
            if args.leaf_patience > 0 and stale >= args.leaf_patience:
                print(f"leaf: early stopped at epoch {epoch}")
                break
    pd.DataFrame(history).to_csv(output_dir / "leaf_training_history.csv", index=False)
    leaf_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    return {"best_epoch": best_epoch, "best_source_task_query_loss": best_loss, "model_path": str(model_path)}


def leaf_predict_fn(leaf_model, latent):
    leaf_model.eval()

    def predict(rb_seq, sinr_seq, doppler=None):
        return leaf_model(rb_seq, sinr_seq, latent=latent, sample_adjust=True)

    return predict


def evaluate_method(
    output_dir,
    target_hz,
    method,
    horizons,
    metrics,
    horizon_rows,
    extra_payload,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "target_hz": target_hz,
        "method": method,
        "sample_interval_ms": 10,
        "hist_window": extra_payload.get("hist_window"),
        "horizons": horizons,
        "support_size": extra_payload.get("support_size"),
        "query_size": extra_payload.get("query_size"),
        "support_query_gap": extra_payload.get("support_query_gap"),
        "query_start_window": extra_payload.get("query_start_window"),
        "metrics": metrics,
        "extra": extra_payload,
    }
    write_json(output_dir / "metrics.json", payload)
    write_horizon_csv(output_dir / "per_horizon_metrics.csv", target_hz, method, horizons, horizon_rows)
    return payload


def run_target(args, target_hz, all_hzs, methods, horizons, device):
    source_hzs = resolve_source_hzs(args, target_hz, all_hzs)
    run_dir = args.output_dir / f"target_{target_hz}Hz"
    run_dir.mkdir(parents=True, exist_ok=True)
    support_query_gap = int(max(horizons))
    query_start = int(args.support_size + support_query_gap)

    source_sinr_mean, source_sinr_std = compute_sinr_stats(args.dataset_root, source_hzs, split="train")
    support_ds = make_dataset(
        args.dataset_root,
        target_hz,
        "test",
        args.hist_window,
        horizons,
        source_sinr_mean,
        source_sinr_std,
        start=0,
        end=args.support_size,
    )
    query_ds = make_dataset(
        args.dataset_root,
        target_hz,
        "test",
        args.hist_window,
        horizons,
        source_sinr_mean,
        source_sinr_std,
        start=query_start,
        end=None,
    )
    support_loader = DataLoader(support_ds, batch_size=min(args.support_size, args.batch_size), shuffle=True, drop_last=False)
    query_loader = DataLoader(query_ds, batch_size=args.eval_batch_size, shuffle=False, drop_last=False)

    objective = SupervisedObjective(args, source_sinr_mean, source_sinr_std, device)
    source_cache_dir = fixed_source_cache_dir(args, source_hzs)
    source_pooled_dir = (source_cache_dir / "pooled") if source_cache_dir is not None else (run_dir / "pooled")
    base_state_path = source_pooled_dir / "best_pooled.pth"
    pooled_payload = None

    if any(method in methods for method in ["pooled", "finetune", "leaf"]):
        source_train_sets = []
        source_val_sets = []
        for hz in source_hzs:
            train_end = None if args.max_source_windows <= 0 else args.max_source_windows
            source_train_sets.append(
                make_dataset(
                    args.dataset_root,
                    hz,
                    "train",
                    args.hist_window,
                    horizons,
                    source_sinr_mean,
                    source_sinr_std,
                    start=0,
                    end=train_end,
                )
            )
            source_val_sets.append(
                make_dataset(
                    args.dataset_root,
                    hz,
                    "test",
                    args.hist_window,
                    horizons,
                    source_sinr_mean,
                    source_sinr_std,
                    start=0,
                    end=None,
                )
            )
        pooled_eval_dir = run_dir / "pooled"
        pooled_metrics_path = pooled_eval_dir / "metrics.json"
        pooled_train_info_path = source_pooled_dir / "train_info.json"
        pooled_model = build_direct_model(args, horizons, device)
        if base_state_path.exists() and not args.force:
            pooled_model.load_state_dict(torch.load(base_state_path, map_location=device, weights_only=True))
            if pooled_train_info_path.exists():
                train_info = json.loads(pooled_train_info_path.read_text(encoding="utf-8"))
            else:
                train_info = {"model_path": str(base_state_path), "reused_source_checkpoint": True}
        else:
            train_loader = DataLoader(
                ConcatDataset(source_train_sets),
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True,
            )
            val_loader = DataLoader(
                ConcatDataset(source_val_sets),
                batch_size=args.eval_batch_size,
                shuffle=False,
                drop_last=False,
            )
            train_info = train_direct_supervised(
                args,
                pooled_model,
                train_loader,
                val_loader,
                objective,
                source_pooled_dir,
                source_sinr_mean,
                source_sinr_std,
                tag="pooled",
            )
            write_json(pooled_train_info_path, train_info)
        if pooled_metrics_path.exists() and not args.force:
            pooled_payload = json.loads(pooled_metrics_path.read_text(encoding="utf-8"))
        else:
            target_metrics, target_by_h = evaluate_predictions(
                direct_predict_fn(pooled_model),
                query_loader,
                source_sinr_mean,
                source_sinr_std,
                device,
            )
            pooled_payload = evaluate_method(
                pooled_eval_dir,
                target_hz,
                "pooled",
                horizons,
                target_metrics,
                target_by_h,
                {
                    "hist_window": args.hist_window,
                    "support_size": args.support_size,
                    "query_size": len(query_ds),
                    "support_query_gap": support_query_gap,
                    "query_start_window": query_start,
                    "source_hzs": source_hzs,
                    "sinr_mean": source_sinr_mean,
                    "sinr_std": source_sinr_std,
                    "train_info": train_info,
                    "model_path": str(base_state_path),
                    "source_cache_dir": str(source_cache_dir) if source_cache_dir is not None else None,
                    "no_target_adaptation": True,
                },
            )
        base_state = torch.load(base_state_path, map_location=device, weights_only=True)
    else:
        base_state = None

    results = []
    if "pooled" in methods and pooled_payload is not None:
        results.append(pooled_payload)

    if "finetune" in methods:
        finetune_dir = run_dir / "finetune"
        metrics_path = finetune_dir / "metrics.json"
        if metrics_path.exists() and not args.force:
            results.append(json.loads(metrics_path.read_text(encoding="utf-8")))
        else:
            finetune_dir.mkdir(parents=True, exist_ok=True)
            model = build_direct_model(args, horizons, device)
            history = finetune_direct_model(args, base_state, model, support_loader, objective)
            torch.save(model.state_dict(), finetune_dir / "finetuned_from_pooled.pth")
            pd.DataFrame(history).to_csv(finetune_dir / "finetune_history.csv", index=False)
            metrics, by_h = evaluate_predictions(direct_predict_fn(model), query_loader, source_sinr_mean, source_sinr_std, device)
            payload = evaluate_method(
                finetune_dir,
                target_hz,
                "pooled_finetune",
                horizons,
                metrics,
                by_h,
                {
                    "hist_window": args.hist_window,
                    "support_size": args.support_size,
                    "query_size": len(query_ds),
                    "support_query_gap": support_query_gap,
                    "query_start_window": query_start,
                    "source_hzs": source_hzs,
                    "sinr_mean": source_sinr_mean,
                    "sinr_std": source_sinr_std,
                    "base_model_path": str(base_state_path),
                    "finetune_epochs": args.finetune_epochs,
                    "finetune_lr": args.finetune_lr,
                    "finetune_scope": args.finetune_scope,
                },
            )
            results.append(payload)

    if "leaf" in methods:
        leaf_eval_dir = run_dir / "leaf"
        leaf_train_dir = (source_cache_dir / "leaf") if source_cache_dir is not None else leaf_eval_dir
        metrics_path = leaf_eval_dir / "metrics.json"
        leaf_model_path = leaf_train_dir / "best_leaf_crossstitch.pth"
        leaf_train_info_path = leaf_train_dir / "train_info.json"
        if metrics_path.exists() and not args.force:
            results.append(json.loads(metrics_path.read_text(encoding="utf-8")))
        else:
            leaf_eval_dir.mkdir(parents=True, exist_ok=True)
            leaf_base = build_direct_model(args, horizons, device)
            leaf_base.load_state_dict(base_state)
            leaf_model = LeafCrossStitch(
                leaf_base,
                horizons=horizons,
                latent_dim=args.leaf_latent_dim,
                hidden_dim=args.leaf_hidden_dim,
                rb_bias_scale=args.leaf_rb_bias_scale,
                sinr_shift_scale=args.leaf_sinr_shift_scale,
                use_adjustment=not args.leaf_disable_adjustment,
            ).to(device)
            if leaf_model_path.exists() and not args.force:
                leaf_model.load_state_dict(torch.load(leaf_model_path, map_location=device, weights_only=True))
                if leaf_train_info_path.exists():
                    train_info = json.loads(leaf_train_info_path.read_text(encoding="utf-8"))
                else:
                    train_info = {"model_path": str(leaf_model_path), "reused_source_checkpoint": True}
            else:
                leaf_tasks = make_leaf_tasks(args, args.dataset_root, source_hzs, horizons, source_sinr_mean, source_sinr_std)
                train_info = train_leaf_model(args, leaf_model, leaf_tasks, objective, leaf_train_dir)
                write_json(leaf_train_info_path, train_info)

            support_batch = one_batch_loader(support_ds, min(args.support_size, args.batch_size), shuffle=False)
            adapted_latent = adapt_leaf_latent(
                leaf_model,
                support_batch,
                objective,
                target_hz,
                args.leaf_target_inner_steps,
                args.leaf_inner_lr,
                first_order=True,
            ).detach()
            torch.save({"latent": adapted_latent.cpu(), "target_hz": target_hz}, leaf_eval_dir / "target_adapted_latent.pth")
            metrics, by_h = evaluate_predictions(
                leaf_predict_fn(leaf_model, adapted_latent),
                query_loader,
                source_sinr_mean,
                source_sinr_std,
                device,
            )
            payload = evaluate_method(
                leaf_eval_dir,
                target_hz,
                "leaf_crossstitch",
                horizons,
                metrics,
                by_h,
                {
                    "hist_window": args.hist_window,
                    "support_size": args.support_size,
                    "query_size": len(query_ds),
                    "support_query_gap": support_query_gap,
                    "query_start_window": query_start,
                    "source_hzs": source_hzs,
                    "sinr_mean": source_sinr_mean,
                    "sinr_std": source_sinr_std,
                    "base_model_path": str(base_state_path),
                    "source_cache_dir": str(source_cache_dir) if source_cache_dir is not None else None,
                    "train_info": train_info,
                    "leaf_latent_dim": args.leaf_latent_dim,
                    "leaf_inner_steps": args.leaf_inner_steps,
                    "leaf_target_inner_steps": args.leaf_target_inner_steps,
                    "leaf_inner_lr": args.leaf_inner_lr,
                    "leaf_support_size": args.leaf_support_size,
                    "leaf_query_size": args.leaf_query_size,
                    "leaf_support_query_gap": support_query_gap,
                    "leaf_num_task_segments": args.leaf_num_task_segments,
                    "leaf_freeze_base": args.leaf_freeze_base,
                    "leaf_adjustment": not args.leaf_disable_adjustment,
                },
            )
            results.append(payload)

    if "oracle" in methods:
        oracle_dir = run_dir / "oracle"
        metrics_path = oracle_dir / "metrics.json"
        if metrics_path.exists() and not args.force:
            results.append(json.loads(metrics_path.read_text(encoding="utf-8")))
        else:
            oracle_sinr_mean, oracle_sinr_std = compute_sinr_stats(args.dataset_root, [target_hz], split="train")
            target_train_full = make_dataset(
                args.dataset_root,
                target_hz,
                "train",
                args.hist_window,
                horizons,
                oracle_sinr_mean,
                oracle_sinr_std,
            )
            val_count = max(1, int(len(target_train_full) * args.oracle_val_fraction))
            train_count = len(target_train_full) - val_count
            oracle_train = make_dataset(
                args.dataset_root,
                target_hz,
                "train",
                args.hist_window,
                horizons,
                oracle_sinr_mean,
                oracle_sinr_std,
                start=0,
                end=train_count,
            )
            oracle_val = make_dataset(
                args.dataset_root,
                target_hz,
                "train",
                args.hist_window,
                horizons,
                oracle_sinr_mean,
                oracle_sinr_std,
                start=train_count,
                end=None,
            )
            oracle_query = make_dataset(
                args.dataset_root,
                target_hz,
                "test",
                args.hist_window,
                horizons,
                oracle_sinr_mean,
                oracle_sinr_std,
                start=query_start,
                end=None,
            )
            oracle_objective = SupervisedObjective(args, oracle_sinr_mean, oracle_sinr_std, device)
            oracle_model = build_direct_model(args, horizons, device)
            train_info = train_direct_supervised(
                args,
                oracle_model,
                DataLoader(oracle_train, batch_size=args.batch_size, shuffle=True, drop_last=True),
                DataLoader(oracle_val, batch_size=args.eval_batch_size, shuffle=False, drop_last=False),
                oracle_objective,
                oracle_dir,
                oracle_sinr_mean,
                oracle_sinr_std,
                tag="oracle",
            )
            metrics, by_h = evaluate_predictions(
                direct_predict_fn(oracle_model),
                DataLoader(oracle_query, batch_size=args.eval_batch_size, shuffle=False, drop_last=False),
                oracle_sinr_mean,
                oracle_sinr_std,
                device,
            )
            payload = evaluate_method(
                oracle_dir,
                target_hz,
                "target_oracle",
                horizons,
                metrics,
                by_h,
                {
                    "hist_window": args.hist_window,
                    "support_size": args.support_size,
                    "query_size": len(oracle_query),
                    "support_query_gap": support_query_gap,
                    "query_start_window": query_start,
                    "target_train_windows": len(oracle_train),
                    "target_val_windows": len(oracle_val),
                    "sinr_mean": oracle_sinr_mean,
                    "sinr_std": oracle_sinr_std,
                    "train_info": train_info,
                    "oracle_val_fraction": args.oracle_val_fraction,
                },
            )
            results.append(payload)

    return results


def collect_existing_results(output_dir):
    payloads = []
    for metrics_path in sorted(output_dir.glob("target_*Hz/*/metrics.json")):
        try:
            payloads.append(json.loads(metrics_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            print(f"skip malformed metrics file: {metrics_path}")
    return payloads


def write_overall_summary(output_dir, results):
    output_dir.mkdir(parents=True, exist_ok=True)
    by_key = {}
    for payload in collect_existing_results(output_dir) + list(results):
        by_key[(payload["target_hz"], payload["method"])] = payload
    rows = []
    for payload in by_key.values():
        metrics = payload["metrics"]
        rows.append(
            {
                "target_hz": payload["target_hz"],
                "method": payload["method"],
                "throughput_mae": metrics["throughput_mae"],
                "throughput_mse": metrics["throughput_mse"],
                "rb_mae": metrics["rb_mae"],
                "sinr_mae": metrics["sinr_mae"],
                "sinr_mse": metrics["sinr_mse"],
                "support_size": payload["support_size"],
                "query_size": payload["query_size"],
                "support_query_gap": payload.get("support_query_gap"),
                "query_start_window": payload.get("query_start_window"),
                "source_hzs": ",".join(str(hz) for hz in payload.get("extra", {}).get("source_hzs", [])),
            }
        )
    summary = pd.DataFrame(rows).sort_values(["target_hz", "method"])
    summary.to_csv(output_dir / "summary_by_target_method.csv", index=False)
    if not summary.empty:
        avg = (
            summary.groupby("method", as_index=False)
            .agg(
                throughput_mae_mean=("throughput_mae", "mean"),
                throughput_mae_std=("throughput_mae", "std"),
                rb_mae_mean=("rb_mae", "mean"),
                sinr_mae_mean=("sinr_mae", "mean"),
                num_targets=("target_hz", "count"),
            )
            .sort_values("throughput_mae_mean")
        )
        avg.to_csv(output_dir / "summary_average_by_method.csv", index=False)


def main():
    args = parse_args()
    seed_everything(args.seed)
    all_hzs = parse_int_list(args.dopplers)
    target_hzs = all_hzs if args.target_dopplers == "all" else parse_int_list(args.target_dopplers)
    methods = parse_methods(args.methods)
    horizons = parse_int_list(args.horizons)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    all_results = []
    for target_hz in target_hzs:
        if target_hz not in all_hzs:
            raise ValueError(f"Target {target_hz} is not in doppler set {all_hzs}")
        source_hzs = resolve_source_hzs(args, target_hz, all_hzs)
        split_name = "fixed-source" if args.source_dopplers is not None else "leave-one-out"
        print(f"\n=== {split_name} target {target_hz}Hz | source {source_hzs} ===")
        all_results.extend(run_target(args, target_hz, all_hzs, methods, horizons, device))
        write_overall_summary(args.output_dir, all_results)

    write_overall_summary(args.output_dir, all_results)
    print(f"wrote {args.output_dir / 'summary_by_target_method.csv'}")
    print(f"wrote {args.output_dir / 'summary_average_by_method.csv'}")


if __name__ == "__main__":
    main()
