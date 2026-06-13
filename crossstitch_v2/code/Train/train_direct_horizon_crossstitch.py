import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mtl_model import CrossStitchUnit
from run_single_dataset_experiment import calculate_throughput_from_predictions_parallel


def parse_args():
    parser = argparse.ArgumentParser(description="Direct horizon Cross-Stitch model without decoder recursion.")
    parser.add_argument("--dataset_dir", type=Path, default=Path("/4T/xty/new_mcs_dataset/single_user_pf_5Hz"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--hist_window", type=int, default=16)
    parser.add_argument(
        "--horizons",
        type=str,
        default="1,2,3,4,5,6,7,8,9,10",
        help="Comma-separated future frame offsets. h=10 means +100 ms for 10 ms frames.",
    )
    parser.add_argument("--skip_train_frames", type=int, default=0)
    parser.add_argument("--rb_hidden_dim", type=int, default=256)
    parser.add_argument("--sinr_hidden_dim", type=int, default=64)
    parser.add_argument("--rb_layers", type=int, default=3)
    parser.add_argument("--sinr_layers", type=int, default=1)
    parser.add_argument("--cross_stitch_mode", choices=["learn", "identity", "zeros", "none"], default="learn")
    parser.add_argument("--head_type", choices=["linear", "lstm", "split_lstm"], default="linear")
    parser.add_argument("--head_hidden_dim", type=int, default=256)
    parser.add_argument("--rb_head_hidden_dim", type=int, default=None)
    parser.add_argument("--sinr_head_hidden_dim", type=int, default=None)
    parser.add_argument("--head_layers", type=int, default=1)
    parser.add_argument("--rb_head_layers", type=int, default=None)
    parser.add_argument("--sinr_head_layers", type=int, default=None)
    parser.add_argument("--horizon_embed_dim", type=int, default=32)
    parser.add_argument("--head_dropout", type=float, default=0.1)
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
    parser.add_argument("--selection_metric", choices=["throughput_mae", "sinr_mae"], default="throughput_mae")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


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


class DirectHorizonDataset(Dataset):
    def __init__(
        self,
        path,
        hist_window,
        horizons,
        skip_initial_frames=0,
        sinr_mean=None,
        sinr_std=None,
    ):
        payload = pd.read_pickle(path)
        rb = flatten_series(payload, ["RB", "rb", "Rb", "rB"])
        sinr = flatten_series(payload, ["SINR", "sinr", "Sinr", "SNR", "snr", "eff_SINR", "eff_sinr"])
        if skip_initial_frames > 0:
            rb = rb[skip_initial_frames:]
            sinr = sinr[skip_initial_frames:]

        self.rb_raw = torch.as_tensor(rb[:, None], dtype=torch.float32)
        self.sinr_raw = torch.as_tensor(sinr[:, None], dtype=torch.float32)
        if sinr_mean is None or sinr_std is None:
            self.sinr_mean = float(self.sinr_raw.mean())
            self.sinr_std = float(self.sinr_raw.std() + 1e-8)
        else:
            self.sinr_mean = float(sinr_mean)
            self.sinr_std = float(sinr_std)
        self.sinr_norm = (self.sinr_raw - self.sinr_mean) / self.sinr_std

        self.hist_window = int(hist_window)
        self.horizons = torch.as_tensor(horizons, dtype=torch.long)
        self.max_horizon = int(max(horizons))
        self.num_samples = len(self.rb_raw) - self.hist_window - self.max_horizon + 1
        if self.num_samples <= 0:
            raise ValueError("Not enough frames for requested direct horizons.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_end = idx + self.hist_window
        target_idx = input_end + self.horizons - 1
        rb_seq = self.rb_raw[idx:input_end]
        sinr_seq = self.sinr_norm[idx:input_end]
        rb_label = self.rb_raw[target_idx]
        sinr_label = self.sinr_norm[target_idx]
        return rb_seq, sinr_seq, rb_label, sinr_label


class DirectHorizonCrossStitch(nn.Module):
    def __init__(
        self,
        rb_hidden_size,
        sinr_hidden_size,
        rb_num_layers,
        sinr_num_layers,
        horizons,
        rb_output_size=106,
        cross_stitch_mode="learn",
        head_type="linear",
        head_hidden_dim=256,
        rb_head_hidden_dim=None,
        sinr_head_hidden_dim=None,
        head_layers=1,
        rb_head_layers=None,
        sinr_head_layers=None,
        horizon_embed_dim=32,
        head_dropout=0.1,
    ):
        super().__init__()
        self.horizons = tuple(horizons)
        self.num_horizons = len(horizons)
        self.rb_output_size = rb_output_size
        self.cross_stitch_mode = cross_stitch_mode
        self.head_type = head_type
        self.rb_head_layers = int(head_layers if rb_head_layers is None else rb_head_layers)
        self.sinr_head_layers = int(head_layers if sinr_head_layers is None else sinr_head_layers)
        self.rb_head_hidden_dim = int(head_hidden_dim if rb_head_hidden_dim is None else rb_head_hidden_dim)
        self.sinr_head_hidden_dim = int(head_hidden_dim if sinr_head_hidden_dim is None else sinr_head_hidden_dim)
        if self.rb_head_layers < 1 or self.sinr_head_layers < 1:
            raise ValueError("RB/SINR head layers must be positive.")
        if self.rb_head_hidden_dim < 1 or self.sinr_head_hidden_dim < 1:
            raise ValueError("RB/SINR head hidden dims must be positive.")

        self.rb_lstms = nn.ModuleList(
            [nn.LSTM(1, rb_hidden_size, batch_first=True)]
            + [nn.LSTM(rb_hidden_size, rb_hidden_size, batch_first=True) for _ in range(rb_num_layers - 1)]
        )
        self.sinr_lstms = nn.ModuleList(
            [nn.LSTM(1, sinr_hidden_size, batch_first=True)]
            + [nn.LSTM(sinr_hidden_size, sinr_hidden_size, batch_first=True) for _ in range(sinr_num_layers - 1)]
        )
        self.num_cross_stitch = 0 if cross_stitch_mode == "none" else min(rb_num_layers, sinr_num_layers)
        if self.num_cross_stitch > 0:
            self.cross_stitch_dim = max(rb_hidden_size, sinr_hidden_size)
            self.rb_projections_up = nn.ModuleList()
            self.sinr_projections_up = nn.ModuleList()
            self.cross_stitch_units = nn.ModuleList()
            self.rb_projections_down = nn.ModuleList()
            self.sinr_projections_down = nn.ModuleList()
            for _ in range(self.num_cross_stitch):
                self.rb_projections_up.append(nn.Linear(rb_hidden_size, self.cross_stitch_dim))
                self.sinr_projections_up.append(nn.Linear(sinr_hidden_size, self.cross_stitch_dim))
                unit = CrossStitchUnit(self.cross_stitch_dim)
                if cross_stitch_mode == "identity":
                    unit.alpha.data = torch.eye(2, 2)
                    unit.alpha.requires_grad = False
                elif cross_stitch_mode == "zeros":
                    unit.alpha.data = torch.zeros(2, 2)
                    unit.alpha.requires_grad = False
                self.cross_stitch_units.append(unit)
                self.rb_projections_down.append(nn.Linear(self.cross_stitch_dim, rb_hidden_size))
                self.sinr_projections_down.append(nn.Linear(self.cross_stitch_dim, sinr_hidden_size))

        direct_dim = max(rb_hidden_size, sinr_hidden_size)
        self.proj_rb = nn.Linear(rb_hidden_size, direct_dim) if rb_hidden_size != direct_dim else nn.Identity()
        self.proj_sinr = nn.Linear(sinr_hidden_size, direct_dim) if sinr_hidden_size != direct_dim else nn.Identity()
        self.fusion = nn.Sequential(
            nn.LayerNorm(direct_dim),
            nn.Linear(direct_dim, direct_dim),
            nn.ReLU(),
            nn.Dropout(head_dropout),
        )
        if head_type == "linear":
            self.rb_head = nn.Linear(direct_dim, self.num_horizons * rb_output_size)
            self.sinr_head = nn.Linear(direct_dim, self.num_horizons)
        elif head_type == "lstm":
            self.horizon_embedding = nn.Embedding(self.num_horizons, horizon_embed_dim)
            lstm_dropout = head_dropout if head_layers > 1 else 0.0
            self.head_lstm = nn.LSTM(
                direct_dim + horizon_embed_dim,
                head_hidden_dim,
                num_layers=head_layers,
                batch_first=True,
                dropout=lstm_dropout,
            )
            self.head_norm = nn.LayerNorm(head_hidden_dim)
            self.head_dropout = nn.Dropout(head_dropout)
            self.rb_head = nn.Linear(head_hidden_dim, rb_output_size)
            self.sinr_head = nn.Linear(head_hidden_dim, 1)
        elif head_type == "split_lstm":
            self.horizon_embedding = nn.Embedding(self.num_horizons, horizon_embed_dim)
            self.rb_context = nn.Sequential(
                nn.LayerNorm(direct_dim),
                nn.Linear(direct_dim, self.rb_head_hidden_dim),
                nn.ReLU(),
                nn.Dropout(head_dropout),
            )
            self.sinr_context = nn.Sequential(
                nn.LayerNorm(direct_dim),
                nn.Linear(direct_dim, self.sinr_head_hidden_dim),
                nn.ReLU(),
                nn.Dropout(head_dropout),
            )
            self.rb_head_lstms = nn.ModuleList()
            self.sinr_head_lstms = nn.ModuleList()
            for i in range(self.rb_head_layers):
                input_dim = self.rb_head_hidden_dim + horizon_embed_dim if i == 0 else self.rb_head_hidden_dim
                self.rb_head_lstms.append(nn.LSTM(input_dim, self.rb_head_hidden_dim, batch_first=True))
            for i in range(self.sinr_head_layers):
                input_dim = self.sinr_head_hidden_dim + horizon_embed_dim if i == 0 else self.sinr_head_hidden_dim
                self.sinr_head_lstms.append(nn.LSTM(input_dim, self.sinr_head_hidden_dim, batch_first=True))
            self.num_head_cross_stitch = 0 if cross_stitch_mode == "none" else min(
                self.rb_head_layers,
                self.sinr_head_layers,
            )
            if self.num_head_cross_stitch > 0:
                self.head_cross_stitch_dim = max(self.rb_head_hidden_dim, self.sinr_head_hidden_dim)
                self.rb_head_projections_up = nn.ModuleList()
                self.sinr_head_projections_up = nn.ModuleList()
                self.head_cross_stitch_units = nn.ModuleList()
                self.rb_head_projections_down = nn.ModuleList()
                self.sinr_head_projections_down = nn.ModuleList()
                for _ in range(self.num_head_cross_stitch):
                    self.rb_head_projections_up.append(
                        nn.Linear(self.rb_head_hidden_dim, self.head_cross_stitch_dim)
                        if self.rb_head_hidden_dim != self.head_cross_stitch_dim
                        else nn.Identity()
                    )
                    self.sinr_head_projections_up.append(
                        nn.Linear(self.sinr_head_hidden_dim, self.head_cross_stitch_dim)
                        if self.sinr_head_hidden_dim != self.head_cross_stitch_dim
                        else nn.Identity()
                    )
                    unit = CrossStitchUnit(self.head_cross_stitch_dim)
                    if cross_stitch_mode == "identity":
                        unit.alpha.data = torch.eye(2, 2)
                        unit.alpha.requires_grad = False
                    elif cross_stitch_mode == "zeros":
                        unit.alpha.data = torch.zeros(2, 2)
                        unit.alpha.requires_grad = False
                    self.head_cross_stitch_units.append(unit)
                    self.rb_head_projections_down.append(
                        nn.Linear(self.head_cross_stitch_dim, self.rb_head_hidden_dim)
                        if self.rb_head_hidden_dim != self.head_cross_stitch_dim
                        else nn.Identity()
                    )
                    self.sinr_head_projections_down.append(
                        nn.Linear(self.head_cross_stitch_dim, self.sinr_head_hidden_dim)
                        if self.sinr_head_hidden_dim != self.head_cross_stitch_dim
                        else nn.Identity()
                    )
            self.rb_head_norm = nn.LayerNorm(self.rb_head_hidden_dim)
            self.sinr_head_norm = nn.LayerNorm(self.sinr_head_hidden_dim)
            self.head_dropout = nn.Dropout(head_dropout)
            self.rb_head = nn.Linear(self.rb_head_hidden_dim, rb_output_size)
            self.sinr_head = nn.Linear(self.sinr_head_hidden_dim, 1)
        else:
            raise ValueError(head_type)

    def forward(self, rb_seq, sinr_seq):
        rb_out, sinr_out = rb_seq, sinr_seq
        rb_h_tuple, sinr_h_tuple = None, None
        max_layers = max(len(self.rb_lstms), len(self.sinr_lstms))
        for i in range(max_layers):
            if i < len(self.rb_lstms):
                rb_out, rb_h_tuple = self.rb_lstms[i](rb_out)
            if i < len(self.sinr_lstms):
                sinr_out, sinr_h_tuple = self.sinr_lstms[i](sinr_out)
            if i < self.num_cross_stitch:
                rb_proj = self.rb_projections_up[i](rb_out)
                sinr_proj = self.sinr_projections_up[i](sinr_out)
                rb_cross, sinr_cross = self.cross_stitch_units[i](rb_proj, sinr_proj)
                rb_out = rb_out + self.rb_projections_down[i](rb_cross)
                sinr_out = sinr_out + self.sinr_projections_down[i](sinr_cross)

        # Use the sequence outputs after Cross-Stitch residual mixing.  The
        # LSTM hidden tuples were captured before the per-layer Cross-Stitch
        # update, so using them would bypass the final Cross-Stitch unit.
        rb_last = self.proj_rb(rb_out[:, -1])
        sinr_last = self.proj_sinr(sinr_out[:, -1])
        fused = self.fusion((rb_last + sinr_last) / 2.0)
        if self.head_type == "linear":
            rb_logits = self.rb_head(fused).view(-1, self.num_horizons, self.rb_output_size)
            sinr_pred = self.sinr_head(fused).view(-1, self.num_horizons, 1)
        elif self.head_type == "lstm":
            horizon_idx = torch.arange(self.num_horizons, device=fused.device)
            horizon_emb = self.horizon_embedding(horizon_idx).unsqueeze(0).expand(fused.shape[0], -1, -1)
            repeated_context = fused.unsqueeze(1).expand(-1, self.num_horizons, -1)
            head_in = torch.cat([repeated_context, horizon_emb], dim=2)
            head_out, _ = self.head_lstm(head_in)
            head_out = self.head_dropout(self.head_norm(head_out))
            rb_logits = self.rb_head(head_out)
            sinr_pred = self.sinr_head(head_out)
        else:
            horizon_idx = torch.arange(self.num_horizons, device=fused.device)
            horizon_emb = self.horizon_embedding(horizon_idx).unsqueeze(0).expand(fused.shape[0], -1, -1)
            rb_context = self.rb_context(rb_last).unsqueeze(1).expand(-1, self.num_horizons, -1)
            sinr_context = self.sinr_context(sinr_last).unsqueeze(1).expand(-1, self.num_horizons, -1)
            rb_head_out = torch.cat([rb_context, horizon_emb], dim=2)
            sinr_head_out = torch.cat([sinr_context, horizon_emb], dim=2)
            max_head_layers = max(len(self.rb_head_lstms), len(self.sinr_head_lstms))
            for i in range(max_head_layers):
                if i < len(self.rb_head_lstms):
                    rb_head_out, _ = self.rb_head_lstms[i](rb_head_out)
                if i < len(self.sinr_head_lstms):
                    sinr_head_out, _ = self.sinr_head_lstms[i](sinr_head_out)
                if i < self.num_head_cross_stitch:
                    rb_proj = self.rb_head_projections_up[i](rb_head_out)
                    sinr_proj = self.sinr_head_projections_up[i](sinr_head_out)
                    rb_cross, sinr_cross = self.head_cross_stitch_units[i](rb_proj, sinr_proj)
                    rb_head_out = self.rb_head_projections_down[i](rb_cross)
                    sinr_head_out = self.sinr_head_projections_down[i](sinr_cross)
            rb_head_out = self.head_dropout(self.rb_head_norm(rb_head_out))
            sinr_head_out = self.head_dropout(self.sinr_head_norm(sinr_head_out))
            rb_logits = self.rb_head(rb_head_out)
            sinr_pred = self.sinr_head(sinr_head_out)
        return rb_logits, sinr_pred


def val_to_index(rb):
    return (rb - 1).long()


def build_sinr_criterion(kind, beta):
    if kind == "mse":
        return nn.MSELoss()
    if kind == "mae":
        return nn.L1Loss()
    if kind == "huber":
        return nn.SmoothL1Loss(beta=beta)
    raise ValueError(kind)


def rate_proxy(rb_value, sinr_norm, sinr_mean, sinr_std):
    sinr_db = sinr_norm * sinr_std + sinr_mean
    snr_linear = torch.pow(10.0, sinr_db / 10.0)
    return rb_value * torch.log2(1.0 + snr_linear).clamp(max=8.0)


@torch.no_grad()
def evaluate(model, loader, sinr_mean, sinr_std, device):
    model.eval()
    rb_correct = 0
    rb_abs = 0.0
    total = 0
    sinr_abs = 0.0
    sinr_sq = 0.0
    thr_abs = 0.0
    thr_sq = 0.0
    by_h = None

    for rb_seq, sinr_seq, rb_label, sinr_label in loader:
        rb_seq = rb_seq.to(device)
        sinr_seq = sinr_seq.to(device)
        rb_label = rb_label.to(device).squeeze(-1)
        sinr_label = sinr_label.to(device)
        rb_logits, sinr_pred = model(rb_seq, sinr_seq)
        rb_class = torch.argmax(rb_logits, dim=2) + 1
        sinr_pred_denorm = (sinr_pred * sinr_std + sinr_mean).squeeze(-1)
        sinr_label_denorm = (sinr_label * sinr_std + sinr_mean).squeeze(-1)

        rb_correct += (rb_class == rb_label).sum().item()
        rb_abs += torch.abs(rb_class - rb_label).sum().item()
        err_sinr = sinr_pred_denorm - sinr_label_denorm
        sinr_abs += torch.abs(err_sinr).sum().item()
        sinr_sq += (err_sinr ** 2).sum().item()

        thr_pred = calculate_throughput_from_predictions_parallel(
            rb_class.reshape(-1),
            sinr_pred_denorm.reshape(-1),
            device,
        ).view_as(sinr_pred_denorm)
        thr_true = calculate_throughput_from_predictions_parallel(
            rb_label.reshape(-1),
            sinr_label_denorm.reshape(-1),
            device,
        ).view_as(sinr_pred_denorm)
        err_thr = thr_pred - thr_true
        thr_abs += torch.abs(err_thr).sum().item()
        thr_sq += (err_thr ** 2).sum().item()
        total += rb_label.numel()

        batch_by_h = {
            "count": torch.full((rb_label.shape[1],), rb_label.shape[0], device=device, dtype=torch.float32),
            "rb_abs": torch.abs(rb_class - rb_label).float().sum(dim=0),
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
        "rb_mae": rb_abs / total,
        "sinr_mae": sinr_abs / total,
        "sinr_mse": sinr_sq / total,
        "throughput_mae": thr_abs / total,
        "throughput_mse": thr_sq / total,
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


@torch.no_grad()
def evaluate_last_value(loader, sinr_mean, sinr_std, device):
    total = 0
    sums = {"rb_abs": 0.0, "sinr_abs": 0.0, "sinr_sq": 0.0, "thr_abs": 0.0, "thr_sq": 0.0}
    by_h = None
    for rb_seq, sinr_seq, rb_label, sinr_label in loader:
        rb_seq = rb_seq.to(device)
        sinr_seq = sinr_seq.to(device)
        rb_label = rb_label.to(device).squeeze(-1)
        sinr_label = sinr_label.to(device)
        rb_pred = rb_seq[:, -1:, 0].repeat(1, rb_label.shape[1]).round()
        sinr_pred_denorm = (sinr_seq[:, -1:, 0] * sinr_std + sinr_mean).repeat(1, rb_label.shape[1])
        sinr_label_denorm = (sinr_label * sinr_std + sinr_mean).squeeze(-1)

        err_sinr = sinr_pred_denorm - sinr_label_denorm
        thr_pred = calculate_throughput_from_predictions_parallel(
            rb_pred.reshape(-1),
            sinr_pred_denorm.reshape(-1),
            device,
        ).view_as(sinr_pred_denorm)
        thr_true = calculate_throughput_from_predictions_parallel(
            rb_label.reshape(-1),
            sinr_label_denorm.reshape(-1),
            device,
        ).view_as(sinr_pred_denorm)
        err_thr = thr_pred - thr_true
        rb_abs = torch.abs(rb_pred - rb_label)
        total += rb_label.numel()
        sums["rb_abs"] += rb_abs.sum().item()
        sums["sinr_abs"] += torch.abs(err_sinr).sum().item()
        sums["sinr_sq"] += (err_sinr ** 2).sum().item()
        sums["thr_abs"] += torch.abs(err_thr).sum().item()
        sums["thr_sq"] += (err_thr ** 2).sum().item()

        batch_by_h = {
            "count": torch.full((rb_label.shape[1],), rb_label.shape[0], device=device, dtype=torch.float32),
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


def plot_curves(rows, output_dir):
    horizons = [row["h"] for row in rows]
    for metric, ylabel, filename in [
        ("throughput_mae", "Throughput MAE", "direct_throughput_mae_by_horizon.png"),
        ("sinr_mae", "SINR MAE (dB)", "direct_sinr_mae_by_horizon.png"),
        ("rb_mae", "RB MAE", "direct_rb_mae_by_horizon.png"),
    ]:
        plt.figure(figsize=(6.4, 4.0))
        plt.plot(horizons, [row[f"direct_{metric}"] for row in rows], marker="o", label="Direct Cross-Stitch")
        plt.plot(horizons, [row[f"last_{metric}"] for row in rows], marker="s", label="Last Value")
        plt.xlabel("Prediction horizon h (10 ms frame)")
        plt.ylabel(ylabel)
        plt.xticks(horizons)
        plt.grid(True, linestyle="--", alpha=0.35)
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=220, bbox_inches="tight")
        plt.close()


def main():
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.json"
    if metrics_path.exists() and not args.force:
        print(metrics_path.read_text())
        return

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    if any(h <= 0 for h in horizons):
        raise ValueError("Horizons must be positive frame offsets.")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    train_path = args.dataset_dir / "train_9000_HDF5.pkl"
    test_path = args.dataset_dir / "test_1000_HDF5.pkl"
    train_dataset = DirectHorizonDataset(
        train_path,
        hist_window=args.hist_window,
        horizons=horizons,
        skip_initial_frames=args.skip_train_frames,
    )
    test_dataset = DirectHorizonDataset(
        test_path,
        hist_window=args.hist_window,
        horizons=horizons,
        sinr_mean=train_dataset.sinr_mean,
        sinr_std=train_dataset.sinr_std,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    eval_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)

    model = DirectHorizonCrossStitch(
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion_rb = nn.CrossEntropyLoss()
    criterion_sinr = build_sinr_criterion(args.sinr_loss, args.huber_beta)
    classes = torch.arange(1, 107, device=device).float()

    best_value = float("inf")
    best_epoch = 0
    stale = 0
    history = []
    model_path = args.output_dir / "best_direct_horizon_crossstitch.pth"
    print(
        f"Training direct Cross-Stitch on {args.dataset_dir} horizons={horizons} "
        f"frames, hist={args.hist_window}, head={args.head_type} on {device}"
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", ncols=110)
        for rb_seq, sinr_seq, rb_label, sinr_label in pbar:
            rb_seq = rb_seq.to(device)
            sinr_seq = sinr_seq.to(device)
            rb_label = rb_label.to(device)
            sinr_label = sinr_label.to(device)
            optimizer.zero_grad()
            rb_logits, sinr_pred = model(rb_seq, sinr_seq)
            loss_rb = criterion_rb(rb_logits.reshape(-1, 106), val_to_index(rb_label.reshape(-1)))
            loss_sinr = criterion_sinr(sinr_pred, sinr_label)
            loss = loss_rb + loss_sinr

            if args.gamma_rb_dist > 0:
                probs = F.softmax(rb_logits, dim=2)
                rb_expect = (probs * classes).sum(dim=2)
                loss = loss + args.gamma_rb_dist * F.l1_loss(rb_expect, rb_label.squeeze(-1).float())

            if args.gamma_rate_proxy > 0:
                probs = F.softmax(rb_logits, dim=2)
                rb_expect = (probs * classes).sum(dim=2)
                pred_proxy = rate_proxy(
                    rb_expect,
                    sinr_pred.squeeze(-1),
                    train_dataset.sinr_mean,
                    train_dataset.sinr_std,
                )
                true_proxy = rate_proxy(
                    rb_label.squeeze(-1).float(),
                    sinr_label.squeeze(-1),
                    train_dataset.sinr_mean,
                    train_dataset.sinr_std,
                )
                loss = loss + args.gamma_rate_proxy * F.smooth_l1_loss(pred_proxy / 50.0, true_proxy / 50.0, beta=0.05)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(rb_seq)
            seen += len(rb_seq)
        scheduler.step()

        eval_metrics, _ = evaluate(model, eval_loader, train_dataset.sinr_mean, train_dataset.sinr_std, device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(seen, 1),
            **{k: float(v) for k, v in eval_metrics.items()},
        }
        history.append(row)
        select_value = eval_metrics[args.selection_metric]
        print(
            f"epoch {epoch}: loss={row['train_loss']:.4f} "
            f"RB MAE={eval_metrics['rb_mae']:.4f} "
            f"SINR MAE={eval_metrics['sinr_mae']:.4f} "
            f"Throughput MAE={eval_metrics['throughput_mae']:.4f}"
        )
        if select_value < best_value:
            best_value = float(select_value)
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), model_path)
            print(f"saved best checkpoint at epoch {epoch}")
        else:
            stale += 1
            if args.patience > 0 and stale >= args.patience:
                print(f"early stopped at epoch {epoch}")
                break

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    final_metrics, final_by_h = evaluate(model, eval_loader, train_dataset.sinr_mean, train_dataset.sinr_std, device)
    last_metrics, last_by_h = evaluate_last_value(eval_loader, train_dataset.sinr_mean, train_dataset.sinr_std, device)

    rows = []
    for horizon, direct_row, last_row in zip(horizons, final_by_h, last_by_h):
        row = {"h": int(horizon), "time_ms": int(horizon * 10)}
        for key in ["rb_mae", "sinr_mae", "sinr_mse", "throughput_mae", "throughput_mse"]:
            row[f"direct_{key}"] = float(direct_row[key])
            row[f"last_{key}"] = float(last_row[key])
        rows.append(row)

    pd.DataFrame(rows).to_csv(args.output_dir / "direct_horizon_metrics.csv", index=False)
    pd.DataFrame(history).to_csv(args.output_dir / "training_history.csv", index=False)
    plot_curves(rows, args.output_dir)
    payload = {
        "dataset_dir": str(args.dataset_dir),
        "sample_interval_ms": 10,
        "hist_window": args.hist_window,
        "history_span_ms": (args.hist_window - 1) * 10,
        "horizons": horizons,
        "best_epoch": best_epoch,
        "selection_metric": args.selection_metric,
        "direct_metrics": final_metrics,
        "last_value_metrics": last_metrics,
        "by_horizon": rows,
        "model_path": str(model_path),
        "sinr_mean": train_dataset.sinr_mean,
        "sinr_std": train_dataset.sinr_std,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
