# mtl_dataset.py
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

def _to_numpy(x):
    if isinstance(x, pd.DataFrame): return x.values
    if isinstance(x, pd.Series): return x.to_numpy()
    return np.asarray(x)

class MTL_Dataset(Dataset):
    def __init__(
        self,
        rb_file_path: str,
        sinr_file_path: str,
        obs_window: int = 16,
        pre_window: int = 1,
        skip_initial_frames: int = 0,
        sinr_mean: float = None,
        sinr_std: float = None,
    ):
        # RB Data
        rb_raw_all = pd.read_pickle(rb_file_path)
        rb_np = None
        if isinstance(rb_raw_all, dict):
            for key in ["RB", "rb", "Rb", "rB"]:
                if key in rb_raw_all: rb_np = _to_numpy(rb_raw_all[key]); break
        if rb_np is None: rb_np = _to_numpy(rb_raw_all)

        if rb_np.ndim == 2 and rb_np.shape[0] == 1 and rb_np.shape[1] > 1:
            rb_np = rb_np.T
        if rb_np.ndim == 1: rb_np = rb_np[:, None]
        if skip_initial_frames > 0: rb_np = rb_np[skip_initial_frames:, ...]
        self.rb_data = torch.as_tensor(rb_np, dtype=torch.float32)

        # SINR Data
        sinr_raw_all = pd.read_pickle(sinr_file_path)
        sinr_np = None
        if isinstance(sinr_raw_all, dict):
            for key in ["SINR", "sinr", "Sinr", "SNR", "snr"]:
                if key in sinr_raw_all: sinr_np = _to_numpy(sinr_raw_all[key]); break
        if sinr_np is None: sinr_np = _to_numpy(sinr_raw_all)

        if sinr_np.ndim == 2 and sinr_np.shape[0] == 1 and sinr_np.shape[1] > 1:
             sinr_np = sinr_np.T
        if sinr_np.ndim == 1: sinr_np = sinr_np[:, None]
        if skip_initial_frames > 0: sinr_np = sinr_np[skip_initial_frames:, ...]
        self.sinr_data_raw = torch.as_tensor(sinr_np, dtype=torch.float32)

        # SINR Standardization
        if sinr_mean is None or sinr_std is None:
            self.sinr_mean = self.sinr_data_raw.mean()
            self.sinr_std = self.sinr_data_raw.std()
            print(f"[Dataset] Calculated SINR stats for standardization: mean={self.sinr_mean:.4f}, std={self.sinr_std:.4f}")
        else:
            self.sinr_mean = sinr_mean
            self.sinr_std = sinr_std
        self.sinr_data = (self.sinr_data_raw - self.sinr_mean) / (self.sinr_std + 1e-8)

        # Config
        self.obs_window = int(obs_window)
        self.pre_window = int(pre_window)
        self.data_len = int(min(len(self.rb_data), len(self.sinr_data)))
        self.num_samples = self.data_len - self.obs_window - self.pre_window + 1

        if self.num_samples <= 0:
            raise ValueError("Not enough data to create samples.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        input_start = idx
        input_end = input_start + self.obs_window
        rb_seq = self.rb_data[input_start:input_end]
        sinr_seq = self.sinr_data[input_start:input_end]

        label_start = input_end
        label_end = label_start + self.pre_window
        rb_label = self.rb_data[label_start:label_end]
        sinr_label = self.sinr_data[label_start:label_end]

        return rb_seq, sinr_seq, rb_label, sinr_label