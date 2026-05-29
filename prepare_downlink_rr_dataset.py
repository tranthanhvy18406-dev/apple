import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


SOURCE_DIR = Path("/4T/xty/down_link/dataset_SINR_predict_rr")
OUTPUT_DIR = Path("/4T/xty/down_link/dataset_SINR_predict_rr")


def _read_payload(mat_path: Path, group_name: str) -> dict:
    with h5py.File(mat_path, "r") as handle:
        group = handle[group_name]
        return {
            "RB": np.asarray(group["RB"], dtype=np.float32).reshape(-1),
            "SINR": np.asarray(group["eff_SINR"], dtype=np.float32).reshape(-1),
            "Throughput": np.asarray(group["Throughput"], dtype=np.float32).reshape(-1),
        }


def _read_raw_sinr(mat_path: Path, dataset_name: str) -> dict:
    with h5py.File(mat_path, "r") as handle:
        raw = np.asarray(handle[dataset_name], dtype=np.float32)
    return {
        "shape": list(raw.shape),
        "min": float(raw.min()),
        "max": float(raw.max()),
        "mean": float(raw.mean()),
    }


def _write_pickle(payload: dict, out_path: Path) -> None:
    pd.to_pickle(payload, out_path)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_mat = SOURCE_DIR / "train.mat"
    test_mat = SOURCE_DIR / "test.mat"
    train_payload = _read_payload(train_mat, "train_dataset")
    test_payload = _read_payload(test_mat, "test_dataset")

    shutil.copy2(train_mat, OUTPUT_DIR / "train_9000_HDF5.mat")
    shutil.copy2(test_mat, OUTPUT_DIR / "test_1000_HDF5.mat")

    train_pkl = OUTPUT_DIR / "train_9000_HDF5.pkl"
    test_pkl = OUTPUT_DIR / "test_1000_HDF5.pkl"
    _write_pickle(train_payload, train_pkl)
    _write_pickle(test_payload, test_pkl)

    metadata = {
        "dataset": "dataset_SINR_predict_rr",
        "source_train_mat": str(train_mat),
        "source_test_mat": str(test_mat),
        "prepared_train_pkl": str(train_pkl),
        "prepared_test_pkl": str(test_pkl),
        "train_length": int(train_payload["RB"].shape[0]),
        "test_length": int(test_payload["RB"].shape[0]),
        "rb_range": [float(train_payload["RB"].min()), float(train_payload["RB"].max())],
        "sinr_range": [float(train_payload["SINR"].min()), float(train_payload["SINR"].max())],
        "throughput_range": [
            float(train_payload["Throughput"].min()),
            float(train_payload["Throughput"].max()),
        ],
        "train_sinr_matrix": _read_raw_sinr(SOURCE_DIR / "train_SINR.mat", "train_dataset_SINR"),
        "test_sinr_matrix": _read_raw_sinr(SOURCE_DIR / "test_SINR.mat", "test_dataset_SINR"),
    }
    (OUTPUT_DIR / "metadata_cross_stitch.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
