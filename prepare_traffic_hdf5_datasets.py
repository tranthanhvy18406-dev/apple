import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


DEFAULT_DATASETS = {
    "ftp": {
        "train_mat": Path(
            "/4T/xty/Code_ThroughPut_TrafficModel(1)/Code_ThroughPut_TrafficModel/"
            "link_simulation/new_mcs_dataset/diff_trafficModel/single_user_pf_5Hz/FTP/"
            "train_9000_HDF5.mat"
        ),
        "test_mat": Path(
            "/4T/xty/Code_ThroughPut_TrafficModel(1)/Code_ThroughPut_TrafficModel/"
            "link_simulation/new_mcs_dataset/diff_trafficModel/single_user_pf_5Hz/FTP/"
            "test_1000_HDF5.mat"
        ),
    },
    "http": {
        "train_mat": Path("/4T/xty/train_9000_Http.mat"),
        "test_mat": Path("/4T/xty/test_1000_HDF5_http.mat"),
    },
    "volp": {
        "train_mat": Path("/4T/xty/train_9000_HDF5_volp.mat"),
        "test_mat": Path("/4T/xty/test_1000_volp.mat"),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare local FTP/HTTP/VoIP HDF5 MAT traffic datasets for Cross-Stitch evaluation."
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("/4T/xty/traffic_hdf5_datasets"),
        help="Root directory that will contain one subdirectory per traffic dataset.",
    )
    return parser.parse_args()


def _read_series(mat_path: Path):
    with h5py.File(mat_path, "r") as handle:
        if "train_dataset" in handle:
            group = handle["train_dataset"]
            split = "train"
        elif "test_dataset" in handle:
            group = handle["test_dataset"]
            split = "test"
        else:
            raise KeyError(f"Unsupported MAT structure in {mat_path}")

        rb = np.asarray(group["RB"], dtype=np.float32).reshape(-1)
        sinr = np.asarray(group["eff_SINR"], dtype=np.float32).reshape(-1)
        throughput = np.asarray(group["Throughput"], dtype=np.float32).reshape(-1)

    return split, {"RB": rb, "SINR": sinr, "Throughput": throughput}


def _write_pickle(payload, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(payload, out_path)


def prepare_one(dataset_name: str, cfg: dict, output_root: Path):
    dataset_dir = output_root / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    train_split, train_payload = _read_series(cfg["train_mat"])
    test_split, test_payload = _read_series(cfg["test_mat"])
    if train_split != "train" or test_split != "test":
        raise ValueError(f"Unexpected split mapping for {dataset_name}")

    train_mat_std = dataset_dir / "train_9000_HDF5.mat"
    test_mat_std = dataset_dir / "test_1000_HDF5.mat"
    shutil.copy2(cfg["train_mat"], train_mat_std)
    shutil.copy2(cfg["test_mat"], test_mat_std)
    shutil.copy2(cfg["train_mat"], dataset_dir / cfg["train_mat"].name)
    shutil.copy2(cfg["test_mat"], dataset_dir / cfg["test_mat"].name)

    train_pkl = dataset_dir / "train_9000_HDF5.pkl"
    test_pkl = dataset_dir / "test_1000_HDF5.pkl"
    _write_pickle(train_payload, train_pkl)
    _write_pickle(test_payload, test_pkl)

    metadata = {
        "dataset": dataset_name,
        "source_train_mat": str(cfg["train_mat"]),
        "source_test_mat": str(cfg["test_mat"]),
        "prepared_train_mat": str(train_mat_std),
        "prepared_test_mat": str(test_mat_std),
        "prepared_train_pkl": str(train_pkl),
        "prepared_test_pkl": str(test_pkl),
        "train_length": int(train_payload["RB"].shape[0]),
        "test_length": int(test_payload["RB"].shape[0]),
        "rb_range": [float(train_payload["RB"].min()), float(train_payload["RB"].max())],
        "sinr_range": [float(train_payload["SINR"].min()), float(train_payload["SINR"].max())],
    }
    (dataset_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metadata


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for dataset_name, cfg in DEFAULT_DATASETS.items():
        summaries.append(prepare_one(dataset_name, cfg, args.output_root))

    summary_path = args.output_root / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
