import argparse
import glob
import os
import pickle
import re
import sys

import numpy as np
from sklearn.metrics import accuracy_score


def project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_dataset(data_dir: str):
    eeg_path = os.path.join(data_dir, "eeg_data.npy")
    if not os.path.exists(eeg_path):
        raise FileNotFoundError(f"Missing {eeg_path}")

    labels_candidates = sorted(
        glob.glob(os.path.join(data_dir, "labels_*_per-class_run-*.npy"))
        + glob.glob(os.path.join(data_dir, "labels_*-per-class_run-*.npy"))
        + glob.glob(os.path.join(data_dir, "labels_*.npy"))
    )
    labels_candidates = [p for p in labels_candidates if os.path.basename(p).startswith("labels_")]
    if not labels_candidates:
        raise FileNotFoundError(f"No labels_*.npy found in {data_dir}")

    labels_path = labels_candidates[-1]
    eeg_data = np.load(eeg_path)
    labels = np.load(labels_path).reshape(-1)
    return eeg_data, labels, labels_path


def segment_stim_trials(
    eeg_data: np.ndarray,
    labels: np.ndarray,
    sampling_rate: int,
    stim_duration: float,
    baseline_duration: float,
    tol_samples: int = 5,
):
    if eeg_data.ndim != 2:
        raise ValueError(f"Expected eeg_data with shape (n_channels, total_samples), got {eeg_data.shape}")

    n_channels, total_samples = eeg_data.shape
    n_trials = int(labels.shape[0])
    if n_trials <= 0:
        raise ValueError("labels array is empty")

    trial_len = total_samples // n_trials
    remainder = total_samples - trial_len * n_trials
    if remainder != 0:
        eeg_data = eeg_data[:, : trial_len * n_trials]

    expected_stim_samples = int(round(stim_duration * sampling_rate))
    expected_baseline_samples = int(round(baseline_duration * sampling_rate))
    has_baseline = abs(trial_len - (expected_stim_samples + expected_baseline_samples)) <= tol_samples
    baseline_samples = expected_baseline_samples if has_baseline else 0

    eeg_trials = eeg_data.reshape(n_channels, n_trials, trial_len).transpose(1, 0, 2)
    eeg_trials_stim = np.zeros((n_trials, n_channels, expected_stim_samples), dtype=eeg_trials.dtype)

    for i in range(n_trials):
        trial = eeg_trials[i]
        if baseline_samples > 0:
            baseline = trial[:, :baseline_samples]
            trial = trial - baseline.mean(axis=-1, keepdims=True)
            trial_stim = trial[:, baseline_samples : baseline_samples + expected_stim_samples]
        else:
            trial_stim = trial[:, :expected_stim_samples]

        if trial_stim.shape[-1] < expected_stim_samples:
            pad = expected_stim_samples - trial_stim.shape[-1]
            trial_stim = np.pad(trial_stim, ((0, 0), (0, pad)), mode="constant")
        eeg_trials_stim[i] = trial_stim[:, :expected_stim_samples]

    return eeg_trials_stim


def infer_n_classes(labels: np.ndarray, configured_n_classes: int) -> int:
    max_label = int(np.max(labels)) if labels.size else -1
    if max_label < configured_n_classes:
        return configured_n_classes
    return max_label + 1


def main():
    parser = argparse.ArgumentParser(description="Evaluate cached FBTRCA model on saved eeg_data.npy.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/sub-02/ses-01",
        help="Directory containing eeg_data.npy and labels_*.npy",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="cache/FBTRCA_model.pkl",
        help="Path to trained FBTRCA model pickle file",
    )
    args = parser.parse_args()

    root = project_root()
    sys.path.insert(0, root)
    # Unpickling may require local brainda source imports.
    sys.path.insert(0, os.path.join(root, "brainda"))

    from src.config import SAMPLING_RATE, STIM_DURATION, BASELINE_DURATION, N_CLASSES

    data_dir = os.path.abspath(os.path.join(root, args.data_dir))
    model_path = os.path.abspath(os.path.join(root, args.model_path))
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    eeg_data, labels, labels_path = load_dataset(data_dir)
    eeg_trials_stim = segment_stim_trials(
        eeg_data=eeg_data,
        labels=labels,
        sampling_rate=int(SAMPLING_RATE),
        stim_duration=float(STIM_DURATION),
        baseline_duration=float(BASELINE_DURATION),
    )

    # Match training-time preprocessing: remove per-trial channel mean.
    X = eeg_trials_stim - np.mean(eeg_trials_stim, axis=-1, keepdims=True)

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    y_true = labels.astype(int)
    n_classes = infer_n_classes(y_true, int(N_CLASSES))
    y_true = y_true % n_classes

    y_pred = model.predict(X).astype(int)
    acc = accuracy_score(y_true, y_pred)

    m = re.search(r"labels_(\d+)-per-class", os.path.basename(labels_path))
    n_per_class = int(m.group(1)) if m else None

    print("Evaluation summary:")
    print(f"  data_dir: {data_dir}")
    print(f"  model_path: {model_path}")
    print(f"  labels_file: {labels_path}")
    print(f"  n_trials: {len(y_true)}")
    if n_per_class is not None:
        print(f"  n_per_class (from filename): {n_per_class}")
    print(f"  X shape for prediction: {X.shape}")
    print(f"  total accuracy: {acc:.4f} ({acc * 100:.2f}%)")


if __name__ == "__main__":
    main()
