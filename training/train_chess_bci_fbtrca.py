import argparse
import glob
import os
import pickle
import re
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score

from brainda.algorithms.decomposition import FBTRCA, generate_filterbank
from brainda.algorithms.utils.model_selection import (
    set_random_seeds,
    generate_loo_indices,
    match_loo_indices,
)


def project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def parse_chess_bci_constants(chess_bci_path: str) -> dict:
    """
    Parse a few constants from legacy/chess_bci.py so we can locate its SAVE_DIR
    without importing pygame/serial-heavy code.
    """
    with open(chess_bci_path, "r", encoding="utf-8") as f:
        txt = f.read()

    def m(pattern: str, cast):
        mm = re.search(pattern, txt, flags=re.MULTILINE)
        if not mm:
            raise ValueError(f"Missing pattern in {chess_bci_path}: {pattern}")
        return cast(mm.group(1))

    stim_type = m(r"^\s*STIM_TYPE\s*=\s*'([^']+)'", str)
    stim_duration = m(r"^\s*STIM_DURATION\s*=\s*([0-9.]+)", float)
    subject = m(r"^\s*SUBJECT\s*=\s*(\d+)", int)
    session = m(r"^\s*SESSION\s*=\s*(\d+)", int)
    n_per_class = m(r"^\s*N_PER_CLASS\s*=\s*(\d+)", int)

    save_dir = (
        f"data/chess_bci_{stim_type}-vep_32-class_{stim_duration}s-"
        f"/sub-{subject:02d}/ses-{session:02d}/"
    )
    return {
        "stim_type": stim_type,
        "stim_duration": stim_duration,
        "subject": subject,
        "session": session,
        "n_per_class": n_per_class,
        "save_dir": save_dir,
    }


def load_chess_bci_dataset(data_dir: str):
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

    eeg_data = np.load(eeg_path)  # expected shape: (n_channels, total_samples)
    labels = np.load(labels_path)  # expected shape: (n_trials,)
    return eeg_data, labels, labels_path


def segment_trials_from_concatenated_eeg(eeg_data: np.ndarray, labels: np.ndarray, sampling_rate: int, stim_duration: float,
                                          baseline_duration: float, tol_samples: int = 5):
    """
    chess_bci.py concatenates per-trial EEG along time (axis=1) and saves:
      - eeg_data.npy: (n_channels, total_samples)
      - labels_*.npy: (n_trials,)
    This function splits eeg_data into trials in order, then crops to the stimulus window.
    """
    if eeg_data.ndim != 2:
        raise ValueError(f"Expected eeg_data.npy to have 2 dims, got {eeg_data.shape}")

    n_channels, total_samples = eeg_data.shape
    labels = np.asarray(labels).reshape(-1)
    n_trials = len(labels)
    if n_trials <= 0:
        raise ValueError("labels array is empty")

    # Split by assuming fixed per-trial sample count.
    trial_len_float = total_samples / n_trials
    trial_len = int(total_samples // n_trials)
    remainder = total_samples - trial_len * n_trials
    if remainder != 0:
        # Drop remainder samples at the end to keep equal-length trials.
        eeg_data = eeg_data[:, : trial_len * n_trials]

    expected_stim_samples = int(round(stim_duration * sampling_rate))
    expected_baseline_samples = int(round(baseline_duration * sampling_rate))

    # Detect whether baseline is included in the saved trials.
    trial_len_is_baseline_plus_stim = abs(trial_len - (expected_baseline_samples + expected_stim_samples)) <= tol_samples
    if trial_len_is_baseline_plus_stim:
        baseline_samples = expected_baseline_samples
    else:
        baseline_samples = 0

    stim_samples = expected_stim_samples

    eeg_trials = eeg_data.reshape(n_channels, n_trials, trial_len).transpose(1, 0, 2)  # (n_trials, n_channels, trial_len)

    # Create (n_trials, n_channels, stim_samples)
    eeg_trials_stim = np.zeros((n_trials, n_channels, stim_samples), dtype=eeg_trials.dtype)

    for i in range(n_trials):
        trial = eeg_trials[i]

        if baseline_samples > 0:
            baseline = trial[:, :baseline_samples]
            baseline_mean = baseline.mean(axis=-1, keepdims=True)
            trial = trial - baseline_mean
            trial_stim = trial[:, baseline_samples : baseline_samples + stim_samples]
        else:
            trial_stim = trial[:, :stim_samples]

        if trial_stim.shape[-1] >= stim_samples:
            eeg_trials_stim[i] = trial_stim[:, :stim_samples]
        else:
            # Pad if we ended up with fewer than expected samples.
            pad_width = stim_samples - trial_stim.shape[-1]
            eeg_trials_stim[i] = np.pad(trial_stim, ((0, 0), (0, pad_width)), mode="constant")

    meta = {
        "n_trials": n_trials,
        "n_channels": n_channels,
        "total_samples": total_samples,
        "trial_len": trial_len,
        "trial_len_float": trial_len_float,
        "baseline_samples_used": baseline_samples,
        "stim_samples": stim_samples,
        "remainder_dropped": remainder,
    }
    return eeg_trials_stim, meta


def build_eeg_tensor_for_fbtrca(
    eeg_trials_stim: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
    expected_n_per_class: int | None,
):
    """
    Build eeg tensor shaped:
      (n_reps, n_classes, n_channels, n_samples)

    where n_reps corresponds to the repetition index within each class
    (e.g., the 2nd time we saw class c).
    """
    labels = np.asarray(labels).reshape(-1)
    if eeg_trials_stim.shape[0] != labels.shape[0]:
        raise ValueError(f"Mismatch: eeg_trials_stim has {eeg_trials_stim.shape[0]} trials but labels has {labels.shape[0]}")

    n_trials_total, n_channels, n_samples = eeg_trials_stim.shape

    # Fold labels if they exceed our chosen n_classes (e.g., chess_bci may use more freqs than config).
    labels_folded = labels % n_classes

    class_indices = []
    class_counts = []
    for c in range(n_classes):
        idx = np.where(labels_folded == c)[0]
        class_indices.append(idx)
        class_counts.append(len(idx))

    min_count = int(min(class_counts)) if class_counts else 0
    if expected_n_per_class is not None:
        n_reps = min(min_count, int(expected_n_per_class))
    else:
        n_reps = min_count

    if n_reps <= 0:
        raise ValueError(
            f"Not enough data to build per-class repetitions. min_count={min_count}, expected_n_per_class={expected_n_per_class}"
        )

    if n_reps < 2:
        print(f"Warning: only {n_reps} repetition(s) available across all classes; LOO evaluation may be limited.")

    eeg_tensor = np.zeros((n_reps, n_classes, n_channels, n_samples), dtype=eeg_trials_stim.dtype)
    for c in range(n_classes):
        # Take the first n_reps occurrences for this class.
        selected_trial_idxs = class_indices[c][:n_reps]
        for r in range(n_reps):
            eeg_tensor[r, c] = eeg_trials_stim[selected_trial_idxs[r]]

    return eeg_tensor, labels_folded, class_counts, n_reps


def train_fbtrca(
    eeg_tensor: np.ndarray,
    stimulus_classes,
    sampling_rate: int,
    stim_duration: float,
    model_save_path: str,
    evaluate_loo: bool = False,
    seed: int = 64,
):
    """
    eeg_tensor: (n_reps, n_classes, n_channels, n_samples)
    """
    n_reps, n_classes, n_channels, _ = eeg_tensor.shape

    target_tab = {tuple(map(float, cls)): idx for idx, cls in enumerate(stimulus_classes)}
    target_by_trial = [stimulus_classes] * n_reps

    def run_fit_once(eeg_tensor_local: np.ndarray):
        set_random_seeds(seed)
        eeg = np.copy(eeg_tensor_local)
        np.random.seed(seed)
        np.random.shuffle(eeg)  # shuffle repetition order only

        classes = range(n_classes)
        n_trials = eeg.shape[0]

        y = np.array([list(target_tab.values())] * n_trials).T.reshape(-1)
        eeg_temp = eeg[:n_trials, classes, :, :]  # onset_delay=0
        X = eeg_temp.swapaxes(0, 1).reshape(-1, *eeg_temp.shape[2:])  # (n_trials*n_classes, n_channels, n_samples)

        duration_samples = int(round(sampling_rate * stim_duration))
        filterX = np.copy(X[..., :duration_samples])
        filterX = filterX - np.mean(filterX, axis=-1, keepdims=True)
        filterY = np.copy(y)

        n_bands = 3
        wp = [[8 * i, 90] for i in range(1, n_bands + 1)]
        ws = [[8 * i - 2, 95] for i in range(1, n_bands + 1)]
        filterbank = generate_filterbank(wp, ws, sampling_rate, order=4, rp=1)
        filterweights = np.arange(1, len(filterbank) + 1) ** (-1.25) + 0.25

        models = OrderedDict(
            [
                (
                    "fbtrca",
                    FBTRCA(filterbank, filterweights=filterweights, ensemble=True),
                )
            ]
        )

        model = clone(models["fbtrca"]).fit(filterX, filterY)
        return model

    model = run_fit_once(eeg_tensor)

    if evaluate_loo:
        set_random_seeds(seed)
        classes = range(n_classes)
        n_trials = eeg_tensor.shape[0]

        # Build X/y in the same way as run_fit_once (but without shuffling for reporting).
        y = np.array([list(target_tab.values())] * n_trials).T.reshape(-1)
        eeg_temp = eeg_tensor[:, classes, :, :]
        X = eeg_temp.swapaxes(0, 1).reshape(-1, *eeg_temp.shape[2:])
        duration_samples = int(round(sampling_rate * stim_duration))
        filterX = np.copy(X[..., :duration_samples])
        filterX = filterX - np.mean(filterX, axis=-1, keepdims=True)

        events = []
        for j_class in classes:
            events.extend([str(target_by_trial[i_trial][j_class]) for i_trial in range(n_trials)])
        events = np.array(events)
        subjects = ["1"] * (n_classes * n_trials)
        meta = pd.DataFrame(data=np.array([subjects, events]).T, columns=["subject", "event"])

        loo_indices = generate_loo_indices(meta)

        # Recreate model components for each fold.
        n_bands = 3
        wp = [[8 * i, 90] for i in range(1, n_bands + 1)]
        ws = [[8 * i - 2, 95] for i in range(1, n_bands + 1)]
        filterbank = generate_filterbank(wp, ws, sampling_rate, order=4, rp=1)
        filterweights = np.arange(1, len(filterbank) + 1) ** (-1.25) + 0.25
        models = OrderedDict(
            [
                (
                    "fbtrca",
                    FBTRCA(filterbank, filterweights=filterweights, ensemble=True),
                )
            ]
        )

        n_loo = len(loo_indices["1"][events[0]])
        loo_accs = []
        pred_labels_all = []
        test_labels_all = []

        for k in range(n_loo):
            train_ind, validate_ind, test_ind = match_loo_indices(k, meta, loo_indices)
            train_ind = np.concatenate([train_ind, validate_ind])

            trainX, trainY = filterX[train_ind], y[train_ind]
            testX, testY = filterX[test_ind], y[test_ind]

            loo_model = clone(models["fbtrca"]).fit(trainX, trainY)
            pred_labels = loo_model.predict(testX)

            loo_accs.append(balanced_accuracy_score(testY, pred_labels))
            pred_labels_all.extend(pred_labels.tolist())
            test_labels_all.extend(testY.tolist())

        cm = confusion_matrix(test_labels_all, pred_labels_all, normalize="true")
        acc = accuracy_score(test_labels_all, pred_labels_all)
        bal_acc = float(np.mean(loo_accs)) if loo_accs else float("nan")
        print(f"LOO balanced accuracy: {bal_acc:.4f}")
        print(f"LOO accuracy: {acc:.4f}")
        print("LOO normalized confusion matrix:")
        print(cm)

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    with open(model_save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved FBTRCA model to: {model_save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train FBTRCA model for EEG-Chess (chess_bci.py data).")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing eeg_data.npy and labels_*.npy.")
    parser.add_argument("--model_save_path", type=str, default=None, help="Where to save the trained model pkl.")
    parser.add_argument("--evaluate_loo", action="store_true", help="Also run LOOCV evaluation (may be slow).")
    parser.add_argument("--seed", type=int, default=64)
    parser.add_argument("--n_classes", type=int, default=None, help="Override number of classes (default: from src/config.py).")
    parser.add_argument("--n_per_class_expected", type=int, default=None, help="Expected repetition per class (default: from labels filename / chess_bci).")
    args = parser.parse_args()

    root_dir = project_root()
    sys.path.insert(0, root_dir)

    from src.config import (
        MODEL_PATH,
        N_CLASSES as CFG_N_CLASSES,
        N_CHANNELS as CFG_N_CHANNELS,
        BASELINE_DURATION,
        SAMPLING_RATE,
        STIM_DURATION,
        SSVEP_CLASSES,
    )

    chess_bci_path = os.path.join(root_dir, "legacy", "chess_bci.py")
    if not os.path.exists(chess_bci_path):
        raise FileNotFoundError(f"Cannot find {chess_bci_path}")

    chess_consts = parse_chess_bci_constants(chess_bci_path)
    default_data_dir = os.path.join(root_dir, chess_consts["save_dir"])

    data_dir = args.data_dir if args.data_dir is not None else default_data_dir
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")

    model_save_path = args.model_save_path if args.model_save_path is not None else os.path.join(root_dir, MODEL_PATH)

    n_classes = args.n_classes if args.n_classes is not None else int(CFG_N_CLASSES)
    n_channels_expected = int(CFG_N_CHANNELS)
    sampling_rate = int(SAMPLING_RATE)
    stim_duration = float(chess_consts["stim_duration"])

    if len(SSVEP_CLASSES) < n_classes:
        raise ValueError(f"src/config.py only defines {len(SSVEP_CLASSES)} SSVEP_CLASSES, but --n_classes={n_classes}.")
    stimulus_classes = SSVEP_CLASSES[:n_classes]

    eeg_data, labels, labels_path = load_chess_bci_dataset(data_dir)

    labels = np.asarray(labels).reshape(-1)
    if eeg_data.shape[0] != n_channels_expected:
        print(f"Warning: eeg_data has {eeg_data.shape[0]} channel(s), but config expects {n_channels_expected}. Continuing anyway.")

    expected_n_per_class = args.n_per_class_expected
    if expected_n_per_class is None:
        # Try to extract from labels filename like: labels_2-per-class_run-1.npy
        m = re.search(r"labels_(\d+)-per-class", os.path.basename(labels_path))
        if m:
            expected_n_per_class = int(m.group(1))
        else:
            expected_n_per_class = chess_consts["n_per_class"]

    eeg_trials_stim, meta = segment_trials_from_concatenated_eeg(
        eeg_data=eeg_data,
        labels=labels,
        sampling_rate=sampling_rate,
        stim_duration=stim_duration,
        baseline_duration=float(BASELINE_DURATION),
    )

    print("Dataset summary:")
    print(f"  data_dir: {data_dir}")
    print(f"  n_trials_total: {meta['n_trials']}")
    print(f"  n_channels: {meta['n_channels']}")
    print(f"  trial_len(samples): {meta['trial_len']} (float {meta['trial_len_float']:.2f})")
    print(f"  baseline_samples_used: {meta['baseline_samples_used']}")
    print(f"  stim_samples: {meta['stim_samples']}")

    eeg_tensor, labels_folded, class_counts, n_reps = build_eeg_tensor_for_fbtrca(
        eeg_trials_stim=eeg_trials_stim,
        labels=labels,
        n_classes=n_classes,
        expected_n_per_class=expected_n_per_class,
    )

    print("Per-class availability:")
    print(f"  expected_n_per_class: {expected_n_per_class}")
    print(f"  n_reps_used: {n_reps}")
    print(f"  min_count_across_classes: {int(min(class_counts))}")

    train_fbtrca(
        eeg_tensor=eeg_tensor,
        stimulus_classes=stimulus_classes,
        sampling_rate=sampling_rate,
        stim_duration=stim_duration,
        model_save_path=model_save_path,
        evaluate_loo=args.evaluate_loo,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

