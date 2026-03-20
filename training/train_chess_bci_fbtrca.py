import argparse
import glob
import os
import pickle
import re
import sys
from collections import OrderedDict
from typing import Optional, Tuple

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


def parse_stim_duration_from_legacy(chess_bci_path: str) -> float:
    """
    Extract STIM_DURATION (seconds) from legacy/chess_bci.py via regex.
    We avoid importing/exec-ing the chess file.
    """
    with open(chess_bci_path, "r", encoding="utf-8") as f:
        txt = f.read()

    m = re.search(r"^\s*STIM_DURATION\s*=\s*([0-9.]+)", txt, flags=re.MULTILINE)
    if not m:
        raise ValueError(f"Could not parse STIM_DURATION from {chess_bci_path}")
    return float(m.group(1))


def parse_frequency_classes_from_legacy(chess_bci_path: str):
    """
    Parse the frequency/phase tuples used by legacy/chess_bci.py.

    chess_bci.py defines:
      - SSVEP_FREQUENCIES: 32 tuples
      - EXTENDED_FREQUENCIES = SSVEP_FREQUENCIES + [ ... 32 more tuples ]

    Returns:
      list[(freq_hz:int, phase_offset_pi:float)] ordered as in legacy code.
    """
    with open(chess_bci_path, "r", encoding="utf-8") as f:
        txt = f.read()

    def extract_tuples(block: str):
        tuples = re.findall(r"\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)", block)
        out = []
        for a, b in tuples:
            out.append((int(float(a)), float(b)))
        return out

    # Base list
    m_base = re.search(r"SSVEP_FREQUENCIES\s*=\s*\[(.*?)\]\s*\n", txt, flags=re.DOTALL)
    if not m_base:
        # Fallback: match up to the next "\n\n# ---" comment block.
        m_base = re.search(r"SSVEP_FREQUENCIES\s*=\s*\[(.*?)\]\s*\n\n#\s*---", txt, flags=re.DOTALL)
    if not m_base:
        raise ValueError(f"Could not parse SSVEP_FREQUENCIES from {chess_bci_path}")
    base_block = m_base.group(1)
    base_tuples = extract_tuples(base_block)

    # Extended appended list
    m_ext = re.search(
        r"EXTENDED_FREQUENCIES\s*=\s*SSVEP_FREQUENCIES\s*\+\s*\[(.*?)\]\s*\n",
        txt,
        flags=re.DOTALL,
    )
    if not m_ext:
        raise ValueError(f"Could not parse EXTENDED_FREQUENCIES from {chess_bci_path}")
    ext_block = m_ext.group(1)
    ext_tuples = extract_tuples(ext_block)

    return base_tuples + ext_tuples

def _parse_dataset_name_components(stem_dir_name: str) -> dict:
    """
    Parse dataset metadata from directory name like:
      chess_bci_alternating-vep_32-class_1.2s-
    """
    # Example:
    # chess_bci_alternating-vep_32-class_1.2s-
    m = re.match(
        r"^chess_bci_(?P<stim_type>.+?)-vep_(?P<n_classes>\d+)-class_(?P<stim_duration>[0-9.]+)s-?$",
        stem_dir_name,
    )
    if not m:
        raise ValueError(f"Unrecognized chess_bci dataset dir name: {stem_dir_name}")
    return {
        "stim_type": m.group("stim_type"),
        "n_classes": int(m.group("n_classes")),
        "stim_duration": float(m.group("stim_duration")),
    }


def _parse_subject_session(data_dir: str) -> Tuple[int, int]:
    """
    Parse subject/session from .../sub-XX/ses-YY/ .
    """
    m_sub = re.search(r"sub-(\d+)", data_dir)
    m_ses = re.search(r"ses-(\d+)", data_dir)
    if not (m_sub and m_ses):
        return (0, 0)
    return (int(m_sub.group(1)), int(m_ses.group(1)))


def _try_parse_meta_from_path(path: str) -> dict:
    """
    Try to parse chess_bci dataset metadata from any ancestor folder name.
    Returns {} if not found / not parseable.
    """
    parts = os.path.normpath(path).split(os.sep)
    for part in parts:
        if part.startswith("chess_bci_"):
            try:
                return _parse_dataset_name_components(part)
            except Exception:
                continue
    return {}


def discover_chess_bci_data_dir(
    data_root: str,
    preferred_data_dir: Optional[str] = None,
) -> Tuple[str, dict]:
    """
    Auto-discover the chess_bci dataset directory that contains eeg_data.npy.
    Returns: (data_dir, parsed_metadata_from_dirname)
    """
    if preferred_data_dir is not None:
        data_dir = os.path.abspath(preferred_data_dir)
        eeg_path = os.path.join(data_dir, "eeg_data.npy")
        if not os.path.exists(eeg_path):
            raise FileNotFoundError(f"Missing {eeg_path}")
        meta = _try_parse_meta_from_path(data_dir)
        return data_dir, meta

    candidates = glob.glob(os.path.join(data_root, "sub-*", "ses-*"))
    valid = []
    for d in candidates:
        if not os.path.exists(os.path.join(d, "eeg_data.npy")):
            continue
        meta = _try_parse_meta_from_path(d)
        subject, session = _parse_subject_session(d)
        valid.append((subject, session, d, meta))

    if not valid:
        raise FileNotFoundError(f"No chess_bci dataset found under {data_root}")

    # Pick the highest (subject, session).
    valid.sort(key=lambda x: (x[0], x[1]))
    subject, session, best_dir, best_meta = valid[-1]
    return best_dir, best_meta


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
    expected_n_per_class,
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
    present_class_ids = []
    for c in range(n_classes):
        idx = np.where(labels_folded == c)[0]
        class_indices.append(idx)
        count_c = len(idx)
        class_counts.append(count_c)
        if count_c > 0:
            present_class_ids.append(c)

    if not present_class_ids:
        raise ValueError("No class IDs are present in labels; cannot build training tensor.")

    min_count_present = int(min(class_counts[c] for c in present_class_ids))
    if expected_n_per_class is not None:
        n_reps = min(min_count_present, int(expected_n_per_class))
    else:
        n_reps = min_count_present

    if n_reps <= 0:
        raise ValueError(
            "Not enough data to build per-class repetitions "
            f"(min_count_present={min_count_present}, expected_n_per_class={expected_n_per_class})."
        )

    if n_reps < 2:
        print(
            f"Warning: only {n_reps} repetition(s) available for at least one present class; "
            "LOO evaluation may be limited."
        )

    # Drop classes that are absent so we don't require min_count across all n_classes.
    class_ids_used = sorted(present_class_ids)
    n_classes_eff = len(class_ids_used)

    eeg_tensor = np.zeros(
        (n_reps, n_classes_eff, n_channels, n_samples), dtype=eeg_trials_stim.dtype
    )
    for c_eff, c_orig in enumerate(class_ids_used):
        selected_trial_idxs = class_indices[c_orig][:n_reps]
        for r in range(n_reps):
            eeg_tensor[r, c_eff] = eeg_trials_stim[selected_trial_idxs[r]]

    return eeg_tensor, class_ids_used, class_counts, n_reps


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
    parser.add_argument("--stim_duration", type=float, default=None, help="Override stim duration in seconds (default: parsed from data folder name).")
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
    )

    data_root = os.path.join(root_dir, "data")
    data_dir, discovered_meta = discover_chess_bci_data_dir(
        data_root=data_root,
        preferred_data_dir=args.data_dir,
    )
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")

    model_save_path = args.model_save_path if args.model_save_path is not None else os.path.join(root_dir, MODEL_PATH)

    legacy_chess_bci_path = os.path.join(root_dir, "legacy", "chess_bci.py")
    stimulus_classes_all = parse_frequency_classes_from_legacy(legacy_chess_bci_path)
    n_classes_total = (
        args.n_classes
        if args.n_classes is not None
        else len(stimulus_classes_all)
    )
    n_channels_expected = int(CFG_N_CHANNELS)
    sampling_rate = int(SAMPLING_RATE)
    if args.stim_duration is not None:
        stim_duration = float(args.stim_duration)
    else:
        discovered_stim_duration = discovered_meta.get("stim_duration")
        if discovered_stim_duration is not None:
            stim_duration = float(discovered_stim_duration)
        else:
            stim_duration = parse_stim_duration_from_legacy(legacy_chess_bci_path)

    # If you override n_classes_total, slice the frequency table accordingly.
    if n_classes_total > len(stimulus_classes_all):
        raise ValueError(
            f"--n_classes={n_classes_total} exceeds the number of classes parsed from legacy/chess_bci.py "
            f"({len(stimulus_classes_all)})."
        )
    stimulus_classes_total = stimulus_classes_all[:n_classes_total]

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
            # Fallback: chess_bci.py uses N_PER_CLASS=2 by default.
            expected_n_per_class = 2
            print("Warning: could not infer n_per_class from labels filename; using 2.")

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

    eeg_tensor, class_ids_used, class_counts, n_reps = build_eeg_tensor_for_fbtrca(
        eeg_trials_stim=eeg_trials_stim,
        labels=labels,
        n_classes=n_classes_total,
        expected_n_per_class=expected_n_per_class,
    )

    print("Per-class availability:")
    print(f"  expected_n_per_class: {expected_n_per_class}")
    print(f"  n_reps_used: {n_reps}")
    present_counts = [class_counts[c] for c in class_ids_used]
    print(f"  n_classes_eff: {len(class_ids_used)}")
    print(f"  min_count_present: {int(min(present_counts))}")

    stimulus_classes_eff = [stimulus_classes_total[cid] for cid in class_ids_used]

    train_fbtrca(
        eeg_tensor=eeg_tensor,
        stimulus_classes=stimulus_classes_eff,
        sampling_rate=sampling_rate,
        stim_duration=stim_duration,
        model_save_path=model_save_path,
        evaluate_loo=args.evaluate_loo,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

