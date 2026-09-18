"""
Module D - RNN Anomaly Detection using the SKAB dataset.

Dataset source:
https://www.kaggle.com/datasets/yuriykatser/skoltech-anomaly-benchmark-skab

IMPORTANT:
- The dataset must be downloaded manually.
- This code does NOT download the dataset.
- You can provide either:
    1. A single SKAB CSV file
    2. A folder containing multiple SKAB CSV files

Example:
    python connectionist/rnn_anomaly.py --dataset data/skab

or:
    python connectionist/rnn_anomaly.py --dataset data/skab/valve1/0.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

FEATURE_COLUMNS = [
    "Accelerometer1RMS",
    "Accelerometer2RMS",
    "Current",
    "Pressure",
    "Temperature",
    "Thermocouple",
    "Voltage",
    "Volume Flow RateRMS",
]

DEFAULT_DATASET = (Path(__file__).resolve().parents[1]/ "data"/ "skab")

class SensorRNN(nn.Module):
    """
    GRU-based anomaly detector.
    Input:
        sequence of sensor readings
    Output:
        anomaly logit for the final point in the sequence
    """

    def __init__(
        self,
        input_size=len(FEATURE_COLUMNS),
        hidden_size=48,
        num_layers=1,
        dropout=0.20,
    ):
        super().__init__()

        # Dropout inside GRU is only active when num_layers > 1.
        gru_dropout = dropout if num_layers > 1 else 0.0

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 24),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(24, 1),
        )

    def forward(self, x):
        output, _ = self.rnn(x)
        last_output = output[:, -1, :]
        last_output = self.dropout(last_output)
        return self.fc(last_output).squeeze(1)

def _read_single_csv(path):
    """
    Read one SKAB CSV file.
    SKAB CSV files use semicolon separators.
    """

    path = Path(path)
    df = pd.read_csv(path, sep=";")
    required_columns = FEATURE_COLUMNS + ["anomaly"]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Dataset file '{path}' is missing required columns: "
            f"{missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    df = df[required_columns].copy()

    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(df[column],errors="coerce",)

    df["anomaly"] = pd.to_numeric(df["anomaly"],errors="coerce",)
    df = df.dropna().reset_index(drop=True)

    df["anomaly"] = (
        df["anomaly"]
        .astype(int)
        .clip(0, 1)
    )

    return df

def _find_csv_files(path):
    """
    Find CSV files.
    If path is a CSV:
        return [path]

    If path is a directory:
        recursively find all CSV files.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Expected a CSV file or dataset folder, got: {path}")

        return [path]

    csv_files = sorted(path.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found inside dataset folder: {path}")
    return csv_files


def load_skab_experiments(path):
    """
    Load one or multiple SKAB experiments.
    Returns:
        list of dictionaries:
        [
            {
                "name": filename,
                "path": path,
                "df": dataframe
            },
            ...
        ]
    """
    csv_files = _find_csv_files(path)
    experiments = []

    for csv_file in csv_files:
        try:
            df = _read_single_csv(csv_file)
        except ValueError as exc:
            print(f"Skipping {csv_file}: {exc}")
            continue

        if len(df) < 20:
            print(
                f"Skipping {csv_file}: "
                f"not enough rows ({len(df)})"
            )
            continue

        experiments.append(
            {
                "name": csv_file.name,
                "path": str(csv_file),
                "df": df,
            }
        )

    if not experiments:
        raise ValueError("No valid SKAB experiment CSV files were found.")

    return experiments

def load_skab_dataset(path):
    """
    Backwards-compatible loader.

    Loads a single CSV file and returns its DataFrame.
    If a folder is supplied, all valid CSV files are combined.

    This function is kept so existing project code does not break.
    """

    experiments = load_skab_experiments(path)
    if len(experiments) == 1:
        return experiments[0]["df"]

    frames = []

    for experiment in experiments:
        frame = experiment["df"].copy()
        frame["_experiment"] = experiment["name"]
        frames.append(frame)

    return pd.concat(
        frames,
        ignore_index=True,
    )

def make_windows(df, window=10):
    """
    Create temporal windows.

    Label:
        anomaly value of the final point in each window.
    """

    if len(df) <= window:
        raise ValueError(
            f"Dataset has only {len(df)} rows, "
            f"but window size is {window}."
        )

    values = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    labels = df["anomaly"].to_numpy(
        dtype=np.float32
    )

    x = []
    y = []

    for end in range(window, len(df) + 1):
        x.append(values[end - window:end])
        y.append(labels[end - 1])

    return (
        np.asarray(x, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
    )

def _experiment_has_both_classes(experiment):
    labels = experiment["df"]["anomaly"].to_numpy()
    return (
        np.any(labels == 0)
        and np.any(labels == 1)
    )

def split_experiments(experiments):
    """
    Split complete SKAB experiments.
    The preferred split is:
        70% train
        15% validation
        15% test

    Experiments are never mixed between splits.
    When only a small number of CSVs is available, the function
    ensures that the test set contains an anomaly whenever possible.
    """

    n = len(experiments)
    if n == 1:
        return (experiments,[],[],)

    experiments = sorted(
        experiments,
        key=lambda e: (
            not _experiment_has_both_classes(e),
            e["name"],
        ),
    )

    if n == 2:
        return ([experiments[0]],[],[experiments[1]],)

    if n == 3:
        return ([experiments[0]],[experiments[1]],[experiments[2]],)

    train_count = max(1,int(round(n * 0.70)),)
    val_count = max(1,int(round(n * 0.15)),)

    if train_count + val_count >= n:
        train_count = n - 2
        val_count = 1

    train = experiments[:train_count]
    validation = experiments[train_count:train_count + val_count]

    test = experiments[train_count + val_count:]
    return train, validation, test

def split_single_experiment(x,y,train_ratio=0.60,val_ratio=0.20,):
    """
    Fallback for the case where only one SKAB CSV is supplied.

    We use chronological chunks to preserve temporal ordering.

    If the final test section contains no anomaly, we adjust the
    boundaries so that an anomaly-containing section is available
    for testing.

    This fallback is less ideal than training/testing on separate
    experiments. For the bonus experiment, using multiple SKAB CSVs
    is recommended.
    """

    n = len(x)
    if n < 30:
        raise ValueError("Not enough windows for train/validation/test split.")

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_end = max(1, train_end)
    val_end = min(n - 1, max(train_end + 1, val_end))

    train_x = x[:train_end]
    train_y = y[:train_end]

    val_x = x[train_end:val_end]
    val_y = y[train_end:val_end]

    test_x = x[val_end:]
    test_y = y[val_end:]

    if (np.sum(test_y) == 0 and np.sum(y) > 0):
        anomaly_indices = np.where(y == 1)[0]
        last_anomaly = anomaly_indices[-1]

        if last_anomaly >= val_end:
            pass

        else:
            candidate = max(
                train_end + 1,
                last_anomaly - max(1, int(n * 0.05)),
            )
            val_x = x[train_end:candidate]
            val_y = y[train_end:candidate]

            test_x = x[candidate:]
            test_y = y[candidate:]

    return (
        (train_x, train_y),
        (val_x, val_y),
        (test_x, test_y),
    )

def standardize(train_x, *other_x):
    """
    Fit normalization only on training data.

    This prevents test-data leakage.
    """

    flat_train = train_x.reshape(-1,train_x.shape[-1],)
    mean = flat_train.mean(axis=0)
    std = flat_train.std(axis=0)

    std[std < 1e-6] = 1.0
    transformed = [(train_x - mean) / std]

    transformed.extend((x - mean) / std for x in other_x)

    return (
        *transformed,
        mean,
        std,
    )

def metrics(
    y_true,
    probabilities,
    threshold=0.5,
):
    """
    Calculate classification metrics.
    """

    y_true = np.asarray(y_true).astype(int)

    probabilities = np.asarray(probabilities)

    y_pred = (probabilities >= threshold).astype(int)

    tp = int(
        (
            (y_true == 1)
            & (y_pred == 1)
        ).sum()
    )

    tn = int(
        (
            (y_true == 0)
            & (y_pred == 0)
        ).sum()
    )

    fp = int(
        (
            (y_true == 0)
            & (y_pred == 1)
        ).sum()
    )

    fn = int(
        (
            (y_true == 1)
            & (y_pred == 0)
        ).sum()
    )

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    accuracy = (
        (tp + tn)
        / max(len(y_true), 1)
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }

def find_best_threshold(
    y_true,
    probabilities,
):
    """
    Select anomaly threshold using validation data.

    We search thresholds from 0.10 to 0.90.
    """

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in np.arange(
        0.10,
        0.91,
        0.01,
    ):

        result = metrics(
            y_true,
            probabilities,
            threshold=float(threshold),
        )

        if result["f1"] > best_f1:

            best_f1 = result["f1"]
            best_threshold = float(
                threshold
            )

    return best_threshold

def _probabilities(model,x,batch_size=256,):
    """
    Generate anomaly probabilities in batches.
    """

    model.eval()

    loader = DataLoader(
        TensorDataset(
            torch.tensor(
                x,
                dtype=torch.float32,
            )
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    probabilities = []
    with torch.no_grad():
        for (xb,) in loader:
            logits = model(xb)
            probs = torch.sigmoid(logits)
            probabilities.extend(probs.cpu().numpy())

    return np.asarray(
        probabilities,
        dtype=np.float32,
    )

def _build_dataset_from_experiments(experiments,window,):
    """
    Convert a list of SKAB experiments into windows.
    """

    all_x = []
    all_y = []

    for experiment in experiments:
        df = experiment["df"]
        try:
            x, y = make_windows(
                df,
                window=window,
            )
        except ValueError:
            continue

        if len(x) == 0:
            continue

        all_x.append(x)
        all_y.append(y)

    if not all_x:
        raise ValueError("No usable temporal windows were created.")

    return (
        np.concatenate(all_x, axis=0),
        np.concatenate(all_y, axis=0),
    )

def train_model(
    dataset_path=DEFAULT_DATASET,
    window=10,
    epochs=30,
    batch_size=64,
    lr=0.001,
):
    """
    Train the re-tuned GRU on manually downloaded SKAB data.
    dataset_path may be:
        data/skab

    or:

        data/skab/valve1/0.csv
    """

    torch.manual_seed(3)
    np.random.seed(3)

    dataset_path = Path(dataset_path)
    experiments = load_skab_experiments(dataset_path)

    print(
        f"Loaded {len(experiments)} SKAB experiment(s)."
    )

    if len(experiments) >= 3:

        train_experiments, val_experiments, test_experiments = (
            split_experiments(experiments)
        )

        train_x, train_y = (
            _build_dataset_from_experiments(
                train_experiments,
                window,
            )
        )

        val_x, val_y = (
            _build_dataset_from_experiments(
                val_experiments,
                window,
            )
        )

        test_x, test_y = (
            _build_dataset_from_experiments(
                test_experiments,
                window,
            )
        )

        split_type = "experiment-level"

    elif len(experiments) == 2:

        train_experiments, _, test_experiments = (
            split_experiments(experiments)
        )

        train_x, train_y = (
            _build_dataset_from_experiments(
                train_experiments,
                window,
            )
        )

        test_x, test_y = (
            _build_dataset_from_experiments(
                test_experiments,
                window,
            )
        )

        split = split_single_experiment(
            train_x,
            train_y,
            train_ratio=0.80,
            val_ratio=0.20,
        )

        (
            (train_x, train_y),
            (val_x, val_y),
            _,
        ) = split

        split_type = "two-experiment"

    else:
        df = experiments[0]["df"]
        x, y = make_windows(
            df,
            window=window,
        )

        (
            (train_x, train_y),
            (val_x, val_y),
            (test_x, test_y),
        ) = split_single_experiment(
            x,
            y,
        )

        split_type = "single-experiment chronological"

    print(
        "Train labels:",
        dict(
            zip(
                *np.unique(
                    train_y,
                    return_counts=True,
                )
            )
        ),
    )

    print(
        "Validation labels:",
        dict(
            zip(
                *np.unique(
                    val_y,
                    return_counts=True,
                )
            )
        ),
    )

    print(
        "Test labels:",
        dict(
            zip(
                *np.unique(
                    test_y,
                    return_counts=True,
                )
            )
        ),
    )

    if np.sum(train_y) == 0:
        raise ValueError(
            "Training data contains no anomaly samples. "
            "Choose a SKAB folder containing anomaly experiments."
        )

    if np.sum(test_y) == 0:
        print(
            "WARNING: Test set contains no anomaly samples. "
            "F1/recall may be zero."
        )

    (
        train_x,
        val_x,
        test_x,
        mean,
        std,
    ) = standardize(
        train_x,
        val_x,
        test_x,
    )

    positive = float(
        np.sum(train_y == 1)
    )

    negative = float(
        np.sum(train_y == 0)
    )

    pos_weight_value = (
        negative / max(positive, 1.0)
    )

    pos_weight_value = min(
        pos_weight_value,
        20.0,
    )

    pos_weight = torch.tensor(
        [pos_weight_value],
        dtype=torch.float32,
    )

    print(
        f"Positive training windows: {int(positive)}"
    )

    print(
        f"Negative training windows: {int(negative)}"
    )

    print(
        f"Positive class weight: "
        f"{pos_weight_value:.3f}"
    )

    train_dataset = TensorDataset(
        torch.tensor(
            train_x,
            dtype=torch.float32,
        ),
        torch.tensor(
            train_y,
            dtype=torch.float32,
        ),
    )

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    model = SensorRNN()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    best_state = None
    best_val_f1 = -1.0

    patience = 7
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(
                logits,
                yb,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )
            optimizer.step()
            epoch_loss += float(
                loss.item()
            )

        val_prob = _probabilities(
            model,
            val_x,
        )
        val_threshold = find_best_threshold(
            val_y,
            val_prob,
        )
        val_result = metrics(
            val_y,
            val_prob,
            threshold=val_threshold,
        )

        val_f1 = val_result["f1"]
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {
                key: value.detach().clone()
                for key, value
                in model.state_dict().items()
            }
            patience_counter = 0

        else:
            patience_counter += 1

        if (epoch == 0 or (epoch + 1) % 5 == 0
        ):
            print(
                f"Epoch {epoch + 1:02d}/{epochs} "
                f"loss={epoch_loss / max(len(loader), 1):.4f} "
                f"val_f1={val_f1:.3f} "
                f"threshold={val_threshold:.2f}"
            )

        if patience_counter >= patience:
            print(
                f"Early stopping at epoch {epoch + 1}."
            )
            break

    if best_state is not None:
        model.load_state_dict(
            best_state
        )

    val_prob = _probabilities(
        model,
        val_x,
    )

    threshold = find_best_threshold(
        val_y,
        val_prob,
    )

    test_prob = _probabilities(
        model,
        test_x,
    )

    test_metrics = metrics(
        test_y,
        test_prob,
        threshold=threshold,
    )

    total_rows = sum(
        len(experiment["df"])
        for experiment in experiments
    )

    info = {
        "dataset": str(dataset_path),
        "experiments": len(experiments),
        "rows": total_rows,
        "windows": (
            len(train_x)
            + len(val_x)
            + len(test_x)
        ),
        "train_windows": len(train_x),
        "validation_windows": len(val_x),
        "test_windows": len(test_x),
        "window": window,
        "features": FEATURE_COLUMNS,
        "split_type": split_type,
        "threshold": threshold,
        "metrics": test_metrics,
        "mean": mean,
        "std": std,
    }

    print()
    print("Final RNN test results")
    print("----------------------")
    print(
        f"Threshold : {threshold:.2f}"
    )
    print(
        f"Accuracy  : {test_metrics['accuracy']:.3f}"
    )
    print(
        f"Precision : {test_metrics['precision']:.3f}"
    )
    print(
        f"Recall    : {test_metrics['recall']:.3f}"
    )
    print(
        f"F1 Score  : {test_metrics['f1']:.3f}"
    )

    print()
    print(
        "Confusion matrix:"
    )
    print(
        f"TP={test_metrics['tp']} "
        f"TN={test_metrics['tn']} "
        f"FP={test_metrics['fp']} "
        f"FN={test_metrics['fn']}"
    )

    return model, info

def predict_stream(
    model,
    readings,
    mean=None,
    std=None,
    window=10,
    threshold=0.5,
):
    """
    Predict anomalies from a sequence of sensor readings.
    readings can be:
        pandas.DataFrame

    or:
        list of dictionaries
    The input must contain all SKAB sensor columns.
    """

    if isinstance(readings,pd.DataFrame,):
        df = readings.copy()

    else:
        df = pd.DataFrame(readings)

    missing = [
        column
        for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "predict_stream expects SKAB sensor columns. "
            f"Missing: {missing}"
        )

    values = df[FEATURE_COLUMNS].copy()
    for column in FEATURE_COLUMNS:
        values[column] = pd.to_numeric(
            values[column],
            errors="coerce",
        )

    values = values.ffill().bfill()
    values = values.to_numpy(dtype=np.float32)

    if mean is not None and std is not None:
        mean = np.asarray(mean,dtype=np.float32,)
        std = np.asarray(std,dtype=np.float32,)
        std = np.where(
            std < 1e-6,
            1.0,
            std,
        )
        values = (values - mean) / std

    flags = []
    model.eval()

    with torch.no_grad():
        for end in range(window,len(values) + 1,):
            sequence = values[end - window:end]
            tensor = torch.tensor(
                sequence[None, ...],
                dtype=torch.float32,
            )

            probability = float(
                torch.sigmoid(
                    model(tensor)
                )[0]
            )

            flags.append(
                (
                    end,
                    probability,
                    probability >= threshold,
                )
            )

    return flags

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=(
            "Train Module D RNN on manually "
            "downloaded SKAB data."
        )
    )

    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help=(
            "Path to a SKAB CSV file or "
            "folder containing SKAB CSV files."
        ),
    )

    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="Number of previous sensor readings.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate.",
    )

    args = parser.parse_args()

    _, info = train_model(
        dataset_path=args.dataset,
        window=args.window,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    print()
    print("Dataset:", info["dataset"])
    print("Experiments:", info["experiments"])
    print("Rows:", info["rows"])
    print("Windows:", info["windows"])
    print("Split:", info["split_type"])
    print("Threshold:", round(info["threshold"], 3))

    print(
        "Test metrics:",
        {
            key: round(value, 3)
            for key, value
            in info["metrics"].items()
            if key in [
                "accuracy",
                "precision",
                "recall",
                "f1",
            ]
        },
    )