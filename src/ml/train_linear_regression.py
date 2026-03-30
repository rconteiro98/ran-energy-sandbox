"""Train a simple linear regression model for hourly site energy demand.

The goal of this module is to keep the implementation deterministic,
readable, and easy to explain in an interview. It uses ordinary least
squares with a NumPy pseudo-inverse instead of a heavier ML framework.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_INPUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "simulated_kpis.csv"
)
DEFAULT_PREDICTIONS_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "linear_regression_predictions.csv"
)
DEFAULT_COEFFICIENTS_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "linear_regression_coefficients.csv"
)

REQUIRED_COLUMNS = [
    "timestamp",
    "site_id",
    "active_users",
    "traffic_mbps",
    "prb_utilization",
    "avg_sinr",
    "energy_watts",
    "hour_of_day",
    "is_weekend",
]
FEATURE_COLUMNS = [
    "active_users",
    "traffic_mbps",
    "prb_utilization",
    "avg_sinr",
    "hour_of_day",
    "is_weekend",
    "site_id",
]
TARGET_COLUMN = "energy_watts"


def load_simulated_kpis(input_path: str | Path = DEFAULT_INPUT_PATH) -> pd.DataFrame:
    """Load the simulated KPI dataset and validate the required columns."""

    dataset = pd.read_csv(input_path, parse_dates=["timestamp"])
    missing_columns = [
        column_name
        for column_name in REQUIRED_COLUMNS
        if column_name not in dataset.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return dataset.sort_values(["timestamp", "site_id"], kind="stable").reset_index(
        drop=True
    )


def encode_model_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Return the deterministic encoded feature frame used by the model."""

    missing_columns = [
        column_name for column_name in FEATURE_COLUMNS if column_name not in dataset.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing feature columns: {missing_columns}")

    feature_frame = dataset[FEATURE_COLUMNS].copy()
    feature_frame["is_weekend"] = feature_frame["is_weekend"].astype(int)

    site_dummies = pd.get_dummies(
        feature_frame["site_id"],
        prefix="site_id",
        dtype=int,
    )
    feature_frame = pd.concat(
        [feature_frame.drop(columns=["site_id"]), site_dummies],
        axis=1,
    )
    return feature_frame


def build_feature_matrix(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return model features and target using simple deterministic encoding."""

    if TARGET_COLUMN not in dataset.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    feature_frame = encode_model_features(dataset)
    target = dataset[TARGET_COLUMN].copy()
    return feature_frame, target


def split_train_test_by_time(
    feature_frame: pd.DataFrame,
    target: pd.Series,
    timestamps: pd.Series,
    train_fraction: float = 0.8,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Split the dataset using timestamp order to avoid future leakage."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    ordered_timestamps = np.sort(timestamps.unique())
    split_index = int(len(ordered_timestamps) * train_fraction)
    split_index = min(max(split_index, 1), len(ordered_timestamps) - 1)
    split_timestamp = ordered_timestamps[split_index]

    train_mask = timestamps < split_timestamp
    test_mask = ~train_mask

    return {
        "X_train": feature_frame.loc[train_mask].reset_index(drop=True),
        "X_test": feature_frame.loc[test_mask].reset_index(drop=True),
        "y_train": target.loc[train_mask].reset_index(drop=True),
        "y_test": target.loc[test_mask].reset_index(drop=True),
        "train_timestamps": timestamps.loc[train_mask].reset_index(drop=True),
        "test_timestamps": timestamps.loc[test_mask].reset_index(drop=True),
        "split_timestamp": pd.Series([split_timestamp]),
    }


def add_intercept_column(feature_frame: pd.DataFrame) -> np.ndarray:
    """Add a column of ones so the fitted model learns an intercept term."""

    intercept = np.ones((len(feature_frame), 1), dtype=float)
    numeric_features = feature_frame.astype(float).to_numpy()
    return np.hstack([intercept, numeric_features])


def fit_linear_regression(
    feature_frame: pd.DataFrame, target: pd.Series
) -> np.ndarray:
    """Fit ordinary least squares using the Moore-Penrose pseudo-inverse."""

    design_matrix = add_intercept_column(feature_frame)
    target_vector = target.astype(float).to_numpy()
    return np.linalg.pinv(design_matrix) @ target_vector


def predict_linear_regression(
    feature_frame: pd.DataFrame, coefficients: np.ndarray
) -> np.ndarray:
    """Generate predictions from a fitted linear regression coefficient vector."""

    design_matrix = add_intercept_column(feature_frame)
    return design_matrix @ coefficients


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Return compact regression metrics for easy reporting."""

    residuals = y_true.astype(float).to_numpy() - y_pred
    absolute_errors = np.abs(residuals)
    squared_errors = residuals**2
    total_variance = (
        y_true.astype(float).to_numpy() - float(y_true.astype(float).mean())
    ) ** 2

    return {
        "mae": float(np.mean(absolute_errors)),
        "rmse": float(np.sqrt(np.mean(squared_errors))),
        "r2": float(1.0 - np.sum(squared_errors) / np.sum(total_variance)),
    }


def build_coefficients_frame(
    feature_columns: list[str], coefficients: np.ndarray
) -> pd.DataFrame:
    """Return a readable coefficient table including the intercept."""

    coefficient_names = ["intercept", *feature_columns]
    return pd.DataFrame(
        {
            "feature": coefficient_names,
            "coefficient": np.round(coefficients, 6),
        }
    )


def build_predictions_frame(
    dataset: pd.DataFrame,
    test_index: pd.Index,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """Return a compact table of predictions for the held-out test rows."""

    predictions = dataset.loc[test_index, ["timestamp", "site_id"]].reset_index(
        drop=True
    )
    predictions["actual_energy_watts"] = y_true.reset_index(drop=True)
    predictions["predicted_energy_watts"] = np.round(y_pred, 2)
    predictions["absolute_error_watts"] = np.round(
        np.abs(predictions["actual_energy_watts"] - predictions["predicted_energy_watts"]),
        2,
    )
    return predictions


def save_training_outputs(
    predictions_frame: pd.DataFrame,
    coefficient_frame: pd.DataFrame,
    predictions_output_path: str | Path,
    coefficients_output_path: str | Path,
) -> None:
    """Persist the compact prediction and coefficient outputs to CSV."""

    predictions_output_path = Path(predictions_output_path)
    coefficients_output_path = Path(coefficients_output_path)
    predictions_output_path.parent.mkdir(parents=True, exist_ok=True)
    coefficients_output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_frame.to_csv(predictions_output_path, index=False)
    coefficient_frame.to_csv(coefficients_output_path, index=False)


def train_and_evaluate_model_from_frame(
    dataset: pd.DataFrame,
    predictions_output_path: str | Path | None = None,
    coefficients_output_path: str | Path | None = None,
) -> dict[str, pd.DataFrame | dict[str, float] | pd.Series]:
    """Train the linear regression model from an in-memory KPI dataframe."""

    ordered_dataset = dataset.sort_values(["timestamp", "site_id"], kind="stable").reset_index(
        drop=True
    )
    feature_frame, target = build_feature_matrix(ordered_dataset)
    split_results = split_train_test_by_time(
        feature_frame=feature_frame,
        target=target,
        timestamps=ordered_dataset["timestamp"],
    )

    coefficients = fit_linear_regression(
        feature_frame=split_results["X_train"],
        target=split_results["y_train"],
    )
    test_predictions = predict_linear_regression(
        feature_frame=split_results["X_test"],
        coefficients=coefficients,
    )
    metrics = compute_metrics(
        y_true=split_results["y_test"],
        y_pred=test_predictions,
    )

    coefficient_frame = build_coefficients_frame(
        feature_columns=split_results["X_train"].columns.tolist(),
        coefficients=coefficients,
    )

    test_mask = ordered_dataset["timestamp"] >= split_results["split_timestamp"].iloc[0]
    predictions_frame = build_predictions_frame(
        dataset=ordered_dataset,
        test_index=ordered_dataset.index[test_mask],
        y_true=split_results["y_test"],
        y_pred=test_predictions,
    )

    if predictions_output_path is not None and coefficients_output_path is not None:
        save_training_outputs(
            predictions_frame=predictions_frame,
            coefficient_frame=coefficient_frame,
            predictions_output_path=predictions_output_path,
            coefficients_output_path=coefficients_output_path,
        )

    return {
        "dataset": ordered_dataset,
        "metrics": metrics,
        "coefficients": coefficient_frame,
        "predictions": predictions_frame,
        "split_timestamp": split_results["split_timestamp"],
    }


def train_and_evaluate_model(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    predictions_output_path: str | Path = DEFAULT_PREDICTIONS_PATH,
    coefficients_output_path: str | Path = DEFAULT_COEFFICIENTS_PATH,
) -> dict[str, pd.DataFrame | dict[str, float] | pd.Series]:
    """Train the linear regression model, save outputs, and return results."""

    dataset = load_simulated_kpis(input_path=input_path)
    return train_and_evaluate_model_from_frame(
        dataset=dataset,
        predictions_output_path=predictions_output_path,
        coefficients_output_path=coefficients_output_path,
    )


def align_feature_frame(
    feature_frame: pd.DataFrame, required_feature_columns: list[str]
) -> pd.DataFrame:
    """Align an encoded feature frame to the exact columns used during training."""

    aligned_frame = feature_frame.copy()
    for column_name in required_feature_columns:
        if column_name not in aligned_frame.columns:
            aligned_frame[column_name] = 0.0

    return aligned_frame[required_feature_columns].astype(float)


def predict_dataset_energy(
    dataset: pd.DataFrame, coefficient_frame: pd.DataFrame
) -> pd.DataFrame:
    """Predict energy usage for a KPI dataframe using learned coefficients."""

    if coefficient_frame.empty:
        raise ValueError("coefficient_frame must not be empty")

    ordered_coefficients = coefficient_frame.reset_index(drop=True)
    if str(ordered_coefficients.loc[0, "feature"]) != "intercept":
        raise ValueError("coefficient_frame must start with the intercept row")

    required_feature_columns = (
        ordered_coefficients.loc[ordered_coefficients["feature"] != "intercept", "feature"]
        .astype(str)
        .tolist()
    )
    encoded_features = encode_model_features(dataset)
    aligned_features = align_feature_frame(encoded_features, required_feature_columns)
    coefficients = ordered_coefficients["coefficient"].astype(float).to_numpy()
    predicted_energy = predict_linear_regression(
        feature_frame=aligned_features,
        coefficients=coefficients,
    )

    prediction_frame = dataset.copy()
    prediction_frame["predicted_energy_watts"] = np.round(predicted_energy, 2)
    return prediction_frame


def print_training_summary(
    results: dict[str, pd.DataFrame | dict[str, float] | pd.Series]
) -> None:
    """Print a short summary that is easy to read in the terminal."""

    metrics = results["metrics"]
    coefficients = results["coefficients"]
    predictions = results["predictions"]
    split_timestamp = results["split_timestamp"].iloc[0]

    print("Linear regression training summary")
    print(f"Train/test split timestamp: {split_timestamp}")
    print(f"MAE:  {metrics['mae']:.2f} watts")
    print(f"RMSE: {metrics['rmse']:.2f} watts")
    print(f"R2:   {metrics['r2']:.4f}")
    print()
    print("Top coefficients by absolute magnitude:")
    print(
        coefficients.reindex(
            coefficients["coefficient"].abs().sort_values(ascending=False).index
        )
        .head(10)
        .to_string(index=False)
    )
    print()
    print("Prediction sample:")
    print(predictions.head(10).to_string(index=False))


def main() -> None:
    """Train the default model and print a concise report."""

    results = train_and_evaluate_model()
    print_training_summary(results)


if __name__ == "__main__":
    main()
