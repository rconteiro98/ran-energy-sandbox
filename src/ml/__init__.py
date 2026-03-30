"""ML module for simple, interview-ready modeling experiments."""

from importlib import import_module

__all__ = [
    "DEFAULT_COEFFICIENTS_PATH",
    "DEFAULT_INPUT_PATH",
    "DEFAULT_PREDICTIONS_PATH",
    "build_feature_matrix",
    "compute_metrics",
    "fit_linear_regression",
    "load_simulated_kpis",
    "predict_linear_regression",
    "split_train_test_by_time",
    "train_and_evaluate_model",
]


def __getattr__(name: str):
    """Lazily expose ML helpers without importing the training module eagerly."""

    if name in __all__:
        module = import_module(".train_linear_regression", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
