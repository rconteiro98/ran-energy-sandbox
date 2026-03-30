"""Synthetic KPI simulation module for the UCL RAN energy sandbox."""

from importlib import import_module

__all__ = [
    "DEFAULT_NUM_DAYS",
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_RANDOM_SEED",
    "DEFAULT_START_TIMESTAMP",
    "DEFAULT_STEP_MINUTES",
    "build_hourly_timestamps",
    "build_live_kpi_snapshot",
    "build_live_timestamp",
    "generate_simulated_kpis",
    "load_live_study_area_sites",
    "save_simulated_kpis",
]


def __getattr__(name: str):
    """Lazily expose simulation helpers without importing on package load."""

    if name in __all__:
        if name in {
            "DEFAULT_STEP_MINUTES",
            "build_live_kpi_snapshot",
            "build_live_timestamp",
            "load_live_study_area_sites",
        }:
            module = import_module(".live_kpi_stream", __name__)
            return getattr(module, name)

        module = import_module(".generate_kpis", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
