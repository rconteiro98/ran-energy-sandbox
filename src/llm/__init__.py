"""LLM helpers for optional explanation features in the dashboard.

This package is intentionally narrow: the local model may summarize or explain
existing outputs, but it must not drive tower-control decisions.
"""

from importlib import import_module

__all__ = [
    "DEFAULT_LOCAL_LLM_ENDPOINT",
    "DEFAULT_LOCAL_LLM_MODEL",
    "LocalLlmError",
    "build_forecast_summary",
    "build_simulation_summary_prompt",
    "build_training_results_summary",
    "generate_local_llm_text",
    "list_local_llm_models",
]


def __getattr__(name: str):
    """Lazily expose LLM helpers without importing them on package load."""

    if name in {
        "DEFAULT_LOCAL_LLM_ENDPOINT",
        "DEFAULT_LOCAL_LLM_MODEL",
        "LocalLlmError",
        "generate_local_llm_text",
        "list_local_llm_models",
    }:
        module = import_module(".local_llama_client", __name__)
        return getattr(module, name)

    if name in {
        "build_forecast_summary",
        "build_simulation_summary_prompt",
        "build_training_results_summary",
    }:
        module = import_module(".explain_simulation_results", __name__)
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
