"""Rule-based control helpers for site power-state experiments."""

from importlib import import_module

__all__ = [
    "DEFAULT_STANDBY_POWER_WATTS",
    "apply_tower_power_rules",
]


def __getattr__(name: str):
    """Lazily expose rule helpers without importing on package load."""

    if name in __all__:
        module = import_module(".tower_power_rules", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
