"""Geo module for the UCL RAN energy sandbox.

This package contains simple, interview-ready utilities for defining the
approximate UCL Bloomsbury study area and loading approximate candidate sites
for research prototyping only.
"""

from importlib import import_module

__all__ = [
    "CAMPUS_CENTER_LATITUDE",
    "CAMPUS_CENTER_LONGITUDE",
    "STUDY_RADIUS_METERS",
    "annotate_sites_with_distance",
    "filter_sites_within_study_area",
    "load_candidate_sites",
    "load_study_area_sites",
]


def __getattr__(name: str):
    """Lazily expose campus-area symbols without importing on package load."""

    if name in __all__:
        module = import_module(".campus_area", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
