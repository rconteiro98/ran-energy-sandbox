"""Utilities for the approximate UCL Bloomsbury study area.

The campus center and candidate site data in this module are approximate and
intended for research prototyping only. They are suitable for an early MVP and
interview discussion, not for operational network planning.
"""

from __future__ import annotations

from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import pandas as pd

EARTH_RADIUS_METERS = 6_371_000
CAMPUS_CENTER_LATITUDE = 51.5246
CAMPUS_CENTER_LONGITUDE = -0.1340
STUDY_RADIUS_METERS = 1_200
DEFAULT_CANDIDATE_SITES_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "raw" / "candidate_sites.csv"
)


def haversine_distance_meters(
    latitude_a: float,
    longitude_a: float,
    latitude_b: float,
    longitude_b: float,
) -> float:
    """Return the great-circle distance in meters between two lat/lon points."""

    latitude_a_rad = radians(latitude_a)
    longitude_a_rad = radians(longitude_a)
    latitude_b_rad = radians(latitude_b)
    longitude_b_rad = radians(longitude_b)

    latitude_delta = latitude_b_rad - latitude_a_rad
    longitude_delta = longitude_b_rad - longitude_a_rad

    haversine_term = (
        sin(latitude_delta / 2) ** 2
        + cos(latitude_a_rad) * cos(latitude_b_rad) * sin(longitude_delta / 2) ** 2
    )
    central_angle = 2 * asin(sqrt(haversine_term))
    return EARTH_RADIUS_METERS * central_angle


def load_candidate_sites(csv_path: str | Path = DEFAULT_CANDIDATE_SITES_PATH) -> pd.DataFrame:
    """Load approximate candidate sites for research prototyping from CSV."""

    dataframe = pd.read_csv(csv_path)
    dataframe = dataframe.copy()
    dataframe["site_id"] = dataframe["site_id"].astype(str)
    dataframe["site_name"] = dataframe["site_name"].astype(str)
    dataframe["site_type"] = dataframe["site_type"].astype(str)
    dataframe["latitude"] = pd.to_numeric(dataframe["latitude"], errors="raise")
    dataframe["longitude"] = pd.to_numeric(dataframe["longitude"], errors="raise")

    return dataframe[
        ["site_id", "site_name", "site_type", "latitude", "longitude"]
    ].sort_values("site_id", kind="stable").reset_index(drop=True)


def annotate_sites_with_distance(candidate_sites: pd.DataFrame) -> pd.DataFrame:
    """Return a clean DataFrame with distance from the campus center in meters."""

    dataframe = candidate_sites.copy()
    dataframe["distance_to_campus_m"] = dataframe.apply(
        lambda row: haversine_distance_meters(
            CAMPUS_CENTER_LATITUDE,
            CAMPUS_CENTER_LONGITUDE,
            row["latitude"],
            row["longitude"],
        ),
        axis=1,
    )

    return dataframe.sort_values(
        ["distance_to_campus_m", "site_id"], kind="stable"
    ).reset_index(drop=True)


def filter_sites_within_study_area(
    candidate_sites_with_distance: pd.DataFrame,
    radius_meters: int = STUDY_RADIUS_METERS,
) -> pd.DataFrame:
    """Filter candidate sites to the approximate Bloomsbury study radius."""

    study_area_sites = candidate_sites_with_distance.loc[
        candidate_sites_with_distance["distance_to_campus_m"] <= radius_meters
    ].copy()
    return study_area_sites.reset_index(drop=True)


def load_study_area_sites(
    csv_path: str | Path = DEFAULT_CANDIDATE_SITES_PATH,
    radius_meters: int = STUDY_RADIUS_METERS,
) -> pd.DataFrame:
    """Load, annotate, and filter approximate candidate sites for the study area."""

    candidate_sites = load_candidate_sites(csv_path)
    candidate_sites_with_distance = annotate_sites_with_distance(candidate_sites)
    return filter_sites_within_study_area(candidate_sites_with_distance, radius_meters)


def main() -> None:
    """Print the approximate candidate sites that fall within the study radius."""

    study_area_sites = load_study_area_sites()
    print("Approximate study-area candidate sites for research prototyping only:")
    print(study_area_sites.to_string(index=False))


if __name__ == "__main__":
    main()
