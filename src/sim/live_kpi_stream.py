"""Generate deterministic live KPI snapshots for the dashboard demo.

This module simulates a small stream of site KPIs that updates one step at a
time. It is intentionally simple and interview-friendly: each snapshot is
derived from a step index, the site metadata, and a fixed random seed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.geo import load_study_area_sites
from src.sim.generate_kpis import (
    DEFAULT_START_TIMESTAMP,
    WEEKDAY_HOURLY_PROFILE,
    WEEKEND_HOURLY_PROFILE,
    get_site_type_parameters,
)

DEFAULT_LIVE_RANDOM_SEED = 123
DEFAULT_STEP_MINUTES = 15
DEFAULT_LIVE_INPUT_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "raw" / "candidate_sites.csv"
)


def build_live_timestamp(
    step_index: int,
    start_timestamp: str = DEFAULT_START_TIMESTAMP,
    step_minutes: int = DEFAULT_STEP_MINUTES,
) -> pd.Timestamp:
    """Return the simulated live timestamp for a given dashboard step."""

    if step_index < 0:
        raise ValueError("step_index must be non-negative")
    if step_minutes <= 0:
        raise ValueError("step_minutes must be positive")

    start_time = pd.Timestamp(start_timestamp)
    return start_time + pd.Timedelta(minutes=step_index * step_minutes)


def load_live_study_area_sites(
    csv_path: str | Path = DEFAULT_LIVE_INPUT_PATH,
) -> pd.DataFrame:
    """Load the filtered study-area sites used by the live simulator."""

    return load_study_area_sites(csv_path=csv_path).reset_index(drop=True)


def get_profile_value(timestamp: pd.Timestamp) -> float:
    """Return a simple hourly demand multiplier for the simulated timestamp."""

    hour_of_day = int(timestamp.hour)
    profile = WEEKEND_HOURLY_PROFILE if timestamp.dayofweek >= 5 else WEEKDAY_HOURLY_PROFILE
    return float(profile[hour_of_day])


def build_live_kpi_snapshot(
    study_area_sites: pd.DataFrame,
    step_index: int,
    random_seed: int = DEFAULT_LIVE_RANDOM_SEED,
) -> pd.DataFrame:
    """Generate one KPI snapshot for all study-area sites at a given step."""

    timestamp = build_live_timestamp(step_index=step_index)
    profile_value = get_profile_value(timestamp)
    rows: list[dict[str, object]] = []

    for site_index, site_row in study_area_sites.reset_index(drop=True).iterrows():
        parameters = get_site_type_parameters(site_row["site_type"])
        site_rng = np.random.default_rng(random_seed + step_index * 101 + site_index * 17)

        distance_factor = float(
            np.clip(1.02 - float(site_row["distance_to_campus_m"]) / 12_000.0, 0.92, 1.05)
        )
        weekday_scale = parameters["weekday_scale"]
        weekend_scale = parameters["weekend_scale"]
        day_type_scale = weekend_scale if timestamp.dayofweek >= 5 else weekday_scale

        active_users = int(
            max(
                1,
                round(
                    parameters["base_users"]
                    * profile_value
                    * day_type_scale
                    * distance_factor
                    + site_rng.normal(0.0, 4.0)
                ),
            )
        )

        per_user_traffic = max(
            0.6,
            0.7 + 1.3 * profile_value + float(site_rng.normal(0.0, 0.08)),
        )
        traffic_mbps = max(1.0, active_users * per_user_traffic)
        capacity_mbps = parameters["capacity_mbps"] * float(site_rng.uniform(0.96, 1.04))
        prb_utilization = float(
            np.clip(
                8.0
                + (traffic_mbps / capacity_mbps) * 100.0
                + site_rng.normal(0.0, 2.5),
                5.0,
                98.0,
            )
        )
        avg_sinr = float(
            np.clip(
                20.5
                - 0.08 * prb_utilization
                - float(site_row["distance_to_campus_m"]) / 900.0
                + parameters["sinr_offset"]
                + site_rng.normal(0.0, 0.6),
                5.0,
                30.0,
            )
        )
        simulated_energy_watts = float(
            np.clip(
                parameters["idle_power_watts"]
                + prb_utilization * parameters["power_per_prb"]
                + active_users * 0.9
                + site_rng.normal(0.0, 6.0),
                parameters["idle_power_watts"] * 0.95,
                None,
            )
        )

        rows.append(
            {
                "timestamp": timestamp,
                "site_id": site_row["site_id"],
                "site_name": site_row["site_name"],
                "site_type": site_row["site_type"],
                "latitude": float(site_row["latitude"]),
                "longitude": float(site_row["longitude"]),
                "distance_to_campus_m": float(site_row["distance_to_campus_m"]),
                "active_users": active_users,
                "traffic_mbps": round(traffic_mbps, 1),
                "prb_utilization": round(prb_utilization, 1),
                "avg_sinr": round(avg_sinr, 1),
                "simulated_energy_watts": round(simulated_energy_watts, 1),
                "hour_of_day": int(timestamp.hour),
                "is_weekend": bool(timestamp.dayofweek >= 5),
                "simulation_step": step_index,
            }
        )

    return pd.DataFrame(rows).sort_values("site_id", kind="stable").reset_index(drop=True)

