"""Generate simple synthetic hourly KPIs for the filtered study-area sites.

The generated data is intended for research prototyping only. It is designed to
be deterministic, readable, and realistic enough for downstream rules and ML
experiments in this sandbox project.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.geo import load_study_area_sites

DEFAULT_START_TIMESTAMP = "2026-01-05 00:00:00"
DEFAULT_NUM_DAYS = 14
DEFAULT_RANDOM_SEED = 42
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "simulated_kpis.csv"
)

# These profiles create a simple Bloomsbury-like pattern: quiet overnight,
# increasing activity in the morning, a daytime peak, and a softer evening tail.
WEEKDAY_HOURLY_PROFILE = np.array(
    [
        0.10,
        0.08,
        0.07,
        0.07,
        0.09,
        0.13,
        0.22,
        0.38,
        0.58,
        0.76,
        0.89,
        0.97,
        1.00,
        0.98,
        0.95,
        0.92,
        0.88,
        0.81,
        0.70,
        0.57,
        0.44,
        0.31,
        0.21,
        0.14,
    ]
)
WEEKEND_HOURLY_PROFILE = np.array(
    [
        0.12,
        0.10,
        0.09,
        0.08,
        0.08,
        0.10,
        0.15,
        0.24,
        0.34,
        0.46,
        0.59,
        0.72,
        0.83,
        0.90,
        0.93,
        0.91,
        0.85,
        0.77,
        0.66,
        0.54,
        0.42,
        0.31,
        0.22,
        0.16,
    ]
)


def build_hourly_timestamps(
    start_timestamp: str = DEFAULT_START_TIMESTAMP,
    num_days: int = DEFAULT_NUM_DAYS,
) -> pd.DatetimeIndex:
    """Return an hourly timestamp range for the simulation horizon."""

    if num_days <= 0:
        raise ValueError("num_days must be a positive integer")

    return pd.date_range(start=start_timestamp, periods=num_days * 24, freq="h")


def get_site_type_parameters(site_type: str) -> dict[str, float]:
    """Return simple baseline parameters for each supported site type."""

    if site_type == "rooftop":
        return {
            "base_users": 150.0,
            "weekday_scale": 1.04,
            "weekend_scale": 0.82,
            "capacity_mbps": 340.0,
            "idle_power_watts": 680.0,
            "power_per_prb": 5.8,
            "sinr_offset": 1.8,
        }
    if site_type == "street_furniture":
        return {
            "base_users": 120.0,
            "weekday_scale": 0.98,
            "weekend_scale": 0.92,
            "capacity_mbps": 260.0,
            "idle_power_watts": 540.0,
            "power_per_prb": 4.9,
            "sinr_offset": 0.9,
        }

    return {
        "base_users": 110.0,
        "weekday_scale": 1.0,
        "weekend_scale": 0.9,
        "capacity_mbps": 240.0,
        "idle_power_watts": 520.0,
        "power_per_prb": 4.8,
        "sinr_offset": 1.0,
    }


def build_site_kpi_frame(
    site_row: pd.Series,
    timestamps: pd.DatetimeIndex,
    random_seed: int,
    site_index: int,
) -> pd.DataFrame:
    """Generate hourly KPIs for one site over the requested time horizon."""

    parameters = get_site_type_parameters(site_row["site_type"])
    site_rng = np.random.default_rng(random_seed + site_index * 97)

    profile_shift = int(site_rng.integers(-1, 2))
    demand_multiplier = float(site_rng.uniform(0.92, 1.10))
    traffic_multiplier = float(site_rng.uniform(0.95, 1.08))
    capacity_mbps = parameters["capacity_mbps"] * float(site_rng.uniform(0.96, 1.08))
    idle_power_watts = parameters["idle_power_watts"] * float(site_rng.uniform(0.98, 1.03))
    power_per_prb = parameters["power_per_prb"] * float(site_rng.uniform(0.96, 1.04))
    sinr_offset = parameters["sinr_offset"] + float(site_rng.normal(0.0, 0.5))
    distance_factor = float(
        np.clip(1.02 - float(site_row["distance_to_campus_m"]) / 12_000.0, 0.92, 1.05)
    )

    site_frame = pd.DataFrame({"timestamp": timestamps})
    site_frame["site_id"] = site_row["site_id"]
    site_frame["hour_of_day"] = site_frame["timestamp"].dt.hour
    site_frame["is_weekend"] = site_frame["timestamp"].dt.dayofweek >= 5

    hours = site_frame["hour_of_day"].to_numpy()
    is_weekend = site_frame["is_weekend"].to_numpy()
    day_index = np.arange(len(site_frame)) // 24
    daily_factors = site_rng.normal(loc=1.0, scale=0.05, size=len(timestamps) // 24)

    weekday_profile = np.roll(WEEKDAY_HOURLY_PROFILE, profile_shift)
    weekend_profile = np.roll(WEEKEND_HOURLY_PROFILE, profile_shift)
    base_profile = np.where(is_weekend, weekend_profile[hours], weekday_profile[hours])
    day_type_scale = np.where(
        is_weekend, parameters["weekend_scale"], parameters["weekday_scale"]
    )
    demand_signal = (
        parameters["base_users"]
        * demand_multiplier
        * distance_factor
        * base_profile
        * day_type_scale
        * daily_factors[day_index]
    )

    active_users = np.maximum(
        np.rint(demand_signal + site_rng.normal(loc=0.0, scale=4.0, size=len(site_frame))),
        1,
    ).astype(int)

    per_user_traffic = (
        0.65
        + 1.35 * base_profile * traffic_multiplier
        + site_rng.normal(loc=0.0, scale=0.08, size=len(site_frame))
    )
    traffic_mbps = np.clip(active_users * per_user_traffic, 1.0, None)

    prb_utilization = np.clip(
        8.0
        + (traffic_mbps / capacity_mbps) * 100.0
        + site_rng.normal(loc=0.0, scale=3.0, size=len(site_frame)),
        5.0,
        98.0,
    )

    avg_sinr = np.clip(
        20.5
        - 0.08 * prb_utilization
        - float(site_row["distance_to_campus_m"]) / 900.0
        + sinr_offset
        + site_rng.normal(loc=0.0, scale=0.7, size=len(site_frame)),
        5.0,
        30.0,
    )

    energy_watts = np.clip(
        idle_power_watts
        + prb_utilization * power_per_prb
        + active_users * 0.9
        + site_rng.normal(loc=0.0, scale=8.0, size=len(site_frame)),
        idle_power_watts * 0.95,
        None,
    )

    site_frame["active_users"] = active_users
    site_frame["traffic_mbps"] = np.round(traffic_mbps, 1)
    site_frame["prb_utilization"] = np.round(prb_utilization, 1)
    site_frame["avg_sinr"] = np.round(avg_sinr, 1)
    site_frame["energy_watts"] = np.round(energy_watts, 1)

    return site_frame[
        [
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
    ]


def generate_simulated_kpis(
    start_timestamp: str = DEFAULT_START_TIMESTAMP,
    num_days: int = DEFAULT_NUM_DAYS,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """Generate hourly KPIs for each filtered study-area site."""

    study_area_sites = load_study_area_sites()
    timestamps = build_hourly_timestamps(
        start_timestamp=start_timestamp,
        num_days=num_days,
    )

    site_frames = [
        build_site_kpi_frame(
            site_row=site_row,
            timestamps=timestamps,
            random_seed=random_seed,
            site_index=site_index,
        )
        for site_index, (_, site_row) in enumerate(study_area_sites.iterrows())
    ]

    return (
        pd.concat(site_frames, ignore_index=True)
        .sort_values(["timestamp", "site_id"], kind="stable")
        .reset_index(drop=True)
    )


def save_simulated_kpis(
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    start_timestamp: str = DEFAULT_START_TIMESTAMP,
    num_days: int = DEFAULT_NUM_DAYS,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """Generate synthetic KPIs and save them to the processed data directory."""

    simulated_kpis = generate_simulated_kpis(
        start_timestamp=start_timestamp,
        num_days=num_days,
        random_seed=random_seed,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    simulated_kpis.to_csv(output_path, index=False)
    return simulated_kpis


def print_simulation_summary(simulated_kpis: pd.DataFrame, output_path: Path) -> None:
    """Print a short preview and summary statistics for the generated KPIs."""

    print("Synthetic KPI preview:")
    print(simulated_kpis.head(12).to_string(index=False))
    print()
    print(
        "Rows: "
        f"{len(simulated_kpis)}, "
        f"Sites: {simulated_kpis['site_id'].nunique()}, "
        f"Time range: {simulated_kpis['timestamp'].min()} to "
        f"{simulated_kpis['timestamp'].max()}"
    )
    print(f"Saved to: {output_path}")
    print()
    print("Summary statistics:")
    print(
        simulated_kpis[
            [
                "active_users",
                "traffic_mbps",
                "prb_utilization",
                "avg_sinr",
                "energy_watts",
            ]
        ]
        .agg(["min", "mean", "max"])
        .round(1)
        .to_string()
    )


def main() -> None:
    """Generate the default KPI dataset and print a short summary."""

    output_path = Path(DEFAULT_OUTPUT_PATH)
    simulated_kpis = save_simulated_kpis(output_path=output_path)
    print_simulation_summary(simulated_kpis=simulated_kpis, output_path=output_path)


if __name__ == "__main__":
    main()
