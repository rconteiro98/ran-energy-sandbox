"""Apply simple rule-based tower on/off decisions to live KPI snapshots."""

from __future__ import annotations

import pandas as pd

LOW_USERS_THRESHOLD = 18
LOW_TRAFFIC_THRESHOLD_MBPS = 20.0
LOW_PRB_THRESHOLD = 20.0
WAKE_USERS_THRESHOLD = 24
WAKE_TRAFFIC_THRESHOLD_MBPS = 28.0
WAKE_PRB_THRESHOLD = 28.0
LOW_LOAD_STREAK_TO_TURN_OFF = 3
DEFAULT_STANDBY_POWER_WATTS = 80.0

ON_STATE_COLOR = [46, 160, 67, 220]
OFF_STATE_COLOR = [220, 53, 69, 220]


def build_previous_state_lookup(previous_state: pd.DataFrame | None) -> dict[str, dict[str, object]]:
    """Return the prior tower state keyed by site_id for hysteresis handling."""

    if previous_state is None or previous_state.empty:
        return {}

    indexed_state = previous_state.set_index("site_id")
    return indexed_state[["tower_state", "low_load_streak"]].to_dict(orient="index")


def apply_tower_power_rules(
    live_snapshot: pd.DataFrame,
    previous_state: pd.DataFrame | None = None,
    standby_power_watts: float = DEFAULT_STANDBY_POWER_WATTS,
) -> pd.DataFrame:
    """Attach tower on/off decisions and controlled energy estimates."""

    state_lookup = build_previous_state_lookup(previous_state)
    output_rows: list[dict[str, object]] = []

    for _, row in live_snapshot.sort_values("site_id", kind="stable").iterrows():
        site_id = row["site_id"]
        previous_site_state = state_lookup.get(
            site_id,
            {"tower_state": "ON", "low_load_streak": 0},
        )
        previous_tower_state = str(previous_site_state["tower_state"])
        previous_streak = int(previous_site_state["low_load_streak"])

        low_load = (
            int(row["active_users"]) < LOW_USERS_THRESHOLD
            and float(row["traffic_mbps"]) < LOW_TRAFFIC_THRESHOLD_MBPS
            and float(row["prb_utilization"]) < LOW_PRB_THRESHOLD
        )
        wake_required = (
            int(row["active_users"]) >= WAKE_USERS_THRESHOLD
            or float(row["traffic_mbps"]) >= WAKE_TRAFFIC_THRESHOLD_MBPS
            or float(row["prb_utilization"]) >= WAKE_PRB_THRESHOLD
        )

        if previous_tower_state == "ON":
            low_load_streak = previous_streak + 1 if low_load else 0
            if low_load_streak >= LOW_LOAD_STREAK_TO_TURN_OFF:
                tower_state = "OFF"
                decision_reason = "Low load streak reached shutdown threshold"
            else:
                tower_state = "ON"
                decision_reason = "Traffic or utilization still supports service"
        else:
            low_load_streak = previous_streak + 1 if low_load else previous_streak
            if wake_required:
                tower_state = "ON"
                low_load_streak = 0
                decision_reason = "Load increased above wake threshold"
            else:
                tower_state = "OFF"
                decision_reason = "Low-load sleep mode remains active"

        raw_energy_watts = float(row["simulated_energy_watts"])
        controlled_energy_watts = (
            raw_energy_watts if tower_state == "ON" else standby_power_watts
        )
        energy_saved_watts = max(0.0, raw_energy_watts - controlled_energy_watts)

        output_row = row.to_dict()
        output_row["tower_state"] = tower_state
        output_row["low_load_streak"] = low_load_streak
        output_row["decision_reason"] = decision_reason
        output_row["state_color"] = ON_STATE_COLOR if tower_state == "ON" else OFF_STATE_COLOR
        output_row["controlled_energy_watts"] = round(controlled_energy_watts, 1)
        output_row["energy_saved_watts"] = round(energy_saved_watts, 1)
        output_rows.append(output_row)

    return pd.DataFrame(output_rows).reset_index(drop=True)
