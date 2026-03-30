"""Simple Streamlit dashboard for the approximate UCL Bloomsbury study area.

The site locations shown here are approximate and intended for research
prototyping only. This dashboard is designed to be readable and demo-friendly.
"""

from __future__ import annotations

import time

import pandas as pd
import pydeck as pdk
import streamlit as st

from src.geo import (
    CAMPUS_CENTER_LATITUDE,
    CAMPUS_CENTER_LONGITUDE,
    STUDY_RADIUS_METERS,
    load_study_area_sites,
)
from src.rules import apply_tower_power_rules
from src.sim import build_live_kpi_snapshot


st.set_page_config(page_title="UCL RAN Energy Sandbox", layout="wide")

DEFAULT_SITE_COLOR = [117, 117, 117, 210]
STUDY_RADIUS_COLOR = [255, 152, 0, 160]
CAMPUS_CENTER_COLOR = [220, 53, 69, 220]
STATE_LABELS = {"ON": "green", "OFF": "red"}


def build_campus_dataframe() -> pd.DataFrame:
    """Return the campus center in the same schema used by site layers."""

    return pd.DataFrame(
        [
            {
                "site_id": "CAMPUS",
                "site_name": "Approximate UCL Bloomsbury Campus Center",
                "site_type": "campus_center",
                "distance_to_campus_m": 0.0,
                "latitude": CAMPUS_CENTER_LATITUDE,
                "longitude": CAMPUS_CENTER_LONGITUDE,
            }
        ]
    )


def add_tower_state_colors(site_states: pd.DataFrame) -> pd.DataFrame:
    """Attach map colors using the current tower on/off state."""

    display_sites = site_states.copy()
    display_sites["fill_color"] = display_sites["state_color"].apply(
        lambda color: color if isinstance(color, list) else DEFAULT_SITE_COLOR
    )
    return display_sites


def format_distance(value: float | None) -> str:
    """Return a readable metric string for a distance value in meters."""

    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.1f} m"


def build_study_radius_layer() -> pdk.Layer:
    """Create a simple outline showing the approximate study radius."""

    radius_circle = pd.DataFrame(
        [
            {
                "latitude": CAMPUS_CENTER_LATITUDE,
                "longitude": CAMPUS_CENTER_LONGITUDE,
                "radius_m": STUDY_RADIUS_METERS,
            }
        ]
    )

    return pdk.Layer(
        "ScatterplotLayer",
        data=radius_circle,
        get_position="[longitude, latitude]",
        get_radius="radius_m",
        get_fill_color=[0, 0, 0, 0],
        get_line_color=STUDY_RADIUS_COLOR,
        pickable=False,
        stroked=True,
        filled=False,
        line_width_min_pixels=2,
    )


def build_site_layer(study_area_sites: pd.DataFrame) -> pdk.Layer:
    """Create a point layer for approximate candidate study-area sites."""

    return pdk.Layer(
        "ScatterplotLayer",
        data=study_area_sites,
        get_position="[longitude, latitude]",
        get_radius=35,
        get_fill_color="fill_color",
        pickable=True,
    )


def build_campus_layer() -> pdk.Layer:
    """Create a separate point layer for the approximate campus center."""

    campus_center = build_campus_dataframe()

    return pdk.Layer(
        "ScatterplotLayer",
        data=campus_center,
        get_position="[longitude, latitude]",
        get_radius=30,
        get_fill_color=CAMPUS_CENTER_COLOR,
        pickable=True,
    )


def build_deck(site_states: pd.DataFrame) -> pdk.Deck:
    """Build the interactive map centered on the approximate campus area."""

    tooltip = {
        "html": (
            "<b>site_id:</b> {site_id}<br/>"
            "<b>site_name:</b> {site_name}<br/>"
            "<b>site_type:</b> {site_type}<br/>"
            "<b>tower_state:</b> {tower_state}<br/>"
            "<b>active_users:</b> {active_users}<br/>"
            "<b>traffic_mbps:</b> {traffic_mbps}<br/>"
            "<b>prb_utilization:</b> {prb_utilization}<br/>"
            "<b>controlled_energy_watts:</b> {controlled_energy_watts}"
        )
    }

    return pdk.Deck(
        map_style="light",
        initial_view_state=pdk.ViewState(
            latitude=CAMPUS_CENTER_LATITUDE,
            longitude=CAMPUS_CENTER_LONGITUDE,
            zoom=14,
            pitch=0,
        ),
        layers=[
            build_study_radius_layer(),
            build_site_layer(site_states),
            build_campus_layer(),
        ],
        tooltip=tooltip,
    )


def build_site_type_filter(study_area_sites: pd.DataFrame) -> list[str]:
    """Render a simple sidebar filter for site types."""

    available_site_types = sorted(study_area_sites["site_type"].unique().tolist())
    return st.sidebar.multiselect(
        "Site type",
        options=available_site_types,
        default=available_site_types,
    )


def filter_sites_by_type(
    study_area_sites: pd.DataFrame, selected_site_types: list[str]
) -> pd.DataFrame:
    """Filter the study-area sites by the selected site types."""

    if not selected_site_types:
        return study_area_sites.iloc[0:0].copy()

    return study_area_sites.loc[
        study_area_sites["site_type"].isin(selected_site_types)
    ].copy()


def render_metrics(filtered_sites: pd.DataFrame) -> None:
    """Render a compact summary section above the map."""

    on_sites = int((filtered_sites["tower_state"] == "ON").sum()) if not filtered_sites.empty else 0
    off_sites = int((filtered_sites["tower_state"] == "OFF").sum()) if not filtered_sites.empty else 0
    controlled_energy = (
        filtered_sites["controlled_energy_watts"].sum() if not filtered_sites.empty else 0.0
    )
    energy_saved = (
        filtered_sites["energy_saved_watts"].sum() if not filtered_sites.empty else 0.0
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric("Filtered towers", len(filtered_sites))
    metric_columns[1].metric("Towers ON", on_sites)
    metric_columns[2].metric("Towers OFF", off_sites)
    metric_columns[3].metric("Energy saved", f"{energy_saved:.1f} W")

    energy_columns = st.columns(3)
    energy_columns[0].metric("Controlled energy", f"{controlled_energy:.1f} W")
    energy_columns[1].metric("Study radius", f"{STUDY_RADIUS_METERS} m")
    energy_columns[2].metric(
        "Simulated timestamp",
        filtered_sites["timestamp"].iloc[0].strftime("%Y-%m-%d %H:%M")
        if not filtered_sites.empty
        else "N/A",
    )


def initialize_live_state(study_area_sites: pd.DataFrame) -> None:
    """Initialize the simulated live network state in Streamlit session state."""

    if "live_tick" in st.session_state and "live_network_state" in st.session_state:
        return

    initial_snapshot = build_live_kpi_snapshot(study_area_sites, step_index=0)
    st.session_state.live_tick = 0
    st.session_state.live_network_state = apply_tower_power_rules(initial_snapshot)


def advance_live_state(study_area_sites: pd.DataFrame) -> None:
    """Advance the live simulator by one step and apply the rule controller."""

    next_tick = int(st.session_state.get("live_tick", 0)) + 1
    previous_state = st.session_state.get("live_network_state")
    next_snapshot = build_live_kpi_snapshot(study_area_sites, step_index=next_tick)
    next_network_state = apply_tower_power_rules(
        live_snapshot=next_snapshot,
        previous_state=previous_state,
    )
    st.session_state.live_tick = next_tick
    st.session_state.live_network_state = next_network_state


def reset_live_state() -> None:
    """Clear the current live simulation so it restarts from step zero."""

    st.session_state.pop("live_tick", None)
    st.session_state.pop("live_network_state", None)


def render_tower_cards(filtered_sites: pd.DataFrame) -> None:
    """Render a simple card per tower with the key live KPI values."""

    st.subheader("Live Tower States")
    columns = st.columns(3)

    for site_position, (_, site_row) in enumerate(filtered_sites.iterrows()):
        column = columns[site_position % 3]
        state_text = str(site_row["tower_state"])
        state_color = "#2ea043" if state_text == "ON" else "#dc3545"
        state_label = STATE_LABELS.get(state_text, state_text.lower())
        column.markdown(
            f"""
            <div style="border:1px solid #d9d9d9;border-radius:12px;padding:16px;margin-bottom:12px;background:#fafafa;">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                    <div>
                        <div style="font-size:0.9rem;color:#6b7280;">{site_row["site_id"]}</div>
                        <div style="font-size:1.05rem;font-weight:700;">{site_row["site_name"]}</div>
                    </div>
                    <div style="padding:4px 10px;border-radius:999px;background:{state_color};color:white;font-weight:700;">
                        {state_text}
                    </div>
                </div>
                <div><b>Users:</b> {int(site_row["active_users"])}</div>
                <div><b>Traffic:</b> {float(site_row["traffic_mbps"]):.1f} Mbps</div>
                <div><b>PRB:</b> {float(site_row["prb_utilization"]):.1f}%</div>
                <div><b>SINR:</b> {float(site_row["avg_sinr"]):.1f} dB</div>
                <div><b>Energy:</b> {float(site_row["controlled_energy_watts"]):.1f} W</div>
                <div style="margin-top:8px;color:{state_color};font-weight:600;">
                    Tower state is {state_label}.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    """Render the dashboard using simulated live KPIs and rule-based control."""

    study_area_sites = load_study_area_sites()
    initialize_live_state(study_area_sites)

    st.sidebar.header("Filters")
    selected_site_types = build_site_type_filter(study_area_sites)
    live_auto_refresh = st.sidebar.toggle("Live auto-refresh", value=False)
    refresh_seconds = st.sidebar.slider("Refresh interval (seconds)", 1, 10, 3)
    advance_once = st.sidebar.button("Advance one step", use_container_width=True)
    reset_simulation = st.sidebar.button("Reset simulation", use_container_width=True)

    if reset_simulation:
        reset_live_state()
        initialize_live_state(study_area_sites)

    if advance_once:
        advance_live_state(study_area_sites)

    current_network_state = st.session_state["live_network_state"].copy()
    filtered_sites = filter_sites_by_type(current_network_state, selected_site_types)
    table_sites = filtered_sites.copy()
    table_sites["distance_to_campus_m"] = table_sites["distance_to_campus_m"].round(1)
    map_sites = add_tower_state_colors(table_sites)

    st.title("UCL RAN Energy Sandbox")
    st.write(
        "Approximate candidate sites within the Bloomsbury study radius around UCL. "
        "This dashboard now simulates live KPIs and applies a simple rule-based "
        "tower ON/OFF controller."
    )
    st.caption(
        "Green means the tower is ON. Red means the tower is in low-load sleep mode."
    )

    render_metrics(table_sites)
    render_tower_cards(table_sites)
    st.pydeck_chart(build_deck(map_sites), use_container_width=True)
    st.subheader("Live Site Table")
    st.dataframe(
        table_sites[
            [
                "timestamp",
                "site_id",
                "site_type",
                "active_users",
                "traffic_mbps",
                "prb_utilization",
                "avg_sinr",
                "tower_state",
                "controlled_energy_watts",
                "energy_saved_watts",
                "decision_reason",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    if live_auto_refresh:
        time.sleep(refresh_seconds)
        advance_live_state(study_area_sites)
        st.rerun()


if __name__ == "__main__":
    main()
