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
from src.ml import predict_dataset_energy, train_and_evaluate_model_from_frame
from src.rules import apply_tower_power_rules
from src.sim import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_START_TIMESTAMP,
    build_live_kpi_snapshot,
    generate_simulated_kpis,
)


st.set_page_config(page_title="UCL RAN Energy Sandbox", layout="wide")

DEFAULT_SITE_COLOR = [117, 117, 117, 210]
STUDY_RADIUS_COLOR = [255, 152, 0, 160]
CAMPUS_CENTER_COLOR = [220, 53, 69, 220]
STATE_LABELS = {"ON": "green", "OFF": "red"}
LIVE_VIEW_LABEL = "Live Control"
SIMULATION_VIEW_LABEL = "Simulation & ML"
DEFAULT_FORECAST_DAYS = 7


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
    """Render a compact summary section above the live map."""

    on_sites = (
        int((filtered_sites["tower_state"] == "ON").sum()) if not filtered_sites.empty else 0
    )
    off_sites = (
        int((filtered_sites["tower_state"] == "OFF").sum()) if not filtered_sites.empty else 0
    )
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


def merge_site_metadata(
    kpi_frame: pd.DataFrame, study_area_sites: pd.DataFrame
) -> pd.DataFrame:
    """Attach explicit site metadata to a KPI frame for display purposes."""

    metadata_columns = [
        "site_id",
        "site_name",
        "site_type",
        "latitude",
        "longitude",
        "distance_to_campus_m",
    ]
    metadata = study_area_sites[metadata_columns].drop_duplicates("site_id")
    merged_frame = kpi_frame.copy()
    missing_columns = [
        column_name
        for column_name in metadata_columns[1:]
        if column_name not in merged_frame.columns
    ]
    if missing_columns:
        merged_frame = merged_frame.drop(
            columns=[column_name for column_name in metadata_columns if column_name in merged_frame.columns and column_name != "site_id"],
            errors="ignore",
        ).merge(metadata, on="site_id", how="left")

    return merged_frame.sort_values(["timestamp", "site_id"], kind="stable").reset_index(
        drop=True
    )


def clear_simulation_ml_state() -> None:
    """Remove model results that depend on the current simulation controls."""

    st.session_state.pop("ml_training_results", None)
    st.session_state.pop("ml_forecast_frame", None)
    st.session_state.pop("ml_forecast_start", None)
    st.session_state.pop("ml_forecast_end", None)


def build_simulation_start_timestamp(start_date: object) -> str:
    """Convert the selected date into the hourly simulator timestamp format."""

    return pd.Timestamp(start_date).strftime("%Y-%m-%d 00:00:00")


def render_simulation_metrics(simulated_kpis: pd.DataFrame) -> None:
    """Render a compact summary for the simulated KPI dataset."""

    summary_columns = st.columns(4)
    summary_columns[0].metric("Rows", len(simulated_kpis))
    summary_columns[1].metric("Sites", simulated_kpis["site_id"].nunique())
    summary_columns[2].metric(
        "Mean energy",
        f"{simulated_kpis['energy_watts'].mean():.1f} W",
    )
    summary_columns[3].metric(
        "Peak traffic",
        f"{simulated_kpis['traffic_mbps'].max():.1f} Mbps",
    )

    range_columns = st.columns(3)
    range_columns[0].metric(
        "Start",
        simulated_kpis["timestamp"].min().strftime("%Y-%m-%d %H:%M"),
    )
    range_columns[1].metric(
        "End",
        simulated_kpis["timestamp"].max().strftime("%Y-%m-%d %H:%M"),
    )
    range_columns[2].metric("Study radius", f"{STUDY_RADIUS_METERS} m")


def render_simulation_charts(simulated_kpis: pd.DataFrame) -> None:
    """Render simple visual summaries of the simulated KPI dataset."""

    total_energy = (
        simulated_kpis.groupby("timestamp", as_index=True)["energy_watts"]
        .sum()
        .to_frame("total_energy_watts")
    )
    per_site_energy = simulated_kpis.pivot(
        index="timestamp",
        columns="site_id",
        values="energy_watts",
    )
    total_users = (
        simulated_kpis.groupby("timestamp", as_index=True)["active_users"]
        .sum()
        .to_frame("total_active_users")
    )

    chart_columns = st.columns(2)
    chart_columns[0].subheader("Total Simulated Energy")
    chart_columns[0].line_chart(total_energy, use_container_width=True)
    chart_columns[1].subheader("Total Simulated Users")
    chart_columns[1].line_chart(total_users, use_container_width=True)

    st.subheader("Per-Site Simulated Energy")
    st.line_chart(per_site_energy, use_container_width=True)


def render_simulated_kpi_table(simulated_kpis: pd.DataFrame) -> None:
    """Render the detailed simulated KPI table."""

    st.subheader("Simulated KPI Table")
    st.dataframe(
        simulated_kpis[
            [
                "timestamp",
                "site_id",
                "site_name",
                "site_type",
                "active_users",
                "traffic_mbps",
                "prb_utilization",
                "avg_sinr",
                "energy_watts",
                "hour_of_day",
                "is_weekend",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


def render_training_results(training_results: dict[str, object]) -> None:
    """Render model metrics, coefficients, and fit plots."""

    metrics = training_results["metrics"]
    coefficients = training_results["coefficients"]
    predictions = training_results["predictions"]
    split_timestamp = training_results["split_timestamp"].iloc[0]

    st.subheader("Linear Regression Training")
    st.caption(
        "The model is multivariate, so the clearest fit views are actual-vs-predicted "
        "comparisons and test-period time series rather than a single regression line."
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric("Train/test split", str(split_timestamp))
    metric_columns[1].metric("MAE", f"{metrics['mae']:.2f} W")
    metric_columns[2].metric("RMSE", f"{metrics['rmse']:.2f} W")
    metric_columns[3].metric("R2", f"{metrics['r2']:.4f}")

    comparison_frame = (
        predictions.groupby("timestamp", as_index=True)[
            ["actual_energy_watts", "predicted_energy_watts"]
        ]
        .sum()
        .rename(
            columns={
                "actual_energy_watts": "actual_total_watts",
                "predicted_energy_watts": "predicted_total_watts",
            }
        )
    )

    chart_columns = st.columns(2)
    chart_columns[0].subheader("Test Period Total Watts")
    chart_columns[0].line_chart(comparison_frame, use_container_width=True)
    chart_columns[1].subheader("Actual vs Predicted Scatter")
    chart_columns[1].scatter_chart(
        predictions,
        x="actual_energy_watts",
        y="predicted_energy_watts",
        use_container_width=True,
    )

    table_columns = st.columns(2)
    table_columns[0].subheader("Learned Coefficients")
    table_columns[0].dataframe(coefficients, use_container_width=True, hide_index=True)
    table_columns[1].subheader("Prediction Sample")
    table_columns[1].dataframe(
        predictions.head(20),
        use_container_width=True,
        hide_index=True,
    )


def render_forecast_results(forecast_frame: pd.DataFrame) -> None:
    """Render the next-week energy forecast outputs."""

    st.subheader("Next-Week Energy Forecast")

    metric_columns = st.columns(4)
    metric_columns[0].metric("Forecast rows", len(forecast_frame))
    metric_columns[1].metric("Forecast sites", forecast_frame["site_id"].nunique())
    metric_columns[2].metric(
        "Mean predicted watts",
        f"{forecast_frame['predicted_energy_watts'].mean():.1f} W",
    )
    metric_columns[3].metric(
        "Peak predicted watts",
        f"{forecast_frame['predicted_energy_watts'].max():.1f} W",
    )

    total_forecast = (
        forecast_frame.groupby("timestamp", as_index=True)["predicted_energy_watts"]
        .sum()
        .to_frame("total_predicted_watts")
    )
    per_site_forecast = forecast_frame.pivot(
        index="timestamp",
        columns="site_id",
        values="predicted_energy_watts",
    )

    chart_columns = st.columns(2)
    chart_columns[0].subheader("Total Forecast Watts")
    chart_columns[0].line_chart(total_forecast, use_container_width=True)
    chart_columns[1].subheader("Per-Site Forecast Watts")
    chart_columns[1].line_chart(per_site_forecast, use_container_width=True)

    st.dataframe(
        forecast_frame[
            [
                "timestamp",
                "site_id",
                "site_name",
                "site_type",
                "active_users",
                "traffic_mbps",
                "prb_utilization",
                "avg_sinr",
                "predicted_energy_watts",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


def render_live_view(study_area_sites: pd.DataFrame) -> None:
    """Render the current live-control dashboard view."""

    initialize_live_state(study_area_sites)

    st.sidebar.header("Live controls")
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

    st.write(
        "Approximate candidate sites within the Bloomsbury study radius around UCL. "
        "This dashboard simulates live KPIs and applies a simple rule-based tower "
        "ON/OFF controller."
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


def render_simulation_ml_view(study_area_sites: pd.DataFrame) -> None:
    """Render the simulation, training, and forecasting dashboard view."""

    st.write(
        "Use this view to inspect the synthetic KPI dataset, train the baseline "
        "linear regression model, and forecast next-week tower energy from "
        "simulated future KPIs."
    )

    st.sidebar.header("Simulation settings")
    start_date = st.sidebar.date_input(
        "Simulation start date",
        value=pd.Timestamp(DEFAULT_START_TIMESTAMP).date(),
    )
    num_days = st.sidebar.slider("Simulation days", 7, 28, 14)
    random_seed = int(
        st.sidebar.number_input(
            "Simulation random seed",
            min_value=0,
            value=int(DEFAULT_RANDOM_SEED),
            step=1,
        )
    )

    simulation_signature = (
        f"{pd.Timestamp(start_date).date().isoformat()}|{num_days}|{random_seed}"
    )
    if st.session_state.get("ml_simulation_signature") != simulation_signature:
        st.session_state["ml_simulation_signature"] = simulation_signature
        clear_simulation_ml_state()

    simulated_kpis = merge_site_metadata(
        generate_simulated_kpis(
            start_timestamp=build_simulation_start_timestamp(start_date),
            num_days=num_days,
            random_seed=random_seed,
        ),
        study_area_sites,
    )

    render_simulation_metrics(simulated_kpis)
    render_simulation_charts(simulated_kpis)
    render_simulated_kpi_table(simulated_kpis)

    action_columns = st.columns(2)
    train_model = action_columns[0].button(
        "Train linear regression",
        use_container_width=True,
    )
    predict_next_week = action_columns[1].button(
        "Predict next week watts",
        use_container_width=True,
        disabled="ml_training_results" not in st.session_state,
    )

    if train_model:
        st.session_state["ml_training_results"] = train_and_evaluate_model_from_frame(
            simulated_kpis
        )
        st.session_state.pop("ml_forecast_frame", None)
        st.session_state.pop("ml_forecast_start", None)
        st.session_state.pop("ml_forecast_end", None)

    training_results = st.session_state.get("ml_training_results")
    if training_results is not None:
        render_training_results(training_results)
    else:
        st.info("Train the linear regression model to see fit metrics, coefficients, and plots.")

    if predict_next_week and training_results is not None:
        forecast_start = pd.Timestamp(simulated_kpis["timestamp"].max()) + pd.Timedelta(hours=1)
        forecast_frame = merge_site_metadata(
            generate_simulated_kpis(
                start_timestamp=forecast_start.strftime("%Y-%m-%d %H:%M:%S"),
                num_days=DEFAULT_FORECAST_DAYS,
                random_seed=random_seed,
            ),
            study_area_sites,
        )
        st.session_state["ml_forecast_frame"] = predict_dataset_energy(
            forecast_frame,
            training_results["coefficients"],
        )
        st.session_state["ml_forecast_start"] = forecast_start
        st.session_state["ml_forecast_end"] = forecast_frame["timestamp"].max()

    forecast_frame = st.session_state.get("ml_forecast_frame")
    if forecast_frame is not None:
        forecast_start = st.session_state.get("ml_forecast_start")
        forecast_end = st.session_state.get("ml_forecast_end")
        if forecast_start is not None and forecast_end is not None:
            st.caption(
                "Forecast window: "
                f"{pd.Timestamp(forecast_start).strftime('%Y-%m-%d %H:%M')} to "
                f"{pd.Timestamp(forecast_end).strftime('%Y-%m-%d %H:%M')}"
            )
        render_forecast_results(forecast_frame)


def main() -> None:
    """Render the dashboard using simulated live KPIs and model experimentation."""

    study_area_sites = load_study_area_sites()

    st.title("UCL RAN Energy Sandbox")
    selected_view = st.sidebar.radio(
        "View",
        options=[LIVE_VIEW_LABEL, SIMULATION_VIEW_LABEL],
    )

    if selected_view == LIVE_VIEW_LABEL:
        render_live_view(study_area_sites)
    else:
        render_simulation_ml_view(study_area_sites)


if __name__ == "__main__":
    main()
