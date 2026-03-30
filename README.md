# ucl-ran-energy-sandbox

Modular RAN energy optimization sandbox focused on the UCL Bloomsbury campus area.

## Current State

This repository currently contains an early working prototype with the following implemented pieces:

- `geo`: defines the approximate UCL Bloomsbury study area and loads candidate nearby sites
- `sim`: generates synthetic KPI data and a live simulated KPI stream for the dashboard
- `rules`: applies a simple deterministic tower ON/OFF rule based on live load conditions
- `ml`: includes a standalone baseline linear regression script to predict `energy_watts` from simulated KPI features
- `app`: provides a Streamlit dashboard that shows:
  - study-area sites on a map
  - live simulated KPI values per site
  - green state when a tower is `ON`
  - red state when a tower is `OFF`
  - simple rule-based energy-saving behavior

At the moment, the dashboard uses simulated live data and rule-based control. The ML model has been implemented and trained offline, but it is not yet connected to the live dashboard decision loop.

## What The Project Does Today

- Loads approximate candidate sites around the UCL Bloomsbury campus
- Simulates hourly and live network KPIs such as:
  - `active_users`
  - `traffic_mbps`
  - `prb_utilization`
  - `avg_sinr`
  - `energy_watts`
- Applies simple low-load sleep logic to switch towers between `ON` and `OFF`
- Visualizes the current simulated network state in Streamlit

## Current Limitations

- All KPI data is synthetic
- Tower switching is currently rule-based, not ML-driven
- The trained linear regression model is not yet used for live inference
- No compare, llm, or interface modules beyond the current Streamlit MVP have been implemented

## Future Implementation

The planned next steps are:

1. Connect the linear regression model to the live simulation so the dashboard can show predicted wattage per site in real time.
2. Use predicted wattage as an input to smarter switching decisions, while keeping clear safety guardrails.
3. Add comparison logic to evaluate baseline operation versus controlled operation.
4. Expand the dashboard into a more complete interface for scenario testing and demonstration.
5. Optionally introduce LLM-assisted explanation features later, once the core simulation and control flow are stable.

## Run The Dashboard

From the project root:

```powershell
python -m streamlit run app/streamlit_app.py
```

## Run The Baseline Linear Regression

```powershell
python -m src.ml.train_linear_regression
```

## Project Goal

The goal is to build a simple, modular, and explainable sandbox for demonstrating how RAN energy optimization could be explored around a small urban study area before moving to more advanced control logic.
