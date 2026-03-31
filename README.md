# ucl-ran-energy-sandbox

Modular RAN energy optimization sandbox focused on the UCL Bloomsbury campus area.

## Current State

This repository currently contains an early working prototype with the following implemented pieces:

- `geo`: defines the approximate UCL Bloomsbury study area and loads candidate nearby sites
- `sim`: generates synthetic KPI data and a live simulated KPI stream for the dashboard
- `rules`: applies a simple deterministic tower ON/OFF rule based on live load conditions
- `ml`: includes a standalone baseline linear regression script to predict `energy_watts` from simulated KPI features
- `llm`: provides an optional local-LLM explanation layer for summarizing model training results and forecast trends in plain English
- `app`: provides a Streamlit dashboard that shows:
  - study-area sites on a map
  - live simulated KPI values per site
  - green state when a tower is `ON`
  - red state when a tower is `OFF`
  - simple rule-based energy-saving behavior
  - simulation charts, model metrics, and next-week energy forecasts
  - an optional local-LLM summary for the `Simulation & ML` view

At the moment, the dashboard uses simulated live data and rule-based control. The ML model can be trained and used for next-week energy forecasting inside the simulation workflow, but it is not yet connected to the live dashboard decision loop. The local LLM feature is explanation-only and does not control predictions or tower states.

## What The Project Does Today

- Loads approximate candidate sites around the UCL Bloomsbury campus
- Simulates hourly and live network KPIs such as:
  - `active_users`
  - `traffic_mbps`
  - `prb_utilization`
  - `avg_sinr`
  - `energy_watts`
- Applies simple low-load sleep logic to switch towers between `ON` and `OFF`
- Trains a multivariate linear regression baseline on simulated KPI data
- Predicts next-week energy demand from simulated future KPIs
- Summarizes training results and forecast trends with an optional local LLM
- Visualizes the current simulated network state in Streamlit

## Current Limitations

- All KPI data is synthetic
- Tower switching is currently rule-based, not ML-driven
- The trained linear regression model is not yet used for live inference
- The local LLM only explains existing outputs and does not make control decisions
- The local LLM depends on a separately running local model endpoint
- No compare or interface modules beyond the current Streamlit MVP have been implemented

## Future Implementation

The planned next steps are:

1. Connect the linear regression model to the live simulation so the dashboard can show predicted wattage per site in real time.
2. Use predicted wattage as an input to smarter switching decisions, while keeping clear safety guardrails.
3. Add comparison logic to evaluate baseline operation versus controlled operation.
4. Expand the local LLM layer from summary text into more targeted explanation features, such as per-site forecast interpretation and clearer model diagnostics.
5. Expand the dashboard into a more complete interface for scenario testing and demonstration.

## Run The Dashboard

From the project root:

```powershell
python -m streamlit run app/streamlit_app.py
```

If you want to use the optional LLM summary in the `Simulation & ML` view, start a compatible local model runtime first and enter the local endpoint and installed model tag in the dashboard. The LLM is used only to explain training and forecast outputs in plain English.

## Run The Baseline Linear Regression

```powershell
python -m src.ml.train_linear_regression
```

## Dashboard Walkthrough

For a step-by-step explanation of the two dashboard views and the key concepts behind them, see [docs/dashboard_views_guide.md](docs/dashboard_views_guide.md).

## Project Goal

The goal is to build a simple, modular, and explainable sandbox for demonstrating how RAN energy optimization could be explored around a small urban study area before moving to more advanced control logic.
