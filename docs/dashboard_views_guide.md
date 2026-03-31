# Dashboard Views Guide

This note explains the two Streamlit dashboard views in simple, interview-friendly language.

## The Two Views

- `Live Control`: shows a simulated live network snapshot and a simple rule-based ON/OFF decision for each tower.
- `Simulation & ML`: shows the full synthetic KPI dataset, trains the baseline linear regression model, and predicts future tower energy.

## View 1: Live Control

### What this view is for

Use this view to explain how the sandbox behaves like a small live operations screen.

### Step by step

1. The app loads the approximate study-area sites from the `geo` module.
2. It creates a live KPI snapshot for the current time step.
3. It applies the rule-based controller from the `rules` module.
4. Each tower is marked as `ON` or `OFF`.
5. The dashboard shows summary metrics such as towers ON, towers OFF, and energy saved.
6. It renders tower cards so each site can be inspected quickly.
7. It plots the sites on the map around the UCL Bloomsbury study area.
8. It shows the detailed live table with the decision reason for each tower.
9. The user can advance one step, reset the simulation, or enable auto-refresh.

### Important concepts in this view

- `Study area`: the approximate Bloomsbury radius around the campus center used to keep the MVP focused.
- `Candidate site`: a possible nearby tower or site in the sandbox dataset.
- `Live KPI snapshot`: one simulated moment in time for all sites.
- `Active users`: the number of simulated users attached to a site.
- `Traffic Mbps`: the simulated traffic load passing through the site.
- `PRB utilization`: a simple radio load indicator showing how much radio resource is being used.
- `SINR`: signal quality, included as a telecom-relevant KPI.
- `Tower state`: whether the rule keeps the tower `ON` or puts it into low-load sleep mode as `OFF`.
- `Controlled energy watts`: the power used after the rule decision is applied.
- `Energy saved watts`: the difference between baseline simulated energy and the controlled energy.
- `Decision reason`: the explicit rule output that explains why the tower changed or kept its state.

### How to describe it in one sentence

The `Live Control` view is the operational view: it takes simulated live KPIs and shows how a simple controller would switch towers on or off in real time.

## View 2: Simulation & ML

### What this view is for

Use this view to explain how the sandbox generates data, evaluates the baseline model, and produces a simple forecast.

### Step by step

1. The user chooses a simulation start date, number of days, and random seed.
2. The app generates a synthetic KPI dataset for all study-area sites.
3. It merges the KPI data with site metadata from the `geo` module.
4. The dashboard shows summary metrics such as row count, site count, mean energy, and peak traffic.
5. It plots total simulated energy, total simulated users, and per-site energy over time.
6. It shows the detailed KPI table for inspection.
7. The user clicks `Train linear regression`.
8. The app splits the data into train and test periods and fits the baseline model.
9. It displays model metrics such as `MAE`, `RMSE`, and `R2`.
10. It shows learned coefficients and actual-vs-predicted charts.
11. The user clicks `Predict next week watts`.
12. The app generates a future synthetic KPI window and uses the trained coefficients to estimate future energy demand.
13. It displays forecast metrics, forecast charts, and the forecast table.

### Important concepts in this view

- `Synthetic KPI dataset`: generated data used for prototyping before real network telemetry exists.
- `Random seed`: keeps the simulation reproducible so the same settings give the same results.
- `Baseline model`: the first simple ML model, used for explainability and easy debugging.
- `Linear regression`: predicts `energy_watts` from the KPI features with a transparent weighted formula.
- `Train/test split`: separates past rows used for fitting from later rows used for evaluation.
- `MAE`: average absolute prediction error in watts.
- `RMSE`: error metric that penalizes larger mistakes more strongly.
- `R2`: a rough measure of how much variance the model explains.
- `Coefficients`: the learned weights showing how each feature influences predicted energy.
- `Forecast window`: the next simulated period after the training data ends.

### How to describe it in one sentence

The `Simulation & ML` view is the analysis view: it generates a reproducible synthetic dataset, trains a transparent baseline model, and uses it to forecast future energy.

## The Difference Between The Two Views

- `Live Control` is about current behavior and rule-based decisions.
- `Simulation & ML` is about dataset generation, model training, and future prediction.
- `Live Control` answers, "What is happening now in the simulated network?"
- `Simulation & ML` answers, "What does the model learn from the simulated data, and what might happen next?"

## Simple Interview Framing

You can explain the dashboard as a split between:

- an operational view for rule-based control
- an analytical view for simulation, model training, and forecasting

That split keeps the MVP modular and makes each part easier to test and explain.
