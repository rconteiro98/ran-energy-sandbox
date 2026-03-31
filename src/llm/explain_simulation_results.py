"""Build compact summaries and prompts for Simulation & ML explanations."""

from __future__ import annotations

import pandas as pd


def build_training_results_summary(
    training_results: dict[str, object]
) -> dict[str, object]:
    """Return a compact, prompt-friendly summary of model training outputs."""

    metrics = training_results["metrics"]
    coefficients = training_results["coefficients"]
    predictions = training_results["predictions"]
    split_timestamp = pd.Timestamp(training_results["split_timestamp"].iloc[0])

    coefficient_frame = coefficients.loc[
        coefficients["feature"] != "intercept",
        ["feature", "coefficient"],
    ].copy()
    strongest_coefficients = coefficient_frame.reindex(
        coefficient_frame["coefficient"].abs().sort_values(ascending=False).index
    ).head(4)

    strongest_terms = [
        {
            "feature": str(row["feature"]),
            "coefficient": round(float(row["coefficient"]), 3),
        }
        for _, row in strongest_coefficients.iterrows()
    ]

    prediction_frame = predictions.copy()
    prediction_frame["prediction_error_watts"] = (
        prediction_frame["predicted_energy_watts"]
        - prediction_frame["actual_energy_watts"]
    )

    total_actual = float(prediction_frame["actual_energy_watts"].sum())
    total_predicted = float(prediction_frame["predicted_energy_watts"].sum())
    total_error = total_predicted - total_actual

    return {
        "split_timestamp": split_timestamp.strftime("%Y-%m-%d %H:%M"),
        "test_rows": int(len(prediction_frame)),
        "test_sites": int(prediction_frame["site_id"].nunique()),
        "mae_watts": round(float(metrics["mae"]), 2),
        "rmse_watts": round(float(metrics["rmse"]), 2),
        "r2": round(float(metrics["r2"]), 4),
        "total_actual_watts": round(total_actual, 1),
        "total_predicted_watts": round(total_predicted, 1),
        "total_bias_watts": round(total_error, 1),
        "top_coefficients": strongest_terms,
    }


def build_forecast_summary(forecast_frame: pd.DataFrame) -> dict[str, object]:
    """Return a compact, prompt-friendly summary of forecast behavior."""

    ordered_forecast = forecast_frame.sort_values(
        ["timestamp", "site_id"], kind="stable"
    ).reset_index(drop=True)
    total_forecast = (
        ordered_forecast.groupby("timestamp", as_index=False)["predicted_energy_watts"]
        .sum()
        .rename(columns={"predicted_energy_watts": "total_predicted_watts"})
    )
    total_forecast["date"] = total_forecast["timestamp"].dt.strftime("%Y-%m-%d")

    daily_totals = (
        total_forecast.groupby("date", as_index=False)["total_predicted_watts"]
        .sum()
        .sort_values("date", kind="stable")
        .reset_index(drop=True)
    )
    peak_row = total_forecast.loc[
        total_forecast["total_predicted_watts"].idxmax()
    ]
    low_row = total_forecast.loc[
        total_forecast["total_predicted_watts"].idxmin()
    ]

    per_site_average = (
        ordered_forecast.groupby("site_id", as_index=False)["predicted_energy_watts"]
        .mean()
        .rename(columns={"predicted_energy_watts": "mean_predicted_watts"})
        .sort_values("mean_predicted_watts", ascending=False, kind="stable")
        .reset_index(drop=True)
    )

    top_sites = [
        {
            "site_id": str(row["site_id"]),
            "mean_predicted_watts": round(float(row["mean_predicted_watts"]), 1),
        }
        for _, row in per_site_average.head(3).iterrows()
    ]

    first_day_total = float(daily_totals.iloc[0]["total_predicted_watts"])
    last_day_total = float(daily_totals.iloc[-1]["total_predicted_watts"])
    day_change_pct = (
        ((last_day_total - first_day_total) / first_day_total) * 100.0
        if first_day_total != 0.0
        else 0.0
    )

    return {
        "forecast_start": pd.Timestamp(ordered_forecast["timestamp"].min()).strftime(
            "%Y-%m-%d %H:%M"
        ),
        "forecast_end": pd.Timestamp(ordered_forecast["timestamp"].max()).strftime(
            "%Y-%m-%d %H:%M"
        ),
        "rows": int(len(ordered_forecast)),
        "sites": int(ordered_forecast["site_id"].nunique()),
        "mean_predicted_watts": round(
            float(ordered_forecast["predicted_energy_watts"].mean()),
            1,
        ),
        "peak_total_watts": round(float(peak_row["total_predicted_watts"]), 1),
        "peak_timestamp": pd.Timestamp(peak_row["timestamp"]).strftime("%Y-%m-%d %H:%M"),
        "lowest_total_watts": round(float(low_row["total_predicted_watts"]), 1),
        "lowest_timestamp": pd.Timestamp(low_row["timestamp"]).strftime("%Y-%m-%d %H:%M"),
        "first_day_total_watts": round(first_day_total, 1),
        "last_day_total_watts": round(last_day_total, 1),
        "day_change_pct": round(day_change_pct, 1),
        "top_sites": top_sites,
    }


def build_simulation_summary_prompt(
    training_summary: dict[str, object],
    forecast_summary: dict[str, object],
) -> str:
    """Build a short prompt for a plain-English simulation summary."""

    return (
        "You are helping explain a research prototype for RAN energy forecasting.\n"
        "Write exactly two short paragraphs in plain English.\n"
        "Paragraph 1 must summarize model training quality.\n"
        "Paragraph 2 must summarize forecast trends.\n"
        "Mention that the data is simulated.\n"
        "Do not invent values or recommendations.\n"
        "Keep the full answer under 140 words.\n\n"
        f"Training summary: {training_summary}\n"
        f"Forecast summary: {forecast_summary}\n"
    )
