"""Tests for compact Simulation & ML LLM summary builders."""

from __future__ import annotations

import unittest

from src.llm import (
    build_forecast_summary,
    build_simulation_summary_prompt,
    build_training_results_summary,
)
from src.ml import predict_dataset_energy, train_and_evaluate_model_from_frame
from src.sim import generate_simulated_kpis


class LlmSimulationSummaryTests(unittest.TestCase):
    """Check that prompt builders stay compact and deterministic."""

    def test_training_summary_contains_core_metrics(self) -> None:
        training_frame = generate_simulated_kpis(num_days=7, random_seed=42)
        training_results = train_and_evaluate_model_from_frame(training_frame)

        summary = build_training_results_summary(training_results)

        self.assertIn("mae_watts", summary)
        self.assertIn("rmse_watts", summary)
        self.assertIn("r2", summary)
        self.assertIn("top_coefficients", summary)
        self.assertGreater(len(summary["top_coefficients"]), 0)

    def test_forecast_summary_contains_trend_indicators(self) -> None:
        training_frame = generate_simulated_kpis(num_days=7, random_seed=42)
        training_results = train_and_evaluate_model_from_frame(training_frame)
        forecast_frame = generate_simulated_kpis(
            start_timestamp="2026-01-12 00:00:00",
            num_days=7,
            random_seed=42,
        )
        predicted_forecast = predict_dataset_energy(
            forecast_frame,
            training_results["coefficients"],
        )

        summary = build_forecast_summary(predicted_forecast)

        self.assertIn("peak_timestamp", summary)
        self.assertIn("day_change_pct", summary)
        self.assertIn("top_sites", summary)
        self.assertEqual(len(summary["top_sites"]), 3)

    def test_prompt_mentions_simulated_data_and_both_sections(self) -> None:
        prompt = build_simulation_summary_prompt(
            training_summary={"mae_watts": 12.3, "r2": 0.95},
            forecast_summary={"peak_total_watts": 1234.5, "day_change_pct": -1.2},
        )

        self.assertIn("Paragraph 1", prompt)
        self.assertIn("Paragraph 2", prompt)
        self.assertIn("simulated", prompt.lower())


if __name__ == "__main__":
    unittest.main()
