"""Tests for the simple regression training and forecast flow."""

from __future__ import annotations

import unittest

from src.ml import predict_dataset_energy, train_and_evaluate_model_from_frame
from src.sim import generate_simulated_kpis


class LinearRegressionPipelineTests(unittest.TestCase):
    """Check that the model can train on simulated data and predict future rows."""

    def test_training_from_dataframe_returns_expected_outputs(self) -> None:
        simulated_kpis = generate_simulated_kpis(num_days=7, random_seed=42)

        results = train_and_evaluate_model_from_frame(simulated_kpis)

        self.assertIn("metrics", results)
        self.assertIn("coefficients", results)
        self.assertIn("predictions", results)
        self.assertFalse(results["coefficients"].empty)
        self.assertFalse(results["predictions"].empty)
        self.assertEqual(results["coefficients"].iloc[0]["feature"], "intercept")

    def test_prediction_uses_trained_coefficients_for_future_rows(self) -> None:
        training_frame = generate_simulated_kpis(num_days=7, random_seed=42)
        forecast_frame = generate_simulated_kpis(
            start_timestamp="2026-01-12 00:00:00",
            num_days=7,
            random_seed=42,
        )

        training_results = train_and_evaluate_model_from_frame(training_frame)
        predicted_frame = predict_dataset_energy(
            forecast_frame,
            training_results["coefficients"],
        )

        self.assertEqual(len(predicted_frame), len(forecast_frame))
        self.assertIn("predicted_energy_watts", predicted_frame.columns)
        self.assertFalse(predicted_frame["predicted_energy_watts"].isna().any())
        self.assertTrue((predicted_frame["predicted_energy_watts"] > 0).all())


if __name__ == "__main__":
    unittest.main()
