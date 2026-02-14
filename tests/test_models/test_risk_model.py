"""Tests for the XGBoost risk prediction model."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config.settings import ModelSettings
from src.features.feature_engineering import (
    compute_team_features,
    engineer_jira_features,
    get_model_feature_columns,
    merge_sentiment_features,
)
from src.models.risk_model import RiskModel


@pytest.fixture
def trained_model(sample_jira_tickets) -> tuple[RiskModel, pd.DataFrame]:
    """Return a trained model and the training DataFrame."""
    df = engineer_jira_features(sample_jira_tickets)
    df = compute_team_features(df)
    df = merge_sentiment_features(df, [], "PROJ")

    # Add target variable
    df["is_delayed"] = (df["sprint_rollover_count"] > 0).astype(int)

    settings = ModelSettings(
        xgboost_n_estimators=10,  # Small for fast tests
        xgboost_max_depth=3,
    )
    model = RiskModel(settings)
    model.train(df, target_col="is_delayed", date_col="created")

    return model, df


class TestRiskModel:
    def test_train_returns_metrics(self, trained_model):
        model, df = trained_model
        assert model.is_trained

    def test_predict_adds_risk_columns(self, trained_model):
        model, df = trained_model
        predictions = model.predict(df)

        assert "risk_score" in predictions.columns
        assert "risk_level" in predictions.columns
        assert predictions["risk_score"].between(0, 1).all()
        assert set(predictions["risk_level"].dropna().unique()).issubset({"Low", "Medium", "High"})

    def test_predict_single(self, trained_model):
        model, df = trained_model
        features = df.iloc[0].to_dict()

        result = model.predict_single(features)

        assert "risk_score" in result
        assert "risk_level" in result
        assert "top_risk_factors" in result
        assert 0 <= result["risk_score"] <= 1
        assert result["risk_level"] in ("Low", "Medium", "High")

    def test_feature_importance(self, trained_model):
        model, _ = trained_model
        importance = model.get_feature_importance()

        assert isinstance(importance, dict)
        assert len(importance) > 0

    def test_save_and_load(self, trained_model):
        model, df = trained_model

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)
            assert (Path(tmpdir) / "model.json").exists()
            assert (Path(tmpdir) / "metadata.json").exists()

            loaded_model = RiskModel()
            loaded_model.load(tmpdir)
            assert loaded_model.is_trained

            # Predictions should be identical
            original_preds = model.predict(df)["risk_score"].values
            loaded_preds = loaded_model.predict(df)["risk_score"].values
            np.testing.assert_array_almost_equal(original_preds, loaded_preds)

    def test_untrained_model_raises(self):
        model = RiskModel()
        df = pd.DataFrame({"x": [1, 2, 3]})

        with pytest.raises(RuntimeError, match="trained"):
            model.predict(df)

    def test_untrained_predict_single_raises(self):
        model = RiskModel()
        with pytest.raises(RuntimeError, match="trained"):
            model.predict_single({"story_points": 5})


class TestModelThresholds:
    def test_risk_level_boundaries(self, trained_model):
        model, df = trained_model
        settings = model._settings

        predictions = model.predict(df)
        for _, row in predictions.iterrows():
            score = row["risk_score"]
            level = row["risk_level"]
            if score >= settings.risk_threshold_high:
                assert level == "High"
            elif score >= settings.risk_threshold_medium:
                assert level == "Medium"
            else:
                assert level == "Low"
