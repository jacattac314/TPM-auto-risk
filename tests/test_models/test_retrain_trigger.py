"""Tests for the model retraining trigger (SNS-invoked Lambda)."""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest

from scripts.train_model import _extract_project_keys_from_sns, lambda_handler


class TestExtractProjectKeysFromSns:
    def test_drift_alert_extracts_keys(self):
        event = {
            "Records": [{
                "Sns": {
                    "Message": json.dumps({
                        "alert_type": "MODEL_DRIFT_DETECTED",
                        "overall_drift_pct": 25.0,
                        "drifted_features": ["story_points", "days_in_status"],
                    }),
                },
            }],
        }

        with patch.dict("os.environ", {"TRAINING_PROJECT_KEYS": "PROJ,ALPHA"}):
            keys = _extract_project_keys_from_sns(event)
            assert keys == ["PROJ", "ALPHA"]

    def test_non_drift_alert_returns_empty(self):
        event = {
            "Records": [{
                "Sns": {
                    "Message": json.dumps({
                        "alert_type": "DATA_QUALITY_VIOLATION",
                        "project_key": "PROJ",
                    }),
                },
            }],
        }
        keys = _extract_project_keys_from_sns(event)
        assert keys == []

    def test_malformed_sns_message_returns_empty(self):
        event = {
            "Records": [{"Sns": {"Message": "not valid json"}}],
        }
        keys = _extract_project_keys_from_sns(event)
        assert keys == []

    def test_empty_records_returns_empty(self):
        keys = _extract_project_keys_from_sns({"Records": []})
        assert keys == []


class TestLambdaHandler:
    @patch("scripts.train_model.TrainingPipeline")
    @patch("scripts.train_model.get_settings")
    def test_direct_invocation(self, mock_settings, mock_pipeline_cls):
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {"status": "trained", "recall": 0.85}
        mock_pipeline_cls.return_value = mock_pipeline

        result = lambda_handler({"project_keys": ["PROJ"]}, None)

        assert result["statusCode"] == 200
        mock_pipeline.run.assert_called_once_with(["PROJ"])

    @patch("scripts.train_model.TrainingPipeline")
    @patch("scripts.train_model.get_settings")
    def test_sns_triggered_invocation(self, mock_settings, mock_pipeline_cls):
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {"status": "retrained"}
        mock_pipeline_cls.return_value = mock_pipeline

        event = {
            "Records": [{
                "Sns": {
                    "Message": json.dumps({
                        "alert_type": "MODEL_DRIFT_DETECTED",
                        "overall_drift_pct": 30.0,
                    }),
                },
            }],
        }

        with patch.dict("os.environ", {"TRAINING_PROJECT_KEYS": "PROJ"}):
            result = lambda_handler(event, None)

        assert result["statusCode"] == 200
        mock_pipeline.run.assert_called_once_with(["PROJ"])

    def test_no_project_keys_returns_400(self):
        result = lambda_handler({}, None)
        assert result["statusCode"] == 400
