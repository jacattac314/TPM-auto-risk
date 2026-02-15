"""Tests for the training pipeline orchestration.

Verifies the end-to-end flow: Silver layer loading, feature engineering,
target computation, Gold layer persistence, and model training/saving.
"""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch, call

import pandas as pd
import pytest

from src.models.training_pipeline import TrainingPipeline, _compute_target


@pytest.fixture
def silver_df(sample_jira_tickets):
    """Build a DataFrame that simulates Silver-layer Jira records."""
    return pd.DataFrame(sample_jira_tickets)


class TestComputeTarget:
    """Tests for the binary target variable derivation.

    The _compute_target function labels a ticket as delayed when:
    1. It was resolved after its due date, OR
    2. It was rolled over into multiple sprints, OR
    3. It has been in-progress longer than 2x the median for the project.
    """

    def test_sprint_rollover_marks_delayed(self, sample_jira_tickets):
        from src.features.feature_engineering import engineer_jira_features, compute_team_features

        df = engineer_jira_features(sample_jira_tickets)
        df = compute_team_features(df)
        df = _compute_target(df)

        # Tickets with sprint_rollover_count > 0 should be delayed
        rolled = df[df["sprint_rollover_count"] > 0]
        assert (rolled["is_delayed"] == 1).all()

    def test_no_delayed_when_all_on_track(self):
        from src.features.feature_engineering import engineer_jira_features, compute_team_features

        tickets = [
            {
                "issue_id": "1", "issue_key": "OK-1", "project_key": "OK",
                "summary": "Fine", "description": "On track ticket",
                "status": "Done", "status_category": "Done",
                "priority": "Medium", "issue_type": "Task",
                "assignee": "eng_1", "reporter": "pm_1",
                "created": "2024-01-01T00:00:00Z", "updated": "2024-01-10T00:00:00Z",
                "due_date": "2024-02-01T00:00:00Z",
                "resolution_date": "2024-01-15T00:00:00Z",
                "story_points": 3, "labels": [], "components": [],
                "sprint_history": [{"sprint_id": "1", "sprint_name": "Sprint 1",
                                     "sprint_state": "closed", "start_date": "2024-01-01",
                                     "end_date": "2024-01-14", "complete_date": "2024-01-14"}],
                "issue_links": [], "comment_count": 2,
                "ingested_at": "2024-01-20T00:00:00Z",
            },
        ]
        df = engineer_jira_features(tickets)
        df = compute_team_features(df)
        df = _compute_target(df)

        # Resolved before due date, single sprint, done => not delayed
        assert df["is_delayed"].iloc[0] == 0

    def test_overdue_resolution_marks_delayed(self):
        from src.features.feature_engineering import engineer_jira_features, compute_team_features

        tickets = [
            {
                "issue_id": "2", "issue_key": "LATE-1", "project_key": "LATE",
                "summary": "Late delivery", "description": "Resolved after due date",
                "status": "Done", "status_category": "Done",
                "priority": "High", "issue_type": "Story",
                "assignee": "eng_2", "reporter": "pm_1",
                "created": "2024-01-01T00:00:00Z", "updated": "2024-03-15T00:00:00Z",
                "due_date": "2024-02-01T00:00:00Z",
                "resolution_date": "2024-03-15T00:00:00Z",
                "story_points": 8, "labels": [], "components": [],
                "sprint_history": [{"sprint_id": "1", "sprint_name": "Sprint 1",
                                     "sprint_state": "closed", "start_date": "2024-01-01",
                                     "end_date": "2024-01-14", "complete_date": "2024-01-14"}],
                "issue_links": [], "comment_count": 5,
                "ingested_at": "2024-03-16T00:00:00Z",
            },
        ]
        df = engineer_jira_features(tickets)
        df = compute_team_features(df)
        df = _compute_target(df)

        assert df["is_delayed"].iloc[0] == 1

    def test_target_positive_rate_logged(self, sample_jira_tickets, caplog):
        from src.features.feature_engineering import engineer_jira_features, compute_team_features

        df = engineer_jira_features(sample_jira_tickets)
        df = compute_team_features(df)
        import logging
        with caplog.at_level(logging.INFO):
            _compute_target(df)
        assert any("Target computed" in msg for msg in caplog.messages)


class TestTrainingPipeline:
    """Tests for the TrainingPipeline.run() orchestration."""

    @patch("src.models.training_pipeline.DataLakeWriter")
    def test_raises_on_empty_silver(self, mock_datalake_cls, settings):
        """Pipeline should raise ValueError when Silver layer has no data."""
        mock_datalake = MagicMock()
        mock_datalake.read_silver.return_value = pd.DataFrame()
        mock_datalake_cls.return_value = mock_datalake

        pipeline = TrainingPipeline(settings)

        with pytest.raises(ValueError, match="No training data"):
            pipeline.run(["EMPTY"])

    @patch("src.models.training_pipeline.DataLakeWriter")
    def test_full_pipeline_runs(self, mock_datalake_cls, settings, sample_jira_tickets):
        """Pipeline loads Silver data, engineers features, trains, and saves to S3."""
        mock_datalake = MagicMock()
        mock_datalake.read_silver.return_value = pd.DataFrame(sample_jira_tickets)
        mock_datalake_cls.return_value = mock_datalake

        pipeline = TrainingPipeline(settings)

        with patch.object(pipeline._model, "save_to_s3", return_value="s3://bucket/model.tar.gz"):
            result = pipeline.run(["PROJ"])

        assert result["status"] == "success"
        assert result["training_samples"] > 0
        assert "model_version" in result
        assert "metrics" in result

        # Should have written to Gold layer
        mock_datalake.write_gold.assert_called()

    @patch("src.models.training_pipeline.DataLakeWriter")
    def test_multi_project_training(self, mock_datalake_cls, settings, sample_jira_tickets):
        """Pipeline concatenates data from multiple projects."""
        mock_datalake = MagicMock()

        # Return data for first project, empty for second
        df = pd.DataFrame(sample_jira_tickets)
        mock_datalake.read_silver.side_effect = [df, pd.DataFrame()]
        mock_datalake_cls.return_value = mock_datalake

        pipeline = TrainingPipeline(settings)

        with patch.object(pipeline._model, "save_to_s3", return_value="s3://bucket/model"):
            result = pipeline.run(["PROJ", "EMPTY"])

        assert result["status"] == "success"
        # Only PROJ had data
        assert result["training_samples"] == len(sample_jira_tickets)

    @patch("src.models.training_pipeline.DataLakeWriter")
    def test_sentiment_data_merged(self, mock_datalake_cls, settings, sample_jira_tickets):
        """Pipeline merges sentiment timeseries when provided."""
        mock_datalake = MagicMock()
        mock_datalake.read_silver.return_value = pd.DataFrame(sample_jira_tickets)
        mock_datalake_cls.return_value = mock_datalake

        sentiment = {
            "PROJ": [
                {"window_start": "2024-01-15T00:00:00Z", "avg_sentiment": 0.3,
                 "message_count": 20, "std_dev": 0.2},
                {"window_start": "2024-01-16T00:00:00Z", "avg_sentiment": -0.1,
                 "message_count": 18, "std_dev": 0.3},
            ],
        }

        pipeline = TrainingPipeline(settings)

        with patch.object(pipeline._model, "save_to_s3", return_value="s3://bucket/model"):
            result = pipeline.run(["PROJ"], sentiment_data=sentiment)

        assert result["status"] == "success"

    @patch("src.models.training_pipeline.DataLakeWriter")
    def test_model_saved_as_latest(self, mock_datalake_cls, settings, sample_jira_tickets):
        """Pipeline saves model with both timestamped version and 'latest' alias."""
        mock_datalake = MagicMock()
        mock_datalake.read_silver.return_value = pd.DataFrame(sample_jira_tickets)
        mock_datalake_cls.return_value = mock_datalake

        pipeline = TrainingPipeline(settings)

        with patch.object(pipeline._model, "save_to_s3", return_value="s3://bucket/model") as mock_save:
            pipeline.run(["PROJ"])

        # Called twice: once with timestamped version, once with "latest"
        assert mock_save.call_count == 2
        versions = [c.kwargs.get("model_version") or c.args[1] for c in mock_save.call_args_list]
        assert "latest" in versions
