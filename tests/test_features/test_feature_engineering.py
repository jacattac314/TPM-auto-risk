"""Tests for feature engineering module."""

from __future__ import annotations

import pytest
import pandas as pd

from src.features.feature_engineering import (
    compute_team_features,
    engineer_jira_features,
    get_model_feature_columns,
    merge_sentiment_features,
    prepare_model_input,
)


class TestEngineerJiraFeatures:
    def test_empty_input(self):
        result = engineer_jira_features([])
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_basic_feature_creation(self, sample_jira_tickets):
        df = engineer_jira_features(sample_jira_tickets)

        assert not df.empty
        assert len(df) == len(sample_jira_tickets)

        # Verify key feature columns exist
        expected_features = [
            "story_points", "has_dependency", "dependency_count",
            "priority_numeric", "priority_interaction",
            "days_in_status", "days_since_created",
            "description_ambiguity", "sprint_rollover_count",
            "ticket_churn", "issue_type_risk", "status_category_encoded",
        ]
        for col in expected_features:
            assert col in df.columns, f"Missing feature: {col}"

    def test_has_dependency_binary(self, sample_jira_tickets):
        df = engineer_jira_features(sample_jira_tickets)
        assert set(df["has_dependency"].unique()).issubset({0, 1})

    def test_priority_mapping(self, sample_jira_tickets):
        df = engineer_jira_features(sample_jira_tickets)
        assert df["priority_numeric"].between(1, 5).all()

    def test_description_ambiguity_scoring(self, sample_jira_tickets):
        df = engineer_jira_features(sample_jira_tickets)
        assert df["description_ambiguity"].between(0, 1).all()

    def test_sprint_rollover_count(self, sample_jira_tickets):
        df = engineer_jira_features(sample_jira_tickets)
        assert (df["sprint_rollover_count"] >= 0).all()
        # Tickets with 2 sprints should have rollover count of 1
        has_rollover = df["sprint_rollover_count"] > 0
        assert has_rollover.any()


class TestComputeTeamFeatures:
    def test_complexity_per_person(self, sample_jira_tickets):
        df = engineer_jira_features(sample_jira_tickets)
        df = compute_team_features(df)

        assert "complexity_per_person" in df.columns
        assert (df["complexity_per_person"] >= 0).all()

    def test_team_wip(self, sample_jira_tickets):
        df = engineer_jira_features(sample_jira_tickets)
        df = compute_team_features(df)

        assert "team_wip" in df.columns
        assert (df["team_wip"] >= 0).all()

    def test_assignee_load(self, sample_jira_tickets):
        df = engineer_jira_features(sample_jira_tickets)
        df = compute_team_features(df)

        assert "assignee_load" in df.columns
        assert (df["assignee_load"] >= 0).all()

    def test_empty_dataframe(self):
        empty_df = pd.DataFrame()
        result = compute_team_features(empty_df)
        assert result.empty


class TestMergeSentimentFeatures:
    def test_adds_sentiment_columns(self, sample_jira_tickets, sample_sentiment_timeseries):
        df = engineer_jira_features(sample_jira_tickets)
        df = merge_sentiment_features(df, sample_sentiment_timeseries, "PROJ")

        assert "avg_sentiment" in df.columns
        assert "sentiment_velocity" in df.columns
        assert "sentiment_anomaly" in df.columns

    def test_empty_sentiment_defaults_to_zero(self, sample_jira_tickets):
        df = engineer_jira_features(sample_jira_tickets)
        df = merge_sentiment_features(df, [], "PROJ")

        assert (df["avg_sentiment"] == 0.0).all()
        assert (df["sentiment_velocity"] == 0.0).all()


class TestPrepareModelInput:
    def test_returns_correct_columns(self, sample_jira_tickets):
        df = engineer_jira_features(sample_jira_tickets)
        df = compute_team_features(df)
        df = merge_sentiment_features(df, [], "PROJ")
        df["comment_count"] = 0

        model_input = prepare_model_input(df)
        expected_cols = get_model_feature_columns()

        assert list(model_input.columns) == expected_cols

    def test_all_numeric(self, sample_jira_tickets):
        df = engineer_jira_features(sample_jira_tickets)
        df = compute_team_features(df)
        df = merge_sentiment_features(df, [], "PROJ")
        df["comment_count"] = 0

        model_input = prepare_model_input(df)
        for col in model_input.columns:
            assert pd.api.types.is_numeric_dtype(model_input[col]), f"{col} is not numeric"

    def test_no_missing_values(self, sample_jira_tickets):
        df = engineer_jira_features(sample_jira_tickets)
        df = compute_team_features(df)
        df = merge_sentiment_features(df, [], "PROJ")
        df["comment_count"] = 0

        model_input = prepare_model_input(df)
        assert model_input.isna().sum().sum() == 0
