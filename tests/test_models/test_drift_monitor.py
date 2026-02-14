"""Tests for drift monitoring and data quality checking."""

from __future__ import annotations

import pandas as pd
import pytest

from src.models.drift_monitor import DataQualityChecker


class TestDataQualityChecker:
    def setup_method(self):
        self.checker = DataQualityChecker()

    def test_valid_data_passes(self, sample_jira_tickets):
        df = pd.DataFrame(sample_jira_tickets)
        result = self.checker.validate(df)
        # The sample data should pass most checks
        assert isinstance(result, dict)
        assert "passed" in result
        assert "violations" in result
        assert "metrics" in result
        assert result["row_count"] == len(sample_jira_tickets)

    def test_missing_column_flagged(self):
        df = pd.DataFrame({"random_col": [1, 2, 3]})
        result = self.checker.validate(df)
        assert not result["passed"]
        assert result["violations_count"] > 0

        # At least the required columns should be flagged
        missing_violations = [
            v for v in result["violations"] if v["rule"] == "column_exists"
        ]
        assert len(missing_violations) > 0

    def test_negative_story_points_flagged(self):
        checker = DataQualityChecker(constraints={
            "story_points": {"type": "numeric", "min": 0, "max": 100, "completeness": 0.5},
        })
        df = pd.DataFrame({"story_points": [-5, 3, 8, None, 13]})
        result = checker.validate(df)
        assert not result["passed"]

        min_violations = [v for v in result["violations"] if v["rule"] == "min_value"]
        assert len(min_violations) == 1

    def test_invalid_status_values_flagged(self):
        checker = DataQualityChecker(constraints={
            "status": {
                "type": "categorical",
                "allowed_values": ["Open", "In Progress", "Done"],
                "completeness": 0.95,
            },
        })
        df = pd.DataFrame({"status": ["Open", "Done", "InvalidStatus", "In Progress"]})
        result = checker.validate(df)
        assert not result["passed"]

        cat_violations = [v for v in result["violations"] if v["rule"] == "allowed_values"]
        assert len(cat_violations) == 1
        assert "InvalidStatus" in cat_violations[0]["invalid_values"]

    def test_uniqueness_violation(self):
        checker = DataQualityChecker(constraints={
            "issue_key": {"type": "string", "uniqueness": 1.0, "completeness": 1.0},
        })
        df = pd.DataFrame({"issue_key": ["PROJ-1", "PROJ-1", "PROJ-2"]})
        result = checker.validate(df)
        assert not result["passed"]

        unique_violations = [v for v in result["violations"] if v["rule"] == "uniqueness"]
        assert len(unique_violations) == 1

    def test_completeness_violation(self):
        checker = DataQualityChecker(constraints={
            "created": {"type": "datetime", "completeness": 1.0},
        })
        df = pd.DataFrame({"created": ["2024-01-01", None, "2024-01-03"]})
        result = checker.validate(df)
        assert not result["passed"]

        comp_violations = [v for v in result["violations"] if v["rule"] == "completeness"]
        assert len(comp_violations) == 1

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = self.checker.validate(df)
        # Empty DataFrame should still report missing columns
        assert result["row_count"] == 0
