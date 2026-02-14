"""Tests for prompt template formatting."""

from __future__ import annotations

import pytest

from src.reporting.prompt_templates import (
    format_jira_risk_summary,
    format_risk_scores_summary,
    format_sentiment_summary,
)


class TestFormatJiraRiskSummary:
    def test_empty_tickets(self):
        result = format_jira_risk_summary([])
        assert "No high-risk tickets" in result

    def test_formats_tickets(self):
        tickets = [
            {
                "issue_key": "PROJ-101",
                "summary": "Fix critical auth bug",
                "status": "In Progress",
                "risk_score": 0.85,
                "story_points": 8,
                "days_in_status": 5,
                "dependency_count": 2,
            }
        ]
        result = format_jira_risk_summary(tickets)
        assert "PROJ-101" in result
        assert "Fix critical auth bug" in result
        assert "85%" in result

    def test_limits_to_15_tickets(self):
        tickets = [
            {"issue_key": f"PROJ-{i}", "summary": f"Ticket {i}",
             "status": "Open", "risk_score": 0.5, "story_points": 3,
             "days_in_status": 1, "dependency_count": 0}
            for i in range(20)
        ]
        result = format_jira_risk_summary(tickets)
        lines = [l for l in result.strip().split("\n") if l.strip()]
        assert len(lines) <= 15


class TestFormatRiskScoresSummary:
    def test_empty_predictions(self):
        result = format_risk_scores_summary([])
        assert "No risk predictions" in result

    def test_formats_distribution(self):
        predictions = [
            {"risk_level": "High", "risk_score": 0.8},
            {"risk_level": "High", "risk_score": 0.75},
            {"risk_level": "Medium", "risk_score": 0.5},
            {"risk_level": "Low", "risk_score": 0.2},
            {"risk_level": "Low", "risk_score": 0.1},
        ]
        result = format_risk_scores_summary(predictions)
        assert "2 High" in result
        assert "1 Medium" in result
        assert "2 Low" in result
        assert "Total Tickets Scored: 5" in result


class TestFormatSentimentSummary:
    def test_empty_data(self):
        result = format_sentiment_summary([])
        assert "No Slack sentiment data" in result

    def test_positive_sentiment(self):
        data = [
            {"avg_sentiment": 0.5, "message_count": 10},
            {"avg_sentiment": 0.3, "message_count": 8},
        ]
        result = format_sentiment_summary(data)
        assert "positive" in result

    def test_negative_sentiment(self):
        data = [
            {"avg_sentiment": -0.5, "message_count": 10},
            {"avg_sentiment": -0.3, "message_count": 8},
        ]
        result = format_sentiment_summary(data)
        assert "negative" in result

    def test_anomalies_included(self):
        data = [{"avg_sentiment": 0.1, "message_count": 10}]
        anomalies = [
            {"window_start": "2024-01-18", "avg_sentiment": -0.8, "z_score": -3.2}
        ]
        result = format_sentiment_summary(data, anomalies)
        assert "anomal" in result.lower()
