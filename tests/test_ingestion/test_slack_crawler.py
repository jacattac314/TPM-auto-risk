"""Tests for Slack ingestion and sentiment analysis."""

from __future__ import annotations

import pytest

from src.ingestion.slack_crawler import (
    SentimentAnalyzer,
    compute_sentiment_timeseries,
    detect_sentiment_anomalies,
)


class TestSentimentAnalyzer:
    def setup_method(self):
        self.analyzer = SentimentAnalyzer()

    def test_positive_sentiment(self):
        score = self.analyzer.score("Great work! The feature shipped and everything looks good.")
        assert score > 0

    def test_negative_sentiment(self):
        score = self.analyzer.score("This is broken. We have a major blocker and an outage.")
        assert score < 0

    def test_neutral_sentiment(self):
        score = self.analyzer.score("Updated the configuration file.")
        assert -0.5 < score < 0.5

    def test_engineering_lexicon_blocked(self):
        score = self.analyzer.score("I'm blocked on the API integration.")
        assert score < 0

    def test_engineering_lexicon_shipped(self):
        score = self.analyzer.score("We shipped the new dashboard.")
        assert score > 0

    def test_empty_string(self):
        score = self.analyzer.score("")
        assert score == 0.0


class TestComputeSentimentTimeseries:
    def test_empty_messages(self):
        result = compute_sentiment_timeseries([])
        assert result == []

    def test_single_message(self):
        messages = [{"message_ts": "1700000000.000000", "sentiment_score": 0.5}]
        result = compute_sentiment_timeseries(messages)
        assert len(result) == 1
        assert result[0]["avg_sentiment"] == 0.5
        assert result[0]["message_count"] == 1

    def test_multiple_messages_same_day(self):
        messages = [
            {"message_ts": "1700000000.000000", "sentiment_score": 0.5},
            {"message_ts": "1700001000.000000", "sentiment_score": -0.3},
            {"message_ts": "1700002000.000000", "sentiment_score": 0.2},
        ]
        result = compute_sentiment_timeseries(messages, window_hours=24)
        assert len(result) == 1
        assert result[0]["message_count"] == 3

    def test_invalid_timestamp_skipped(self):
        messages = [
            {"message_ts": "invalid", "sentiment_score": 0.5},
            {"message_ts": "1700000000.000000", "sentiment_score": 0.3},
        ]
        result = compute_sentiment_timeseries(messages)
        assert len(result) == 1


class TestDetectSentimentAnomalies:
    def test_no_anomalies_stable_sentiment(self):
        timeseries = [
            {"window_start": f"2024-01-{i+1:02d}", "avg_sentiment": 0.3, "message_count": 10, "std_dev": 0.1}
            for i in range(7)
        ]
        anomalies = detect_sentiment_anomalies(timeseries)
        assert len(anomalies) == 0

    def test_detects_significant_drop(self):
        timeseries = [
            {"window_start": "2024-01-01", "avg_sentiment": 0.5, "message_count": 10, "std_dev": 0.1},
            {"window_start": "2024-01-02", "avg_sentiment": 0.4, "message_count": 10, "std_dev": 0.1},
            {"window_start": "2024-01-03", "avg_sentiment": 0.3, "message_count": 10, "std_dev": 0.1},
            {"window_start": "2024-01-04", "avg_sentiment": -0.8, "message_count": 10, "std_dev": 0.1},
            {"window_start": "2024-01-05", "avg_sentiment": 0.4, "message_count": 10, "std_dev": 0.1},
        ]
        anomalies = detect_sentiment_anomalies(timeseries, threshold_std_devs=1.5)
        assert len(anomalies) >= 1
        assert anomalies[0]["anomaly_type"] == "negative_sentiment_drop"

    def test_too_few_datapoints(self):
        timeseries = [
            {"window_start": "2024-01-01", "avg_sentiment": 0.5, "message_count": 10, "std_dev": 0.1},
        ]
        assert detect_sentiment_anomalies(timeseries) == []
