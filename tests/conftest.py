"""Shared test fixtures for the Auto-TPM Risk Radar test suite."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from src.config.settings import AppSettings


@pytest.fixture
def settings() -> AppSettings:
    """Return test settings with safe defaults."""
    return AppSettings(
        debug=True,
        log_level="DEBUG",
    )


@pytest.fixture
def sample_jira_tickets() -> list[dict[str, Any]]:
    """Return a list of sample Jira ticket records for testing."""
    base_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    tickets = []
    for i in range(20):
        is_delayed = i % 3 == 0
        tickets.append({
            "issue_id": str(1000 + i),
            "issue_key": f"PROJ-{100 + i}",
            "project_key": "PROJ",
            "summary": f"Implement feature {chr(65 + i % 26)}",
            "description": f"Description for feature {i}. " * (3 if i % 4 != 0 else 1),
            "status": ["To Do", "In Progress", "In Review", "Done"][i % 4],
            "status_category": ["To Do", "In Progress", "In Progress", "Done"][i % 4],
            "priority": ["High", "Medium", "Low", "Highest", "Medium"][i % 5],
            "issue_type": ["Story", "Bug", "Task", "Spike"][i % 4],
            "assignee": f"engineer_{i % 5}",
            "reporter": f"pm_{i % 2}",
            "created": (base_date.replace(day=1 + i % 28)).isoformat(),
            "updated": (base_date.replace(month=1 + i % 3, day=1 + i % 28)).isoformat(),
            "due_date": (base_date.replace(month=2, day=1 + i % 28)).isoformat() if i % 2 == 0 else None,
            "resolution_date": (
                base_date.replace(month=3 if is_delayed else 2, day=1 + i % 28).isoformat()
                if i % 4 == 3
                else None
            ),
            "story_points": [1, 2, 3, 5, 8, 13][i % 6],
            "labels": ["backend"] if i % 3 == 0 else ["frontend"],
            "components": ["api"] if i % 2 == 0 else ["ui"],
            "sprint_history": [
                {"sprint_id": "1", "sprint_name": "Sprint 1", "sprint_state": "closed",
                 "start_date": "2024-01-01", "end_date": "2024-01-14", "complete_date": "2024-01-14"},
            ] + (
                [{"sprint_id": "2", "sprint_name": "Sprint 2", "sprint_state": "active",
                  "start_date": "2024-01-15", "end_date": "2024-01-28", "complete_date": None}]
                if is_delayed else []
            ),
            "issue_links": [
                {"type": "Blocks", "direction": "outward", "linked_issue": f"PROJ-{200 + i}"}
            ] if i % 4 == 0 else [],
            "comment_count": i * 2,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        })
    return tickets


@pytest.fixture
def sample_slack_messages() -> list[dict[str, Any]]:
    """Return sample Slack message records for testing."""
    messages = []
    sentiments = [0.5, 0.3, -0.2, -0.8, 0.1, -0.5, 0.4, -0.9, 0.2, -0.1]
    for i in range(10):
        ts = f"170000{i:04d}.000000"
        messages.append({
            "channel_id": "C123456",
            "message_ts": ts,
            "user_id": f"U{i:06d}",
            "text": f"Test message {i}",
            "sentiment_score": sentiments[i],
            "thread_ts": None,
            "reply_count": 0,
            "reaction_count": i,
            "datetime_utc": datetime(2024, 1, 15, 10 + i, 0, tzinfo=timezone.utc).isoformat(),
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        })
    return messages


@pytest.fixture
def sample_sentiment_timeseries() -> list[dict[str, Any]]:
    """Return sample sentiment timeseries data."""
    return [
        {"window_start": "2024-01-15T00:00:00Z", "avg_sentiment": 0.3, "message_count": 20, "std_dev": 0.2},
        {"window_start": "2024-01-16T00:00:00Z", "avg_sentiment": 0.2, "message_count": 18, "std_dev": 0.3},
        {"window_start": "2024-01-17T00:00:00Z", "avg_sentiment": -0.1, "message_count": 25, "std_dev": 0.4},
        {"window_start": "2024-01-18T00:00:00Z", "avg_sentiment": -0.5, "message_count": 30, "std_dev": 0.3},
        {"window_start": "2024-01-19T00:00:00Z", "avg_sentiment": -0.3, "message_count": 15, "std_dev": 0.2},
    ]
