"""Tests for Jira ingestion module."""

from __future__ import annotations

import pytest

from src.ingestion.jira_crawler import (
    _extract_ticket,
    _parse_sprint_history,
    _parse_sprint_string,
)


class TestParseSprintString:
    def test_parse_valid_sprint_string(self):
        sprint_str = (
            "com.atlassian.greenhopper.service.sprint.Sprint@1234"
            "[id=42,name=Sprint 7,state=ACTIVE,"
            "startDate=2024-01-15,endDate=2024-01-28,completeDate=<null>]"
        )
        result = _parse_sprint_string(sprint_str)
        assert result is not None
        assert result["sprint_id"] == "42"
        assert result["sprint_name"] == "Sprint 7"
        assert result["sprint_state"] == "ACTIVE"
        assert result["start_date"] == "2024-01-15"
        assert result["complete_date"] is None

    def test_parse_invalid_string_returns_none(self):
        assert _parse_sprint_string("not a sprint string") is None

    def test_parse_empty_string_returns_none(self):
        assert _parse_sprint_string("") is None


class TestParseSprintHistory:
    def test_none_input(self):
        assert _parse_sprint_history(None) == []

    def test_empty_list(self):
        assert _parse_sprint_history([]) == []

    def test_dict_list_input(self):
        sprints = [
            {"id": "1", "name": "Sprint 1", "state": "closed",
             "startDate": "2024-01-01", "endDate": "2024-01-14", "completeDate": "2024-01-14"},
            {"id": "2", "name": "Sprint 2", "state": "active",
             "startDate": "2024-01-15", "endDate": "2024-01-28", "completeDate": None},
        ]
        result = _parse_sprint_history(sprints)
        assert len(result) == 2
        assert result[0]["sprint_name"] == "Sprint 1"
        assert result[1]["sprint_state"] == "active"


class TestExtractTicket:
    def test_basic_extraction(self):
        issue = {
            "id": "10001",
            "key": "PROJ-101",
            "fields": {
                "project": {"key": "PROJ"},
                "summary": "Fix login bug",
                "description": "Users cannot log in when...",
                "status": {"name": "In Progress", "statusCategory": {"name": "In Progress"}},
                "priority": {"name": "High"},
                "issuetype": {"name": "Bug"},
                "assignee": {"displayName": "Alice"},
                "reporter": {"displayName": "Bob"},
                "created": "2024-01-10T10:00:00.000+0000",
                "updated": "2024-01-15T14:30:00.000+0000",
                "duedate": "2024-01-20",
                "resolutiondate": None,
                "customfield_10016": 5,
                "customfield_10020": [],
                "labels": ["urgent"],
                "components": [{"name": "auth"}],
                "issuelinks": [
                    {"type": {"name": "Blocks"}, "outwardIssue": {"key": "PROJ-102"}}
                ],
                "comment": {"total": 3},
            },
        }

        ticket = _extract_ticket(issue, "customfield_10016", "customfield_10020")

        assert ticket["issue_key"] == "PROJ-101"
        assert ticket["project_key"] == "PROJ"
        assert ticket["summary"] == "Fix login bug"
        assert ticket["status"] == "In Progress"
        assert ticket["priority"] == "High"
        assert ticket["story_points"] == 5
        assert ticket["assignee"] == "Alice"
        assert ticket["comment_count"] == 3
        assert len(ticket["issue_links"]) == 1
        assert ticket["issue_links"][0]["linked_issue"] == "PROJ-102"

    def test_null_assignee(self):
        issue = {
            "id": "10002",
            "key": "PROJ-102",
            "fields": {
                "project": {"key": "PROJ"},
                "summary": "Unassigned ticket",
                "description": None,
                "status": {"name": "Open", "statusCategory": {"name": "To Do"}},
                "priority": {"name": "Low"},
                "issuetype": {"name": "Task"},
                "assignee": None,
                "reporter": None,
                "created": "2024-01-01",
                "updated": "2024-01-01",
                "duedate": None,
                "resolutiondate": None,
                "customfield_10016": None,
                "customfield_10020": None,
                "labels": [],
                "components": [],
                "issuelinks": [],
                "comment": {"total": 0},
            },
        }
        ticket = _extract_ticket(issue, "customfield_10016", "customfield_10020")
        assert ticket["assignee"] is None
        assert ticket["story_points"] is None
