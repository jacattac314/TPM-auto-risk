"""Tests for the GenAI report generator."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.config.settings import AppSettings
from src.reporting.report_generator import BedrockClient, ReportGenerator


@pytest.fixture
def mock_bedrock():
    with patch("src.reporting.report_generator.boto3") as mock_boto3:
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        yield mock_client


@pytest.fixture
def report_gen(settings, mock_bedrock):
    return ReportGenerator(settings)


def _make_bedrock_response(text: str) -> dict:
    """Build a mock Bedrock invoke_model response."""
    body = MagicMock()
    body.read.return_value = json.dumps({
        "content": [{"text": text}],
    }).encode()
    return {"body": body}


class TestBedrockClient:
    def test_invoke_sonnet(self, settings, mock_bedrock):
        mock_bedrock.invoke_model.return_value = _make_bedrock_response("Hello world")

        client = BedrockClient(settings)
        result = client.invoke("test prompt", model_tier="sonnet")

        assert result == "Hello world"
        call_kwargs = mock_bedrock.invoke_model.call_args[1]
        assert settings.aws.bedrock_model_id in call_kwargs["modelId"]

    def test_invoke_haiku(self, settings, mock_bedrock):
        mock_bedrock.invoke_model.return_value = _make_bedrock_response("Classification result")

        client = BedrockClient(settings)
        result = client.invoke("test prompt", model_tier="haiku")

        assert result == "Classification result"
        call_kwargs = mock_bedrock.invoke_model.call_args[1]
        assert settings.aws.bedrock_haiku_model_id in call_kwargs["modelId"]

    def test_invoke_empty_response(self, settings, mock_bedrock):
        body = MagicMock()
        body.read.return_value = json.dumps({"content": []}).encode()
        mock_bedrock.invoke_model.return_value = {"body": body}

        client = BedrockClient(settings)
        result = client.invoke("test")
        assert result == ""


class TestReportGenerator:
    def test_generate_executive_report(self, report_gen, mock_bedrock):
        report_json = json.dumps({
            "executive_summary": "All is well.",
            "key_accomplishments": ["Shipped feature A"],
            "critical_risks": [],
            "blockers": [],
            "sentiment_outlook": "Positive",
            "mitigation_plan": [],
            "next_week_focus": ["Deploy feature B"],
        })
        mock_bedrock.invoke_model.return_value = _make_bedrock_response(report_json)

        report = report_gen.generate_executive_report(
            project_name="TestProject",
            reporting_period="2024-01-15 to 2024-01-22",
            predictions=[
                {"risk_level": "High", "risk_score": 0.8, "issue_key": "PROJ-1"},
                {"risk_level": "Low", "risk_score": 0.2, "issue_key": "PROJ-2"},
            ],
        )

        assert "executive_summary" in report
        assert report["_metadata"]["project_name"] == "TestProject"
        assert report["_metadata"]["total_tickets"] == 2

    def test_generate_report_with_anomalies(self, report_gen, mock_bedrock):
        # First call: executive report; second call: anomaly analysis
        report_json = json.dumps({
            "executive_summary": "Risks detected.",
            "critical_risks": [],
        })
        analysis_json = json.dumps({
            "hypothesis": "Team frustration from blocked PR.",
            "recommended_action": "Escalate to engineering manager.",
            "severity": "high",
        })

        mock_bedrock.invoke_model.side_effect = [
            _make_bedrock_response(report_json),
            _make_bedrock_response(analysis_json),
        ]

        anomalies = [
            {"window_start": "2024-01-18T00:00:00Z", "avg_sentiment": -0.8,
             "z_score": -3.2, "message_count": 30},
        ]

        report = report_gen.generate_executive_report(
            project_name="TestProject",
            reporting_period="2024-01-15 to 2024-01-22",
            predictions=[{"risk_level": "Medium", "risk_score": 0.5}],
            anomalies=anomalies,
        )

        assert "sentiment_anomaly_analysis" in report
        assert len(report["sentiment_anomaly_analysis"]) == 1
        assert report["sentiment_anomaly_analysis"][0]["analysis"]["severity"] == "high"
        assert report["_metadata"]["anomalies_analyzed"] == 1

    def test_classify_ticket(self, report_gen, mock_bedrock):
        classification = json.dumps({
            "category": "bug_fix",
            "complexity_assessment": "moderate",
            "rationale": "Auth regression needs investigation.",
        })
        mock_bedrock.invoke_model.return_value = _make_bedrock_response(classification)

        result = report_gen.classify_ticket({
            "issue_key": "PROJ-99",
            "summary": "Fix auth regression",
            "description": "Login fails after deploy",
            "issue_type": "Bug",
            "status": "In Progress",
            "story_points": 5,
        })

        assert result["category"] == "bug_fix"
        assert result["complexity_assessment"] == "moderate"

    def test_explain_risk(self, report_gen, mock_bedrock):
        explanation = "This ticket is at risk because it has been in review for 15 days."
        mock_bedrock.invoke_model.return_value = _make_bedrock_response(explanation)

        result = report_gen.explain_risk(
            ticket={"issue_key": "PROJ-1", "summary": "Test", "status": "In Review",
                    "risk_score": 0.85, "risk_level": "High"},
            risk_factors=[{"feature": "days_in_status", "value": 15.0, "importance": 0.8}],
        )

        assert "risk" in result.lower() or "15 days" in result

    def test_parse_json_strips_code_fences(self, report_gen):
        raw = '```json\n{"key": "value"}\n```'
        result = report_gen._parse_json_response(raw)
        assert result == {"key": "value"}

    def test_parse_json_extracts_from_text(self, report_gen):
        raw = 'Here is the result: {"key": "value"} and some trailing text.'
        result = report_gen._parse_json_response(raw)
        assert result == {"key": "value"}

    def test_parse_json_fallback_on_invalid(self, report_gen):
        raw = "This is not JSON at all."
        result = report_gen._parse_json_response(raw)
        assert result["parse_error"] is True
        assert result["raw_response"] == raw

    def test_analyze_sentiment_anomalies_empty(self, report_gen):
        result = report_gen.analyze_sentiment_anomalies([], "TestProject")
        assert result == []

    def test_analyze_sentiment_anomalies_bedrock_down(self, report_gen, mock_bedrock):
        mock_bedrock.invoke_model.side_effect = ConnectionError("Bedrock unavailable")

        anomalies = [
            {"window_start": "2024-01-18", "avg_sentiment": -0.8,
             "z_score": -3.0, "message_count": 25},
        ]
        result = report_gen.analyze_sentiment_anomalies(anomalies, "TestProject")

        assert len(result) == 1
        assert result[0]["analysis"]["severity"] == "unknown"
