"""Tests for the API handler / Flask application."""

from __future__ import annotations

import json
import os
from unittest.mock import patch, MagicMock

import pytest
from flask import Flask
from flask.testing import FlaskClient

from src.api.handler import (
    _validate_project_key,
    _validate_issue_key,
    _check_rate_limit,
    _rate_limit_state,
    _serialize_predictions,
    create_app,
)
from src.config.settings import AppSettings


@pytest.fixture
def app(settings) -> Flask:
    """Create a test Flask application."""
    app = create_app(settings)
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(app) -> FlaskClient:
    """Create a test client."""
    return app.test_client()


class TestInputValidation:
    def test_valid_project_keys(self):
        assert _validate_project_key("PROJ") is None
        assert _validate_project_key("MY_PROJECT") is None
        assert _validate_project_key("AB") is None
        assert _validate_project_key("PROJ123") is None

    def test_invalid_project_keys(self):
        assert _validate_project_key("") is not None
        assert _validate_project_key("a") is not None  # lowercase
        assert _validate_project_key("proj") is not None  # lowercase
        assert _validate_project_key("P") is not None  # too short
        assert _validate_project_key("../../etc/passwd") is not None  # path traversal
        assert _validate_project_key("A" * 21) is not None  # too long
        assert _validate_project_key("PROJ-123") is not None  # issue key format

    def test_valid_issue_keys(self):
        assert _validate_issue_key("PROJ-1") is None
        assert _validate_issue_key("PROJ-123") is None
        assert _validate_issue_key("MY_PROJ-9999999") is None

    def test_invalid_issue_keys(self):
        assert _validate_issue_key("") is not None
        assert _validate_issue_key("PROJ") is not None  # no dash
        assert _validate_issue_key("PROJ-") is not None  # no number
        assert _validate_issue_key("../../path-1") is not None
        assert _validate_issue_key("proj-123") is not None  # lowercase
        assert _validate_issue_key("PROJ-12345678") is not None  # number too long


class TestRateLimiting:
    def setup_method(self):
        _rate_limit_state.clear()

    def test_allows_within_limit(self):
        assert _check_rate_limit("test_client") is False
        assert _check_rate_limit("test_client") is False

    def test_blocks_over_limit(self):
        for _ in range(120):
            _check_rate_limit("test_client")
        assert _check_rate_limit("test_client") is True

    def test_separate_clients(self):
        for _ in range(120):
            _check_rate_limit("client_a")
        assert _check_rate_limit("client_a") is True
        assert _check_rate_limit("client_b") is False


class TestSerializePredictions:
    def test_basic_serialization(self):
        preds = [{"issue_key": "PROJ-1", "risk_score": 0.123456, "_internal": "hidden"}]
        result = _serialize_predictions(preds)
        assert len(result) == 1
        assert result[0]["risk_score"] == 0.1235
        assert "_internal" not in result[0]

    def test_nan_handling(self):
        import math
        preds = [{"risk_score": float("nan"), "other": float("inf")}]
        result = _serialize_predictions(preds)
        assert result[0]["risk_score"] == 0.0
        assert result[0]["other"] == 0.0


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

        data = response.get_json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_health_bypasses_auth(self, settings):
        """Health endpoint should work even when API_KEY is set."""
        with patch.dict(os.environ, {"API_KEY": "secret-key"}):
            app = create_app(settings)
            app.config["TESTING"] = True
            client = app.test_client()

            response = client.get("/health")
            assert response.status_code == 200


class TestAuthentication:
    def test_rejects_missing_api_key(self, settings):
        with patch.dict(os.environ, {"API_KEY": "test-secret"}):
            app = create_app(settings)
            app.config["TESTING"] = True
            client = app.test_client()

            response = client.get("/api/v1/projects/PROJ/risk")
            assert response.status_code == 401

    def test_rejects_wrong_api_key(self, settings):
        with patch.dict(os.environ, {"API_KEY": "test-secret"}):
            app = create_app(settings)
            app.config["TESTING"] = True
            client = app.test_client()

            response = client.get(
                "/api/v1/projects/PROJ/risk",
                headers={"Authorization": "Bearer wrong-key"},
            )
            assert response.status_code == 401

    def test_accepts_valid_api_key(self, settings):
        with patch.dict(os.environ, {"API_KEY": "test-secret"}):
            with patch("src.datalake.storage.boto3"):
                app = create_app(settings)
                app.config["TESTING"] = True
                client = app.test_client()

                # Will return 404 since no data, but auth passes
                response = client.get(
                    "/api/v1/projects/PROJ/risk",
                    headers={"Authorization": "Bearer test-secret"},
                )
                # Either 404 (no data) or 503 (no model) - but NOT 401
                assert response.status_code != 401


class TestProjectRiskValidation:
    def test_invalid_project_key_returns_400(self, client):
        response = client.get("/api/v1/projects/../../etc/risk")
        assert response.status_code in (400, 404)

    def test_lowercase_project_key_returns_400(self, client):
        response = client.get("/api/v1/projects/proj/risk")
        assert response.status_code in (400, 404)


class TestTicketRiskValidation:
    def test_invalid_issue_key_returns_400(self, client):
        response = client.get("/api/v1/tickets/invalid/risk")
        assert response.status_code in (400, 404)


class TestFeedbackEndpoint:
    def test_missing_body_returns_400(self, client):
        response = client.post(
            "/api/v1/reports/PROJ/feedback",
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_invalid_feedback_type_returns_400(self, client):
        response = client.post(
            "/api/v1/reports/PROJ/feedback",
            data=json.dumps({"feedback_type": "invalid_type"}),
            content_type="application/json",
        )
        assert response.status_code == 400
        data = response.get_json()
        assert "Invalid feedback_type" in data["error"]

    @patch("src.datalake.storage.boto3")
    def test_valid_feedback_accepted(self, mock_boto3, settings):
        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3

        app = create_app(settings)
        app.config["TESTING"] = True
        client = app.test_client()

        response = client.post(
            "/api/v1/reports/PROJ/feedback",
            data=json.dumps({
                "original_report": {"summary": "test"},
                "edited_report": {"summary": "edited test"},
                "feedback_type": "edited",
            }),
            content_type="application/json",
        )
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "feedback_recorded"

    def test_invalid_project_key_in_feedback(self, client):
        response = client.post(
            "/api/v1/reports/../../hack/feedback",
            data=json.dumps({"feedback_type": "approved"}),
            content_type="application/json",
        )
        assert response.status_code in (400, 404)
