"""Tests for the API handler / Flask application."""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest
from flask import Flask
from flask.testing import FlaskClient

from src.api.handler import create_app
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


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

        data = response.get_json()
        assert data["status"] == "healthy"
        assert "timestamp" in data


class TestFeedbackEndpoint:
    def test_missing_body_returns_400(self, client):
        response = client.post(
            "/api/v1/reports/PROJ/feedback",
            content_type="application/json",
        )
        assert response.status_code == 400

    @patch("src.datalake.storage.boto3")
    def test_valid_feedback_accepted(self, mock_boto3, settings):
        """Test feedback endpoint with mocked S3."""
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
