"""API orchestration layer (Lambda handler + Flask app).

Serves as the integration point between the Dashboard UI and the
backend services (SageMaker, Bedrock, Data Lake). Supports both
AWS Lambda deployment and local Flask development.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import numpy as np
from flask import Flask, Response, jsonify, request

from src.config.settings import AppSettings, get_settings
from src.datalake.storage import DataLakeWriter
from src.features.feature_engineering import (
    compute_team_features,
    engineer_jira_features,
    merge_sentiment_features,
)
from src.ingestion.slack_crawler import (
    compute_sentiment_timeseries,
    detect_sentiment_anomalies,
)
from src.models.risk_model import RiskModel
from src.reporting.report_generator import ReportGenerator

logger = logging.getLogger(__name__)

# Input validation patterns
_PROJECT_KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,19}$")
_ISSUE_KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,19}-\d{1,7}$")

# Rate limiting: simple in-memory token bucket (per-Lambda instance)
_RATE_LIMIT_WINDOW = 60  # seconds
_RATE_LIMIT_MAX_REQUESTS = 120
_rate_limit_state: dict[str, list[float]] = {}


def _validate_project_key(project_key: str) -> str | None:
    """Return an error message if the project key is invalid, else None."""
    if not _PROJECT_KEY_RE.match(project_key):
        return f"Invalid project key format: '{project_key}'. Must be 2-20 uppercase alphanumeric characters."
    return None


def _validate_issue_key(issue_key: str) -> str | None:
    """Return an error message if the issue key is invalid, else None."""
    if not _ISSUE_KEY_RE.match(issue_key):
        return f"Invalid issue key format: '{issue_key}'. Expected format: PROJ-123."
    return None


def _check_rate_limit(client_id: str) -> bool:
    """Return True if the request should be rate-limited."""
    now = time.monotonic()
    timestamps = _rate_limit_state.get(client_id, [])
    # Prune old entries
    timestamps = [t for t in timestamps if now - t < _RATE_LIMIT_WINDOW]
    if len(timestamps) >= _RATE_LIMIT_MAX_REQUESTS:
        _rate_limit_state[client_id] = timestamps
        return True
    timestamps.append(now)
    _rate_limit_state[client_id] = timestamps
    return False


def _error_response(message: str, status_code: int, **extra: Any) -> tuple[Response, int]:
    """Build a consistent error JSON response."""
    body: dict[str, Any] = {"error": message, **extra}
    return jsonify(body), status_code


def create_app(settings: AppSettings | None = None) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    settings = settings or get_settings()

    # API key from environment (empty = auth disabled for local dev)
    api_key = os.environ.get("API_KEY", "")

    # Initialize services
    model = RiskModel(settings.model)
    datalake = DataLakeWriter(settings)
    report_gen = ReportGenerator(settings)

    @app.before_request
    def _authenticate_and_rate_limit() -> tuple[Response, int] | None:
        # Skip auth for health check
        if request.path == "/health":
            return None

        # API key authentication (disabled when API_KEY env is empty)
        if api_key:
            provided = request.headers.get("Authorization", "")
            if not provided.startswith("Bearer ") or provided[7:] != api_key:
                return _error_response("Invalid or missing API key", 401)

        # Rate limiting by source IP
        client_id = request.remote_addr or "unknown"
        if _check_rate_limit(client_id):
            return _error_response("Rate limit exceeded", 429)

        return None

    def _load_model() -> tuple[Response, int] | None:
        """Try to load the model from S3. Returns error response on failure, None on success."""
        try:
            model.load_from_s3(settings)
            return None
        except FileNotFoundError:
            logger.error("Model artifacts not found in S3")
            return _error_response("Risk model not found", 503)
        except (OSError, ConnectionError) as e:
            logger.error("S3 connection error loading model: %s", e)
            return _error_response("Risk model temporarily unavailable", 503)
        except (ValueError, RuntimeError) as e:
            logger.error("Corrupt or incompatible model artifact: %s", e)
            return _error_response("Risk model failed to load", 503)

    @app.route("/health", methods=["GET"])
    def health() -> tuple[Response, int]:
        return jsonify({"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}), 200

    @app.route("/api/v1/projects/<project_key>/risk", methods=["GET"])
    def get_project_risk(project_key: str) -> tuple[Response, int]:
        """Get risk predictions for all tickets in a project."""
        validation_err = _validate_project_key(project_key)
        if validation_err:
            return _error_response(validation_err, 400)

        try:
            features_df = datalake.read_gold(project_key)
            if features_df.empty:
                return _error_response("No feature data found for project", 404, project_key=project_key)

            model_err = _load_model()
            if model_err:
                return model_err

            predictions_df = model.predict(features_df)
            predictions = predictions_df.to_dict("records")

            level_counts = Counter(p.get("risk_level") for p in predictions)
            avg_score = sum(p.get("risk_score", 0) for p in predictions) / max(len(predictions), 1)

            response = {
                "project_key": project_key,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "summary": {
                    "total_tickets": len(predictions),
                    "high_risk": level_counts.get("High", 0),
                    "medium_risk": level_counts.get("Medium", 0),
                    "low_risk": level_counts.get("Low", 0),
                    "avg_risk_score": round(avg_score, 4),
                },
                "tickets": _serialize_predictions(predictions),
            }

            return jsonify(response), 200

        except (OSError, ConnectionError) as e:
            logger.exception("Data lake error for project %s: %s", project_key, e)
            return _error_response("Data source temporarily unavailable", 503)

    @app.route("/api/v1/projects/<project_key>/report", methods=["POST"])
    def generate_report(project_key: str) -> tuple[Response, int]:
        """Generate an AI-drafted executive status report."""
        validation_err = _validate_project_key(project_key)
        if validation_err:
            return _error_response(validation_err, 400)

        try:
            body = request.get_json(silent=True) or {}
            project_name = body.get("project_name", project_key)
            reporting_period = body.get(
                "reporting_period",
                f"Week of {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
            )

            features_df = datalake.read_gold(project_key)
            if features_df.empty:
                return _error_response("No data available for report generation", 404)

            model_err = _load_model()
            if model_err:
                return model_err

            predictions_df = model.predict(features_df)
            predictions = predictions_df.to_dict("records")

            # Load sentiment data if available
            sentiment_data: list[dict] = []
            anomalies: list[dict] = []
            try:
                slack_df = datalake.read_silver("slack", project_key)
                if not slack_df.empty:
                    messages = slack_df.to_dict("records")
                    sentiment_data = compute_sentiment_timeseries(messages)
                    anomalies = detect_sentiment_anomalies(sentiment_data)
            except (OSError, ConnectionError):
                logger.warning("Could not load Slack data for %s (S3 unavailable)", project_key)
            except (KeyError, ValueError) as e:
                logger.warning("Slack data malformed for %s: %s", project_key, e)

            report = report_gen.generate_executive_report(
                project_name=project_name,
                reporting_period=reporting_period,
                predictions=predictions,
                sentiment_data=sentiment_data,
                anomalies=anomalies,
            )

            # Flag if sentiment was unavailable
            if not sentiment_data:
                report.setdefault("_warnings", []).append("Slack sentiment data unavailable")

            return jsonify(report), 200

        except (OSError, ConnectionError) as e:
            logger.exception("Service error generating report for %s: %s", project_key, e)
            return _error_response("Service temporarily unavailable", 503)

    @app.route("/api/v1/tickets/<issue_key>/risk", methods=["GET"])
    def get_ticket_risk(issue_key: str) -> tuple[Response, int]:
        """Get detailed risk analysis for a single ticket."""
        validation_err = _validate_issue_key(issue_key)
        if validation_err:
            return _error_response(validation_err, 400)

        project_key = issue_key.split("-")[0]

        try:
            features_df = datalake.read_gold(project_key)
            if features_df.empty:
                return _error_response("No feature data found", 404, project_key=project_key)

            ticket_row = features_df[features_df["issue_key"] == issue_key]
            if ticket_row.empty:
                return _error_response(f"Ticket {issue_key} not found", 404)

            model_err = _load_model()
            if model_err:
                return model_err

            result = model.predict_single(ticket_row.iloc[0].to_dict())

            # Generate risk explanation (non-critical, degrade gracefully)
            try:
                explanation = report_gen.explain_risk(
                    ticket=ticket_row.iloc[0].to_dict(),
                    risk_factors=result.get("top_risk_factors", []),
                )
                result["explanation"] = explanation
            except (OSError, ConnectionError):
                logger.warning("Bedrock unavailable for risk explanation on %s", issue_key)
                result["explanation"] = None
                result.setdefault("_warnings", []).append("Risk explanation unavailable (LLM service down)")
            except (ValueError, KeyError) as e:
                logger.warning("Failed to generate explanation for %s: %s", issue_key, e)
                result["explanation"] = None

            result["issue_key"] = issue_key
            return jsonify(result), 200

        except (OSError, ConnectionError) as e:
            logger.exception("Data source error analyzing ticket %s: %s", issue_key, e)
            return _error_response("Service temporarily unavailable", 503)

    @app.route("/api/v1/projects/<project_key>/radar", methods=["GET"])
    def get_radar_data(project_key: str) -> tuple[Response, int]:
        """Get data formatted for the Risk Radar scatter plot visualization."""
        validation_err = _validate_project_key(project_key)
        if validation_err:
            return _error_response(validation_err, 400)

        try:
            features_df = datalake.read_gold(project_key)
            if features_df.empty:
                return _error_response("No data available", 404)

            model_err = _load_model()
            if model_err:
                return model_err

            predictions_df = model.predict(features_df)

            # Vectorized radar point construction
            sp = predictions_df.get("story_points", 0).fillna(0).astype(float)
            itr = predictions_df.get("issue_type_risk", 0.3).fillna(0.3).astype(float)
            sent = predictions_df.get("avg_sentiment", 0).fillna(0).astype(float)
            dis = predictions_df.get("days_in_status", 0).fillna(0).astype(float)
            rs = predictions_df.get("risk_score", 0).fillna(0).astype(float)

            radar_points = [
                {
                    "issue_key": row.get("issue_key"),
                    "summary": str(row.get("summary", ""))[:80],
                    "x": round(float(sp.iloc[i] * itr.iloc[i]), 4),
                    "y": round(float(sent.iloc[i]), 4),
                    "size": max(float(sp.iloc[i]), 1.0),
                    "color": row.get("risk_level", "Low"),
                    "risk_score": round(float(rs.iloc[i]), 4),
                    "status": row.get("status"),
                    "assignee": row.get("assignee"),
                    "days_in_status": round(float(dis.iloc[i]), 1),
                }
                for i, (_, row) in enumerate(predictions_df.iterrows())
            ]

            return jsonify({
                "project_key": project_key,
                "radar_points": radar_points,
                "axes": {
                    "x": {"label": "Technical Complexity", "description": "Story Points x Type Risk"},
                    "y": {"label": "Team Sentiment", "description": "Slack-derived sentiment score"},
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }), 200

        except (OSError, ConnectionError) as e:
            logger.exception("Data source error for radar %s: %s", project_key, e)
            return _error_response("Service temporarily unavailable", 503)

    @app.route("/api/v1/reports/<project_key>/feedback", methods=["POST"])
    def submit_report_feedback(project_key: str) -> tuple[Response, int]:
        """Accept human-edited report for future RLHF fine-tuning."""
        validation_err = _validate_project_key(project_key)
        if validation_err:
            return _error_response(validation_err, 400)

        body = request.get_json(silent=True)
        if not body:
            return _error_response("Request body required", 400)

        feedback_type = body.get("feedback_type", "edited")
        if feedback_type not in ("approved", "edited", "rejected"):
            return _error_response(
                f"Invalid feedback_type: '{feedback_type}'. Must be approved, edited, or rejected.",
                400,
            )

        try:
            feedback_record = {
                "project_key": project_key,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "feedback_type": feedback_type,
                "original_report": body.get("original_report"),
                "edited_report": body.get("edited_report"),
            }

            datalake.write_bronze([feedback_record], "feedback", project_key)
            return jsonify({"status": "feedback_recorded"}), 200

        except (OSError, ConnectionError) as e:
            logger.exception("Failed to store feedback for %s: %s", project_key, e)
            return _error_response("Failed to store feedback", 503)

    return app


def _serialize_predictions(predictions: list[dict]) -> list[dict]:
    """Clean predictions for JSON serialization."""
    serializable = []
    for p in predictions:
        clean = {}
        for key, value in p.items():
            if key.startswith("_"):
                continue
            if isinstance(value, float):
                if np.isnan(value) or np.isinf(value):
                    clean[key] = 0.0
                else:
                    clean[key] = round(value, 4)
            elif hasattr(value, "isoformat"):
                clean[key] = value.isoformat()
            else:
                clean[key] = value
        serializable.append(clean)
    return serializable


# Lambda handler for API Gateway integration
def lambda_handler(event: dict, context: Any) -> dict:
    """AWS Lambda entry point (API Gateway proxy integration)."""
    app = create_app()

    # Convert API Gateway event to WSGI
    path = event.get("path", "/")
    method = event.get("httpMethod", "GET")
    headers = event.get("headers", {}) or {}
    body = event.get("body", "")

    with app.test_request_context(path=path, method=method, headers=headers, data=body):
        try:
            response = app.full_dispatch_request()
            return {
                "statusCode": response.status_code,
                "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
                "body": response.get_data(as_text=True),
            }
        except Exception:
            logger.exception("Lambda handler error")
            return {
                "statusCode": 500,
                "body": json.dumps({"error": "Internal server error"}),
            }
