"""API orchestration layer (Lambda handler + Flask app).

Serves as the integration point between the Dashboard UI and the
backend services (SageMaker, Bedrock, Data Lake). Supports both
AWS Lambda deployment and local Flask development.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from flask import Flask, jsonify, request

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


def create_app(settings: AppSettings | None = None) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    settings = settings or get_settings()

    # Initialize services
    model = RiskModel(settings.model)
    datalake = DataLakeWriter(settings)
    report_gen = ReportGenerator(settings)

    @app.route("/health", methods=["GET"])
    def health() -> tuple:
        return jsonify({"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}), 200

    @app.route("/api/v1/projects/<project_key>/risk", methods=["GET"])
    def get_project_risk(project_key: str) -> tuple:
        """Get risk predictions for all tickets in a project.

        Returns ticket-level risk scores and overall project risk summary.
        """
        try:
            # Load latest features from Gold layer
            features_df = datalake.read_gold(project_key)
            if features_df.empty:
                return jsonify({"error": "No feature data found for project", "project_key": project_key}), 404

            # Load model
            try:
                model.load_from_s3(settings)
            except Exception:
                logger.warning("Could not load model from S3, using untrained model")
                return jsonify({"error": "Risk model not available"}), 503

            # Generate predictions
            predictions_df = model.predict(features_df)

            # Build response
            predictions = predictions_df.to_dict("records")
            high_risk = [p for p in predictions if p.get("risk_level") == "High"]

            response = {
                "project_key": project_key,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "summary": {
                    "total_tickets": len(predictions),
                    "high_risk": len(high_risk),
                    "medium_risk": sum(1 for p in predictions if p.get("risk_level") == "Medium"),
                    "low_risk": sum(1 for p in predictions if p.get("risk_level") == "Low"),
                    "avg_risk_score": sum(p.get("risk_score", 0) for p in predictions) / max(len(predictions), 1),
                },
                "tickets": _serialize_predictions(predictions),
            }

            return jsonify(response), 200

        except Exception:
            logger.exception("Error computing risk for project %s", project_key)
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/api/v1/projects/<project_key>/report", methods=["POST"])
    def generate_report(project_key: str) -> tuple:
        """Generate an AI-drafted executive status report.

        Request body (optional):
            - reporting_period: str (e.g. "2024-01-15 to 2024-01-22")
            - project_name: str (human-readable project name)
        """
        try:
            body = request.get_json(silent=True) or {}
            project_name = body.get("project_name", project_key)
            reporting_period = body.get(
                "reporting_period",
                f"Week of {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
            )

            # Load features and generate predictions
            features_df = datalake.read_gold(project_key)
            if features_df.empty:
                return jsonify({"error": "No data available for report generation"}), 404

            try:
                model.load_from_s3(settings)
            except Exception:
                return jsonify({"error": "Risk model not available"}), 503

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
            except Exception:
                logger.warning("Could not load Slack data for %s", project_key)

            # Generate report
            report = report_gen.generate_executive_report(
                project_name=project_name,
                reporting_period=reporting_period,
                predictions=predictions,
                sentiment_data=sentiment_data,
                anomalies=anomalies,
            )

            return jsonify(report), 200

        except Exception:
            logger.exception("Error generating report for project %s", project_key)
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/api/v1/tickets/<issue_key>/risk", methods=["GET"])
    def get_ticket_risk(issue_key: str) -> tuple:
        """Get detailed risk analysis for a single ticket."""
        try:
            project_key = issue_key.split("-")[0] if "-" in issue_key else ""
            features_df = datalake.read_gold(project_key)

            if features_df.empty:
                return jsonify({"error": "No feature data found"}), 404

            ticket_row = features_df[features_df["issue_key"] == issue_key]
            if ticket_row.empty:
                return jsonify({"error": f"Ticket {issue_key} not found"}), 404

            try:
                model.load_from_s3(settings)
            except Exception:
                return jsonify({"error": "Risk model not available"}), 503

            result = model.predict_single(ticket_row.iloc[0].to_dict())

            # Generate risk explanation
            try:
                explanation = report_gen.explain_risk(
                    ticket=ticket_row.iloc[0].to_dict(),
                    risk_factors=result.get("top_risk_factors", []),
                )
                result["explanation"] = explanation
            except Exception:
                logger.warning("Could not generate risk explanation for %s", issue_key)

            result["issue_key"] = issue_key
            return jsonify(result), 200

        except Exception:
            logger.exception("Error analyzing ticket %s", issue_key)
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/api/v1/projects/<project_key>/radar", methods=["GET"])
    def get_radar_data(project_key: str) -> tuple:
        """Get data formatted for the Risk Radar scatter plot visualization.

        Returns tickets positioned by:
            - x: complexity (story_points * issue_type_risk)
            - y: sentiment (avg_sentiment)
            - size: story_points
            - color: risk_level
        """
        try:
            features_df = datalake.read_gold(project_key)
            if features_df.empty:
                return jsonify({"error": "No data available"}), 404

            try:
                model.load_from_s3(settings)
            except Exception:
                return jsonify({"error": "Risk model not available"}), 503

            predictions_df = model.predict(features_df)

            radar_points = []
            for _, row in predictions_df.iterrows():
                radar_points.append({
                    "issue_key": row.get("issue_key"),
                    "summary": row.get("summary", "")[:80],
                    "x": float(row.get("story_points", 0) * row.get("issue_type_risk", 0.3)),
                    "y": float(row.get("avg_sentiment", 0)),
                    "size": max(float(row.get("story_points", 1)), 1),
                    "color": row.get("risk_level", "Low"),
                    "risk_score": float(row.get("risk_score", 0)),
                    "status": row.get("status"),
                    "assignee": row.get("assignee"),
                    "days_in_status": float(row.get("days_in_status", 0)),
                })

            return jsonify({
                "project_key": project_key,
                "radar_points": radar_points,
                "axes": {
                    "x": {"label": "Technical Complexity", "description": "Story Points x Type Risk"},
                    "y": {"label": "Team Sentiment", "description": "Slack-derived sentiment score"},
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }), 200

        except Exception:
            logger.exception("Error generating radar for project %s", project_key)
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/api/v1/reports/<project_key>/feedback", methods=["POST"])
    def submit_report_feedback(project_key: str) -> tuple:
        """Accept human-edited report for future RLHF fine-tuning.

        Request body:
            - original_report: dict (the AI-generated report)
            - edited_report: dict (the human-edited version)
            - feedback_type: str ("approved", "edited", "rejected")
        """
        try:
            body = request.get_json(silent=True)
            if not body:
                return jsonify({"error": "Request body required"}), 400

            feedback_record = {
                "project_key": project_key,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "feedback_type": body.get("feedback_type", "edited"),
                "original_report": body.get("original_report"),
                "edited_report": body.get("edited_report"),
            }

            # Store feedback for future RLHF training
            datalake.write_bronze([feedback_record], "feedback", project_key)

            return jsonify({"status": "feedback_recorded"}), 200

        except Exception:
            logger.exception("Error recording feedback for %s", project_key)
            return jsonify({"error": "Internal server error"}), 500

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
    from werkzeug.serving import WSGIRequestHandler

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
