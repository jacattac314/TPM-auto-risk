"""Dashboard web application for the Risk Radar visualization.

Provides the Flask-based UI layer with:
- Risk Radar scatter plot (complexity vs sentiment)
- Auto-Draft report interface (split-screen with evidence locker)
- Project overview and ticket drill-down views
"""

from __future__ import annotations

import json
import logging
import os

from flask import Flask, render_template, jsonify, request
from src.api.handler import create_app as create_api_app

logger = logging.getLogger(__name__)


def create_dashboard_app() -> Flask:
    """Create the combined dashboard + API Flask app."""
    # Get the API app with all backend routes
    app = create_api_app()

    # Configure template and static directories
    app.template_folder = os.path.join(os.path.dirname(__file__), "templates")
    app.static_folder = os.path.join(os.path.dirname(__file__), "static")

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/project/<project_key>")
    def project_view(project_key: str):
        return render_template("project.html", project_key=project_key)

    @app.route("/report/<project_key>")
    def report_view(project_key: str):
        return render_template("report.html", project_key=project_key)

    return app


if __name__ == "__main__":
    app = create_dashboard_app()
    app.run(debug=True, host="0.0.0.0", port=8080)
