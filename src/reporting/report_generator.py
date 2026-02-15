"""GenAI reporting engine using Amazon Bedrock.

Synthesizes XGBoost risk predictions and Slack sentiment into
human-readable executive status reports via RAG architecture.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import boto3

from src.config.settings import AppSettings
from src.reporting.prompt_templates import (
    EXECUTIVE_REPORT_PROMPT,
    RISK_EXPLANATION_PROMPT,
    TICKET_CLASSIFICATION_PROMPT,
    format_jira_risk_summary,
    format_risk_scores_summary,
    format_sentiment_summary,
)

logger = logging.getLogger(__name__)


class BedrockClient:
    """Wrapper for Amazon Bedrock model invocation with retry and error handling."""

    def __init__(self, settings: AppSettings) -> None:
        self._client = boto3.client("bedrock-runtime", region_name=settings.aws.region)
        self._sonnet_model_id = settings.aws.bedrock_model_id
        self._haiku_model_id = settings.aws.bedrock_haiku_model_id

    def invoke(
        self,
        prompt: str,
        model_tier: str = "sonnet",
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> str:
        """Invoke a Bedrock model and return the text response.

        Args:
            prompt: The formatted prompt string.
            model_tier: 'sonnet' for complex synthesis, 'haiku' for classification.
            temperature: Sampling temperature (low = deterministic).
            max_tokens: Maximum output tokens.

        Returns:
            The model's text response.
        """
        model_id = self._sonnet_model_id if model_tier == "sonnet" else self._haiku_model_id

        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        })

        response = self._client.invoke_model(
            modelId=model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )

        response_body = json.loads(response["body"].read())
        content = response_body.get("content", [])

        if content and isinstance(content, list):
            return content[0].get("text", "")
        return ""


class ReportGenerator:
    """Generates executive status reports using Bedrock LLMs and risk model outputs."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._bedrock = BedrockClient(settings)

    def generate_executive_report(
        self,
        project_name: str,
        reporting_period: str,
        predictions: list[dict[str, Any]],
        sentiment_data: list[dict[str, Any]] | None = None,
        anomalies: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Generate a full executive status report.

        Args:
            project_name: Human-readable project name.
            reporting_period: e.g. "2024-01-15 to 2024-01-22".
            predictions: List of ticket dicts with risk scores.
            sentiment_data: Slack sentiment timeseries.
            anomalies: Detected sentiment anomalies.

        Returns:
            Parsed JSON report with executive_summary, critical_risks, etc.
        """
        # Segment tickets
        high_risk = [p for p in predictions if p.get("risk_level") == "High"]
        completed = [p for p in predictions if p.get("status_category") == "Done"]
        in_progress = [p for p in predictions if p.get("status_category") == "In Progress"]

        # Format prompt data
        jira_summary = format_jira_risk_summary(high_risk)
        risk_summary = format_risk_scores_summary(predictions)
        sentiment_summary = format_sentiment_summary(sentiment_data or [], anomalies)

        prompt = EXECUTIVE_REPORT_PROMPT.format(
            project_name=project_name,
            reporting_period=reporting_period,
            total_tickets=len(predictions),
            completed_count=len(completed),
            in_progress_count=len(in_progress),
            high_risk_count=len(high_risk),
            medium_risk_count=sum(1 for p in predictions if p.get("risk_level") == "Medium"),
            jira_risk_summary=jira_summary,
            slack_sentiment_summary=sentiment_summary,
            risk_scores_summary=risk_summary,
        )

        logger.info("Generating executive report for %s (%d tickets)", project_name, len(predictions))

        raw_response = self._bedrock.invoke(prompt, model_tier="sonnet", temperature=0.1)

        # Parse JSON response
        report = self._parse_json_response(raw_response)

        # Tier 2: LLM analysis of sentiment anomalies
        enriched_anomalies: list[dict[str, Any]] = []
        if anomalies:
            try:
                enriched_anomalies = self.analyze_sentiment_anomalies(
                    anomalies, project_name
                )
                report["sentiment_anomaly_analysis"] = enriched_anomalies
            except (OSError, ConnectionError):
                logger.warning("Tier 2 anomaly analysis unavailable for %s", project_name)

        # Add metadata
        report["_metadata"] = {
            "project_name": project_name,
            "reporting_period": reporting_period,
            "total_tickets": len(predictions),
            "model_tier": "sonnet",
            "risk_distribution": {
                "high": len(high_risk),
                "medium": sum(1 for p in predictions if p.get("risk_level") == "Medium"),
                "low": sum(1 for p in predictions if p.get("risk_level") == "Low"),
            },
            "anomalies_analyzed": len(enriched_anomalies),
        }

        return report

    def classify_ticket(self, ticket: dict[str, Any]) -> dict[str, Any]:
        """Classify a single ticket using Haiku (cost-effective).

        Args:
            ticket: Dict with issue_key, summary, description, etc.

        Returns:
            Classification result with category and complexity_assessment.
        """
        prompt = TICKET_CLASSIFICATION_PROMPT.format(
            issue_key=ticket.get("issue_key", "N/A"),
            summary=ticket.get("summary", ""),
            description=ticket.get("description", "")[:500],  # Truncate for cost
            issue_type=ticket.get("issue_type", ""),
            status=ticket.get("status", ""),
            story_points=ticket.get("story_points", 0),
        )

        raw = self._bedrock.invoke(prompt, model_tier="haiku", temperature=0.0, max_tokens=256)
        return self._parse_json_response(raw)

    def explain_risk(
        self,
        ticket: dict[str, Any],
        risk_factors: list[dict[str, Any]],
        context: str = "",
    ) -> str:
        """Generate a human-readable risk explanation for a single ticket.

        Args:
            ticket: Ticket data with risk score.
            risk_factors: Top contributing features from the model.
            context: Additional context (e.g. Slack excerpts).

        Returns:
            Plain-text explanation string.
        """
        factors_text = "\n".join(
            f"- {f['feature']}: value={f['value']}, importance={f['importance']}"
            for f in risk_factors
        )

        prompt = RISK_EXPLANATION_PROMPT.format(
            issue_key=ticket.get("issue_key", "N/A"),
            summary=ticket.get("summary", ""),
            status=ticket.get("status", ""),
            risk_score=f"{ticket.get('risk_score', 0):.0%}",
            risk_level=ticket.get("risk_level", "Unknown"),
            risk_factors=factors_text,
            additional_context=context or "No additional context available.",
        )

        return self._bedrock.invoke(prompt, model_tier="haiku", temperature=0.1, max_tokens=256)

    def analyze_sentiment_anomalies(
        self,
        anomalies: list[dict[str, Any]],
        project_name: str,
    ) -> list[dict[str, Any]]:
        """Tier 2: Use LLM to provide contextual analysis of sentiment anomalies.

        Sends detected anomaly windows to Haiku for root-cause hypotheses,
        producing actionable insights that feed into the executive report.

        Args:
            anomalies: Detected anomalies from detect_sentiment_anomalies().
            project_name: Project name for context.

        Returns:
            List of enriched anomaly dicts with an 'analysis' field.
        """
        if not anomalies:
            return []

        enriched: list[dict[str, Any]] = []
        for anomaly in anomalies[:5]:  # Cap at 5 to control costs
            prompt = (
                f"You are a senior TPM analyzing team communication signals for "
                f"project '{project_name}'.\n\n"
                f"A sentiment anomaly was detected:\n"
                f"- Time window: {anomaly.get('window_start', 'N/A')}\n"
                f"- Average sentiment: {anomaly.get('avg_sentiment', 0):.3f}\n"
                f"- Z-score: {anomaly.get('z_score', 0):.2f}\n"
                f"- Message count: {anomaly.get('message_count', 0)}\n\n"
                f"Based on this data, provide a 2-sentence hypothesis about "
                f"what might have caused the drop and one recommended action. "
                f"Output valid JSON with keys: \"hypothesis\", \"recommended_action\", "
                f"\"severity\" (one of: low, medium, high)."
            )

            try:
                raw = self._bedrock.invoke(
                    prompt, model_tier="haiku", temperature=0.1, max_tokens=256
                )
                analysis = self._parse_json_response(raw)
            except (OSError, ConnectionError):
                logger.warning("Bedrock unavailable for anomaly analysis")
                analysis = {"hypothesis": "LLM analysis unavailable", "severity": "unknown"}

            enriched.append({**anomaly, "analysis": analysis})

        return enriched

    def _parse_json_response(self, raw: str) -> dict[str, Any]:
        """Parse a JSON response from the LLM, handling common formatting issues."""
        text = raw.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (```json and ```)
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass

            logger.warning("Failed to parse LLM JSON response, returning raw text")
            return {"raw_response": raw, "parse_error": True}
