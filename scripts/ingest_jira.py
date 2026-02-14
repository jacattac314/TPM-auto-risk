"""Jira ingestion Lambda handler / standalone script.

Orchestrates the full Jira ingestion pipeline:
1. Crawl Jira project via OAuth API
2. Redact PII from text fields
3. Store raw data in Bronze layer
4. Store sanitized data in Silver layer
5. Run data quality checks
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from src.config.settings import get_settings
from src.datalake.storage import DataLakeWriter
from src.ingestion.jira_crawler import JiraIngestionState, crawl_project
from src.ingestion.pii_redactor import ComprehendPIIRedactor, redact_record
from src.models.drift_monitor import AlertPublisher, DataQualityChecker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Text fields in Jira records that should be scanned for PII
JIRA_TEXT_FIELDS = ["summary", "description"]


def run_ingestion(project_keys: list[str]) -> dict[str, Any]:
    """Run the Jira ingestion pipeline for the given projects.

    Returns a summary of the ingestion results.
    """
    settings = get_settings()
    datalake = DataLakeWriter(settings)
    redactor = ComprehendPIIRedactor(settings)
    quality_checker = DataQualityChecker()
    alert_publisher = AlertPublisher(settings)
    state = JiraIngestionState(
        table_name=settings.aws.dynamodb_table,
        region=settings.aws.region,
    )

    results = {}

    for project_key in project_keys:
        logger.info("Starting Jira ingestion for project: %s", project_key)

        try:
            # Step 1: Crawl Jira
            raw_tickets = crawl_project(project_key, settings, state)

            if not raw_tickets:
                logger.info("No new tickets for %s", project_key)
                results[project_key] = {"status": "no_new_data", "tickets": 0}
                continue

            # Step 2: Write raw data to Bronze
            datalake.write_bronze(raw_tickets, "jira", project_key)

            # Step 3: Redact PII
            sanitized_tickets = [
                redact_record(ticket, JIRA_TEXT_FIELDS, redactor)
                for ticket in raw_tickets
            ]

            # Step 4: Write sanitized data to Silver
            datalake.write_silver(sanitized_tickets, "jira", project_key)

            # Step 5: Data quality checks
            import pandas as pd

            df = pd.DataFrame(sanitized_tickets)
            quality_report = quality_checker.validate(df)

            if not quality_report["passed"]:
                logger.warning(
                    "Data quality check FAILED for %s: %d violations",
                    project_key,
                    quality_report["violations_count"],
                )
                alert_publisher.publish_quality_alert(quality_report, project_key)

            results[project_key] = {
                "status": "success",
                "tickets_ingested": len(sanitized_tickets),
                "pii_detected": sum(
                    1
                    for t in sanitized_tickets
                    if t.get("_pii_redaction", {}).get("pii_detected")
                ),
                "quality_passed": quality_report["passed"],
                "quality_violations": quality_report["violations_count"],
            }

        except Exception as e:
            logger.exception("Ingestion failed for %s", project_key)
            results[project_key] = {"status": "error", "error": str(e)}

    return results


def lambda_handler(event: dict, context: Any) -> dict:
    """AWS Lambda entry point for scheduled Jira ingestion."""
    project_keys = event.get("project_keys", [])

    if not project_keys:
        # Default: read from environment or event
        project_keys_env = os.environ.get("JIRA_PROJECT_KEYS", "")
        project_keys = [k.strip() for k in project_keys_env.split(",") if k.strip()]

    if not project_keys:
        return {"statusCode": 400, "body": json.dumps({"error": "No project keys specified"})}

    results = run_ingestion(project_keys)

    return {
        "statusCode": 200,
        "body": json.dumps(results, default=str),
    }


if __name__ == "__main__":
    import sys

    keys = sys.argv[1:] if len(sys.argv) > 1 else ["PROJ"]
    result = run_ingestion(keys)
    print(json.dumps(result, indent=2, default=str))
