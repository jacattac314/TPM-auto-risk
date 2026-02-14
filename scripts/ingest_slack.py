"""Slack ingestion Lambda handler / standalone script.

Orchestrates:
1. Crawl Slack channels for messages
2. Compute sentiment timeseries
3. Detect anomalies
4. Redact PII
5. Store in data lake
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from src.config.settings import get_settings
from src.datalake.storage import DataLakeWriter
from src.ingestion.pii_redactor import ComprehendPIIRedactor, redact_record
from src.ingestion.slack_crawler import (
    SlackChannelCrawler,
    compute_sentiment_timeseries,
    detect_sentiment_anomalies,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SLACK_TEXT_FIELDS = ["text"]


def run_ingestion(
    channel_ids: list[str],
    project_key: str,
    lookback_hours: int = 24,
) -> dict[str, Any]:
    """Run Slack ingestion for specified channels.

    Args:
        channel_ids: Slack channel IDs to crawl.
        project_key: Project key to associate the data with.
        lookback_hours: How far back to fetch messages.

    Returns:
        Ingestion summary.
    """
    settings = get_settings()
    crawler = SlackChannelCrawler(settings)
    datalake = DataLakeWriter(settings)
    redactor = ComprehendPIIRedactor(settings)

    now = datetime.now(timezone.utc)
    oldest = str((now - timedelta(hours=lookback_hours)).timestamp())

    all_messages: list[dict[str, Any]] = []
    channel_results = {}

    for channel_id in channel_ids:
        logger.info("Crawling Slack channel: %s", channel_id)

        try:
            messages = crawler.crawl_channel(channel_id, oldest=oldest)

            # Redact PII
            sanitized = [redact_record(msg, SLACK_TEXT_FIELDS, redactor) for msg in messages]
            all_messages.extend(sanitized)

            channel_results[channel_id] = {
                "messages_crawled": len(messages),
                "pii_detected": sum(
                    1 for m in sanitized if m.get("_pii_redaction", {}).get("pii_detected")
                ),
            }

        except Exception as e:
            logger.exception("Failed to crawl channel %s", channel_id)
            channel_results[channel_id] = {"error": str(e)}

    if not all_messages:
        return {"status": "no_data", "channels": channel_results}

    # Store in data lake
    datalake.write_bronze(all_messages, "slack", project_key)
    datalake.write_silver(all_messages, "slack", project_key)

    # Compute sentiment analytics
    timeseries = compute_sentiment_timeseries(all_messages)
    anomalies = detect_sentiment_anomalies(
        timeseries,
        threshold_std_devs=settings.slack.sentiment_anomaly_threshold,
    )

    # Store sentiment analytics
    if timeseries:
        import pandas as pd

        ts_df = pd.DataFrame(timeseries)
        datalake.write_gold(ts_df, project_key, feature_set_name="sentiment_timeseries")

    return {
        "status": "success",
        "project_key": project_key,
        "total_messages": len(all_messages),
        "sentiment_windows": len(timeseries),
        "anomalies_detected": len(anomalies),
        "channels": channel_results,
    }


def lambda_handler(event: dict, context: Any) -> dict:
    """AWS Lambda entry point for scheduled Slack ingestion."""
    channel_ids = event.get("channel_ids", [])
    project_key = event.get("project_key", "")
    lookback_hours = event.get("lookback_hours", 24)

    if not channel_ids:
        channels_env = os.environ.get("SLACK_CHANNELS", "")
        channel_ids = [c.strip() for c in channels_env.split(",") if c.strip()]

    if not project_key:
        project_key = os.environ.get("SLACK_PROJECT_KEY", "DEFAULT")

    if not channel_ids:
        return {"statusCode": 400, "body": json.dumps({"error": "No channel IDs specified"})}

    results = run_ingestion(channel_ids, project_key, lookback_hours)

    return {
        "statusCode": 200,
        "body": json.dumps(results, default=str),
    }


if __name__ == "__main__":
    import sys

    channels = sys.argv[1:] if len(sys.argv) > 1 else []
    if not channels:
        print("Usage: python ingest_slack.py <channel_id1> <channel_id2> ...")
        sys.exit(1)

    result = run_ingestion(channels, project_key="PROJ")
    print(json.dumps(result, indent=2, default=str))
