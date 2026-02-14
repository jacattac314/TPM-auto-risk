"""Slack data ingestion with tiered sentiment analysis.

Tier 1: Lightweight VADER sentiment scoring on every message.
Tier 2: LLM-based contextual analysis triggered on anomaly detection.
"""

from __future__ import annotations

import logging
import statistics
from datetime import datetime, timezone
from typing import Any

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.config.settings import AppSettings

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Tier 1 sentiment analysis using VADER, optimized for engineering lexicon."""

    def __init__(self) -> None:
        self._analyzer = SentimentIntensityAnalyzer()
        # Augment VADER lexicon with engineering-specific terms
        engineering_lexicon = {
            "blocked": -2.5,
            "blocker": -2.5,
            "regression": -2.0,
            "broken": -2.0,
            "outage": -3.0,
            "incident": -2.0,
            "hotfix": -1.5,
            "rollback": -1.5,
            "tech debt": -1.0,
            "workaround": -0.5,
            "hack": -1.0,
            "scope creep": -1.5,
            "shipped": 2.0,
            "deployed": 1.5,
            "merged": 1.0,
            "resolved": 2.0,
            "unblocked": 2.5,
            "milestone": 1.5,
            "on track": 2.0,
            "lgtm": 1.5,
            "approved": 1.5,
        }
        self._analyzer.lexicon.update(engineering_lexicon)

    def score(self, text: str) -> float:
        """Return compound sentiment score in range [-1.0, +1.0]."""
        scores = self._analyzer.polarity_scores(text)
        return scores["compound"]


class SlackChannelCrawler:
    """Crawls a Slack channel and produces sentiment-annotated message records."""

    def __init__(self, settings: AppSettings) -> None:
        self._client = WebClient(token=settings.slack.bot_token)
        self._sentiment = SentimentAnalyzer()
        self._anomaly_threshold = settings.slack.sentiment_anomaly_threshold

    def crawl_channel(
        self,
        channel_id: str,
        oldest: str | None = None,
        latest: str | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch messages from a channel and annotate with sentiment scores.

        Args:
            channel_id: Slack channel ID.
            oldest: Unix timestamp string for the start of the window.
            latest: Unix timestamp string for the end of the window.

        Returns:
            List of message records with sentiment annotations.
        """
        messages: list[dict[str, Any]] = []
        cursor = None

        while True:
            try:
                kwargs: dict[str, Any] = {
                    "channel": channel_id,
                    "limit": 200,
                }
                if oldest:
                    kwargs["oldest"] = oldest
                if latest:
                    kwargs["latest"] = latest
                if cursor:
                    kwargs["cursor"] = cursor

                resp = self._client.conversations_history(**kwargs)
            except SlackApiError as e:
                logger.error("Slack API error for channel %s: %s", channel_id, e.response["error"])
                break

            for msg in resp.get("messages", []):
                if msg.get("subtype") in ("channel_join", "channel_leave", "bot_message"):
                    continue

                text = msg.get("text", "")
                if not text.strip():
                    continue

                sentiment = self._sentiment.score(text)
                ts = msg.get("ts", "")

                messages.append({
                    "channel_id": channel_id,
                    "message_ts": ts,
                    "user_id": msg.get("user"),
                    "text": text,
                    "sentiment_score": sentiment,
                    "thread_ts": msg.get("thread_ts"),
                    "reply_count": msg.get("reply_count", 0),
                    "reaction_count": sum(
                        r.get("count", 0) for r in msg.get("reactions", [])
                    ),
                    "datetime_utc": datetime.fromtimestamp(
                        float(ts), tz=timezone.utc
                    ).isoformat()
                    if ts
                    else None,
                    "ingested_at": datetime.now(timezone.utc).isoformat(),
                })

            if not resp.get("has_more"):
                break
            cursor = resp.get("response_metadata", {}).get("next_cursor")

        logger.info(
            "Crawled %d messages from channel %s", len(messages), channel_id
        )
        return messages


def compute_sentiment_timeseries(
    messages: list[dict[str, Any]],
    window_hours: int = 24,
) -> list[dict[str, Any]]:
    """Aggregate per-message sentiment into time-bucketed averages.

    Returns a list of dicts with keys: window_start, window_end,
    avg_sentiment, message_count, std_dev.
    """
    if not messages:
        return []

    sorted_msgs = sorted(messages, key=lambda m: m.get("message_ts", "0"))
    buckets: dict[str, list[float]] = {}

    for msg in sorted_msgs:
        ts_str = msg.get("message_ts", "0")
        try:
            ts = float(ts_str)
        except (ValueError, TypeError):
            continue
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        bucket_key = dt.strftime("%Y-%m-%dT%H:00:00Z") if window_hours <= 1 else dt.strftime(
            "%Y-%m-%dT00:00:00Z"
        )
        buckets.setdefault(bucket_key, []).append(msg["sentiment_score"])

    timeseries = []
    for window_start, scores in sorted(buckets.items()):
        avg = statistics.mean(scores)
        std = statistics.stdev(scores) if len(scores) > 1 else 0.0
        timeseries.append({
            "window_start": window_start,
            "avg_sentiment": round(avg, 4),
            "message_count": len(scores),
            "std_dev": round(std, 4),
        })

    return timeseries


def detect_sentiment_anomalies(
    timeseries: list[dict[str, Any]],
    threshold_std_devs: float = 2.0,
) -> list[dict[str, Any]]:
    """Identify time windows where sentiment drops significantly.

    An anomaly is flagged when the average sentiment in a window deviates
    by more than `threshold_std_devs` standard deviations below the
    overall mean. These anomalies trigger Tier 2 LLM analysis.
    """
    if len(timeseries) < 3:
        return []

    all_scores = [t["avg_sentiment"] for t in timeseries]
    overall_mean = statistics.mean(all_scores)
    overall_std = statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0

    if overall_std == 0:
        return []

    anomalies = []
    for window in timeseries:
        z_score = (window["avg_sentiment"] - overall_mean) / overall_std
        if z_score < -threshold_std_devs:
            anomalies.append({
                **window,
                "z_score": round(z_score, 4),
                "anomaly_type": "negative_sentiment_drop",
            })

    logger.info("Detected %d sentiment anomalies (threshold=%.1f sigma)", len(anomalies), threshold_std_devs)
    return anomalies
