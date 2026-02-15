"""Feature engineering for risk prediction.

Transforms raw Jira and Slack data into the engineered features
specified in the PRD (Table 1), including complexity_per_person,
has_dependency, sentiment_velocity, priority_interaction, days_in_status,
description_ambiguity, and ticket_churn.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Priority levels mapped to numeric values for interaction features
PRIORITY_MAP = {
    "Highest": 5,
    "High": 4,
    "Medium": 3,
    "Low": 2,
    "Lowest": 1,
}


def engineer_jira_features(tickets: list[dict[str, Any]]) -> pd.DataFrame:
    """Transform raw Jira ticket records into ML-ready features.

    Args:
        tickets: List of flattened Jira ticket dicts from the ingestion layer.

    Returns:
        DataFrame with one row per ticket and all engineered features.
    """
    if not tickets:
        return pd.DataFrame()

    df = pd.DataFrame(tickets)

    # Parse dates
    for col in ["created", "updated", "due_date", "resolution_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    now = pd.Timestamp.now(tz="UTC")

    # ---- Feature: story_points (clean) ----
    df["story_points"] = pd.to_numeric(df.get("story_points"), errors="coerce").fillna(0)

    # ---- Feature: has_dependency ----
    df["has_dependency"] = df["issue_links"].apply(
        lambda links: 1 if isinstance(links, list) and len(links) > 0 else 0
    )
    df["dependency_count"] = df["issue_links"].apply(
        lambda links: len(links) if isinstance(links, list) else 0
    )

    # ---- Feature: priority_numeric ----
    df["priority_numeric"] = df["priority"].map(PRIORITY_MAP).fillna(3).astype(int)

    # ---- Feature: priority_interaction ----
    # High-priority * high-complexity = compounding risk
    df["priority_interaction"] = df["priority_numeric"] * df["story_points"]

    # ---- Feature: days_in_status ----
    # Approximated from the 'updated' field; ideally computed from changelog
    df["days_in_status"] = (now - df["updated"]).dt.total_seconds() / 86400
    df["days_in_status"] = df["days_in_status"].clip(lower=0).fillna(0).round(1)

    # ---- Feature: days_since_created ----
    df["days_since_created"] = (now - df["created"]).dt.total_seconds() / 86400
    df["days_since_created"] = df["days_since_created"].clip(lower=0).fillna(0).round(1)

    # ---- Feature: is_overdue ----
    df["is_overdue"] = 0
    if "due_date" in df.columns:
        has_due = df["due_date"].notna()
        df.loc[has_due, "is_overdue"] = (now > df.loc[has_due, "due_date"]).astype(int)

    # ---- Feature: description_ambiguity ----
    df["description_word_count"] = df["description"].apply(
        lambda d: len(str(d).split()) if pd.notna(d) and d else 0
    )
    # Short descriptions (< 10 words) or missing => ambiguous
    # Very long descriptions (> 500 words) => possible scope creep
    df["description_ambiguity"] = df["description_word_count"].apply(
        lambda wc: 1.0 if wc < 10 else (0.5 if wc > 500 else 0.0)
    )

    # ---- Feature: sprint_rollover_count ----
    df["sprint_rollover_count"] = df["sprint_history"].apply(_count_sprint_rollovers)

    # ---- Feature: current_sprint_name ----
    df["current_sprint_name"] = df["sprint_history"].apply(_get_current_sprint_name)

    # ---- Feature: ticket_churn (status changes) ----
    # Estimated from sprint rollovers and comment activity as a proxy
    # In production, this would use the Jira changelog
    df["ticket_churn"] = df["sprint_rollover_count"] + (df["comment_count"] / 5).astype(int)

    # ---- Feature: assignee_churn ----
    # Counts the number of unique assignees a ticket has had.  High churn
    # signals coordination overhead, unclear ownership, or scope mismatch.
    # In production this comes from the Jira changelog; here we approximate
    # it from comment activity and sprint rollovers as a proxy.
    df["assignee_churn"] = (df["sprint_rollover_count"] > 0).astype(int) + (
        (df["comment_count"] > 10).astype(int)
    )

    # ---- Feature: blocker_dependency_ratio ----
    # Fraction of a ticket's dependencies that are "Blocks" type (vs.
    # "Relates" or "Duplicates").  A high ratio means the ticket has hard
    # upstream/downstream coupling that can cascade delays.
    df["blocker_dependency_ratio"] = df["issue_links"].apply(_compute_blocker_ratio)

    # ---- Feature: issue_type_risk ----
    issue_type_risk = {
        "Bug": 0.7,
        "Story": 0.3,
        "Task": 0.2,
        "Epic": 0.5,
        "Sub-task": 0.1,
        "Spike": 0.4,
    }
    df["issue_type_risk"] = df["issue_type"].map(issue_type_risk).fillna(0.3)

    # ---- Feature: status_category_encoded ----
    status_map = {"To Do": 0, "In Progress": 1, "Done": 2}
    df["status_category_encoded"] = df["status_category"].map(status_map).fillna(0).astype(int)

    logger.info("Engineered %d features for %d tickets", _count_feature_cols(df), len(df))
    return df


def compute_team_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add team-level aggregate features to the ticket DataFrame.

    These features capture resource saturation and team dynamics.

    Args:
        df: DataFrame of ticket-level features (output of engineer_jira_features).

    Returns:
        DataFrame with additional team-level columns.
    """
    if df.empty:
        return df

    # ---- Feature: complexity_per_person ----
    # Group by project to compute team-level metrics
    project_groups = df.groupby("project_key")

    team_size = project_groups["assignee"].transform("nunique").clip(lower=1)
    total_points = project_groups["story_points"].transform("sum")
    df["complexity_per_person"] = (total_points / team_size).round(2)

    # ---- Feature: team_wip (work in progress) ----
    in_progress = df["status_category"] == "In Progress"
    wip_counts = df[in_progress].groupby("project_key")["issue_key"].transform("count")
    df["team_wip"] = 0
    df.loc[in_progress, "team_wip"] = wip_counts
    # Backfill for non-in-progress tickets
    project_wip = df[in_progress].groupby("project_key")["team_wip"].first()
    df["team_wip"] = df["project_key"].map(project_wip).fillna(0).astype(int)

    # ---- Feature: assignee_load ----
    df["assignee_load"] = df.groupby("assignee")["story_points"].transform("sum").fillna(0)

    return df


def merge_sentiment_features(
    ticket_df: pd.DataFrame,
    sentiment_data: list[dict[str, Any]],
    project_key: str,
) -> pd.DataFrame:
    """Merge Slack sentiment features into the ticket DataFrame.

    Args:
        ticket_df: Ticket-level features DataFrame.
        sentiment_data: Sentiment timeseries from Slack analysis.
        project_key: The project these sentiment signals correspond to.

    Returns:
        DataFrame with sentiment columns added.
    """
    if not sentiment_data or ticket_df.empty:
        ticket_df["avg_sentiment"] = 0.0
        ticket_df["sentiment_velocity"] = 0.0
        ticket_df["sentiment_anomaly"] = 0
        return ticket_df

    sent_df = pd.DataFrame(sentiment_data)

    if len(sent_df) >= 2:
        current_avg = sent_df["avg_sentiment"].iloc[-1]
        previous_avg = sent_df["avg_sentiment"].iloc[-2]
        sentiment_velocity = current_avg - previous_avg
    else:
        current_avg = sent_df["avg_sentiment"].mean()
        sentiment_velocity = 0.0

    # Apply project-level sentiment to all tickets in that project
    mask = ticket_df["project_key"] == project_key
    ticket_df.loc[mask, "avg_sentiment"] = current_avg
    ticket_df.loc[mask, "sentiment_velocity"] = sentiment_velocity
    ticket_df.loc[mask, "sentiment_anomaly"] = int(sentiment_velocity < -0.3)

    # Fill NaN for other projects
    for col in ["avg_sentiment", "sentiment_velocity", "sentiment_anomaly"]:
        ticket_df[col] = ticket_df[col].fillna(0.0)

    return ticket_df


def get_model_feature_columns() -> list[str]:
    """Return the ordered list of feature columns expected by the XGBoost model."""
    return [
        "story_points",
        "has_dependency",
        "dependency_count",
        "priority_numeric",
        "priority_interaction",
        "days_in_status",
        "days_since_created",
        "is_overdue",
        "description_ambiguity",
        "description_word_count",
        "sprint_rollover_count",
        "ticket_churn",
        "assignee_churn",
        "blocker_dependency_ratio",
        "issue_type_risk",
        "status_category_encoded",
        "complexity_per_person",
        "team_wip",
        "assignee_load",
        "avg_sentiment",
        "sentiment_velocity",
        "sentiment_anomaly",
        "comment_count",
    ]


def prepare_model_input(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and validate the feature columns for model input.

    Returns a DataFrame containing only the model feature columns,
    with missing values handled.
    """
    feature_cols = get_model_feature_columns()
    missing_cols = set(feature_cols) - set(df.columns)
    for col in missing_cols:
        df[col] = 0

    model_df = df[feature_cols].copy()
    model_df = model_df.fillna(0)

    # Ensure all columns are numeric
    for col in model_df.columns:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce").fillna(0)

    return model_df


def _count_sprint_rollovers(sprint_history: Any) -> int:
    """Count how many sprints a ticket has been rolled over into."""
    if not isinstance(sprint_history, list):
        return 0
    # A ticket in multiple sprints indicates rollover
    return max(0, len(sprint_history) - 1)


def _get_current_sprint_name(sprint_history: Any) -> str | None:
    """Extract the most recent sprint name from sprint history."""
    if not isinstance(sprint_history, list) or not sprint_history:
        return None
    # The last entry in the list is typically the current/most recent sprint
    last = sprint_history[-1]
    if isinstance(last, dict):
        return last.get("sprint_name")
    return None


def _compute_blocker_ratio(links: Any) -> float:
    """Compute the fraction of issue links that are blocker-type dependencies.

    Returns 0.0 when there are no links.
    """
    if not isinstance(links, list) or not links:
        return 0.0
    blocker_count = sum(
        1 for link in links
        if isinstance(link, dict) and link.get("type", "").lower() in ("blocks", "is blocked by")
    )
    return round(blocker_count / len(links), 2)


def _count_feature_cols(df: pd.DataFrame) -> int:
    """Count columns that are engineered features (not metadata)."""
    metadata_cols = {
        "issue_id", "issue_key", "project_key", "summary", "description",
        "status", "assignee", "reporter", "created", "updated",
        "due_date", "resolution_date", "labels", "components",
        "sprint_history", "issue_links", "ingested_at", "issue_type",
        "status_category", "priority", "current_sprint_name", "_pii_redaction",
    }
    return len(set(df.columns) - metadata_cols)
