"""Training pipeline orchestration for the risk model.

Implements the end-to-end workflow: data loading, feature engineering,
model training, evaluation, and registration. Supports both local
execution and SageMaker Pipeline orchestration.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.config.settings import AppSettings, get_settings
from src.datalake.storage import DataLakeWriter
from src.features.feature_engineering import (
    compute_team_features,
    engineer_jira_features,
    merge_sentiment_features,
    prepare_model_input,
)
from src.models.risk_model import RiskModel

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """End-to-end training pipeline for the risk prediction model."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model = RiskModel(self._settings.model)

    def run(
        self,
        project_keys: list[str],
        sentiment_data: dict[str, list[dict[str, Any]]] | None = None,
    ) -> dict[str, Any]:
        """Execute the full training pipeline.

        Args:
            project_keys: List of Jira project keys to train on.
            sentiment_data: Optional dict mapping project_key -> sentiment timeseries.

        Returns:
            Training results including metrics and model metadata.
        """
        logger.info("Starting training pipeline for projects: %s", project_keys)

        # Step 1: Load data from Silver layer
        datalake = DataLakeWriter(self._settings)
        all_tickets: list[pd.DataFrame] = []

        for project_key in project_keys:
            df = datalake.read_silver("jira", project_key)
            if not df.empty:
                all_tickets.append(df)
                logger.info("Loaded %d Silver records for %s", len(df), project_key)

        if not all_tickets:
            raise ValueError("No training data available in the Silver layer")

        raw_df = pd.concat(all_tickets, ignore_index=True)

        # Step 2: Engineer features
        tickets = raw_df.to_dict("records")
        feature_df = engineer_jira_features(tickets)
        feature_df = compute_team_features(feature_df)

        # Step 3: Merge sentiment features
        if sentiment_data:
            for project_key, sent_ts in sentiment_data.items():
                feature_df = merge_sentiment_features(feature_df, sent_ts, project_key)
        else:
            feature_df["avg_sentiment"] = 0.0
            feature_df["sentiment_velocity"] = 0.0
            feature_df["sentiment_anomaly"] = 0

        # Step 4: Compute target variable
        feature_df = _compute_target(feature_df)

        # Step 5: Write features to Gold layer
        for project_key in project_keys:
            project_df = feature_df[feature_df["project_key"] == project_key]
            if not project_df.empty:
                datalake.write_gold(project_df, project_key)

        # Step 6: Train model
        training_metrics = self._model.train(feature_df, target_col="is_delayed")

        # Step 7: Save model
        version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        s3_uri = self._model.save_to_s3(self._settings, model_version=version)
        # Also save as "latest"
        self._model.save_to_s3(self._settings, model_version="latest")

        result = {
            "status": "success",
            "model_version": version,
            "model_s3_uri": s3_uri,
            "training_samples": len(feature_df),
            "metrics": training_metrics.get("avg_metrics", {}),
            "feature_importance": training_metrics.get("feature_importance", {}),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info("Training pipeline complete: %s", json.dumps(result, default=str))
        return result


def _compute_target(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the binary target variable `is_delayed`.

    A ticket is considered delayed if:
    1. It was resolved after its due date, OR
    2. It was rolled over to a subsequent sprint (sprint_rollover_count > 0), OR
    3. It has been in progress for longer than typical (days_in_status > median * 2)

    For tickets without resolution or due dates, sprint rollover is the primary signal.
    """
    df["is_delayed"] = 0

    # Condition 1: Resolved after due date
    has_both = df["resolution_date"].notna() & df["due_date"].notna()
    if has_both.any():
        df.loc[has_both, "is_delayed"] = (
            df.loc[has_both, "resolution_date"] > df.loc[has_both, "due_date"]
        ).astype(int)

    # Condition 2: Sprint rollover
    df.loc[df["sprint_rollover_count"] > 0, "is_delayed"] = 1

    # Condition 3: Stalled tickets (in progress > 2x median)
    in_progress = df["status_category"] == "In Progress"
    if in_progress.any():
        median_days = df.loc[in_progress, "days_in_status"].median()
        if pd.notna(median_days) and median_days > 0:
            stalled = in_progress & (df["days_in_status"] > median_days * 2)
            df.loc[stalled, "is_delayed"] = 1

    positive_rate = df["is_delayed"].mean()
    logger.info(
        "Target computed: %d delayed / %d total (%.1f%%)",
        df["is_delayed"].sum(), len(df), positive_rate * 100,
    )

    return df
