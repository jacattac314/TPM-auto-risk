"""Model and data drift monitoring.

Implements data quality checks (inspired by Deequ), feature attribution
drift detection, and automated retraining triggers as specified in
MLA Domain 4 of the PRD.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import boto3
import numpy as np
import pandas as pd

from src.config.settings import AppSettings
from src.features.feature_engineering import get_model_feature_columns

logger = logging.getLogger(__name__)


# ── Data Quality Constraints ──────────────────────────────────────────

# Inspired by Amazon Deequ / SageMaker Data Quality constraints
DATA_QUALITY_CONSTRAINTS = {
    "story_points": {
        "type": "numeric",
        "min": 0,
        "max": 100,
        "completeness": 0.5,  # At least 50% non-null
    },
    "issue_key": {
        "type": "string",
        "uniqueness": 1.0,  # Must be fully unique
        "completeness": 1.0,  # Must be fully non-null
    },
    "status": {
        "type": "categorical",
        "allowed_values": [
            "Open", "To Do", "In Progress", "In Review", "QA",
            "Done", "Closed", "Reopened", "Blocked",
        ],
        "completeness": 0.95,
    },
    "priority": {
        "type": "categorical",
        "allowed_values": ["Highest", "High", "Medium", "Low", "Lowest"],
        "completeness": 0.8,
    },
    "created": {
        "type": "datetime",
        "completeness": 1.0,
    },
}


class DataQualityChecker:
    """Validates ingested data against predefined quality constraints."""

    def __init__(self, constraints: dict[str, dict] | None = None) -> None:
        self._constraints = constraints or DATA_QUALITY_CONSTRAINTS

    def validate(self, df: pd.DataFrame) -> dict[str, Any]:
        """Run all data quality checks on the DataFrame.

        Returns:
            Dict with 'passed', 'violations', and 'metrics'.
        """
        violations: list[dict[str, Any]] = []
        metrics: dict[str, Any] = {}

        for col_name, rules in self._constraints.items():
            if col_name not in df.columns:
                violations.append({
                    "column": col_name,
                    "rule": "column_exists",
                    "message": f"Required column '{col_name}' not found",
                })
                continue

            col = df[col_name]
            col_metrics: dict[str, Any] = {}

            # Completeness check
            completeness = col.notna().mean()
            col_metrics["completeness"] = round(completeness, 4)
            min_completeness = rules.get("completeness", 0)
            if completeness < min_completeness:
                violations.append({
                    "column": col_name,
                    "rule": "completeness",
                    "expected": min_completeness,
                    "actual": round(completeness, 4),
                    "message": f"Completeness {completeness:.1%} < {min_completeness:.1%}",
                })

            # Type-specific checks
            col_type = rules.get("type")
            if col_type == "numeric":
                non_null = pd.to_numeric(col, errors="coerce").dropna()
                if not non_null.empty:
                    col_metrics["mean"] = round(non_null.mean(), 4)
                    col_metrics["std"] = round(non_null.std(), 4)
                    col_metrics["min"] = round(non_null.min(), 4)
                    col_metrics["max"] = round(non_null.max(), 4)

                    if "min" in rules and non_null.min() < rules["min"]:
                        violations.append({
                            "column": col_name,
                            "rule": "min_value",
                            "expected": rules["min"],
                            "actual": non_null.min(),
                            "message": f"Min value {non_null.min()} < {rules['min']}",
                        })
                    if "max" in rules and non_null.max() > rules["max"]:
                        violations.append({
                            "column": col_name,
                            "rule": "max_value",
                            "expected": rules["max"],
                            "actual": non_null.max(),
                            "message": f"Max value {non_null.max()} > {rules['max']}",
                        })

            elif col_type == "categorical":
                allowed = set(rules.get("allowed_values", []))
                if allowed:
                    actual = set(col.dropna().unique())
                    invalid = actual - allowed
                    if invalid:
                        violations.append({
                            "column": col_name,
                            "rule": "allowed_values",
                            "invalid_values": list(invalid)[:10],
                            "message": f"Found {len(invalid)} invalid categorical values",
                        })

            # Uniqueness check
            if "uniqueness" in rules:
                uniqueness = col.dropna().nunique() / max(col.dropna().count(), 1)
                col_metrics["uniqueness"] = round(uniqueness, 4)
                if uniqueness < rules["uniqueness"]:
                    violations.append({
                        "column": col_name,
                        "rule": "uniqueness",
                        "expected": rules["uniqueness"],
                        "actual": round(uniqueness, 4),
                        "message": f"Uniqueness {uniqueness:.1%} < {rules['uniqueness']:.1%}",
                    })

            metrics[col_name] = col_metrics

        result = {
            "passed": len(violations) == 0,
            "total_checks": sum(len(r) for r in self._constraints.values()),
            "violations_count": len(violations),
            "violations": violations,
            "metrics": metrics,
            "row_count": len(df),
            "column_count": len(df.columns),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if violations:
            logger.warning("Data quality check FAILED: %d violations", len(violations))
        else:
            logger.info("Data quality check PASSED")

        return result


class DriftDetector:
    """Detects feature distribution drift and model performance degradation."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._s3 = boto3.client("s3", region_name=settings.aws.region)
        self._threshold_pct = settings.model.drift_threshold_pct

    def compute_baseline_stats(self, df: pd.DataFrame) -> dict[str, dict[str, float]]:
        """Compute baseline statistics for feature columns.

        These stats are stored and compared against future data batches.
        """
        feature_cols = get_model_feature_columns()
        stats: dict[str, dict[str, float]] = {}

        for col in feature_cols:
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if series.empty:
                continue
            stats[col] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "median": float(series.median()),
                "p25": float(series.quantile(0.25)),
                "p75": float(series.quantile(0.75)),
                "min": float(series.min()),
                "max": float(series.max()),
                "count": int(len(series)),
            }

        return stats

    def detect_drift(
        self,
        current_df: pd.DataFrame,
        baseline_stats: dict[str, dict[str, float]],
    ) -> dict[str, Any]:
        """Compare current data distribution against baseline.

        Returns drift report with per-feature drift scores.
        """
        current_stats = self.compute_baseline_stats(current_df)
        drift_results: list[dict[str, Any]] = []
        drifted_features: list[str] = []

        for feature, baseline in baseline_stats.items():
            current = current_stats.get(feature)
            if not current:
                drift_results.append({
                    "feature": feature,
                    "status": "missing",
                    "message": "Feature not present in current data",
                })
                drifted_features.append(feature)
                continue

            # Compare mean shift as percentage of baseline std
            baseline_std = baseline.get("std", 1.0)
            if baseline_std == 0:
                baseline_std = 1.0

            mean_shift = abs(current["mean"] - baseline["mean"]) / baseline_std
            median_shift = abs(current["median"] - baseline["median"]) / baseline_std

            # Detect distribution change via IQR comparison
            baseline_iqr = baseline["p75"] - baseline["p25"]
            current_iqr = current["p75"] - current["p25"]
            iqr_change_pct = (
                abs(current_iqr - baseline_iqr) / max(baseline_iqr, 0.01) * 100
            )

            is_drifted = (
                mean_shift > (self._threshold_pct / 100 * 3)  # 3x threshold
                or iqr_change_pct > self._threshold_pct * 2
            )

            if is_drifted:
                drifted_features.append(feature)

            drift_results.append({
                "feature": feature,
                "status": "drifted" if is_drifted else "stable",
                "mean_shift_std": round(mean_shift, 4),
                "median_shift_std": round(median_shift, 4),
                "iqr_change_pct": round(iqr_change_pct, 2),
                "baseline_mean": baseline["mean"],
                "current_mean": current["mean"],
            })

        overall_drift_pct = len(drifted_features) / max(len(baseline_stats), 1) * 100
        needs_retrain = overall_drift_pct > self._threshold_pct

        report = {
            "overall_drift_pct": round(overall_drift_pct, 2),
            "needs_retrain": needs_retrain,
            "drifted_features": drifted_features,
            "total_features": len(baseline_stats),
            "drifted_count": len(drifted_features),
            "feature_details": drift_results,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if needs_retrain:
            logger.warning(
                "Drift detected: %.1f%% of features drifted (threshold: %.1f%%)",
                overall_drift_pct, self._threshold_pct,
            )
        else:
            logger.info("No significant drift detected (%.1f%%)", overall_drift_pct)

        return report

    def save_baseline(self, stats: dict[str, dict[str, float]], version: str = "latest") -> None:
        """Persist baseline stats to S3."""
        key = f"models/baseline_stats/{version}/stats.json"
        self._s3.put_object(
            Bucket=self._settings.aws.s3_bucket,
            Key=key,
            Body=json.dumps(stats, indent=2).encode(),
        )
        logger.info("Saved baseline stats to s3://%s/%s", self._settings.aws.s3_bucket, key)

    def load_baseline(self, version: str = "latest") -> dict[str, dict[str, float]]:
        """Load baseline stats from S3."""
        key = f"models/baseline_stats/{version}/stats.json"
        resp = self._s3.get_object(Bucket=self._settings.aws.s3_bucket, Key=key)
        return json.loads(resp["Body"].read())


class AlertPublisher:
    """Publishes alerts via SNS when quality checks fail or drift is detected."""

    def __init__(self, settings: AppSettings) -> None:
        self._sns = boto3.client("sns", region_name=settings.aws.region)
        self._topic_arn = settings.aws.sns_topic_arn

    def publish_quality_alert(self, quality_report: dict[str, Any], project_key: str) -> None:
        """Send an SNS alert for data quality violations."""
        if not self._topic_arn:
            logger.warning("SNS topic ARN not configured, skipping alert")
            return

        message = {
            "alert_type": "DATA_QUALITY_VIOLATION",
            "project_key": project_key,
            "violations_count": quality_report["violations_count"],
            "violations": quality_report["violations"][:10],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self._sns.publish(
            TopicArn=self._topic_arn,
            Subject=f"[Auto-TPM] Data Quality Alert - {project_key}",
            Message=json.dumps(message, indent=2),
        )

    def publish_drift_alert(self, drift_report: dict[str, Any]) -> None:
        """Send an SNS alert when model drift is detected."""
        if not self._topic_arn:
            logger.warning("SNS topic ARN not configured, skipping alert")
            return

        message = {
            "alert_type": "MODEL_DRIFT_DETECTED",
            "overall_drift_pct": drift_report["overall_drift_pct"],
            "drifted_features": drift_report["drifted_features"],
            "recommendation": "Automated retraining recommended",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self._sns.publish(
            TopicArn=self._topic_arn,
            Subject="[Auto-TPM] Model Drift Alert - Retraining Recommended",
            Message=json.dumps(message, indent=2),
            MessageAttributes={
                "alert_type": {
                    "DataType": "String",
                    "StringValue": "MODEL_DRIFT_DETECTED",
                },
            },
        )
