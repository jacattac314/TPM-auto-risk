"""Model training script / Lambda handler.

Can be invoked as:
- A standalone script: python scripts/train_model.py PROJ1 PROJ2
- A Lambda function triggered by schedule or drift detection
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

from src.config.settings import get_settings
from src.models.training_pipeline import TrainingPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _extract_project_keys_from_sns(event: dict) -> list[str]:
    """Extract project keys from an SNS drift alert event.

    SNS-triggered Lambda events wrap the message in Records[].Sns.Message.
    """
    for record in event.get("Records", []):
        sns_msg = record.get("Sns", {}).get("Message", "")
        try:
            payload = json.loads(sns_msg)
        except (json.JSONDecodeError, TypeError):
            continue
        if payload.get("alert_type") == "MODEL_DRIFT_DETECTED":
            logger.info(
                "Drift-triggered retraining: %.1f%% drift detected",
                payload.get("overall_drift_pct", 0),
            )
            # Retrain all configured projects on drift
            keys_env = os.environ.get("TRAINING_PROJECT_KEYS", "")
            return [k.strip() for k in keys_env.split(",") if k.strip()]
    return []


def lambda_handler(event: dict, context: Any) -> dict:
    """AWS Lambda entry point for model training.

    Supports direct invocation (event.project_keys) and SNS-triggered
    invocation from drift alerts.
    """
    project_keys = event.get("project_keys", [])

    # Check for SNS drift alert trigger
    if not project_keys and "Records" in event:
        project_keys = _extract_project_keys_from_sns(event)

    if not project_keys:
        keys_env = os.environ.get("TRAINING_PROJECT_KEYS", "")
        project_keys = [k.strip() for k in keys_env.split(",") if k.strip()]

    if not project_keys:
        return {"statusCode": 400, "body": json.dumps({"error": "No project keys specified"})}

    settings = get_settings()
    pipeline = TrainingPipeline(settings)

    try:
        result = pipeline.run(project_keys)
        return {"statusCode": 200, "body": json.dumps(result, default=str)}
    except Exception as e:
        logger.exception("Training pipeline failed")
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}


if __name__ == "__main__":
    keys = sys.argv[1:] if len(sys.argv) > 1 else ["PROJ"]
    settings = get_settings()
    pipeline = TrainingPipeline(settings)
    result = pipeline.run(keys)
    print(json.dumps(result, indent=2, default=str))
