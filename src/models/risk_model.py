"""XGBoost-based risk prediction model for sprint slippage.

Provides training, evaluation, inference, and model persistence
with time-series cross-validation (walk-forward) strategy.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

from src.config.settings import AppSettings, ModelSettings
from src.features.feature_engineering import get_model_feature_columns, prepare_model_input

logger = logging.getLogger(__name__)


class RiskModel:
    """XGBoost binary classifier predicting ticket-level sprint slippage."""

    def __init__(self, settings: ModelSettings | None = None) -> None:
        self._settings = settings or ModelSettings()
        self._model: xgb.XGBClassifier | None = None
        self._feature_columns = get_model_feature_columns()
        self._metadata: dict[str, Any] = {}

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def train(
        self,
        df: pd.DataFrame,
        target_col: str = "is_delayed",
        date_col: str = "created",
    ) -> dict[str, Any]:
        """Train the XGBoost model using walk-forward time-series CV.

        Args:
            df: DataFrame with feature columns and target variable.
            target_col: Name of the binary target column.
            date_col: Date column for time-series ordering.

        Returns:
            Dict of training metrics (recall, precision, f1, auc per fold).
        """
        # Sort by date for proper time-series splitting
        df = df.sort_values(date_col).reset_index(drop=True)

        X = prepare_model_input(df)
        y = df[target_col].astype(int)

        # Walk-forward cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        fold_metrics: list[dict[str, float]] = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = self._build_model(y_train)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            metrics = {
                "fold": fold,
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "auc_roc": roc_auc_score(y_test, y_proba) if len(y_test.unique()) > 1 else 0.0,
                "test_size": len(y_test),
                "positive_rate": y_test.mean(),
            }
            fold_metrics.append(metrics)
            logger.info("Fold %d - Recall: %.3f, Precision: %.3f, AUC: %.3f",
                        fold, metrics["recall"], metrics["precision"], metrics["auc_roc"])

        # Train final model on full dataset
        self._model = self._build_model(y)
        self._model.fit(X, y, verbose=False)

        # Store metadata
        avg_metrics = {
            k: np.mean([m[k] for m in fold_metrics])
            for k in ["recall", "precision", "f1", "auc_roc"]
        }
        self._metadata = {
            "feature_columns": self._feature_columns,
            "fold_metrics": fold_metrics,
            "avg_metrics": avg_metrics,
            "training_samples": len(X),
            "positive_samples": int(y.sum()),
            "negative_samples": int((1 - y).sum()),
            "feature_importance": self.get_feature_importance(),
        }

        logger.info(
            "Training complete - Avg Recall: %.3f, Avg AUC: %.3f",
            avg_metrics["recall"], avg_metrics["auc_roc"],
        )
        return self._metadata

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate risk predictions for tickets.

        Args:
            df: DataFrame with feature columns.

        Returns:
            Original DataFrame with added columns:
                - risk_score: Probability of delay [0, 1]
                - risk_level: Categorical (High/Medium/Low)
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        X = prepare_model_input(df)
        probabilities = self._model.predict_proba(X)[:, 1]

        result = df.copy()
        result["risk_score"] = np.round(probabilities, 4)
        result["risk_level"] = pd.cut(
            probabilities,
            bins=[-0.01, self._settings.risk_threshold_medium,
                  self._settings.risk_threshold_high, 1.01],
            labels=["Low", "Medium", "High"],
        )

        return result

    def predict_single(self, features: dict[str, Any]) -> dict[str, Any]:
        """Predict risk for a single ticket.

        Args:
            features: Dict of feature name -> value.

        Returns:
            Dict with risk_score, risk_level, and top contributing features.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        df = pd.DataFrame([features])
        X = prepare_model_input(df)
        probability = self._model.predict_proba(X)[0, 1]

        # Determine risk level
        if probability >= self._settings.risk_threshold_high:
            risk_level = "High"
        elif probability >= self._settings.risk_threshold_medium:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        # Get top contributing features using SHAP-like importance
        feature_contributions = self._get_feature_contributions(X.iloc[0])

        return {
            "risk_score": round(float(probability), 4),
            "risk_level": risk_level,
            "top_risk_factors": feature_contributions[:5],
        }

    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importance scores (gain-based)."""
        if not self.is_trained:
            return {}

        importance = self._model.get_booster().get_score(importance_type="gain")
        # Map feature indices back to names
        named_importance = {}
        for key, value in importance.items():
            idx = int(key.replace("f", "")) if key.startswith("f") else None
            if idx is not None and idx < len(self._feature_columns):
                named_importance[self._feature_columns[idx]] = round(value, 4)
            else:
                named_importance[key] = round(value, 4)

        return dict(sorted(named_importance.items(), key=lambda x: x[1], reverse=True))

    def save(self, path: str | Path) -> None:
        """Save model and metadata to local path."""
        if not self.is_trained:
            raise RuntimeError("No trained model to save")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self._model.save_model(str(path / "model.json"))

        with open(path / "metadata.json", "w") as f:
            json.dump(self._metadata, f, indent=2, default=str)

        logger.info("Model saved to %s", path)

    def load(self, path: str | Path) -> None:
        """Load model and metadata from local path."""
        path = Path(path)

        self._model = xgb.XGBClassifier()
        self._model.load_model(str(path / "model.json"))

        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self._metadata = json.load(f)

        logger.info("Model loaded from %s", path)

    def save_to_s3(self, settings: AppSettings, model_version: str = "latest") -> str:
        """Save model artifacts to S3.

        Returns:
            S3 URI of the saved model.
        """
        if not self.is_trained:
            raise RuntimeError("No trained model to save")

        s3 = boto3.client("s3", region_name=settings.aws.region)
        prefix = f"models/risk_model/{model_version}"

        # Save model
        model_buffer = io.BytesIO()
        self._model.save_model(model_buffer)
        model_buffer.seek(0)
        s3.put_object(
            Bucket=settings.aws.s3_bucket,
            Key=f"{prefix}/model.json",
            Body=model_buffer.getvalue(),
        )

        # Save metadata
        s3.put_object(
            Bucket=settings.aws.s3_bucket,
            Key=f"{prefix}/metadata.json",
            Body=json.dumps(self._metadata, indent=2, default=str).encode(),
        )

        s3_uri = f"s3://{settings.aws.s3_bucket}/{prefix}/"
        logger.info("Model saved to %s", s3_uri)
        return s3_uri

    def load_from_s3(self, settings: AppSettings, model_version: str = "latest") -> None:
        """Load model artifacts from S3."""
        s3 = boto3.client("s3", region_name=settings.aws.region)
        prefix = f"models/risk_model/{model_version}"

        # Load model
        resp = s3.get_object(Bucket=settings.aws.s3_bucket, Key=f"{prefix}/model.json")
        model_bytes = resp["Body"].read()
        self._model = xgb.XGBClassifier()
        self._model.load_model(bytearray(model_bytes))

        # Load metadata
        try:
            resp = s3.get_object(Bucket=settings.aws.s3_bucket, Key=f"{prefix}/metadata.json")
            self._metadata = json.loads(resp["Body"].read())
        except s3.exceptions.NoSuchKey:
            logger.warning("No metadata found for model version %s", model_version)

        logger.info("Model loaded from S3 version: %s", model_version)

    def _build_model(self, y_train: pd.Series | None = None) -> xgb.XGBClassifier:
        """Construct an XGBoost classifier with configured hyperparameters."""
        # Compute scale_pos_weight for class imbalance
        scale_pos_weight = self._settings.xgboost_scale_pos_weight
        if y_train is not None:
            n_pos = y_train.sum()
            n_neg = len(y_train) - n_pos
            if n_pos > 0:
                scale_pos_weight = n_neg / n_pos

        return xgb.XGBClassifier(
            max_depth=self._settings.xgboost_max_depth,
            learning_rate=self._settings.xgboost_learning_rate,
            n_estimators=self._settings.xgboost_n_estimators,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            random_state=42,
            enable_categorical=False,
        )

    def _get_feature_contributions(self, features: pd.Series) -> list[dict[str, Any]]:
        """Get per-feature contribution for a single prediction."""
        importance = self.get_feature_importance()
        contributions = []
        for feat_name, imp_score in importance.items():
            feat_value = features.get(feat_name, 0)
            contributions.append({
                "feature": feat_name,
                "value": float(feat_value),
                "importance": imp_score,
            })

        return sorted(contributions, key=lambda x: x["importance"], reverse=True)
