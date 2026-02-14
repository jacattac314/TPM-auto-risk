"""Application configuration using Pydantic settings management."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class AWSSettings(BaseSettings):
    """AWS service configuration."""

    region: str = Field(default="us-east-1", alias="AWS_REGION")
    s3_bucket: str = Field(default="auto-tpm-risk-radar", alias="S3_BUCKET")
    s3_bronze_prefix: str = "bronze"
    s3_silver_prefix: str = "silver"
    s3_gold_prefix: str = "gold"
    dynamodb_table: str = Field(default="tpm-ingestion-state", alias="DYNAMODB_TABLE")
    sagemaker_endpoint: str = Field(
        default="risk-radar-xgboost-endpoint", alias="SAGEMAKER_ENDPOINT"
    )
    bedrock_model_id: str = Field(
        default="anthropic.claude-3-sonnet-20240229-v1:0", alias="BEDROCK_MODEL_ID"
    )
    bedrock_haiku_model_id: str = Field(
        default="anthropic.claude-3-haiku-20240307-v1:0", alias="BEDROCK_HAIKU_MODEL_ID"
    )
    bedrock_embedding_model_id: str = Field(
        default="amazon.titan-embed-text-v1", alias="BEDROCK_EMBEDDING_MODEL_ID"
    )
    comprehend_language_code: str = "en"
    kms_key_id: str = Field(default="", alias="KMS_KEY_ID")
    sns_topic_arn: str = Field(default="", alias="SNS_TOPIC_ARN")


class JiraSettings(BaseSettings):
    """Jira integration configuration."""

    base_url: str = Field(default="", alias="JIRA_BASE_URL")
    consumer_key: str = Field(default="", alias="JIRA_CONSUMER_KEY")
    access_token: str = Field(default="", alias="JIRA_ACCESS_TOKEN")
    access_token_secret: str = Field(default="", alias="JIRA_ACCESS_TOKEN_SECRET")
    private_key_ssm_param: str = Field(
        default="jira_access_private_key", alias="JIRA_PRIVATE_KEY_SSM_PARAM"
    )
    page_size: int = Field(default=100, alias="JIRA_PAGE_SIZE")
    story_points_field_name: str = Field(
        default="Story Points", alias="JIRA_STORY_POINTS_FIELD_NAME"
    )
    sprint_field_name: str = Field(default="Sprint", alias="JIRA_SPRINT_FIELD_NAME")


class SlackSettings(BaseSettings):
    """Slack integration configuration."""

    bot_token: str = Field(default="", alias="SLACK_BOT_TOKEN")
    channels: list[str] = Field(default_factory=list, alias="SLACK_CHANNELS")
    sentiment_anomaly_threshold: float = Field(
        default=2.0, alias="SLACK_SENTIMENT_ANOMALY_THRESHOLD"
    )


class ModelSettings(BaseSettings):
    """ML model configuration."""

    target_recall: float = 0.80
    risk_threshold_high: float = 0.7
    risk_threshold_medium: float = 0.4
    xgboost_max_depth: int = 6
    xgboost_learning_rate: float = 0.1
    xgboost_n_estimators: int = 200
    xgboost_scale_pos_weight: float = 2.0
    drift_threshold_pct: float = 10.0
    retrain_schedule_days: int = 30


class AppSettings(BaseSettings):
    """Top-level application settings."""

    model_config = {"extra": "ignore"}

    app_name: str = "Auto-TPM Risk Radar"
    debug: bool = Field(default=False, alias="DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    aws: AWSSettings = AWSSettings()
    jira: JiraSettings = JiraSettings()
    slack: SlackSettings = SlackSettings()
    model: ModelSettings = ModelSettings()


def get_settings() -> AppSettings:
    """Return application settings singleton."""
    return AppSettings()
