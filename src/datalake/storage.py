"""Data Lake storage layer implementing the Medallion architecture (Bronze/Silver/Gold).

Handles writing records to S3 in Parquet format with proper partitioning,
and managing the AWS Glue Data Catalog for queryability.
"""

from __future__ import annotations

import io
import json
import logging
from datetime import datetime, timezone
from typing import Any

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config.settings import AppSettings

logger = logging.getLogger(__name__)


class DataLakeWriter:
    """Writes data to the S3-based data lake with Medallion architecture tiers."""

    def __init__(self, settings: AppSettings) -> None:
        self._s3 = boto3.client("s3", region_name=settings.aws.region)
        self._bucket = settings.aws.s3_bucket
        self._bronze_prefix = settings.aws.s3_bronze_prefix
        self._silver_prefix = settings.aws.s3_silver_prefix
        self._gold_prefix = settings.aws.s3_gold_prefix
        self._kms_key_id = settings.aws.kms_key_id

    def _s3_put_kwargs(self) -> dict[str, str]:
        """Return common S3 PutObject kwargs including encryption."""
        kwargs: dict[str, str] = {}
        if self._kms_key_id:
            kwargs["ServerSideEncryption"] = "aws:kms"
            kwargs["SSEKMSKeyId"] = self._kms_key_id
        return kwargs

    def write_bronze(
        self,
        data: list[dict[str, Any]],
        source: str,
        project_key: str,
    ) -> str:
        """Write raw JSON data to the Bronze layer.

        Args:
            data: Raw records as received from the source API.
            source: Data source identifier ('jira' or 'slack').
            project_key: Project identifier for partitioning.

        Returns:
            The S3 key where data was written.
        """
        now = datetime.now(timezone.utc)
        date_partition = now.strftime("%Y-%m-%d")
        timestamp = now.strftime("%H%M%S")

        key = (
            f"{self._bronze_prefix}/{source}/"
            f"project={project_key}/date={date_partition}/"
            f"{source}_{timestamp}.json"
        )

        body = json.dumps(data, default=str)
        self._s3.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=body.encode("utf-8"),
            ContentType="application/json",
            **self._s3_put_kwargs(),
        )

        logger.info("Wrote %d records to Bronze: s3://%s/%s", len(data), self._bucket, key)
        return key

    def write_silver(
        self,
        records: list[dict[str, Any]],
        source: str,
        project_key: str,
    ) -> str:
        """Write sanitized/redacted data to the Silver layer in Parquet format.

        Args:
            records: PII-redacted, flattened records.
            source: Data source identifier.
            project_key: Project identifier for partitioning.

        Returns:
            The S3 key where data was written.
        """
        now = datetime.now(timezone.utc)
        date_partition = now.strftime("%Y-%m-%d")
        timestamp = now.strftime("%H%M%S")

        key = (
            f"{self._silver_prefix}/{source}/"
            f"project={project_key}/date={date_partition}/"
            f"{source}_{timestamp}.parquet"
        )

        df = pd.DataFrame(records)
        parquet_buffer = _dataframe_to_parquet_bytes(df)

        self._s3.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=parquet_buffer,
            ContentType="application/octet-stream",
            **self._s3_put_kwargs(),
        )

        logger.info("Wrote %d records to Silver: s3://%s/%s", len(records), self._bucket, key)
        return key

    def write_gold(
        self,
        features_df: pd.DataFrame,
        project_key: str,
        feature_set_name: str = "risk_features",
    ) -> str:
        """Write engineered features to the Gold layer in Parquet format.

        Args:
            features_df: DataFrame of engineered features ready for ML.
            project_key: Project identifier for partitioning.
            feature_set_name: Name of the feature set.

        Returns:
            The S3 key where data was written.
        """
        now = datetime.now(timezone.utc)
        date_partition = now.strftime("%Y-%m-%d")
        timestamp = now.strftime("%H%M%S")

        key = (
            f"{self._gold_prefix}/{feature_set_name}/"
            f"project={project_key}/date={date_partition}/"
            f"features_{timestamp}.parquet"
        )

        parquet_buffer = _dataframe_to_parquet_bytes(features_df)

        self._s3.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=parquet_buffer,
            ContentType="application/octet-stream",
            **self._s3_put_kwargs(),
        )

        logger.info(
            "Wrote %d feature rows to Gold: s3://%s/%s",
            len(features_df), self._bucket, key,
        )
        return key

    def read_silver(self, source: str, project_key: str, date: str | None = None) -> pd.DataFrame:
        """Read Silver layer data back as a DataFrame.

        Args:
            source: Data source identifier.
            project_key: Project identifier.
            date: Optional date partition (YYYY-MM-DD). If None, reads latest.

        Returns:
            Combined DataFrame from matching partitions.
        """
        prefix = f"{self._silver_prefix}/{source}/project={project_key}/"
        if date:
            prefix += f"date={date}/"

        return self._read_parquet_prefix(prefix)

    def read_gold(self, project_key: str, feature_set_name: str = "risk_features") -> pd.DataFrame:
        """Read Gold layer features as a DataFrame."""
        prefix = f"{self._gold_prefix}/{feature_set_name}/project={project_key}/"
        return self._read_parquet_prefix(prefix)

    def _read_parquet_prefix(self, prefix: str) -> pd.DataFrame:
        """List and read all Parquet files under an S3 prefix."""
        paginator = self._s3.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".parquet"):
                    keys.append(obj["Key"])

        if not keys:
            logger.warning("No Parquet files found under s3://%s/%s", self._bucket, prefix)
            return pd.DataFrame()

        frames = []
        for key in keys:
            resp = self._s3.get_object(Bucket=self._bucket, Key=key)
            buf = io.BytesIO(resp["Body"].read())
            frames.append(pd.read_parquet(buf))

        return pd.concat(frames, ignore_index=True)


def _dataframe_to_parquet_bytes(df: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to Parquet bytes with Snappy compression."""
    # Convert complex types (lists, dicts) to JSON strings for Parquet compatibility
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(1)
            if not sample.empty and isinstance(sample.iloc[0], (list, dict)):
                df[col] = df[col].apply(lambda x: json.dumps(x, default=str) if x is not None else None)

    buf = io.BytesIO()
    table = pa.Table.from_pandas(df)
    pq.write_table(table, buf, compression="snappy")
    return buf.getvalue()


class GlueCatalogManager:
    """Manages AWS Glue Data Catalog entries for the data lake tables."""

    def __init__(self, settings: AppSettings) -> None:
        self._glue = boto3.client("glue", region_name=settings.aws.region)
        self._bucket = settings.aws.s3_bucket
        self._database_name = "auto_tpm_risk_radar"

    def ensure_database(self) -> None:
        """Create the Glue database if it doesn't exist."""
        try:
            self._glue.get_database(Name=self._database_name)
        except self._glue.exceptions.EntityNotFoundException:
            self._glue.create_database(
                DatabaseInput={
                    "Name": self._database_name,
                    "Description": "Auto-TPM Risk Radar data lake catalog",
                }
            )
            logger.info("Created Glue database: %s", self._database_name)

    def register_table(
        self,
        table_name: str,
        s3_prefix: str,
        columns: list[dict[str, str]],
        partition_keys: list[dict[str, str]] | None = None,
    ) -> None:
        """Register or update a table in the Glue Data Catalog.

        Args:
            table_name: Name for the Glue table.
            s3_prefix: S3 location prefix for the table data.
            columns: List of column definitions [{"Name": ..., "Type": ...}].
            partition_keys: Optional partition key definitions.
        """
        table_input: dict[str, Any] = {
            "Name": table_name,
            "Description": f"Auto-TPM Risk Radar - {table_name}",
            "StorageDescriptor": {
                "Columns": columns,
                "Location": f"s3://{self._bucket}/{s3_prefix}/",
                "InputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat",
                "OutputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat",
                "SerdeInfo": {
                    "SerializationLibrary": "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe",
                },
                "Compressed": True,
            },
            "TableType": "EXTERNAL_TABLE",
            "Parameters": {"classification": "parquet", "compressionType": "snappy"},
        }

        if partition_keys:
            table_input["PartitionKeys"] = partition_keys

        try:
            self._glue.get_table(DatabaseName=self._database_name, Name=table_name)
            self._glue.update_table(
                DatabaseName=self._database_name, TableInput=table_input
            )
            logger.info("Updated Glue table: %s.%s", self._database_name, table_name)
        except self._glue.exceptions.EntityNotFoundException:
            self._glue.create_table(
                DatabaseName=self._database_name, TableInput=table_input
            )
            logger.info("Created Glue table: %s.%s", self._database_name, table_name)
