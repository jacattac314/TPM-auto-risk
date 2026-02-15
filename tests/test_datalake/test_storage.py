"""Tests for data lake storage layer."""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.config.settings import AppSettings
from src.datalake.storage import DataLakeWriter, GlueCatalogManager, _dataframe_to_parquet_bytes


@pytest.fixture
def mock_s3():
    with patch("src.datalake.storage.boto3") as mock_boto3:
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        yield mock_client


@pytest.fixture
def writer(settings, mock_s3):
    return DataLakeWriter(settings)


class TestDataframeToParquetBytes:
    def test_simple_dataframe(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        result = _dataframe_to_parquet_bytes(df)
        assert isinstance(result, bytes)
        assert len(result) > 0

        # Round-trip should preserve data
        roundtrip = pd.read_parquet(io.BytesIO(result))
        assert list(roundtrip.columns) == ["a", "b"]
        assert len(roundtrip) == 3

    def test_complex_types_serialized_as_json(self):
        df = pd.DataFrame({
            "name": ["ticket1", "ticket2"],
            "labels": [["bug", "p1"], ["feature"]],
            "meta": [{"key": "val"}, {"key": "val2"}],
        })
        result = _dataframe_to_parquet_bytes(df)
        roundtrip = pd.read_parquet(io.BytesIO(result))
        # Lists/dicts should have been JSON-stringified
        assert isinstance(roundtrip["labels"].iloc[0], str)
        parsed = json.loads(roundtrip["labels"].iloc[0])
        assert parsed == ["bug", "p1"]

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = _dataframe_to_parquet_bytes(df)
        assert isinstance(result, bytes)

    def test_none_values_handled(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [None, "y", None]})
        result = _dataframe_to_parquet_bytes(df)
        roundtrip = pd.read_parquet(io.BytesIO(result))
        assert len(roundtrip) == 3


class TestDataLakeWriter:
    def test_write_bronze(self, writer, mock_s3):
        data = [{"key": "PROJ-1", "summary": "Test"}]
        key = writer.write_bronze(data, "jira", "PROJ")

        mock_s3.put_object.assert_called_once()
        call_kwargs = mock_s3.put_object.call_args[1]
        assert call_kwargs["Bucket"] == "auto-tpm-risk-radar"
        assert "bronze/jira/project=PROJ/" in call_kwargs["Key"]
        assert call_kwargs["ContentType"] == "application/json"
        assert "jira" in key

    def test_write_silver(self, writer, mock_s3):
        records = [{"issue_key": "PROJ-1", "summary": "Test ticket"}]
        key = writer.write_silver(records, "jira", "PROJ")

        mock_s3.put_object.assert_called_once()
        call_kwargs = mock_s3.put_object.call_args[1]
        assert "silver/jira/project=PROJ/" in call_kwargs["Key"]
        assert call_kwargs["Key"].endswith(".parquet")

    def test_write_gold(self, writer, mock_s3):
        df = pd.DataFrame({"story_points": [5, 8], "risk_score": [0.3, 0.7]})
        key = writer.write_gold(df, "PROJ")

        mock_s3.put_object.assert_called_once()
        call_kwargs = mock_s3.put_object.call_args[1]
        assert "gold/risk_features/project=PROJ/" in call_kwargs["Key"]

    def test_write_gold_custom_feature_set(self, writer, mock_s3):
        df = pd.DataFrame({"avg_sentiment": [0.5]})
        key = writer.write_gold(df, "PROJ", feature_set_name="sentiment_timeseries")
        call_kwargs = mock_s3.put_object.call_args[1]
        assert "sentiment_timeseries" in call_kwargs["Key"]

    def test_kms_encryption_kwargs(self, settings):
        settings.aws.kms_key_id = "arn:aws:kms:us-east-1:123:key/abc"
        with patch("src.datalake.storage.boto3") as mock_boto3:
            mock_client = MagicMock()
            mock_boto3.client.return_value = mock_client
            w = DataLakeWriter(settings)
            w.write_bronze([{"x": 1}], "jira", "PROJ")

            call_kwargs = mock_client.put_object.call_args[1]
            assert call_kwargs["ServerSideEncryption"] == "aws:kms"
            assert call_kwargs["SSEKMSKeyId"] == "arn:aws:kms:us-east-1:123:key/abc"

    def test_read_gold_no_files(self, writer, mock_s3):
        paginator = MagicMock()
        paginator.paginate.return_value = [{"Contents": []}]
        mock_s3.get_paginator.return_value = paginator

        result = writer.read_gold("NONEXISTENT")
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_read_silver_with_data(self, writer, mock_s3):
        # Create a parquet buffer to return from S3
        df = pd.DataFrame({"text": ["hello"], "sentiment": [0.5]})
        buf = io.BytesIO()
        df.to_parquet(buf)
        parquet_bytes = buf.getvalue()

        paginator = MagicMock()
        paginator.paginate.return_value = [{
            "Contents": [{"Key": "silver/slack/project=PROJ/date=2024-01-01/slack_120000.parquet"}]
        }]
        mock_s3.get_paginator.return_value = paginator

        body_mock = MagicMock()
        body_mock.read.return_value = parquet_bytes
        mock_s3.get_object.return_value = {"Body": body_mock}

        result = writer.read_silver("slack", "PROJ")
        assert not result.empty
        assert "text" in result.columns
        assert result.iloc[0]["text"] == "hello"


class TestGlueCatalogManager:
    def test_ensure_database_creates_if_not_exists(self, settings):
        with patch("src.datalake.storage.boto3") as mock_boto3:
            mock_glue = MagicMock()
            mock_boto3.client.return_value = mock_glue

            # Create a real exception class for EntityNotFoundException
            entity_not_found = type("EntityNotFoundException", (Exception,), {})
            mock_glue.exceptions.EntityNotFoundException = entity_not_found
            mock_glue.get_database.side_effect = entity_not_found("not found")

            manager = GlueCatalogManager(settings)
            manager.ensure_database()
            mock_glue.create_database.assert_called_once()

    def test_register_table_creates_new(self, settings):
        with patch("src.datalake.storage.boto3") as mock_boto3:
            mock_glue = MagicMock()
            mock_boto3.client.return_value = mock_glue

            entity_not_found = type("EntityNotFoundException", (Exception,), {})
            mock_glue.exceptions.EntityNotFoundException = entity_not_found
            mock_glue.get_table.side_effect = entity_not_found("not found")

            manager = GlueCatalogManager(settings)
            manager.register_table(
                "jira_silver",
                "silver/jira",
                columns=[{"Name": "issue_key", "Type": "string"}],
                partition_keys=[{"Name": "project", "Type": "string"}],
            )
            mock_glue.create_table.assert_called_once()
