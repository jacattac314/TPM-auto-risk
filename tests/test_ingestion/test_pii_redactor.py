"""Tests for PII redaction module."""

from __future__ import annotations

import pytest

from src.ingestion.pii_redactor import _chunk_text, _redact_technical_secrets


class TestRedactTechnicalSecrets:
    def test_aws_access_key(self):
        text = "Use key AKIAIOSFODNN7EXAMPLE for access"
        result = _redact_technical_secrets(text)
        assert "AKIAIOSFODNN7EXAMPLE" not in result
        assert "[AWS_ACCESS_KEY]" in result

    def test_database_uri(self):
        text = "Connect to postgres://user:pass@db.example.com:5432/mydb"
        result = _redact_technical_secrets(text)
        assert "postgres://" not in result
        assert "[DATABASE_URI]" in result

    def test_private_key(self):
        text = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAK...\n-----END RSA PRIVATE KEY-----"
        result = _redact_technical_secrets(text)
        assert "PRIVATE KEY" not in result or "[PRIVATE_KEY]" in result

    def test_api_key_assignment(self):
        text = "api_key = 'test_fake_key_0123456789abcdef'"
        result = _redact_technical_secrets(text)
        assert "test_fake_key" not in result
        assert "[REDACTED_SECRET]" in result

    def test_no_secrets(self):
        text = "This is a normal message about the sprint review."
        result = _redact_technical_secrets(text)
        assert result == text


class TestChunkText:
    def test_short_text_returns_single_chunk(self):
        text = "Short text"
        chunks = _chunk_text(text, max_bytes=99_000)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_splits(self):
        text = "Word " * 30_000  # ~150KB
        chunks = _chunk_text(text, max_bytes=50_000)
        assert len(chunks) > 1
        # All chunks together should contain the full text
        reassembled = "".join(chunks)
        assert len(reassembled) == len(text)

    def test_empty_text(self):
        chunks = _chunk_text("", max_bytes=99_000)
        assert len(chunks) == 1
        assert chunks[0] == ""
