"""PII detection and redaction using Amazon Comprehend.

Replaces detected PII entities with typed placeholders (e.g. [NAME], [EMAIL])
to preserve sentence structure for downstream LLM processing. Also applies
regex-based detection for technical secrets like AWS keys.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import boto3

from src.config.settings import AppSettings

logger = logging.getLogger(__name__)

# Regex patterns for technical secrets
_SECRET_PATTERNS = [
    # AWS Access Key ID
    (re.compile(r"(?<![A-Z0-9])(AKIA[0-9A-Z]{16})(?![A-Z0-9])"), "[AWS_ACCESS_KEY]"),
    # AWS Secret Access Key (40-char base64)
    (re.compile(r"(?<![A-Za-z0-9/+=])([A-Za-z0-9/+=]{40})(?![A-Za-z0-9/+=])"), None),
    # Generic API keys/tokens (long hex or alphanumeric strings)
    (re.compile(r"\b(?:api[_-]?key|token|secret|password)\s*[:=]\s*['\"]?([A-Za-z0-9_\-]{20,})['\"]?", re.IGNORECASE), "[REDACTED_SECRET]"),
    # Database connection strings
    (re.compile(r"(?:postgres|mysql|mongodb|redis)://[^\s]+", re.IGNORECASE), "[DATABASE_URI]"),
    # Private keys
    (re.compile(r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC )?PRIVATE KEY-----"), "[PRIVATE_KEY]"),
]

# Mapping of Comprehend PII entity types to placeholder tokens
_PII_PLACEHOLDERS = {
    "NAME": "[NAME]",
    "EMAIL": "[EMAIL]",
    "PHONE": "[PHONE]",
    "ADDRESS": "[ADDRESS]",
    "AGE": "[AGE]",
    "DATE_TIME": "[DATETIME]",
    "SSN": "[SSN]",
    "PASSPORT_NUMBER": "[PASSPORT]",
    "DRIVER_ID": "[DRIVER_ID]",
    "CREDIT_DEBIT_NUMBER": "[CARD_NUMBER]",
    "CREDIT_DEBIT_CVV": "[CVV]",
    "CREDIT_DEBIT_EXPIRY": "[CARD_EXPIRY]",
    "BANK_ACCOUNT_NUMBER": "[BANK_ACCOUNT]",
    "BANK_ROUTING": "[BANK_ROUTING]",
    "IP_ADDRESS": "[IP_ADDRESS]",
    "URL": "[URL]",
    "USERNAME": "[USERNAME]",
    "PASSWORD": "[PASSWORD]",
    "AWS_ACCESS_KEY": "[AWS_ACCESS_KEY]",
    "AWS_SECRET_KEY": "[AWS_SECRET_KEY]",
}


def _redact_technical_secrets(text: str) -> str:
    """Apply regex-based redaction for technical secrets (AWS keys, DB URIs, etc.)."""
    for pattern, replacement in _SECRET_PATTERNS:
        if replacement:
            text = pattern.sub(replacement, text)
    return text


class ComprehendPIIRedactor:
    """Redacts PII from text using Amazon Comprehend's DetectPiiEntities API."""

    def __init__(self, settings: AppSettings) -> None:
        self._client = boto3.client("comprehend", region_name=settings.aws.region)
        self._language_code = settings.aws.comprehend_language_code

    def redact(self, text: str) -> dict[str, Any]:
        """Detect and redact PII entities from the given text.

        Returns a dict with:
            - redacted_text: The text with PII replaced by entity-type placeholders.
            - entities_found: List of detected PII entities with types and positions.
            - pii_detected: Boolean indicating whether any PII was found.
        """
        if not text or not text.strip():
            return {
                "redacted_text": text,
                "entities_found": [],
                "pii_detected": False,
            }

        # Step 1: Redact technical secrets via regex
        text = _redact_technical_secrets(text)

        # Step 2: Detect PII via Comprehend
        # Comprehend has a 100KB limit per request; chunk if needed
        chunks = _chunk_text(text, max_bytes=99_000)
        all_entities: list[dict[str, Any]] = []
        redacted_parts: list[str] = []

        for chunk in chunks:
            try:
                response = self._client.detect_pii_entities(
                    Text=chunk, LanguageCode=self._language_code
                )
            except self._client.exceptions.TextSizeLimitExceededException:
                logger.warning("Text chunk exceeded Comprehend limit, skipping PII detection")
                redacted_parts.append(chunk)
                continue
            except Exception:
                logger.exception("Comprehend PII detection failed")
                redacted_parts.append(chunk)
                continue

            entities = response.get("Entities", [])
            all_entities.extend(entities)

            # Apply redaction in reverse order to preserve offsets
            redacted_chunk = chunk
            for entity in sorted(entities, key=lambda e: e["BeginOffset"], reverse=True):
                pii_type = entity.get("Type", "PII")
                placeholder = _PII_PLACEHOLDERS.get(pii_type, f"[{pii_type}]")
                begin = entity["BeginOffset"]
                end = entity["EndOffset"]
                redacted_chunk = redacted_chunk[:begin] + placeholder + redacted_chunk[end:]

            redacted_parts.append(redacted_chunk)

        redacted_text = "".join(redacted_parts)

        return {
            "redacted_text": redacted_text,
            "entities_found": [
                {"type": e.get("Type"), "score": e.get("Score", 0.0)}
                for e in all_entities
            ],
            "pii_detected": len(all_entities) > 0,
        }


def _chunk_text(text: str, max_bytes: int = 99_000) -> list[str]:
    """Split text into chunks that fit within Comprehend's byte limit."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return [text]

    chunks = []
    current_pos = 0
    while current_pos < len(text):
        # Find a chunk boundary that fits within byte limit
        end_pos = current_pos + max_bytes
        if end_pos >= len(text):
            chunks.append(text[current_pos:])
            break

        # Try to break at a sentence or word boundary
        chunk_candidate = text[current_pos:end_pos]
        while len(chunk_candidate.encode("utf-8")) > max_bytes:
            end_pos -= 1000
            chunk_candidate = text[current_pos:end_pos]

        # Find last sentence break
        last_period = chunk_candidate.rfind(". ")
        last_newline = chunk_candidate.rfind("\n")
        break_point = max(last_period, last_newline)

        if break_point > 0:
            end_pos = current_pos + break_point + 1

        chunks.append(text[current_pos:end_pos])
        current_pos = end_pos

    return chunks


def redact_record(
    record: dict[str, Any],
    text_fields: list[str],
    redactor: ComprehendPIIRedactor,
) -> dict[str, Any]:
    """Redact PII from specified text fields in a data record.

    Args:
        record: A dict (e.g. Jira ticket or Slack message).
        text_fields: List of keys in the record to redact.
        redactor: The ComprehendPIIRedactor instance.

    Returns:
        A copy of the record with specified fields redacted.
    """
    redacted = dict(record)
    pii_summary: list[dict[str, Any]] = []

    for field in text_fields:
        value = redacted.get(field)
        if isinstance(value, str) and value.strip():
            result = redactor.redact(value)
            redacted[field] = result["redacted_text"]
            if result["pii_detected"]:
                pii_summary.extend(result["entities_found"])

    redacted["_pii_redaction"] = {
        "fields_scanned": text_fields,
        "pii_detected": len(pii_summary) > 0,
        "entity_types_found": list({e["type"] for e in pii_summary}),
    }

    return redacted
