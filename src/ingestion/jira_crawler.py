"""Jira data ingestion via OAuth 1.0a with incremental extraction.

Handles paginated extraction, dynamic custom field resolution,
and sprint history parsing from the Jira REST API.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import boto3
import requests
from requests_oauthlib import OAuth1

from src.config.settings import AppSettings

logger = logging.getLogger(__name__)


class JiraFieldResolver:
    """Resolves human-readable Jira field names to their custom field IDs."""

    def __init__(self, base_url: str, auth: OAuth1) -> None:
        self._base_url = base_url.rstrip("/")
        self._auth = auth
        self._field_map: dict[str, str] = {}

    def resolve(self, field_name: str) -> str:
        """Map a human-readable field name (e.g. 'Story Points') to its API field ID."""
        if not self._field_map:
            self._load_fields()
        field_id = self._field_map.get(field_name)
        if field_id is None:
            raise KeyError(
                f"Jira field '{field_name}' not found. "
                f"Available custom fields: {list(self._field_map.keys())[:20]}..."
            )
        return field_id

    def _load_fields(self) -> None:
        resp = requests.get(f"{self._base_url}/rest/api/2/field", auth=self._auth, timeout=30)
        resp.raise_for_status()
        for field in resp.json():
            if field.get("custom"):
                self._field_map[field["name"]] = field["id"]
            else:
                self._field_map[field["name"]] = field["id"]


class JiraIngestionState:
    """Tracks the last ingestion timestamp per project in DynamoDB."""

    def __init__(self, table_name: str, region: str) -> None:
        self._dynamodb = boto3.resource("dynamodb", region_name=region)
        self._table = self._dynamodb.Table(table_name)

    def get_last_ingest_date(self, project_key: str) -> str | None:
        """Return ISO-format date string of last ingestion, or None if first run."""
        resp = self._table.get_item(Key={"project_key": project_key})
        item = resp.get("Item")
        if item:
            return item.get("last_ingest_date")
        return None

    def update_last_ingest_date(self, project_key: str, date_str: str) -> None:
        """Persist the latest ingestion date for a project."""
        self._table.put_item(
            Item={"project_key": project_key, "last_ingest_date": date_str}
        )


def _build_oauth(settings: AppSettings) -> OAuth1:
    """Build an OAuth1 signer using the RSA private key from SSM Parameter Store."""
    ssm = boto3.client("ssm", region_name=settings.aws.region)
    resp = ssm.get_parameter(
        Name=settings.jira.private_key_ssm_param, WithDecryption=True
    )
    private_key = resp["Parameter"]["Value"]

    return OAuth1(
        client_key=settings.jira.consumer_key,
        resource_owner_key=settings.jira.access_token,
        resource_owner_secret=settings.jira.access_token_secret,
        rsa_key=private_key,
        signature_method="RSA-SHA1",
    )


def _parse_sprint_history(sprint_field_value: Any) -> list[dict[str, Any]]:
    """Parse the Jira sprint custom field into structured sprint records.

    Sprint data in Jira is often stored as a serialized string array like:
    'com.atlassian.greenhopper.service.sprint.Sprint@...[id=123,name=Sprint 1,...]'
    or as a list of dicts from newer API versions.
    """
    if sprint_field_value is None:
        return []

    sprints = []
    if isinstance(sprint_field_value, list):
        for item in sprint_field_value:
            if isinstance(item, dict):
                sprints.append({
                    "sprint_id": item.get("id"),
                    "sprint_name": item.get("name"),
                    "sprint_state": item.get("state"),
                    "start_date": item.get("startDate"),
                    "end_date": item.get("endDate"),
                    "complete_date": item.get("completeDate"),
                })
            elif isinstance(item, str):
                parsed = _parse_sprint_string(item)
                if parsed:
                    sprints.append(parsed)
    return sprints


def _parse_sprint_string(sprint_str: str) -> dict[str, Any] | None:
    """Parse a serialized Jira sprint string into a dict."""
    try:
        bracket_start = sprint_str.index("[")
        bracket_end = sprint_str.rindex("]")
        inner = sprint_str[bracket_start + 1 : bracket_end]
        pairs = {}
        for pair in inner.split(","):
            if "=" in pair:
                key, value = pair.split("=", 1)
                pairs[key.strip()] = value.strip() if value.strip() != "<null>" else None
        return {
            "sprint_id": pairs.get("id"),
            "sprint_name": pairs.get("name"),
            "sprint_state": pairs.get("state"),
            "start_date": pairs.get("startDate"),
            "end_date": pairs.get("endDate"),
            "complete_date": pairs.get("completeDate"),
        }
    except (ValueError, IndexError):
        logger.warning("Failed to parse sprint string: %s", sprint_str[:100])
        return None


def _extract_ticket(
    issue: dict[str, Any],
    story_points_field_id: str,
    sprint_field_id: str,
) -> dict[str, Any]:
    """Transform a raw Jira API issue into a flat record for storage."""
    fields = issue.get("fields", {})

    return {
        "issue_id": issue["id"],
        "issue_key": issue["key"],
        "project_key": fields.get("project", {}).get("key"),
        "summary": fields.get("summary"),
        "description": fields.get("description"),
        "status": fields.get("status", {}).get("name"),
        "status_category": fields.get("status", {}).get("statusCategory", {}).get("name"),
        "priority": fields.get("priority", {}).get("name"),
        "issue_type": fields.get("issuetype", {}).get("name"),
        "assignee": fields.get("assignee", {}).get("displayName") if fields.get("assignee") else None,
        "reporter": fields.get("reporter", {}).get("displayName") if fields.get("reporter") else None,
        "created": fields.get("created"),
        "updated": fields.get("updated"),
        "due_date": fields.get("duedate"),
        "resolution_date": fields.get("resolutiondate"),
        "story_points": fields.get(story_points_field_id),
        "labels": fields.get("labels", []),
        "components": [c.get("name") for c in fields.get("components", [])],
        "sprint_history": _parse_sprint_history(fields.get(sprint_field_id)),
        "issue_links": [
            {
                "type": link.get("type", {}).get("name"),
                "direction": "outward" if "outwardIssue" in link else "inward",
                "linked_issue": (
                    link.get("outwardIssue", link.get("inwardIssue", {})).get("key")
                ),
            }
            for link in fields.get("issuelinks", [])
        ],
        "comment_count": fields.get("comment", {}).get("total", 0),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }


def crawl_project(
    project_key: str,
    settings: AppSettings,
    state: JiraIngestionState | None = None,
) -> list[dict[str, Any]]:
    """Incrementally crawl a Jira project and return extracted ticket records.

    Args:
        project_key: The Jira project key (e.g. 'PROJ').
        settings: Application settings.
        state: Optional ingestion state tracker. If None, does a full crawl.

    Returns:
        List of flattened ticket dicts ready for storage.
    """
    auth = _build_oauth(settings)
    base_url = settings.jira.base_url.rstrip("/")

    # Resolve custom field IDs
    resolver = JiraFieldResolver(base_url, auth)
    story_points_id = resolver.resolve(settings.jira.story_points_field_name)
    sprint_id = resolver.resolve(settings.jira.sprint_field_name)

    # Build JQL with incremental filter
    jql = f"project = {project_key}"
    if state:
        last_date = state.get_last_ingest_date(project_key)
        if last_date:
            jql += f' AND updated >= "{last_date}"'

    logger.info("Crawling Jira project %s with JQL: %s", project_key, jql)

    tickets: list[dict[str, Any]] = []
    start_at = 0
    page_size = settings.jira.page_size

    while True:
        resp = requests.get(
            f"{base_url}/rest/api/2/search",
            params={
                "jql": jql,
                "startAt": start_at,
                "maxResults": page_size,
                "fields": "*all",
                "expand": "changelog",
            },
            auth=auth,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

        issues = data.get("issues", [])
        for issue in issues:
            ticket = _extract_ticket(issue, story_points_id, sprint_id)
            tickets.append(ticket)

        total = data.get("total", 0)
        start_at += len(issues)

        logger.info("Fetched %d/%d issues from %s", start_at, total, project_key)

        if start_at >= total or not issues:
            break

    # Update ingestion state
    if state and tickets:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
        state.update_last_ingest_date(project_key, now)

    logger.info("Crawl complete for %s: %d tickets extracted", project_key, len(tickets))
    return tickets
