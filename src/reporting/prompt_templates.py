"""Prompt templates for the GenAI reporting engine.

Uses XML-tagged data isolation and Chain of Thought decomposition
as specified in the PRD prompt engineering strategy.
"""

from __future__ import annotations

EXECUTIVE_REPORT_PROMPT = """\
You are a senior Technical Program Manager. Your goal is to draft a weekly \
executive status report for the project described below. You must be precise, \
data-driven, and actionable.

<project_context>
Project: {project_name}
Reporting Period: {reporting_period}
Total Tickets Tracked: {total_tickets}
Tickets Completed This Period: {completed_count}
Tickets In Progress: {in_progress_count}
Tickets At Risk (High): {high_risk_count}
Tickets At Risk (Medium): {medium_risk_count}
</project_context>

<jira_data>
{jira_risk_summary}
</jira_data>

<slack_context>
{slack_sentiment_summary}
</slack_context>

<risk_scores>
{risk_scores_summary}
</risk_scores>

Instructions:
1. Synthesize the provided data into a cohesive narrative.
2. Highlight the top 3 risks based on the XGBoost risk scores.
3. Correlate Slack sentiment with specific Jira blockers where possible.
4. Suggest concrete mitigation actions for each critical risk.

Output strictly valid JSON with the following keys:
- "executive_summary": A 2-3 sentence high-level overview.
- "key_accomplishments": An array of strings listing completed items.
- "critical_risks": An array of objects, each with "title", "description", \
"risk_score", "affected_tickets", and "recommended_action".
- "blockers": An array of objects with "title" and "details".
- "sentiment_outlook": A brief assessment of team morale based on Slack data.
- "mitigation_plan": An array of objects with "action", "owner_role", and "priority".
- "next_week_focus": An array of strings for upcoming priorities.

Do NOT include any PII. Refer to individuals by role (e.g., "The Backend Engineer") \
if necessary. Do NOT add any text outside the JSON object.
"""

TICKET_CLASSIFICATION_PROMPT = """\
You are a technical project analyst. Classify the following Jira ticket \
based on its content.

<ticket>
Key: {issue_key}
Summary: {summary}
Description: {description}
Type: {issue_type}
Status: {status}
Story Points: {story_points}
</ticket>

Classify into exactly one category and provide a brief rationale.

Output valid JSON:
- "category": One of ["feature", "bug_fix", "tech_debt", "infrastructure", \
"documentation", "testing", "spike"]
- "complexity_assessment": One of ["trivial", "simple", "moderate", "complex", \
"very_complex"]
- "rationale": A 1-sentence explanation.
"""

RISK_EXPLANATION_PROMPT = """\
You are a senior TPM explaining a risk to an executive audience. Given the \
following ticket data and its ML-derived risk score, explain WHY this ticket \
is at risk in plain business language.

<ticket>
Key: {issue_key}
Summary: {summary}
Status: {status}
Risk Score: {risk_score}
Risk Level: {risk_level}
</ticket>

<risk_factors>
{risk_factors}
</risk_factors>

<context>
{additional_context}
</context>

Write a 2-3 sentence explanation of why this ticket is at risk and what \
the likely impact is. Be specific and reference the data provided. \
Output only the explanation text, no JSON.
"""


def format_jira_risk_summary(high_risk_tickets: list[dict]) -> str:
    """Format high-risk Jira tickets for prompt injection."""
    if not high_risk_tickets:
        return "No high-risk tickets identified this period."

    lines = []
    for t in high_risk_tickets[:15]:  # Limit to top 15 for context window
        lines.append(
            f"- [{t.get('issue_key', 'N/A')}] {t.get('summary', 'No summary')} "
            f"| Status: {t.get('status', 'Unknown')} "
            f"| Risk: {t.get('risk_score', 0):.0%} "
            f"| Points: {t.get('story_points', 0)} "
            f"| Days in Status: {t.get('days_in_status', 0):.0f} "
            f"| Dependencies: {t.get('dependency_count', 0)}"
        )
    return "\n".join(lines)


def format_risk_scores_summary(predictions: list[dict]) -> str:
    """Format risk score distribution for the prompt."""
    if not predictions:
        return "No risk predictions available."

    high = sum(1 for p in predictions if p.get("risk_level") == "High")
    medium = sum(1 for p in predictions if p.get("risk_level") == "Medium")
    low = sum(1 for p in predictions if p.get("risk_level") == "Low")
    avg_score = sum(p.get("risk_score", 0) for p in predictions) / len(predictions)

    return (
        f"Risk Distribution: {high} High, {medium} Medium, {low} Low\n"
        f"Average Risk Score: {avg_score:.2%}\n"
        f"Total Tickets Scored: {len(predictions)}"
    )


def format_sentiment_summary(
    sentiment_data: list[dict],
    anomalies: list[dict] | None = None,
) -> str:
    """Format Slack sentiment data for the prompt."""
    if not sentiment_data:
        return "No Slack sentiment data available for this period."

    recent = sentiment_data[-7:] if len(sentiment_data) > 7 else sentiment_data
    avg = sum(s.get("avg_sentiment", 0) for s in recent) / len(recent)

    sentiment_label = "positive" if avg > 0.1 else "negative" if avg < -0.1 else "neutral"
    lines = [f"Overall team sentiment is {sentiment_label} (avg: {avg:.3f})."]

    if anomalies:
        lines.append(f"\n{len(anomalies)} sentiment anomalies detected:")
        for a in anomalies[:5]:
            lines.append(
                f"- {a.get('window_start', 'N/A')}: "
                f"sentiment={a.get('avg_sentiment', 0):.3f}, "
                f"z-score={a.get('z_score', 0):.2f}"
            )

    return "\n".join(lines)
