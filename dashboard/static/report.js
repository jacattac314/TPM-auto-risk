/**
 * Auto-Draft report interface logic.
 * Handles report generation, editing, and publishing with evidence display.
 */

let currentReport = null;

/**
 * Generate an executive status report via the API.
 */
async function generateReport() {
    const btn = document.getElementById('generate-btn');
    const status = document.getElementById('draft-status');
    const draftEl = document.getElementById('report-draft');

    btn.disabled = true;
    btn.textContent = 'Drafting...';
    status.textContent = 'Generating';
    status.className = 'badge generating';

    try {
        const resp = await fetch(`/api/v1/projects/${projectKey}/report`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                project_name: projectKey,
                reporting_period: `Week of ${new Date().toISOString().split('T')[0]}`,
            }),
        });

        if (!resp.ok) {
            throw new Error(`API returned ${resp.status}`);
        }

        currentReport = await resp.json();
        renderReport(currentReport);
        renderEvidence(currentReport);

        status.textContent = 'Generated';
        status.className = 'badge generated';
        document.getElementById('publish-btn').disabled = false;
    } catch (err) {
        console.error('Report generation failed:', err);
        draftEl.innerHTML = `<div class="placeholder-message">Report generation failed: ${err.message}. Ensure data has been ingested and the model is trained.</div>`;
        status.textContent = 'Failed';
    } finally {
        btn.disabled = false;
        btn.textContent = 'Regenerate';
    }
}

/**
 * Render the generated report in the draft panel.
 */
function renderReport(report) {
    const el = document.getElementById('report-draft');
    if (report.parse_error) {
        el.innerHTML = `<pre>${report.raw_response || 'No response'}</pre>`;
        el.contentEditable = 'true';
        return;
    }

    let html = '';

    if (report.executive_summary) {
        html += `<h4>Executive Summary</h4><p>${report.executive_summary}</p>`;
    }

    if (report.key_accomplishments && report.key_accomplishments.length) {
        html += '<h4>Key Accomplishments</h4><ul>';
        report.key_accomplishments.forEach(a => { html += `<li>${a}</li>`; });
        html += '</ul>';
    }

    if (report.critical_risks && report.critical_risks.length) {
        html += '<h4>Critical Risks</h4>';
        report.critical_risks.forEach(r => {
            html += `<div class="evidence-section" style="border-left:3px solid var(--color-high);padding-left:12px;margin:8px 0;">
                <strong>${r.title || 'Risk'}</strong>
                <span class="risk-badge high" style="margin-left:8px;">${((r.risk_score || 0) * 100).toFixed(0)}%</span>
                <p>${r.description || ''}</p>
                ${r.recommended_action ? `<p><em>Action: ${r.recommended_action}</em></p>` : ''}
            </div>`;
        });
    }

    if (report.blockers && report.blockers.length) {
        html += '<h4>Blockers</h4><ul>';
        report.blockers.forEach(b => { html += `<li><strong>${b.title}:</strong> ${b.details}</li>`; });
        html += '</ul>';
    }

    if (report.sentiment_outlook) {
        html += `<h4>Sentiment Outlook</h4><p>${report.sentiment_outlook}</p>`;
    }

    if (report.mitigation_plan && report.mitigation_plan.length) {
        html += '<h4>Mitigation Plan</h4><ul>';
        report.mitigation_plan.forEach(m => {
            html += `<li><strong>[${m.priority || '-'}]</strong> ${m.action} <em>(${m.owner_role || 'TBD'})</em></li>`;
        });
        html += '</ul>';
    }

    if (report.next_week_focus && report.next_week_focus.length) {
        html += '<h4>Next Week Focus</h4><ul>';
        report.next_week_focus.forEach(f => { html += `<li>${f}</li>`; });
        html += '</ul>';
    }

    el.innerHTML = html;
    el.contentEditable = 'true';
}

/**
 * Populate the Evidence Locker with cited tickets and sentiment data.
 */
function renderEvidence(report) {
    const ticketList = document.getElementById('cited-tickets');
    const sentimentEl = document.getElementById('sentiment-evidence');

    // Cited tickets from critical risks
    if (report.critical_risks && report.critical_risks.length) {
        let ticketHtml = '';
        report.critical_risks.forEach(r => {
            const tickets = r.affected_tickets || [];
            if (typeof tickets === 'string') {
                ticketHtml += `<li>${tickets}</li>`;
            } else if (Array.isArray(tickets)) {
                tickets.forEach(t => { ticketHtml += `<li>${t}</li>`; });
            }
        });
        ticketList.innerHTML = ticketHtml || '<li class="placeholder-message">No specific tickets cited.</li>';
    }

    // Sentiment evidence
    if (report.sentiment_outlook) {
        sentimentEl.innerHTML = `<p>${report.sentiment_outlook}</p>`;
    }

    // Tier 2 Sentiment Anomaly Analysis
    // Renders the LLM-generated root-cause hypotheses for each detected
    // anomaly window.  Each card shows the anomaly data, a severity badge,
    // the hypothesis, and a recommended action so TPMs can act immediately.
    if (report.sentiment_anomaly_analysis && report.sentiment_anomaly_analysis.length) {
        let anomalyHtml = '<div class="evidence-section" style="margin-top:16px;">';
        anomalyHtml += '<h4>Sentiment Anomaly Analysis</h4>';
        report.sentiment_anomaly_analysis.forEach(a => {
            const analysis = a.analysis || {};
            const severityColors = { high: '#f85149', medium: '#d29922', low: '#3fb950', unknown: '#8b949e' };
            const sevColor = severityColors[analysis.severity] || severityColors.unknown;
            anomalyHtml += `
                <div style="border-left:3px solid ${sevColor};padding:8px 12px;margin:8px 0;background:#161b22;border-radius:4px;">
                    <strong>${a.window_start || 'N/A'}</strong>
                    <span class="risk-badge" style="background:${sevColor};margin-left:8px;padding:2px 8px;border-radius:4px;font-size:11px;">
                        ${(analysis.severity || 'unknown').toUpperCase()}
                    </span>
                    <div style="margin-top:6px;font-size:13px;">
                        Sentiment: ${(a.avg_sentiment || 0).toFixed(2)} | Z-score: ${(a.z_score || 0).toFixed(1)} | Messages: ${a.message_count || 0}
                    </div>
                    <p style="margin:6px 0 2px;"><strong>Hypothesis:</strong> ${analysis.hypothesis || 'N/A'}</p>
                    ${analysis.recommended_action ? `<p style="margin:2px 0;"><em>Action: ${analysis.recommended_action}</em></p>` : ''}
                </div>`;
        });
        anomalyHtml += '</div>';
        sentimentEl.innerHTML += anomalyHtml;
    }

    // Metadata
    if (report._metadata) {
        const meta = report._metadata;
        sentimentEl.innerHTML += `
            <div class="evidence-section" style="margin-top:16px;">
                <h4>Report Metadata</h4>
                <ul class="evidence-list">
                    <li>Total tickets analyzed: ${meta.total_tickets || '-'}</li>
                    <li>High risk: ${meta.risk_distribution?.high || 0}</li>
                    <li>Medium risk: ${meta.risk_distribution?.medium || 0}</li>
                    <li>Low risk: ${meta.risk_distribution?.low || 0}</li>
                    <li>Model tier: ${meta.model_tier || '-'}</li>
                </ul>
            </div>`;
    }
}

/**
 * Publish the (possibly edited) report and submit feedback for RLHF.
 */
async function publishReport() {
    if (!currentReport) return;

    const editedContent = document.getElementById('report-draft').innerText;

    try {
        const resp = await fetch(`/api/v1/reports/${projectKey}/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                original_report: currentReport,
                edited_report: { raw_text: editedContent },
                feedback_type: 'edited',
            }),
        });

        if (resp.ok) {
            alert('Report published and feedback recorded.');
        } else {
            alert('Failed to publish report.');
        }
    } catch (err) {
        console.error('Publish failed:', err);
        alert('Failed to publish: ' + err.message);
    }
}
