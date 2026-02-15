/**
 * Risk Radar scatter plot and project data loading.
 * Renders ticket risk data as an interactive scatter plot on canvas.
 */

const COLORS = {
    High: '#f85149',
    Medium: '#d29922',
    Low: '#3fb950',
    grid: '#2d3548',
    text: '#8b949e',
    axis: '#e1e4e8',
};

/**
 * Load project risk data from the API and populate the dashboard.
 */
/**
 * Load project risk data from the API and populate the dashboard.
 *
 * Shows a spinner overlay while requests are in flight and renders
 * an error banner at the top of the page if any request fails, so
 * users immediately see what went wrong instead of staring at empty tables.
 */
async function loadProjectData(projectKey) {
    showLoadingSpinner(true);
    dismissBanner();

    try {
        const [riskResp, radarResp] = await Promise.all([
            fetch(`/api/v1/projects/${projectKey}/risk`),
            fetch(`/api/v1/projects/${projectKey}/radar`),
        ]);

        if (riskResp.ok) {
            const riskData = await riskResp.json();
            renderRiskSummary(riskData.summary);
            renderRiskTable(riskData.tickets);
        } else if (riskResp.status === 401) {
            showBanner('Authentication failed. Check your API key.', 'error');
        } else if (riskResp.status === 429) {
            showBanner('Rate limit exceeded. Please wait and try again.', 'warn');
        } else {
            renderError('risk-table-body', 'No risk data available. Ingest Jira data first.');
        }

        if (radarResp.ok) {
            const radarData = await radarResp.json();
            renderRadarChart(radarData.radar_points);
        }
    } catch (err) {
        console.error('Failed to load project data:', err);
        showBanner('Failed to connect to the API. Is the backend running?', 'error');
        renderError('risk-table-body', 'Failed to connect to the API.');
    } finally {
        showLoadingSpinner(false);
    }
}

/**
 * Toggle a loading spinner overlay on the main content area.
 * Creates the spinner element lazily on first call.
 */
function showLoadingSpinner(show) {
    let spinner = document.getElementById('loading-spinner');
    if (!spinner) {
        spinner = document.createElement('div');
        spinner.id = 'loading-spinner';
        spinner.style.cssText =
            'position:fixed;top:0;left:0;width:100%;height:100%;' +
            'background:rgba(15,20,25,.7);display:flex;align-items:center;' +
            'justify-content:center;z-index:200;';
        spinner.innerHTML =
            '<div style="border:4px solid #2d3548;border-top:4px solid #58a6ff;' +
            'border-radius:50%;width:40px;height:40px;animation:spin 0.8s linear infinite;"></div>' +
            '<style>@keyframes spin{to{transform:rotate(360deg)}}</style>';
        document.body.appendChild(spinner);
    }
    spinner.style.display = show ? 'flex' : 'none';
}

/**
 * Show a dismissible banner at the top of the page for errors or warnings.
 * Communicates API issues (auth failures, rate limiting, network errors)
 * to users in a prominent but non-blocking way.
 */
function showBanner(message, level) {
    dismissBanner();
    const colors = { error: '#f85149', warn: '#d29922', info: '#58a6ff' };
    const banner = document.createElement('div');
    banner.id = 'status-banner';
    banner.style.cssText =
        `position:fixed;top:0;left:0;width:100%;padding:10px 20px;` +
        `background:${colors[level] || colors.info};color:#fff;font:14px sans-serif;` +
        `text-align:center;z-index:300;cursor:pointer;`;
    banner.textContent = message + '  (click to dismiss)';
    banner.onclick = () => banner.remove();
    document.body.prepend(banner);
}

function dismissBanner() {
    const existing = document.getElementById('status-banner');
    if (existing) existing.remove();
}

/**
 * Update the risk summary stat cards.
 */
function renderRiskSummary(summary) {
    if (!summary) return;
    document.getElementById('high-count').textContent = summary.high_risk || 0;
    document.getElementById('medium-count').textContent = summary.medium_risk || 0;
    document.getElementById('low-count').textContent = summary.low_risk || 0;
    document.getElementById('total-count').textContent = summary.total_tickets || 0;
}

/**
 * Populate the risk details table.
 */
function renderRiskTable(tickets) {
    const tbody = document.getElementById('risk-table-body');
    if (!tickets || tickets.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" class="loading">No tickets found.</td></tr>';
        return;
    }

    // Sort by risk score descending
    tickets.sort((a, b) => (b.risk_score || 0) - (a.risk_score || 0));

    tbody.innerHTML = tickets.map(t => {
        const level = (t.risk_level || 'Low').toLowerCase();
        const score = ((t.risk_score || 0) * 100).toFixed(0);
        return `<tr>
            <td><strong>${t.issue_key || '-'}</strong></td>
            <td>${(t.summary || '').substring(0, 60)}</td>
            <td>${t.status || '-'}</td>
            <td>${score}%</td>
            <td><span class="risk-badge ${level}">${t.risk_level || '-'}</span></td>
            <td>${(t.days_in_status || 0).toFixed(0)}d</td>
            <td>${t.dependency_count || 0}</td>
        </tr>`;
    }).join('');
}

/**
 * Render the Risk Radar scatter plot on the canvas.
 * X: Technical Complexity, Y: Team Sentiment, Size: Story Points, Color: Risk Level
 */
function renderRadarChart(points) {
    const canvas = document.getElementById('radar-chart');
    if (!canvas || !points || points.length === 0) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;

    // Set canvas size accounting for device pixel ratio
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = 500 * dpr;
    canvas.style.width = rect.width + 'px';
    canvas.style.height = '500px';
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = 500;
    const padding = { top: 40, right: 40, bottom: 60, left: 70 };
    const plotW = w - padding.left - padding.right;
    const plotH = h - padding.top - padding.bottom;

    // Compute data ranges
    const xValues = points.map(p => p.x);
    const yValues = points.map(p => p.y);
    const xMin = Math.min(0, ...xValues);
    const xMax = Math.max(...xValues) * 1.1 || 10;
    const yMin = Math.min(-1, ...yValues);
    const yMax = Math.max(1, ...yValues);

    function scaleX(v) { return padding.left + ((v - xMin) / (xMax - xMin)) * plotW; }
    function scaleY(v) { return padding.top + plotH - ((v - yMin) / (yMax - yMin)) * plotH; }

    // Clear
    ctx.fillStyle = '#1a1f2e';
    ctx.fillRect(0, 0, w, h);

    // Grid lines
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 5; i++) {
        const gx = padding.left + (plotW / 5) * i;
        const gy = padding.top + (plotH / 5) * i;
        ctx.beginPath(); ctx.moveTo(gx, padding.top); ctx.lineTo(gx, padding.top + plotH); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(padding.left, gy); ctx.lineTo(padding.left + plotW, gy); ctx.stroke();
    }

    // Zero line for sentiment (y=0)
    const zeroY = scaleY(0);
    ctx.strokeStyle = '#58a6ff44';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(padding.left, zeroY); ctx.lineTo(padding.left + plotW, zeroY); ctx.stroke();
    ctx.setLineDash([]);

    // Axes labels
    ctx.fillStyle = COLORS.axis;
    ctx.font = '13px -apple-system, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Technical Complexity', w / 2, h - 10);
    ctx.save();
    ctx.translate(16, h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Team Sentiment', 0, 0);
    ctx.restore();

    // Axis tick labels
    ctx.fillStyle = COLORS.text;
    ctx.font = '11px monospace';
    ctx.textAlign = 'center';
    for (let i = 0; i <= 5; i++) {
        const xVal = xMin + ((xMax - xMin) / 5) * i;
        ctx.fillText(xVal.toFixed(1), scaleX(xVal), padding.top + plotH + 20);
    }
    ctx.textAlign = 'right';
    for (let i = 0; i <= 5; i++) {
        const yVal = yMin + ((yMax - yMin) / 5) * i;
        ctx.fillText(yVal.toFixed(1), padding.left - 10, scaleY(yVal) + 4);
    }

    // Draw bubbles
    points.forEach(p => {
        const cx = scaleX(p.x);
        const cy = scaleY(p.y);
        const radius = Math.max(4, Math.min(p.size * 3, 30));
        const color = COLORS[p.color] || COLORS.Low;

        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, Math.PI * 2);
        ctx.fillStyle = color + '66';
        ctx.fill();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Label for larger bubbles
        if (radius > 8) {
            ctx.fillStyle = COLORS.axis;
            ctx.font = '10px monospace';
            ctx.textAlign = 'center';
            ctx.fillText(p.issue_key || '', cx, cy - radius - 4);
        }
    });

    // Legend
    const legendX = w - padding.right - 120;
    const legendY = padding.top + 10;
    ['High', 'Medium', 'Low'].forEach((level, i) => {
        ctx.beginPath();
        ctx.arc(legendX, legendY + i * 22, 6, 0, Math.PI * 2);
        ctx.fillStyle = COLORS[level];
        ctx.fill();
        ctx.fillStyle = COLORS.text;
        ctx.font = '12px -apple-system, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(level + ' Risk', legendX + 14, legendY + i * 22 + 4);
    });

    // Attach interactive tooltip for ticket details on hover
    attachRadarTooltip(canvas, points, scaleX, scaleY);
}

/**
 * Attach tooltip hover behavior to the radar canvas.
 *
 * Tracks mouse position and checks proximity to each rendered bubble.
 * When the cursor is within a bubble's radius the tooltip <div> is
 * positioned beside the cursor and populated with ticket details
 * (key, risk score, status, assignee).  Moving away hides the tooltip.
 */
function attachRadarTooltip(canvas, points, scaleX, scaleY) {
    let tooltip = document.getElementById('radar-tooltip');
    if (!tooltip) {
        tooltip = document.createElement('div');
        tooltip.id = 'radar-tooltip';
        tooltip.style.cssText =
            'position:absolute;pointer-events:none;background:#1a1f2e;' +
            'border:1px solid #2d3548;border-radius:6px;padding:8px 12px;' +
            'color:#e1e4e8;font:12px monospace;display:none;z-index:100;' +
            'box-shadow:0 4px 12px rgba(0,0,0,.4);max-width:250px;';
        document.body.appendChild(tooltip);
    }

    canvas.addEventListener('mousemove', (e) => {
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;

        let hit = null;
        for (const p of points) {
            const cx = scaleX(p.x);
            const cy = scaleY(p.y);
            const r = Math.max(4, Math.min(p.size * 3, 30));
            const dist = Math.sqrt((mx - cx) ** 2 + (my - cy) ** 2);
            if (dist <= r) { hit = p; break; }
        }

        if (hit) {
            tooltip.innerHTML =
                `<strong>${hit.issue_key}</strong><br>` +
                `${hit.summary || ''}<br>` +
                `<span style="color:${COLORS[hit.color]}">${hit.color} Risk</span> ` +
                `(${(hit.risk_score * 100).toFixed(0)}%)<br>` +
                `Status: ${hit.status || '-'} (${hit.days_in_status.toFixed(0)}d)<br>` +
                `Assignee: ${hit.assignee || 'Unassigned'}`;
            tooltip.style.display = 'block';
            tooltip.style.left = (e.pageX + 14) + 'px';
            tooltip.style.top = (e.pageY - 10) + 'px';
            canvas.style.cursor = 'pointer';
        } else {
            tooltip.style.display = 'none';
            canvas.style.cursor = 'default';
        }
    });

    canvas.addEventListener('mouseleave', () => {
        tooltip.style.display = 'none';
    });
}

function renderError(elementId, message) {
    const el = document.getElementById(elementId);
    if (el) {
        el.innerHTML = `<tr><td colspan="7" class="loading">${message}</td></tr>`;
    }
}
