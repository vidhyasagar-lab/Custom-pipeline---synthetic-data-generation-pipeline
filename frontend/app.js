/**
 * SynthGen Studio — Frontend Application
 * SPA with hash routing, full API integration, Chart.js visualizations.
 */

const API = '/api/v1';

// ═══════════════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════════════
const app = {
    files: [],
    filePaths: [],
    runId: null,
    sse: null,
    runData: null,
    currentPage: 'dashboard',
    runs: JSON.parse(localStorage.getItem('synthgen_runs') || '[]'),
    charts: {},
    tableData: [],
    tablePage: 0,
    PAGE_SIZE: 30,
};

const $ = s => document.querySelector(s);
const $$ = s => document.querySelectorAll(s);

// ═══════════════════════════════════════════════════════════════
// ROUTING
// ═══════════════════════════════════════════════════════════════
const PAGE_TITLES = { dashboard: 'Dashboard', generate: 'Generate', results: 'Results', experiments: 'Experiments' };

function navigateTo(page) {
    window.location.hash = page;
}

function handleRoute() {
    const hash = (window.location.hash || '#dashboard').slice(1).split('?')[0];
    const page = PAGE_TITLES[hash] ? hash : 'dashboard';
    app.currentPage = page;

    $$('.page').forEach(p => p.classList.remove('active'));
    const el = $(`#page-${page}`);
    if (el) el.classList.add('active');

    $$('.nav-item').forEach(n => {
        n.classList.toggle('active', n.dataset.page === page);
    });

    $('#topbarTitle').textContent = PAGE_TITLES[page] || 'Dashboard';

    // Close mobile sidebar
    $('#sidebar').classList.remove('open');

    // Page init
    if (page === 'dashboard') initDashboard();
    if (page === 'results') initResults();
    if (page === 'experiments') initExperiments();
}

// ═══════════════════════════════════════════════════════════════
// HEALTH CHECK
// ═══════════════════════════════════════════════════════════════
async function checkHealth() {
    const badge = $('#healthBadge');
    try {
        const r = await fetch(`${API}/health`);
        if (r.ok) {
            badge.className = 'health-badge ok';
            badge.querySelector('.health-label').textContent = 'Connected';
        } else throw 0;
    } catch {
        badge.className = 'health-badge err';
        badge.querySelector('.health-label').textContent = 'Offline';
    }
}

// ═══════════════════════════════════════════════════════════════
// DASHBOARD
// ═══════════════════════════════════════════════════════════════
function initDashboard() {
    let totalConv = 0, totalDocs = 0, qualSum = 0, qualCount = 0;
    app.runs.forEach(r => {
        totalConv += r.conversations || 0;
        totalDocs += r.docs || 0;
        if (r.quality) { qualSum += r.quality; qualCount++; }
    });
    $('#statRuns').textContent = app.runs.length;
    $('#statConversations').textContent = totalConv;
    $('#statDocs').textContent = totalDocs;
    $('#statQuality').textContent = qualCount ? (qualSum / qualCount).toFixed(3) : '\u2014';
    renderRecentRuns();
}

function renderRecentRuns() {
    const body = $('#recentRunsBody');
    if (!app.runs.length) {
        body.innerHTML = '<div class="empty-state"><p>No runs yet. Start your first generation!</p></div>';
        return;
    }
    const list = app.runs.slice().reverse().slice(0, 10);
    body.innerHTML = '<div class="run-list">' + list.map(r => `
        <div class="run-row" onclick="viewRun('${r.id}')">
            <span class="run-row-id">${r.id.slice(0, 8)}</span>
            <span class="run-row-name">${esc(r.name || 'Unnamed Run')}</span>
            <div class="run-row-stats">
                <span>${r.conversations || 0} convs</span>
                <span>${r.quality ? r.quality.toFixed(2) : '\u2014'} quality</span>
                <span>${r.time ? r.time.toFixed(0) + 's' : ''}</span>
            </div>
            <span class="run-row-status badge ${r.status === 'completed' ? 'badge-success' : r.status === 'failed' ? 'badge-error' : 'badge-muted'}">${r.status}</span>
        </div>
    `).join('') + '</div>';
}

function viewRun(runId) {
    app.runId = runId;
    navigateTo('results');
}

function clearRunHistory() {
    if (confirm('Clear all run history?')) {
        app.runs = [];
        localStorage.removeItem('synthgen_runs');
        initDashboard();
        toast('History cleared', 'info');
    }
}

function saveRun(summary) {
    const entry = {
        id: summary.run_id || app.runId,
        name: summary.experiment_name || `Run ${new Date().toLocaleDateString()}`,
        status: summary.status || 'completed',
        conversations: summary.summary?.total_conversations || 0,
        docs: summary.summary?.documents_processed || 0,
        quality: summary.quality?.avg_quality_score || 0,
        time: summary.total_time_seconds || 0,
        date: new Date().toISOString(),
    };
    app.runs = app.runs.filter(r => r.id !== entry.id);
    app.runs.push(entry);
    localStorage.setItem('synthgen_runs', JSON.stringify(app.runs.slice(-50)));
}

// ═══════════════════════════════════════════════════════════════
// GENERATE — FILE UPLOAD
// ═══════════════════════════════════════════════════════════════
function setupUpload() {
    const zone = $('#dropzone');
    const input = $('#fileInput');

    zone.addEventListener('click', () => input.click());
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('dragover'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
    zone.addEventListener('drop', e => {
        e.preventDefault(); zone.classList.remove('dragover');
        addFiles(Array.from(e.dataTransfer.files));
    });
    input.addEventListener('change', () => { addFiles(Array.from(input.files)); input.value = ''; });
}

function addFiles(files) {
    const valid = files.filter(f => /\.(docx|pdf)$/i.test(f.name));
    if (!valid.length) { toast('Only .docx and .pdf files are supported', 'error'); return; }
    valid.forEach(f => {
        if (!app.files.find(ef => ef.name === f.name && ef.size === f.size)) app.files.push(f);
    });
    renderFileList();
    $('#runBtn').disabled = !app.files.length;
}

function removeFile(idx) {
    app.files.splice(idx, 1);
    renderFileList();
    $('#runBtn').disabled = !app.files.length;
}

function renderFileList() {
    const el = $('#fileList');
    if (!app.files.length) { el.innerHTML = ''; return; }
    el.innerHTML = app.files.map((f, i) => `
        <div class="file-item">
            <span class="file-item-info">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                ${esc(f.name)} <span class="file-item-size">(${fmtBytes(f.size)})</span>
            </span>
            <button class="file-item-remove" onclick="removeFile(${i})" title="Remove">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
            </button>
        </div>
    `).join('');
}

// ═══════════════════════════════════════════════════════════════
// GENERATE — PIPELINE EXECUTION
// ═══════════════════════════════════════════════════════════════
async function startPipeline() {
    const btn = $('#runBtn');
    const btnText = $('#runBtnText');
    btn.disabled = true;
    btnText.textContent = 'Uploading...';

    try {
        // Upload files
        const paths = [];
        for (const file of app.files) {
            const fd = new FormData();
            fd.append('file', file);
            const r = await fetch(`${API}/upload-document`, { method: 'POST', body: fd });
            if (!r.ok) throw new Error((await r.json()).detail || 'Upload failed');
            const d = await r.json();
            paths.push(d.path);
            termLog(`Uploaded: ${file.name}`, 'success');
        }
        app.filePaths = paths;

        // Build config
        btnText.textContent = 'Starting pipeline...';
        const cfg = {
            document_paths: paths,
            quality_threshold: parseFloat($('#cfgQuality').value),
            max_retries: parseInt($('#cfgRetries').value),
            enable_multihop: $('#cfgMultihop').checked,
            enable_knowledge_graph: $('#cfgKG').checked,
            enable_advanced_metrics: $('#cfgMetrics').checked,
            experiment_name: $('#cfgExperiment').value.trim() || '',
            personas: getActiveChips('personaChips'),
            query_styles: getActiveChips('styleChips'),
        };

        const r = await fetch(`${API}/generate/async`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(cfg),
        });
        if (!r.ok) throw new Error((await r.json()).detail || 'Pipeline start failed');
        const d = await r.json();
        app.runId = d.run_id;

        showPipelineRunning();
        setTimeout(() => connectSSE(app.runId), 500);

    } catch (err) {
        toast(err.message, 'error');
        btn.disabled = false;
        btnText.textContent = 'Start Generation';
    }
}

function getActiveChips(groupId) {
    return Array.from($$(`#${groupId} .chip.active`)).map(c => c.dataset.value);
}

function showPipelineRunning() {
    $('#pipelineStatus').textContent = 'Running';
    $('#pipelineStatus').className = 'badge badge-accent';
    $('#progressWrap').style.display = 'flex';
    $('#runIndicator').style.display = 'flex';
    updateProgress(0);
    resetPipelineSteps();
    clearTerminal();
    termLog('Pipeline started — run: ' + app.runId.slice(0, 8), 'success');
}

function resetPipelineSteps() {
    $$('.p-step').forEach(s => {
        s.className = 'p-step';
        s.querySelector('.p-step-meta').textContent = '';
    });
}

function updateProgress(pct) {
    $('#progressFill').style.width = pct + '%';
    $('#progressPct').textContent = pct + '%';
}

// ═══════════════════════════════════════════════════════════════
// SSE + POLLING
// ═══════════════════════════════════════════════════════════════
function connectSSE(runId) {
    const es = new EventSource(`${API}/progress/${runId}`);
    app.sse = es;

    es.onmessage = e => {
        try { handleSSE(JSON.parse(e.data)); } catch {}
    };
    es.onerror = () => {
        es.close();
        startPolling(runId);
    };
}

function handleSSE(evt) {
    switch (evt.type) {
        case 'phase_start':
            setStep(evt.phase_number, 'active');
            termLog(`\u25B6 ${evt.phase}`, 'info');
            if (evt.progress_pct) updateProgress(evt.progress_pct);
            break;
        case 'phase_complete':
            setStep(evt.phase_number, 'done');
            termLog(`\u2713 ${evt.phase}`, 'success');
            if (evt.progress_pct) updateProgress(evt.progress_pct);
            break;
        case 'step_start': termLog(`  \u2192 ${evt.step}`); break;
        case 'step_complete': termLog(`  \u2713 ${evt.step}`); break;
        case 'log': termLog(`  [${evt.phase || ''}] ${evt.message}`, evt.level); break;
        case 'pipeline_complete':
            updateProgress(100);
            termLog('\u2550\u2550\u2550 Pipeline completed! \u2550\u2550\u2550', 'success');
            if (app.sse) app.sse.close();
            pipelineFinished();
            break;
        case 'pipeline_failed':
            termLog(`\u2717 FAILED: ${evt.error}`, 'error');
            if (app.sse) app.sse.close();
            pipelineFailed();
            break;
        case 'stream_end':
            if (app.sse) app.sse.close();
            break;
    }
}

function startPolling(runId) {
    termLog('Switched to polling...', 'warning');
    const iv = setInterval(async () => {
        try {
            const r = await fetch(`${API}/status/${runId}`);
            const d = await r.json();
            if (d.status === 'completed') { clearInterval(iv); app.runData = d; pipelineFinished(); }
            else if (d.status === 'failed') { clearInterval(iv); pipelineFailed(); }
        } catch {}
    }, 3000);
}

function setStep(num, state) {
    const el = $(`.p-step[data-step="${num}"]`);
    if (!el) return;
    el.className = `p-step ${state}`;
    if (state === 'done') el.querySelector('.p-step-meta').textContent = 'Done';
    if (state === 'active') el.querySelector('.p-step-meta').textContent = 'Running...';
}

async function pipelineFinished() {
    $('#pipelineStatus').textContent = 'Completed';
    $('#pipelineStatus').className = 'badge badge-success';
    $('#runIndicator').style.display = 'none';
    $('#runBtn').disabled = false;
    $('#runBtnText').textContent = 'Start Generation';

    try {
        const r = await fetch(`${API}/status/${app.runId}`);
        app.runData = await r.json();
        saveRun(app.runData);
        toast('Pipeline completed successfully!', 'success');
    } catch {}
}

function pipelineFailed() {
    $('#pipelineStatus').textContent = 'Failed';
    $('#pipelineStatus').className = 'badge badge-error';
    $('#runIndicator').style.display = 'none';
    $('#runBtn').disabled = false;
    $('#runBtnText').textContent = 'Retry';
    toast('Pipeline failed. Check logs.', 'error');
}

// ═══════════════════════════════════════════════════════════════
// TERMINAL
// ═══════════════════════════════════════════════════════════════
function termLog(msg, level = '') {
    const el = $('#terminal');
    const ts = new Date().toLocaleTimeString();
    const div = document.createElement('div');
    div.className = `terminal-line ${level}`;
    div.innerHTML = `<span class="ts">${ts}</span> <span class="msg">${esc(msg)}</span>`;
    el.appendChild(div);
    el.scrollTop = el.scrollHeight;
}

function clearTerminal() {
    $('#terminal').innerHTML = '';
}

// ═══════════════════════════════════════════════════════════════
// RESULTS
// ═══════════════════════════════════════════════════════════════
function initResults() {
    populateRunSelector();
    const runId = app.runId;
    if (runId && app.runData) {
        $('#runSelector').value = runId;
        renderResults(app.runData);
    }
}

function populateRunSelector() {
    const sel = $('#runSelector');
    const current = sel.value;
    sel.innerHTML = '<option value="">Select a run...</option>';
    app.runs.slice().reverse().forEach(r => {
        const opt = document.createElement('option');
        opt.value = r.id;
        opt.textContent = `${r.id.slice(0, 8)} \u2014 ${r.name || 'Unnamed'} (${r.conversations || 0} convs)`;
        sel.appendChild(opt);
    });
    if (current) sel.value = current;
}

async function loadRun(runId) {
    if (!runId) {
        $('#resultsContent').style.display = 'none';
        $('#resultsEmpty').style.display = 'block';
        $('#resultsActions').style.display = 'none';
        return;
    }
    app.runId = runId;
    try {
        const r = await fetch(`${API}/status/${runId}`);
        if (!r.ok) throw new Error('Run not found');
        app.runData = await r.json();
        renderResults(app.runData);
    } catch {
        toast('Could not load run data. It may have expired.', 'error');
        $('#resultsContent').style.display = 'none';
        $('#resultsEmpty').style.display = 'block';
    }
}

function renderResults(data) {
    $('#resultsEmpty').style.display = 'none';
    $('#resultsContent').style.display = 'block';
    $('#resultsActions').style.display = 'flex';

    const s = data.summary || {};
    const q = data.quality || {};

    // Stats
    const statsHtml = [
        statCard('Conversations', s.total_conversations || s.total_triples || 0, 'accent'),
        statCard('Chunks', s.total_chunks || 0, 'blue'),
        statCard('Multi-Hop', s.multihop_conversations || 0, 'amber'),
        statCard('Rejected', s.rejected || 0, 'red'),
        statCard('Avg Quality', (q.avg_quality_score || 0).toFixed(3), 'green'),
        statCard('Duration', (data.total_time_seconds || 0).toFixed(0) + 's', 'slate'),
    ].join('');
    $('#resultsStats').innerHTML = statsHtml;

    // Charts
    renderQualityChart(q);
    renderTypeChart(data.conversations || []);

    // Advanced Metrics
    const adv = data.advanced_metrics || {};
    if (Object.keys(adv).length && adv.claim_faithfulness_score !== undefined) {
        $('#advMetricsCard').style.display = 'block';
        const metrics = [
            { name: 'Claim Faithfulness', val: adv.claim_faithfulness_score, color: '#6366f1' },
            { name: 'Context Precision', val: adv.context_precision, color: '#3b82f6' },
            { name: 'Context Recall', val: adv.context_recall, color: '#22c55e' },
            { name: 'Answer Relevancy', val: adv.answer_relevancy, color: '#f59e0b' },
        ];
        $('#advMetricsGrid').innerHTML = metrics.filter(m => m.val !== undefined).map(m => `
            <div class="adv-metric">
                <div class="adv-metric-header"><span class="adv-metric-name">${m.name}</span><span class="adv-metric-val">${(m.val || 0).toFixed(3)}</span></div>
                <div class="adv-metric-bar"><div class="adv-metric-fill" style="width:${(m.val || 0) * 100}%;background:${m.color}"></div></div>
            </div>
        `).join('');
    } else {
        $('#advMetricsCard').style.display = 'none';
    }

    // KG Stats
    const kg = data.knowledge_graph_stats || {};
    if (kg.num_nodes) {
        $('#kgCard').style.display = 'block';
        $('#kgBody').innerHTML = `<div class="mini-stats">
            <div class="mini-stat"><span class="mini-stat-val">${kg.num_nodes}</span><span class="mini-stat-lbl">Nodes</span></div>
            <div class="mini-stat"><span class="mini-stat-val">${kg.num_relationships}</span><span class="mini-stat-lbl">Relationships</span></div>
            ${kg.relationship_types ? Object.entries(kg.relationship_types).map(([k,v]) => `<div class="mini-stat"><span class="mini-stat-val">${v}</span><span class="mini-stat-lbl">${k.replace(/_/g,' ')}</span></div>`).join('') : ''}
        </div>`;
    } else { $('#kgCard').style.display = 'none'; }

    // Cost
    const cost = data.cost_summary || {};
    if (cost.total_cost_usd) {
        $('#costCard').style.display = 'block';
        $('#costBody').innerHTML = `<div class="mini-stats">
            <div class="mini-stat"><span class="mini-stat-val">$${cost.total_cost_usd.toFixed(4)}</span><span class="mini-stat-lbl">Total Cost</span></div>
            <div class="mini-stat"><span class="mini-stat-val">${cost.total_calls || 0}</span><span class="mini-stat-lbl">API Calls</span></div>
            <div class="mini-stat"><span class="mini-stat-val">${fmtTokens(cost.total_input_tokens)}</span><span class="mini-stat-lbl">Input Tokens</span></div>
            <div class="mini-stat"><span class="mini-stat-val">${fmtTokens(cost.total_output_tokens)}</span><span class="mini-stat-lbl">Output Tokens</span></div>
        </div>`;
    } else { $('#costCard').style.display = 'none'; }

    // Table
    const convs = data.conversations || [];
    const triples = data.qa_triples || [];
    app.tableData = convs.length ? convs.map((c, i) => {
        const msgs = c.messages || [];
        const m = c.metadata || {};
        return { idx: i, q: msgs[0]?.content || '', a: msgs[1]?.content || '', fq: msgs[2]?.content || '', fa: msgs[3]?.content || '', type: m.question_type || '', score: m.quality_score || 0, source: m.source_file || '', scores: m.validation_scores || {} };
    }) : triples.map((t, i) => ({
        idx: i, q: t.question || '', a: t.answer || '', fq: t.follow_up_q || '', fa: t.follow_up_a || '', type: t.question_type || '', score: t.quality_score || 0, source: (t.retrieved_sources || [])[0] || '', scores: t.validation_scores || {}
    }));
    app.tablePage = 0;
    renderTable();
}

function statCard(label, value, color) {
    const colors = { accent: 'var(--accent)', blue: '#3b82f6', green: '#22c55e', amber: '#f59e0b', red: '#ef4444', slate: '#64748b' };
    const bgs = { accent: 'var(--accent-light)', blue: '#dbeafe', green: '#dcfce7', amber: '#fef3c7', red: '#fee2e2', slate: '#f1f5f9' };
    return `<div class="stat-card"><div class="stat-icon" style="background:${bgs[color] || bgs.slate}"><span style="font-size:1.25rem;font-weight:800;color:${colors[color] || colors.slate}">${typeof value === 'number' && value > 99 ? '' : ''}</span></div><div class="stat-info"><span class="stat-value" style="color:${colors[color]}">${value}</span><span class="stat-label">${label}</span></div></div>`;
}

// ═══════════════════════════════════════════════════════════════
// CHARTS
// ═══════════════════════════════════════════════════════════════
function renderQualityChart(q) {
    const ca = q.criteria_averages || {};
    const labels = Object.keys(ca).map(k => k.replace(/_/g, ' '));
    const values = Object.values(ca);
    if (!labels.length) return;

    if (app.charts.quality) app.charts.quality.destroy();
    app.charts.quality = new Chart($('#qualityChart'), {
        type: 'radar',
        data: {
            labels,
            datasets: [{
                label: 'Score',
                data: values,
                backgroundColor: 'rgba(99,102,241,.15)',
                borderColor: '#6366f1',
                borderWidth: 2,
                pointBackgroundColor: '#6366f1',
                pointRadius: 4,
            }]
        },
        options: {
            scales: { r: { min: 0, max: 1, ticks: { stepSize: 0.2, font: { size: 10 } }, pointLabels: { font: { size: 11, weight: 600 } } } },
            plugins: { legend: { display: false } },
            responsive: true, maintainAspectRatio: false,
        }
    });
}

function renderTypeChart(conversations) {
    const counts = {};
    conversations.forEach(c => {
        const t = c.metadata?.question_type || 'unknown';
        counts[t] = (counts[t] || 0) + 1;
    });
    const labels = Object.keys(counts);
    const values = Object.values(counts);
    if (!labels.length) return;

    const palette = ['#6366f1', '#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#14b8a6'];
    if (app.charts.types) app.charts.types.destroy();
    app.charts.types = new Chart($('#typeChart'), {
        type: 'doughnut',
        data: {
            labels: labels.map(l => l.replace(/_/g, ' ')),
            datasets: [{ data: values, backgroundColor: palette.slice(0, labels.length), borderWidth: 0 }]
        },
        options: {
            plugins: { legend: { position: 'bottom', labels: { padding: 14, font: { size: 11 } } } },
            responsive: true, maintainAspectRatio: false,
            cutout: '55%',
        }
    });
}

// ═══════════════════════════════════════════════════════════════
// TABLE
// ═══════════════════════════════════════════════════════════════
function renderTable(filter = '') {
    let data = app.tableData;
    if (filter) {
        const f = filter.toLowerCase();
        data = data.filter(r => r.q.toLowerCase().includes(f) || r.a.toLowerCase().includes(f));
    }

    $('#convCount').textContent = data.length;
    const start = app.tablePage * app.PAGE_SIZE;
    const page = data.slice(start, start + app.PAGE_SIZE);
    const body = $('#convBody');

    body.innerHTML = page.map((r, i) => {
        const sc = r.score >= 0.8 ? 'q-high' : r.score >= 0.5 ? 'q-med' : 'q-low';
        return `<tr onclick="showQADetail(${r.idx})">
            <td>${start + i + 1}</td>
            <td>${esc(r.q.slice(0, 120))}${r.q.length > 120 ? '...' : ''}</td>
            <td>${esc(r.a.slice(0, 100))}${r.a.length > 100 ? '...' : ''}</td>
            <td><span class="badge ${r.type === 'multihop' ? 'badge-accent' : 'badge-muted'}">${esc(r.type || '\u2014')}</span></td>
            <td><span class="q-score ${sc}">${r.score.toFixed(2)}</span></td>
        </tr>`;
    }).join('');

    const totalPages = Math.ceil(data.length / app.PAGE_SIZE);
    $('#tableFooter').innerHTML = totalPages > 1
        ? `<span>Page ${app.tablePage + 1} of ${totalPages}</span>
           <span style="margin-left:auto;display:flex;gap:.5rem">
             <button class="btn btn-ghost btn-sm" onclick="tablePrev()" ${app.tablePage === 0 ? 'disabled' : ''}>\u2190 Prev</button>
             <button class="btn btn-ghost btn-sm" onclick="tableNext()" ${app.tablePage >= totalPages - 1 ? 'disabled' : ''}>Next \u2192</button>
           </span>`
        : `<span>${data.length} conversation${data.length !== 1 ? 's' : ''}</span>`;
}

function tablePrev() { if (app.tablePage > 0) { app.tablePage--; renderTable($('#tableSearch').value); } }
function tableNext() { app.tablePage++; renderTable($('#tableSearch').value); }

function showQADetail(idx) {
    const r = app.tableData[idx];
    if (!r) return;

    const scoresHtml = Object.entries(r.scores).map(([k, v]) => {
        if (typeof v !== 'number') return '';
        const pct = (v * 100).toFixed(0);
        const color = v >= 0.8 ? 'var(--success)' : v >= 0.5 ? 'var(--warning)' : 'var(--error)';
        return `<div class="adv-metric">
            <div class="adv-metric-header"><span class="adv-metric-name">${k.replace(/_/g,' ')}</span><span class="adv-metric-val">${v.toFixed(2)}</span></div>
            <div class="adv-metric-bar"><div class="adv-metric-fill" style="width:${pct}%;background:${color}"></div></div>
        </div>`;
    }).join('');

    const html = `
        <button class="modal-close" onclick="closeModal()">\u2715</button>
        <h3>Conversation #${idx + 1}</h3>
        <div class="modal-field"><div class="modal-field-label">Question</div><div class="modal-field-value">${esc(r.q)}</div></div>
        <div class="modal-field"><div class="modal-field-label">Answer</div><div class="modal-field-value">${esc(r.a)}</div></div>
        ${r.fq ? `<div class="modal-field"><div class="modal-field-label">Follow-up Question</div><div class="modal-field-value">${esc(r.fq)}</div></div>` : ''}
        ${r.fa ? `<div class="modal-field"><div class="modal-field-label">Follow-up Answer</div><div class="modal-field-value">${esc(r.fa)}</div></div>` : ''}
        <div class="modal-field"><div class="modal-field-label">Type</div><div class="modal-field-value"><span class="badge ${r.type === 'multihop' ? 'badge-accent' : 'badge-muted'}">${r.type || 'standard'}</span> &nbsp; Score: <span class="q-score ${r.score >= .8 ? 'q-high' : r.score >= .5 ? 'q-med' : 'q-low'}">${r.score.toFixed(3)}</span></div></div>
        ${scoresHtml ? `<div class="modal-field"><div class="modal-field-label">Validation Scores</div><div class="adv-metrics-grid" style="margin-top:.5rem">${scoresHtml}</div></div>` : ''}
    `;
    openModal(html);
}

// ═══════════════════════════════════════════════════════════════
// EXPORT
// ═══════════════════════════════════════════════════════════════
async function exportJSON() {
    if (!app.runId) return;
    try {
        const r = await fetch(`${API}/results/${app.runId}/export?format=json`);
        if (!r.ok) throw 0;
        const d = await r.json();
        downloadBlob(new Blob([JSON.stringify(d, null, 2)], { type: 'application/json' }), `synthgen_${app.runId.slice(0, 8)}.json`);
    } catch { toast('Export failed', 'error'); }
}

async function exportExcel() {
    if (!app.runId) return;
    try {
        const r = await fetch(`${API}/results/${app.runId}/export?format=excel`);
        if (!r.ok) throw 0;
        downloadBlob(await r.blob(), `synthgen_${app.runId.slice(0, 8)}.xlsx`);
    } catch { toast('Export failed', 'error'); }
}

function downloadBlob(blob, name) {
    const u = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = u; a.download = name; a.click();
    URL.revokeObjectURL(u);
}

// ═══════════════════════════════════════════════════════════════
// EXPERIMENTS
// ═══════════════════════════════════════════════════════════════
async function initExperiments() {
    try {
        const r = await fetch(`${API}/experiments`);
        if (!r.ok) return;
        const d = await r.json();
        renderExperiments(d.experiments || []);
    } catch {}
}

function renderExperiments(exps) {
    const body = $('#expBody');
    if (!exps.length) { body.innerHTML = '<div class="empty-state"><p>No experiments recorded yet.</p></div>'; return; }

    body.innerHTML = `
        <table class="exp-table">
            <thead><tr><th></th><th>Name</th><th>ID</th><th>Conversations</th><th>Quality</th><th>Cost</th><th>Date</th><th></th></tr></thead>
            <tbody>${exps.map(e => {
                const res = e.result || {};
                return `<tr>
                    <td><input type="checkbox" class="exp-check" data-id="${e.experiment_id}"></td>
                    <td><strong>${esc(e.name || 'Unnamed')}</strong></td>
                    <td style="font-family:var(--mono);font-size:.78rem;color:var(--text-muted)">${e.experiment_id}</td>
                    <td>${res.after_validation || 0}</td>
                    <td>${(res.avg_quality_score || 0).toFixed(3)}</td>
                    <td>${res.total_cost_usd ? '$' + res.total_cost_usd.toFixed(4) : '\u2014'}</td>
                    <td style="font-size:.78rem;color:var(--text-muted)">${e.timestamp ? new Date(e.timestamp * 1000).toLocaleDateString() : ''}</td>
                    <td><button class="btn btn-ghost btn-sm" onclick="deleteExperiment('${e.experiment_id}')" title="Delete"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg></button></td>
                </tr>`;
            }).join('')}</tbody>
        </table>
        <div style="margin-top:1rem"><button class="btn btn-outline btn-sm" onclick="compareSelected()">Compare Selected (2)</button></div>
    `;
}

async function compareSelected() {
    const checks = Array.from($$('.exp-check:checked'));
    if (checks.length !== 2) { toast('Select exactly 2 experiments to compare', 'info'); return; }
    const [a, b] = checks.map(c => c.dataset.id);
    try {
        const r = await fetch(`${API}/experiments/compare/${a}/${b}`);
        if (!r.ok) throw 0;
        const d = await r.json();
        renderComparison(d);
    } catch { toast('Comparison failed', 'error'); }
}

function renderComparison(data) {
    const card = $('#compareCard');
    card.style.display = 'block';
    const a = data.experiment_a || {};
    const b = data.experiment_b || {};
    const rd = data.result_diff || {};

    let html = `<div class="compare-grid"><div class="compare-col"><h4>${esc(a.name || a.id)}</h4>`;
    for (const [k, v] of Object.entries(rd)) {
        if (typeof v.a === 'object') continue;
        html += `<div class="compare-item"><span class="compare-key">${k.replace(/_/g, ' ')}</span><span class="compare-val">${fmtVal(v.a)}</span></div>`;
    }
    html += `</div><div class="compare-col"><h4>${esc(b.name || b.id)}</h4>`;
    for (const [k, v] of Object.entries(rd)) {
        if (typeof v.b === 'object') continue;
        const delta = v.delta !== undefined ? `<span class="compare-delta ${v.delta >= 0 ? 'delta-pos' : 'delta-neg'}">${v.delta >= 0 ? '+' : ''}${fmtVal(v.delta)}</span>` : '';
        html += `<div class="compare-item"><span class="compare-key">${k.replace(/_/g, ' ')}</span><span class="compare-val">${fmtVal(v.b)}${delta}</span></div>`;
    }
    html += '</div></div>';
    $('#compareBody').innerHTML = html;
    card.scrollIntoView({ behavior: 'smooth' });
}

async function deleteExperiment(id) {
    if (!confirm('Delete this experiment?')) return;
    try {
        await fetch(`${API}/experiments/${id}`, { method: 'DELETE' });
        toast('Experiment deleted', 'info');
        initExperiments();
    } catch { toast('Delete failed', 'error'); }
}

// ═══════════════════════════════════════════════════════════════
// MODAL
// ═══════════════════════════════════════════════════════════════
function openModal(html) {
    $('#modalContent').innerHTML = html;
    $('#modalOverlay').classList.add('show');
}
function closeModal() { $('#modalOverlay').classList.remove('show'); }

// ═══════════════════════════════════════════════════════════════
// TOAST
// ═══════════════════════════════════════════════════════════════
function toast(msg, type = 'info') {
    const el = document.createElement('div');
    el.className = `toast toast-${type}`;
    el.textContent = msg;
    $('#toastContainer').appendChild(el);
    setTimeout(() => { el.style.opacity = '0'; setTimeout(() => el.remove(), 300); }, 4000);
}

// ═══════════════════════════════════════════════════════════════
// UTILS
// ═══════════════════════════════════════════════════════════════
function esc(s) { const d = document.createElement('div'); d.textContent = s || ''; return d.innerHTML; }
function fmtBytes(b) { if (!b) return '0 B'; const u = ['B','KB','MB']; const i = Math.floor(Math.log(b)/Math.log(1024)); return (b/Math.pow(1024,i)).toFixed(1)+' '+u[i]; }
function fmtTokens(n) { if (!n) return '0'; if (n >= 1e6) return (n/1e6).toFixed(1)+'M'; if (n >= 1e3) return (n/1e3).toFixed(1)+'K'; return n.toString(); }
function fmtVal(v) { if (typeof v === 'number') return v % 1 ? v.toFixed(4) : v.toString(); return v ?? '\u2014'; }

// ═══════════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════════
function init() {
    // Routing
    window.addEventListener('hashchange', handleRoute);
    handleRoute();

    // Mobile menu
    $('#mobileMenuBtn').addEventListener('click', () => $('#sidebar').classList.toggle('open'));
    document.addEventListener('click', e => {
        if (window.innerWidth <= 768 && !e.target.closest('.sidebar') && !e.target.closest('.mobile-menu-btn')) {
            $('#sidebar').classList.remove('open');
        }
    });

    // Upload
    setupUpload();

    // Config range slider
    $('#cfgQuality').addEventListener('input', e => { $('#cfgQualityVal').textContent = parseFloat(e.target.value).toFixed(2); });

    // Chip toggles
    document.addEventListener('click', e => {
        if (e.target.matches('.chip')) { e.preventDefault(); e.target.classList.toggle('active'); }
    });

    // Run button
    $('#runBtn').addEventListener('click', startPipeline);

    // Clear logs
    $('#clearLogsBtn').addEventListener('click', clearTerminal);

    // Results
    $('#runSelector').addEventListener('change', e => loadRun(e.target.value));
    $('#tableSearch').addEventListener('input', e => { app.tablePage = 0; renderTable(e.target.value); });
    $('#exportJsonBtn').addEventListener('click', exportJSON);
    $('#exportExcelBtn').addEventListener('click', exportExcel);

    // Experiments
    $('#refreshExpBtn').addEventListener('click', initExperiments);

    // Modal close
    $('#modalOverlay').addEventListener('click', e => { if (e.target === $('#modalOverlay')) closeModal(); });
    document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });

    // Health check
    checkHealth();
    setInterval(checkHealth, 30000);
}

document.addEventListener('DOMContentLoaded', init);
