// AgentMonitor frontend — vanilla JS, no build step.

const $  = (sel, root=document) => root.querySelector(sel);
const $$ = (sel, root=document) => Array.from(root.querySelectorAll(sel));

const fmtTs = ts => ts ? new Date(ts).toLocaleString() : '';
const escape = s => (s ?? '').toString()
  .replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    const t = await r.text();
    throw new Error(`${r.status} ${r.statusText}: ${t}`);
  }
  const ct = r.headers.get('content-type') || '';
  return ct.includes('json') ? r.json() : r.text();
}

// ---------- nav ----------
function showPanel(name) {
  $$('.panel').forEach(p => p.classList.add('hidden'));
  $('#panel-' + name).classList.remove('hidden');
  $$('.nav-link').forEach(a => a.classList.remove('active'));
  $(`.nav-link[data-panel="${name}"]`).classList.add('active');
  // lazy-load
  if (name === 'overview') loadOverview();
  if (name === 'agents')   loadAgents();
  if (name === 'runs')     loadRuns();
  if (name === 'cost')     loadCost();
  if (name === 'posture')  loadPosture();
  if (name === 'memory')   loadMemoryRecent();
  if (name === 'thoughts') loadThoughts();
  if (name === 'codescan') loadCodeScan();
  if (name === 'scanobs')  loadScanObs();
  if (name === 'detonations') loadDetonations();
  if (name === 'welcome')  loadWelcome();
}
$$('.nav-link').forEach(a => a.addEventListener('click', () => showPanel(a.dataset.panel)));

// ---------- top status pills + footer strip ----------
const _bootTs = Date.now();
function _fmtUptime(ms) {
  const s = Math.floor(ms/1000);
  const h = Math.floor(s/3600), m = Math.floor((s%3600)/60), ss = s%60;
  if (h) return `${h}h${String(m).padStart(2,'0')}m`;
  if (m) return `${m}m${String(ss).padStart(2,'0')}s`;
  return `${ss}s`;
}
async function refreshStatus() {
  try {
    const [s, agents, runs] = await Promise.all([
      api('/api/status'),
      api('/api/agents').catch(()=>({agents:[]})),
      api('/api/runs?limit=200').catch(()=>({runs:[]})),
    ]);

    // -- top pills --
    const pills = $$('#status-pills .pill');
    pills.forEach(p => {
      const k = p.dataset.key;
      let ok=false, txt='?';
      if (k === 'ollama')  { ok = s.ollama.up;     txt = ok ? 'up' : 'down'; }
      if (k === 'probes')  {
        // Llama Guard is the primary signal; toy probes are secondary.
        const lg = s.interp.llama_guard && s.interp.llama_guard.ready;
        ok = lg || s.interp.loaded;
        txt = lg ? 'guard' : (s.interp.loaded ? 'toy' : 'off');
      }
      if (k === 'browser') { ok = s.browser.open;  txt = ok ? 'open' : 'idle'; }
      p.classList.remove('ok','bad','warn');
      p.classList.add(ok ? 'ok' : (k==='browser' ? 'warn' : 'bad'));
      p.querySelector('em').textContent = txt;
    });

    // -- footer strip --
    const errs = agents.agents.reduce((a,b)=>a+(b.runs_error||0), 0);
    const totalRuns = agents.agents.reduce((a,b)=>a+(b.runs_total||0), 0);
    const harmFlagged = (runs.runs||[]).filter(r => (r.harm_max||0) >= 0.5).length;
    const lgReady = s.interp.llama_guard && s.interp.llama_guard.ready;
    const setText = (id, val, cls) => {
      const el = document.getElementById(id);
      if (!el) return;
      el.textContent = val;
      el.classList.remove('amber','good','bad');
      if (cls) el.classList.add(cls);
    };
    setText('f-agents',  agents.agents.length);
    setText('f-runs',    totalRuns);
    setText('f-errors',  errs, errs > 0 ? 'bad' : null);
    setText('f-harm',    harmFlagged, harmFlagged > 0 ? 'amber' : null);
    setText('f-lg',      lgReady ? (s.interp.llama_guard.model || 'ready') : 'off',
                          lgReady ? 'good' : 'bad');
    const nla = s.nla || {};
    let nlaTxt = 'off', nlaCls = 'bad';
    if      (nla.backend === 'remote')             { nlaTxt = 'remote';      nlaCls = 'good'; }
    else if (nla.backend === 'local_activations')  { nlaTxt = 'activations'; nlaCls = 'good'; }
    else if (nla.backend === 'prompted_approx')    { nlaTxt = 'approx';      nlaCls = 'amber'; }
    // surface async queue depth as a small badge
    try {
      const w = await api('/api/nla/queue').catch(()=>null);
      if (w && w.running && (w.queue_depth > 0 || (w.current && w.current.run_id))) {
        nlaTxt += ` · q${w.queue_depth}` + (w.current ? '+1' : '');
      }
    } catch (_) {}
    setText('f-nla', nlaTxt, nlaCls);
    setText('f-ollama',  s.ollama.up ? `${(s.ollama.models||[]).length} models` : 'down',
                          s.ollama.up ? 'good' : 'bad');
    setText('f-browser', s.browser.open ? 'open' : 'idle');
    setText('f-uptime',  _fmtUptime(Date.now() - _bootTs));
  } catch (e) {
    console.warn('status refresh failed', e);
  }
}

// ---------- overview ----------
async function loadOverview() {
  const [agents, runs, status] = await Promise.all([
    api('/api/agents'), api('/api/runs?limit=10'), api('/api/status'),
  ]);
  const totalRuns = agents.agents.reduce((a,b) => a + (b.runs_total||0), 0);
  const errRuns   = agents.agents.reduce((a,b) => a + (b.runs_error||0), 0);
  const probes    = status.interp.probes || {};
  const probesOn  = Object.values(probes).filter(Boolean).length;

  $('#overview-cards').innerHTML = `
    <div class="card"><div class="label">Agents</div><div class="value">${agents.agents.length}</div><div class="sub">classes registered</div></div>
    <div class="card"><div class="label">Runs (all-time)</div><div class="value">${totalRuns}</div><div class="sub">${errRuns} errored</div></div>
    <div class="card"><div class="label">Safety</div><div class="value">${status.interp.llama_guard && status.interp.llama_guard.ready ? '●' : '○'}</div><div class="sub">Llama Guard 3 ${status.interp.llama_guard && status.interp.llama_guard.ready ? 'ready' : 'offline'} · ${probesOn}/3 toy probes</div></div>
    <div class="card"><div class="label">Ollama</div><div class="value">${status.ollama.up ? '●' : '○'}</div><div class="sub">${status.ollama.up ? (status.ollama.models||[]).length+' models' : 'offline'}</div></div>
  `;
  $('#overview-runs').innerHTML = renderRunsTable(runs.runs);
}

// ---------- agents ----------
async function loadAgents() {
  const data = await api('/api/agents');
  if (!data.agents.length) {
    $('#agents-list').innerHTML = '<p class="muted">No agents have run yet. Use <code>python -m agent_monitor.smoke --with-agent</code> or <code>python -m automations.customer_support</code> to populate.</p>';
    return;
  }
  $('#agents-list').innerHTML = `
    <table>
      <thead><tr><th>Name</th><th>Description</th><th>Runs</th><th>Done</th><th>Errors</th><th>Created</th></tr></thead>
      <tbody>
        ${data.agents.map(a => `
          <tr>
            <td class="td-mono">${escape(a.name)}</td>
            <td class="muted">${escape(a.description||'')}</td>
            <td class="td-num">${a.runs_total||0}</td>
            <td class="td-num">${a.runs_done||0}</td>
            <td class="td-num">${a.runs_error||0}</td>
            <td class="td-mono muted">${escape(fmtTs(a.created_at))}</td>
          </tr>`).join('')}
      </tbody>
    </table>`;
}

// ---------- runs ----------
function renderRunsTable(runs) {
  if (!runs.length) return '<p class="muted">No runs yet.</p>';
  return `
    <table>
      <thead><tr>
        <th>#</th><th>Runtime</th><th>Agent</th><th>External</th><th>Status</th>
        <th>Harm</th><th title="USD cost from reported tokens × list price; — = unknown">$</th>
        <th>Elapsed</th><th>Started</th><th></th>
      </tr></thead>
      <tbody>
        ${runs.map(r => {
          const harm = r.harm_max == null ? '—' : Number(r.harm_max).toFixed(2);
          const cls = r.harm_max == null ? 'harm-low'
                    : r.harm_max >= 0.7 ? 'harm-high'
                    : r.harm_max >= 0.4 ? 'harm-mid' : 'harm-low';
          const kind = r.agent_kind || 'qwen-vllm';
          return `
            <tr>
              <td class="td-mono">${r.id}</td>
              <td><span class="kind-pill kind-${escape(kind)}">${escape(kind)}</span></td>
              <td class="td-mono">${escape(r.agent_name)}</td>
              <td class="td-mono muted">${escape(r.external_id||'')}</td>
              <td><span class="badge ${r.status}">${r.status}</span></td>
              <td class="td-num ${cls}">${harm}${r.harm_max>=0.7?' ⚑':''}</td>
              <td class="td-num muted" title="tokens in/out: ${r.tokens_in||0}/${r.tokens_out||0}; model: ${escape(r.model_id||'')}">${r.cost_usd == null ? '—' : '$'+Number(r.cost_usd).toFixed(4)}</td>
              <td class="td-num muted">${r.elapsed_ms == null ? '' : (r.elapsed_ms/1000).toFixed(2)+'s'}</td>
              <td class="td-mono muted">${escape(fmtTs(r.started_at))}</td>
              <td><button data-watch="${r.id}" data-kind="${escape(kind)}">Watch</button></td>
            </tr>`;
        }).join('')}
      </tbody>
    </table>`;
}

async function loadRuns() {
  const limit = $('#runs-limit').value || 50;
  const aid   = $('#runs-agent-filter').value;
  const kind  = $('#runs-kind-filter') ? $('#runs-kind-filter').value : '';
  const params = new URLSearchParams();
  params.set('limit', limit);
  if (aid)  params.set('agent_id', aid);
  if (kind) params.set('kind', kind);
  const [runs, agents] = await Promise.all([
    api('/api/runs?' + params.toString()),
    api('/api/agents'),
  ]);
  // populate agent filter once, with kind suffix so users see what runtime each is
  const sel = $('#runs-agent-filter');
  if (sel.options.length <= 1) {
    agents.agents.forEach(a => {
      const opt = document.createElement('option');
      opt.value = a.id;
      opt.textContent = a.kind && a.kind !== 'qwen-vllm'
        ? `${a.name} · ${a.kind}` : a.name;
      sel.appendChild(opt);
    });
  }
  $('#runs-table').innerHTML = renderRunsTable(runs.runs);
  $$('button[data-watch]').forEach(b => b.addEventListener('click', () => {
    $('#live-run-id').value = b.dataset.watch;
    window._lastWatchKind = b.dataset.kind || 'qwen-vllm';
    showPanel('live');
    connectLive();
  }));
}
$('#runs-refresh').addEventListener('click', loadRuns);

// ---------- cost (v1.7) ----------
async function loadCost() {
  const gb = $('#cost-group-by').value;
  const since = $('#cost-since').value;
  const params = new URLSearchParams({ group_by: gb });
  if (since) params.set('since', since);
  const [summary, pricing] = await Promise.all([
    api('/api/runs/cost_summary?' + params.toString()),
    api('/api/pricing'),
  ]);

  // totals
  const t = summary.total || {};
  const totalCost = t.cost_usd == null ? '—' : '$' + Number(t.cost_usd).toFixed(4);
  const totalTok  = (t.tokens_in || 0) + (t.tokens_out || 0);
  const unknownPct = t.n_runs ? Math.round(100 * (t.unknown_cost_runs || 0) / t.n_runs) : 0;
  $('#cost-totals').innerHTML = `
    <div class="card"><div class="label">Total cost</div><div class="value">${totalCost}</div>
      <div class="sub">${t.n_runs || 0} finished runs</div></div>
    <div class="card"><div class="label">Total tokens</div><div class="value">${totalTok.toLocaleString()}</div>
      <div class="sub">${(t.tokens_in||0).toLocaleString()} in / ${(t.tokens_out||0).toLocaleString()} out</div></div>
    <div class="card"><div class="label">Unpriced runs</div><div class="value">${t.unknown_cost_runs || 0}</div>
      <div class="sub">${unknownPct}% — unknown model or no tokens reported</div></div>
    <div class="card"><div class="label">Avg $/run</div>
      <div class="value">${t.cost_usd != null && t.n_runs ? '$' + (t.cost_usd / Math.max(1, t.n_runs - (t.unknown_cost_runs||0))).toFixed(4) : '—'}</div>
      <div class="sub">excludes unpriced runs</div></div>
  `;

  // breakdown table
  if (!summary.buckets.length) {
    $('#cost-breakdown').innerHTML = '<p class="muted">No finished runs in this window.</p>';
  } else {
    $('#cost-breakdown').innerHTML = `
      <table>
        <thead><tr>
          <th>${escape(gb)}</th><th>Runs</th><th>Tokens (in)</th>
          <th>Tokens (out)</th><th>Total $</th><th>$/run</th>
          <th>Avg elapsed</th><th>Unpriced</th>
        </tr></thead>
        <tbody>
          ${summary.buckets.map(b => {
            const cost = b.cost_usd == null ? '—' : '$' + Number(b.cost_usd).toFixed(4);
            const priced = (b.n_runs || 0) - (b.unknown_cost_runs || 0);
            const perRun = (b.cost_usd != null && priced > 0)
              ? '$' + (b.cost_usd / priced).toFixed(4) : '—';
            const elapsed = b.avg_elapsed_ms == null ? '—'
              : (b.avg_elapsed_ms / 1000).toFixed(2) + 's';
            return `<tr>
              <td class="td-mono">${escape(String(b.bucket || '(none)'))}</td>
              <td class="td-num">${b.n_runs || 0}</td>
              <td class="td-num muted">${(b.tokens_in || 0).toLocaleString()}</td>
              <td class="td-num muted">${(b.tokens_out || 0).toLocaleString()}</td>
              <td class="td-num">${cost}</td>
              <td class="td-num">${perRun}</td>
              <td class="td-num muted">${elapsed}</td>
              <td class="td-num muted">${b.unknown_cost_runs || 0}</td>
            </tr>`;
          }).join('')}
        </tbody>
      </table>`;
  }

  // pricing catalog (collapsed by default)
  const prices = pricing.prices || {};
  const modelIds = Object.keys(prices).sort();
  $('#cost-pricing-note').textContent = `${modelIds.length} models known`;
  $('#cost-pricing-table').innerHTML = `
    <table>
      <thead><tr>
        <th>model</th><th>prompt $/1M</th><th>completion $/1M</th>
        <th>updated</th><th>source</th>
      </tr></thead>
      <tbody>
        ${modelIds.map(m => {
          const p = prices[m];
          return `<tr>
            <td class="td-mono">${escape(m)}</td>
            <td class="td-num">$${Number(p.prompt_per_1m).toFixed(2)}</td>
            <td class="td-num">$${Number(p.completion_per_1m).toFixed(2)}</td>
            <td class="td-mono muted">${escape(p.updated_at)}</td>
            <td class="td-mono muted">${escape(String(p.source).slice(0, 40))}</td>
          </tr>`;
        }).join('')}
      </tbody>
    </table>`;
}
$('#cost-refresh').addEventListener('click', loadCost);
$('#cost-group-by').addEventListener('change', loadCost);
if ($('#runs-kind-filter')) {
  $('#runs-kind-filter').addEventListener('change', loadRuns);
}

// ---------- posture (v1.8: offensive-pattern classifier) ----------
function _domainColor(d) {
  return ({
    re_tooling:        '#9bd6e7',
    kernel_api:        '#e7c89b',
    byovd:             '#e79b9b',
    exploit_lexicon:   '#e7d99b',
    attack_technique:  '#c89be7',
    recon:             '#9be7c8',
  })[d] || '#bbb';
}

async function loadPosture() {
  const minScore = parseFloat($('#posture-min-score').value) || 0;
  const limit    = parseInt($('#posture-limit').value, 10) || 100;
  const [posture, sigs] = await Promise.all([
    api(`/api/classifier/posture?min_score=${minScore}&limit=${limit}`),
    api('/api/classifier/signatures'),
  ]);

  // totals
  const runs = posture.runs || [];
  const byDomain = {};
  let topScore = 0;
  for (const r of runs) {
    if (r.classifier_score > topScore) topScore = r.classifier_score;
    byDomain[r.classifier_kind] = (byDomain[r.classifier_kind] || 0) + 1;
  }
  $('#posture-totals').innerHTML = `
    <div class="card"><div class="label">Flagged runs</div><div class="value">${runs.length}</div>
      <div class="sub">score &ge; ${minScore.toFixed(2)}</div></div>
    <div class="card"><div class="label">Top score</div><div class="value">${topScore.toFixed(3)}</div>
      <div class="sub">cap = 1.000</div></div>
    <div class="card"><div class="label">Active signatures</div><div class="value">${(sigs.signatures||[]).length}</div>
      <div class="sub">${(sigs.classifier || 'offensive_patterns')}</div></div>
    <div class="card"><div class="label">Top domain</div>
      <div class="value">${Object.keys(byDomain).sort((a,b)=>byDomain[b]-byDomain[a])[0] || '—'}</div>
      <div class="sub">${Object.entries(byDomain).map(([k,v])=>`${k}:${v}`).join(' · ') || 'no flagged runs'}</div></div>
  `;

  // ranked runs
  if (!runs.length) {
    $('#posture-runs').innerHTML = `<p class="muted">No runs above the score threshold.
      Either no agent activity has matched any signature, or historical runs predate
      v1.8 — try the <code>Replay over null runs</code> button.</p>`;
  } else {
    $('#posture-runs').innerHTML = runs.map(r => {
      const dColor = _domainColor(r.classifier_kind);
      const sigList = (r.signals || []).map(s => `
        <li title="${escape(s.matched_text || '')}">
          <span class="td-mono" style="color:${_domainColor(s.domain)}">${escape(s.signature_id)}</span>
          <span class="muted">[${escape(s.domain)}, w=${Number(s.weight).toFixed(2)}]</span>
          ${s.matched_text ? `<code class="muted" style="font-size:11px">"${escape(String(s.matched_text).slice(0,80))}"</code>` : ''}
        </li>`).join('');
      return `
        <div style="border:1px solid #333;border-radius:6px;padding:10px;margin:8px 0;">
          <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
            <strong style="font-size:18px;color:${dColor};">${Number(r.classifier_score).toFixed(3)}</strong>
            <span class="kind-pill" style="background:${dColor}22;color:${dColor};border-color:${dColor}55">${escape(r.classifier_kind || '?')}</span>
            <span>run #${r.id}</span>
            <span class="td-mono">${escape(r.agent_name || '?')}</span>
            <span class="kind-pill kind-${escape(r.agent_kind || 'qwen-vllm')}">${escape(r.agent_kind || 'qwen-vllm')}</span>
            <span class="muted">${escape(fmtTs(r.started_at))}</span>
            <button data-watch="${r.id}" data-kind="${escape(r.agent_kind || 'qwen-vllm')}"
                    style="margin-left:auto;">Watch</button>
          </div>
          <ul style="margin:8px 0 0 0;padding-left:20px;font-size:13px;">${sigList}</ul>
        </div>`;
    }).join('');
    // re-bind Watch buttons
    $$('#posture-runs button[data-watch]').forEach(b => b.addEventListener('click', () => {
      $('#live-run-id').value = b.dataset.watch;
      window._lastWatchKind = b.dataset.kind || 'qwen-vllm';
      showPanel('live');
      connectLive();
    }));
  }

  // signature catalog
  const sigRows = (sigs.signatures || []);
  $('#posture-sig-note').textContent = `${sigRows.length} signatures`;
  $('#posture-sig-table').innerHTML = `
    <table>
      <thead><tr>
        <th>id</th><th>domain</th><th>weight</th>
        <th>description</th><th>source</th>
      </tr></thead>
      <tbody>
        ${sigRows.map(s => `
          <tr>
            <td class="td-mono" style="color:${_domainColor(s.domain)}">${escape(s.id)}</td>
            <td class="td-mono">${escape(s.domain)}</td>
            <td class="td-num">${Number(s.weight).toFixed(2)}</td>
            <td>${escape(s.description)}</td>
            <td class="td-mono muted" style="font-size:11px">${
              s.source && s.source.startsWith('http')
                ? `<a href="${escape(s.source)}" target="_blank" rel="noopener" style="color:inherit;">${escape(s.source)}</a>`
                : escape(s.source)
            }</td>
          </tr>`).join('')}
      </tbody>
    </table>`;
}
$('#posture-refresh').addEventListener('click', loadPosture);
// ---------- scanner obs (v1.9: meta-tool over Semgrep/CodeQL/...) ----------
function _fmtSec(s) {
  if (s == null) return '—';
  if (s < 60)   return `${s.toFixed(1)}s`;
  if (s < 3600) return `${(s/60).toFixed(1)}m`;
  if (s < 86400) return `${(s/3600).toFixed(1)}h`;
  return `${(s/86400).toFixed(1)}d`;
}
function _fmtUsd(n, digits=4) {
  if (n == null) return '—';
  return '$' + Number(n).toFixed(digits);
}
function _fmtPct(x) {
  if (x == null) return '—';
  return (x*100).toFixed(1) + '%';
}

const TRIAGE_STATES = ['new','confirmed','false_positive','fixed','wontfix','suppressed'];

async function loadScanObs() {
  const since = $('#so-since').value.trim() || null;
  const qs = since ? `?since=${encodeURIComponent(since)}` : '';
  const [summary, tools, density] = await Promise.all([
    api(`/api/scanner_obs/summary${qs}`),
    api(`/api/scanner_obs/tools${qs}`),
    api(`/api/scanner_obs/density${qs}`),
  ]);

  // --- Fleet KPI cards --------------------------------------------------
  const s = summary;
  $('#so-summary').innerHTML = `
    <div class="card">
      <div class="label">Scans</div>
      <div class="value">${s.n_scans}</div>
      <div class="sub">${s.total_findings ?? 0} findings total</div>
    </div>
    <div class="card">
      <div class="label">$/finding</div>
      <div class="value">${s.dollar_per_finding != null ? _fmtUsd(s.dollar_per_finding) : '—'}</div>
      <div class="sub">total cost ${_fmtUsd(s.total_cost_usd, 2)}</div>
    </div>
    <div class="card">
      <div class="label">FP rate</div>
      <div class="value">${_fmtPct(s.fp_rate)}</div>
      <div class="sub">${(s.triage?.false_positive || 0)} FP of ${
        (s.triage?.confirmed||0)+(s.triage?.false_positive||0)+(s.triage?.fixed||0)+(s.triage?.wontfix||0)
      } triaged</div>
    </div>
    <div class="card">
      <div class="label">Time-to-fix</div>
      <div class="value">${_fmtSec(s.ttf_p50_seconds)}</div>
      <div class="sub">p90 ${_fmtSec(s.ttf_p90_seconds)}</div>
    </div>
    <div class="card">
      <div class="label">CI minutes</div>
      <div class="value">${s.total_ci_minutes != null ? Number(s.total_ci_minutes).toFixed(1) : '—'}</div>
      <div class="sub">${s.total_ci_minutes != null ? 'sum across scans' : 'not reported'}</div>
    </div>
  `;

  // --- Per-tool table ---------------------------------------------------
  const tt = tools.tools || [];
  $('#so-tools').innerHTML = !tt.length
    ? `<p class="muted">No scans ingested yet. Try
       <code>POST /api/scan/external</code> with a few findings.</p>`
    : `
    <table>
      <thead><tr>
        <th>tool</th><th>version</th><th>scans</th><th>findings</th>
        <th>$/finding</th><th>FP rate</th><th>last run</th>
      </tr></thead>
      <tbody>
        ${tt.map(t => `
          <tr>
            <td class="td-mono"><strong>${escape(t.tool || '?')}</strong></td>
            <td class="td-mono muted">${escape(t.scanner_version || '—')}</td>
            <td class="td-num">${t.n_scans}</td>
            <td class="td-num">${t.n_findings}</td>
            <td class="td-num">${_fmtUsd(t.dollar_per_finding)}</td>
            <td class="td-num">${_fmtPct(t.fp_rate)}</td>
            <td class="muted">${escape(fmtTs(t.last_run) || '—')}</td>
          </tr>`).join('')}
      </tbody>
    </table>`;

  // populate the drift-tool dropdown from observed tools
  const sel = $('#so-drift-tool');
  sel.innerHTML = tt.map(t => `<option value="${escape(t.tool)}">${escape(t.tool)}</option>`).join('');

  // --- Density table ----------------------------------------------------
  const k = density.kinds || [];
  $('#so-density').innerHTML = !k.length
    ? `<p class="muted">No findings yet.</p>`
    : `
    <table>
      <thead><tr>
        <th>kind</th><th>total</th><th>new</th><th>confirmed</th>
        <th>FP</th><th>fixed</th><th>avg severity (0..4)</th>
      </tr></thead>
      <tbody>
        ${k.map(r => `
          <tr>
            <td class="td-mono">${escape(r.kind)}</td>
            <td class="td-num">${r.n}</td>
            <td class="td-num muted">${r.new_c||0}</td>
            <td class="td-num">${r.confirmed||0}</td>
            <td class="td-num" style="color:#e79b9b">${r.fp||0}</td>
            <td class="td-num" style="color:#9be7c8">${r.fixed_c||0}</td>
            <td class="td-num muted">${r.avg_rank != null ? Number(r.avg_rank).toFixed(2) : '—'}</td>
          </tr>`).join('')}
      </tbody>
    </table>`;
}

async function loadScanObsDrift() {
  const tool = $('#so-drift-tool').value;
  if (!tool) {
    $('#so-drift-note').textContent = 'no tool selected';
    return;
  }
  const r = await api(`/api/scanner_obs/drift?tool=${encodeURIComponent(tool)}`);
  if (r.note) {
    $('#so-drift-note').textContent = r.note;
    $('#so-drift').innerHTML = '';
    return;
  }
  $('#so-drift-note').textContent =
    `latest = scan #${r.latest.id} (${r.latest.started_at}) vs previous = scan #${r.previous.id} (${r.previous.started_at})`;
  const block = (label, color, items) => `
    <div style="border:1px solid #333;border-radius:6px;padding:10px;margin:8px 0;">
      <strong style="color:${color}">${label} (${items.length})</strong>
      ${items.length ? `<ul style="margin:6px 0 0 16px;font-size:13px;">${
        items.slice(0,30).map(f => `
          <li><span class="td-mono">${escape(f.file_path)}</span>
              <span class="muted">[${escape(f.kind)}, ${escape(f.severity)}]</span>
              <code class="muted" style="font-size:11px">fp=${escape(f.fingerprint?.slice(0,12))}</code></li>`).join('')
      }${items.length > 30 ? `<li class="muted">… and ${items.length - 30} more</li>` : ''}</ul>` : '<p class="muted" style="margin:4px 0 0">none</p>'}
    </div>`;
  $('#so-drift').innerHTML =
    block('NEW (latest only)',       '#9be7c8', r.new) +
    block('GONE (previous only)',    '#e79b9b', r.gone) +
    block('PERSISTENT (both scans)', '#e7c89b', r.persistent);
}

async function loadScanObsTriage() {
  const scanId = parseInt($('#so-triage-scan').value, 10);
  if (!scanId) { $('#so-triage').innerHTML = '<p class="muted">enter a scan id</p>'; return; }
  let r;
  try {
    r = await api(`/api/scan/${scanId}/findings?limit=200`);
  } catch (e) {
    $('#so-triage').innerHTML = `<p class="muted">error: ${e.message || e}</p>`;
    return;
  }
  const findings = r.findings || [];
  if (!findings.length) {
    $('#so-triage').innerHTML = `<p class="muted">scan #${scanId} has no findings</p>`;
    return;
  }
  $('#so-triage').innerHTML = `
    <table>
      <thead><tr>
        <th>id</th><th>severity</th><th>kind</th><th>file:line</th>
        <th>excerpt</th><th>state</th><th>set</th>
      </tr></thead>
      <tbody>
        ${findings.map(f => `
          <tr data-fid="${f.id}">
            <td class="td-num muted">${f.id}</td>
            <td><span class="kind-pill kind-${escape(f.severity)}">${escape(f.severity)}</span></td>
            <td class="td-mono">${escape(f.kind)}</td>
            <td class="td-mono muted">${escape(f.file_path)}:${f.line_hint || '?'}</td>
            <td><code style="font-size:11px">${escape(String(f.excerpt || '').slice(0,60))}</code></td>
            <td class="td-mono" id="state-${f.id}">${escape(f.triage_state || 'new')}</td>
            <td>
              <select data-triage="${f.id}">
                ${TRIAGE_STATES.map(s => `<option value="${s}" ${(f.triage_state||'new')===s?'selected':''}>${s}</option>`).join('')}
              </select>
            </td>
          </tr>`).join('')}
      </tbody>
    </table>`;
  $$('#so-triage select[data-triage]').forEach(sel => {
    sel.addEventListener('change', async () => {
      const fid = parseInt(sel.dataset.triage, 10);
      const state = sel.value;
      try {
        await fetch(`/api/findings/${fid}/triage`, {
          method: 'POST',
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify({state, by: 'dashboard-user'}),
        }).then(r => r.json());
        $(`#state-${fid}`).textContent = state;
        // Refresh KPIs since triage changes them
        loadScanObs();
      } catch (e) {
        alert('triage failed: ' + (e.message || e));
      }
    });
  });
}

$('#so-refresh').addEventListener('click', loadScanObs);
$('#so-drift-go').addEventListener('click', loadScanObsDrift);
$('#so-triage-load').addEventListener('click', loadScanObsTriage);

$('#posture-replay').addEventListener('click', async () => {
  $('#posture-replay-status').textContent = 'replaying…';
  try {
    const r = await fetch('/api/classifier/replay', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({only_null: true, limit: 1000}),
    }).then(r => r.json());
    $('#posture-replay-status').textContent =
      `classified ${r.classified}, flagged ${r.scored_nonzero}`;
    loadPosture();
  } catch (e) {
    $('#posture-replay-status').textContent = `error: ${e.message || e}`;
  }
});

// ---------- live trace ----------
let liveWs = null;
function disconnectLive() {
  if (liveWs) { liveWs.close(); liveWs = null; }
  $('#live-connect').disabled = false;
  $('#live-disconnect').disabled = true;
  $('#live-status').textContent = 'disconnected';
}
function connectLive() {
  disconnectLive();
  const id = parseInt($('#live-run-id').value);
  if (!Number.isFinite(id)) { $('#live-status').textContent = 'invalid run id'; return; }
  $('#live-feed').innerHTML = '';
  $('#live-status').textContent = 'connecting…';
  // Fetch run detail so we can label the runtime kind and warn honestly when
  // residual-stream interp will not be available for this run.
  api('/api/runs/' + id).then(d => {
    const k = (d.run && d.run.agent_kind) || 'qwen-vllm';
    let meta = {};
    try { meta = JSON.parse(d.run.meta_json || '{}'); } catch(_) {}
    const note = $('#live-runtime-note') || (() => {
      const el = document.createElement('div');
      el.id = 'live-runtime-note';
      el.className = 'muted';
      el.style.cssText = 'margin:6px 0;font-size:12px;';
      $('#live-feed').parentNode.insertBefore(el, $('#live-feed'));
      return el;
    })();
    let runtimeBit;
    if (k === 'qwen-vllm') {
      runtimeBit = `runtime: <span class="kind-pill kind-qwen-vllm">${k}</span> · residual-stream interp available`;
    } else {
      runtimeBit = `runtime: <span class="kind-pill kind-${escape(k)}">${escape(k)}</span> · <em>residual-stream interp not applicable</em> — only text-level harm/probes will populate`;
    }
    const sensitiveBit = meta.sensitive
      ? `<div style="margin-top:6px;padding:8px;background:#3a2a1f;border:1px solid #6a4a30;border-radius:4px;color:#f0b78a;font-size:12px;"><strong>⚑ Sensitive pipeline:</strong> raw inputs are hashed, not stored. Treat trace events here as user-supplied summaries; the source material is the caller's responsibility.</div>`
      : '';
    note.innerHTML = runtimeBit + sensitiveBit;
  }).catch(()=>{});
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  liveWs = new WebSocket(`${proto}//${location.host}/ws/runs/${id}`);
  liveWs.onopen = () => {
    $('#live-status').textContent = 'connected';
    $('#live-connect').disabled = true;
    $('#live-disconnect').disabled = false;
  };
  liveWs.onmessage = ev => {
    const m = JSON.parse(ev.data);
    if (m.type === 'trace') appendTrace(m);
    if (m.type === 'status') $('#live-status').textContent = `run ${m.status}`;
    if (m.type === 'error')  $('#live-status').textContent = 'err: ' + m.message;
  };
  liveWs.onclose = () => disconnectLive();
}
function appendTrace(t) {
  const row = document.createElement('div');
  row.className = `trace-row kind-${t.kind}`;
  let payload = '';
  try { payload = JSON.stringify(JSON.parse(t.payload_json), null, 2); }
  catch { payload = t.payload_json || ''; }
  row.innerHTML = `<span class="seq">#${t.seq}</span><span class="kind">${escape(t.kind)}</span><pre>${escape(payload)}</pre>`;
  const feed = $('#live-feed');
  feed.appendChild(row);
  feed.scrollTop = feed.scrollHeight;
}
$('#live-connect').addEventListener('click', connectLive);
$('#live-disconnect').addEventListener('click', disconnectLive);

// ---------- memory ----------
async function loadMemoryRecent() {
  const data = await api('/api/memory?limit=30');
  $('#mem-results').innerHTML = renderMemory(data.results, data.mode);
}
function renderMemory(results, mode) {
  if (!results.length) return '<p class="muted">No memory chunks.</p>';
  return results.map(r => `
    <div class="mem-chunk">
      <header>
        <code>#${r.id}</code>
        <code>${escape(r.kind)}</code>
        <code>${escape(r.source)}</code>
        ${r.tags ? `<code class="muted">${escape(r.tags)}</code>` : ''}
        ${r.score != null ? `<span class="score">cos=${Number(r.score).toFixed(3)}</span>` : ''}
        <span class="muted">${escape(fmtTs(r.created_at))}</span>
        ${r.embed_dim ? `<span class="muted">vec=${r.embed_dim}d</span>` : ''}
      </header>
      <div class="chunk-text">${escape(r.text)}</div>
    </div>`).join('');
}
$('#mem-search').addEventListener('click', async () => {
  const q = $('#mem-q').value.trim();
  const semantic = $('#mem-semantic').checked ? '&semantic=true' : '';
  if (!q) { loadMemoryRecent(); return; }
  const data = await api(`/api/memory?q=${encodeURIComponent(q)}${semantic}&limit=30`);
  $('#mem-results').innerHTML = renderMemory(data.results, data.mode);
});
$('#mem-q').addEventListener('keydown', e => { if (e.key === 'Enter') $('#mem-search').click(); });
$('#mem-add-toggle').addEventListener('click', () => $('#mem-add').classList.toggle('hidden'));
$('#mem-add-submit').addEventListener('click', async () => {
  const text = $('#mem-add-text').value.trim();
  if (!text) return;
  const tags = $('#mem-add-tags').value.split(',').map(s=>s.trim()).filter(Boolean);
  await api('/api/memory', {
    method: 'POST', headers: {'content-type':'application/json'},
    body: JSON.stringify({text, tags, source: 'ui', kind: 'note'}),
  });
  $('#mem-add-text').value = '';
  $('#mem-add-tags').value = '';
  $('#mem-add').classList.add('hidden');
  loadMemoryRecent();
});

// ---------- interp ----------
function band(score) { return score >= 0.7 ? 'high' : score >= 0.4 ? 'mid' : 'low'; }
$('#interp-score').addEventListener('click', async () => {
  const text = $('#interp-text').value;
  if (!text.trim()) return;
  $('#interp-status').textContent = 'scoring…';
  try {
    const r = await api('/api/interp/score', {
      method: 'POST', headers: {'content-type':'application/json'},
      body: JSON.stringify({text}),
    });
    $('#interp-status').textContent = `${r.text_chars} chars scored`;
    const s = r.scores || {};
    const meta = s._meta || {};
    const source = meta.primary_source || 'toy';
    const cats = meta.primary_categories || [];
    const lat = meta.primary_latency_ms;
    const labels = {
      harm:     source === 'llama_guard_3' ? 'harm (Llama Guard 3)' : 'harm (toy fallback)',
      harm_toy: 'harm_toy (drift)',
      refusal:  'refusal (toy)',
      hedging:  'hedging (toy)',
    };
    const order = ['harm','harm_toy','refusal','hedging'];
    $('#interp-result').innerHTML = `
      <div class="score-grid">
        ${order.map(k => {
          const v = s[k];
          const cls = v == null ? '' : band(v);
          const text = v == null ? '—' : Number(v).toFixed(3);
          return `<div class="score-card ${cls}">
            <div class="label">${labels[k] || k}</div>
            <div class="value">${text}</div>
          </div>`;
        }).join('')}
      </div>
      <div class="muted" style="margin-top:8px; font-size:12px;">
        primary: <code>${source}</code>${lat != null ? ` · ${lat} ms` : ''}
        ${cats.length ? ` · categories: ${cats.map(c=>`<code>${c}</code>`).join(', ')}` : ''}
      </div>`;
  } catch (e) {
    $('#interp-status').textContent = 'error: ' + e.message;
  }
});

// ---------- thoughts (NLA) ----------
async function loadThoughts() {
  try {
    const s = await api('/api/nla/status');
    const banner = $('#nla-banner');
    if (!s.ready) {
      banner.className = 'banner bad';
      banner.innerHTML = `No NLA backend available. Pull <code>${s.prompted_approx.model}</code> into Ollama (<code>ollama pull ${s.prompted_approx.model}</code>), or set <code>NLA_REMOTE_URL</code> to an SGLang wrapper serving one of the <code>kitft/nla-*</code> checkpoints.`;
    } else if (s.backend === 'remote') {
      banner.className = 'banner good';
      banner.innerHTML = `<strong>Backend: remote NLA</strong> at <code>${s.remote.url}</code> &mdash; activation-grounded explanations.`;
    } else if (s.backend === 'local_activations') {
      const la = s.local_activations || {};
      banner.className = 'banner good';
      banner.innerHTML = `<strong>Backend: local activations</strong> via <code>${escape(la.model)}</code> at layer L${la.layer} on <code>${la.device}</code>. Real residual-stream fingerprint + verbalizer.`;
    } else {
      banner.className = 'banner amber';
      const la = s.local_activations || {};
      const upgrade = la.installed && !la.ready
        ? ` Upgrade path: <button id="nla-enable-local" class="inline-btn">enable local activations (loads ${escape(la.model||'Qwen2.5')})</button>`
        : '';
      const sc = s.prompted_approx.self_consistency_n || 1;
      banner.innerHTML =
        `<strong>Backend: prompted approximation</strong> via <code>${escape(s.prompted_approx.model)}</code> ` +
        `(self-consistency N=${sc}, verbatim-quote validation, abstention).` +
        ` Reads surface text only &mdash; not a real Anthropic NLA.${upgrade}`;
      const enableBtn = document.getElementById('nla-enable-local');
      if (enableBtn) enableBtn.addEventListener('click', _nlaEnableLocal);
    }
    // mini stats line under the banner
    const lat = (s.latency || {})[s.backend] || {};
    const cache = s.cache || {};
    const wq = s.worker || {};
    const hr = cache.hit_rate == null ? '—' : (cache.hit_rate * 100).toFixed(0) + '%';
    const lstr = lat.n ? `${lat.p50}ms p50 / ${lat.p95}ms p95 (n=${lat.n})` : 'no samples yet';
    const qstr = wq.running
      ? `queue ${wq.queue_depth}/${wq.queue_max} · processed ${wq.processed} · dropped ${wq.dropped_full || 0}`
      : 'worker idle';
    $('#nla-banner').insertAdjacentHTML('beforeend',
      `<div class="muted" style="margin-top:6px; font-size:11.5px;">` +
      `latency ${lstr} · cache hits ${hr} (l1 ${cache.l1_size||0} / l2 ${cache.l2_size||0}) · ${qstr}` +
      `</div>`);
  } catch (e) {
    $('#nla-banner').className = 'banner bad';
    $('#nla-banner').textContent = 'failed to query NLA status: ' + e.message;
  }
}

async function _nlaEnableLocal() {
  const btn = document.getElementById('nla-enable-local');
  if (btn) { btn.disabled = true; btn.textContent = 'loading model… (this can take minutes on first run)'; }
  try {
    const r = await api('/api/nla/local/enable', { method: 'POST' });
    if (!r.ok) {
      if (btn) btn.textContent = 'failed: ' + (r.error || 'unknown');
      return;
    }
  } catch (e) {
    if (btn) btn.textContent = 'failed: ' + e.message;
    return;
  }
  loadThoughts();
}

function _nlaBand(v) {
  if (v == null) return '';
  if (v >= 0.7) return 'high';
  if (v >= 0.4) return 'mid';
  return 'low';
}

$('#nla-decode').addEventListener('click', async () => {
  const text = $('#nla-text').value.trim();
  if (!text) return;
  const target = $('#nla-target').value;
  const runIdRaw = $('#nla-run-id').value.trim();
  const runId = runIdRaw ? Number(runIdRaw) : null;
  $('#nla-status').textContent = 'decoding… (cold-start ≈ 60s)';
  $('#nla-result').innerHTML = '';
  try {
    const r = await api('/api/nla/decode', {
      method: 'POST', headers: {'content-type':'application/json'},
      body: JSON.stringify({text, target, run_id: runId, persist: true}),
    });
    const d = r.decoding || {};
    if (!d.ok) {
      $('#nla-status').textContent = 'error: ' + (d.error || 'decode failed');
      return;
    }
    const cachedTag = d.cached ? ' · <span class="chip-good">cached</span>' : '';
    $('#nla-status').innerHTML =
      `${d.source} · ${d.latency_ms} ms` +
      (r.persisted_id ? ` · saved as decoding #${r.persisted_id}` : '') +
      cachedTag;
    const stdev = d.stdev || {};
    const abstain = d.abstain || {};
    const grid = ['evaluation_awareness','hidden_motivation','safety_relevance']
      .map(k => {
        const v = d[k];
        const sd = stdev[k];
        const ab = abstain[k];
        const cls = ab ? 'abstain' : _nlaBand(v);
        const txt = ab ? 'abstain' : (v == null ? '—' : Number(v).toFixed(2));
        const sub = ab
          ? '<div class="score-sub">samples disagreed</div>'
          : (sd != null ? `<div class="score-sub">± ${Number(sd).toFixed(2)}</div>` : '');
        const label = k.replace(/_/g,' ');
        return `<div class="score-card ${cls}"><div class="label">${label}</div><div class="value">${txt}</div>${sub}</div>`;
      }).join('');
    const notes = Array.isArray(d.notes) ? d.notes : (d.notes ? [d.notes] : []);
    const dropped = Number(d.notes_dropped || 0);
    const droppedLine = dropped > 0
      ? `<div class="muted" style="margin-top:6px; font-size:11.5px;">⚠ verbatim-validation rejected ${dropped} note(s) that were not exact substrings of the input.</div>`
      : '';
    const samplingLine = (d.n_samples != null && d.n_requested != null)
      ? `<div class="muted" style="margin-top:4px; font-size:11.5px;">self-consistency: ${d.n_samples}/${d.n_requested} samples valid${d.topic_agreement != null ? ` · topic agreement ${(d.topic_agreement*100).toFixed(0)}%` : ''}</div>`
      : '';
    $('#nla-result').innerHTML = `
      <div class="card standalone">
        <div class="eyebrow">topic</div>
        <div class="value" style="font-size:18px;">${escape(d.topic || '—')}</div>
        <div class="muted" style="margin-top:6px;">${escape(d.explanation || '')}</div>
        ${samplingLine}
      </div>
      <div class="score-grid" style="margin-top:12px;">${grid}</div>
      ${notes.length ? `<div class="card standalone" style="margin-top:12px;">
        <div class="eyebrow">evidence (verbatim from input)</div>
        <ul style="margin:6px 0 0; padding-left:18px;">
          ${notes.map(n => `<li>${escape(n)}</li>`).join('')}
        </ul>
        ${droppedLine}
      </div>` : droppedLine}
      <div class="muted" style="margin-top:8px; font-size:12px;">
        backend: <code>${d.source}</code>${d.model ? ` · model: <code>${escape(d.model)}</code>` : ''}
      </div>`;
  } catch (e) {
    $('#nla-status').textContent = 'error: ' + e.message;
  }
});

// ---------- browser ----------
async function refreshShot() {
  const img = $('#br-shot-img');
  img.src = `/api/browser/screenshot.png?t=${Date.now()}`;
}
$('#br-go').addEventListener('click', async () => {
  const url = $('#br-url').value.trim();
  if (!url) return;
  $('#br-status').textContent = 'loading…';
  try {
    const r = await api('/api/browser/goto', {
      method: 'POST', headers: {'content-type':'application/json'},
      body: JSON.stringify({url, headless: true}),
    });
    $('#br-status').textContent = `${r.title}`;
    refreshShot();
    refreshStatus();
  } catch (e) {
    $('#br-status').textContent = 'error: ' + e.message;
  }
});
$('#br-shot').addEventListener('click', refreshShot);
$('#br-close').addEventListener('click', async () => {
  await api('/api/browser/close', {method:'POST'});
  $('#br-status').textContent = 'closed';
  $('#br-shot-img').removeAttribute('src');
  refreshStatus();
});

// ---------- code scan (v1.5) ----------
const _csState = { activeScanId: null, pollTimer: null };

const _SEV_COLORS = {
  critical: '#7f1d1d', high: '#b91c1c', medium: '#c2410c',
  low: '#a16207', info: '#475569',
};

function _csSevPill(sev) {
  const color = _SEV_COLORS[sev] || '#475569';
  return `<span class="cs-sev" style="background:${color}">${escape(sev)}</span>`;
}

async function loadCodeScan() {
  // 1. status banner
  try {
    const s = await api('/api/scan/status');
    const ready = s.ready ? '✓ ready' : '✗ ollama not reachable';
    $('#cs-status-line').innerHTML =
      `model: <code>${escape(s.model)}</code> · prompt: <code>${escape(s.prompt_version)}</code> · ${ready}`;
  } catch (e) {
    $('#cs-status-line').textContent = 'status query failed: ' + e.message;
  }
  await _csRefreshList();
}

async function _csRefreshList() {
  try {
    const r = await api('/api/scan/list');
    const tbody = $('#cs-scans tbody');
    tbody.innerHTML = (r.scans || []).map(s => `
      <tr data-id="${s.id}">
        <td>${s.id}</td>
        <td>${escape(s.label || '')}</td>
        <td title="${escape(s.root_path)}"><code>${escape((s.root_path||'').slice(-40))}</code></td>
        <td><span class="cs-status cs-status-${escape(s.status)}">${escape(s.status)}</span></td>
        <td>${s.scanned_files}/${s.total_files}${s.skipped_files ? ` <span class="muted">(skip ${s.skipped_files})</span>` : ''}</td>
        <td>${s.findings_count}</td>
        <td class="muted">${escape((s.started_at||'').replace('T',' ').slice(0,19))}</td>
        <td>
          <button class="cs-open" data-id="${s.id}">open</button>
          ${s.status === 'running' ? `<button class="cs-cancel" data-id="${s.id}">cancel</button>` : ''}
        </td>
      </tr>
    `).join('');
    $$('#cs-scans .cs-open').forEach(b =>
      b.addEventListener('click', () => _csOpenDetail(parseInt(b.dataset.id))));
    $$('#cs-scans .cs-cancel').forEach(b =>
      b.addEventListener('click', async () => {
        await api(`/api/scan/${b.dataset.id}/cancel`, {method:'POST'});
        await _csRefreshList();
      }));
  } catch (e) {
    console.warn('scan list failed', e);
  }
}

// Mirrors agent_monitor.code_scan.DEFAULT_EXTENSIONS but grouped so the
// dropdown can subset it. "all" uses the server's default (omits the
// `extensions` field). Anything else sends an explicit subset.
const _CS_LANG_GROUPS = {
  c_cpp:       {'.c':'c','.h':'c','.cc':'cpp','.cpp':'cpp','.cxx':'cpp','.hpp':'cpp','.hh':'cpp'},
  rust:        {'.rs':'rust'},
  go:          {'.go':'go'},
  python:      {'.py':'python'},
  js_ts:       {'.js':'javascript','.mjs':'javascript','.cjs':'javascript','.ts':'typescript','.tsx':'typescript'},
  java_kotlin: {'.java':'java','.kt':'kotlin'},
  ruby:        {'.rb':'ruby'},
  php:         {'.php':'php'},
  shell:       {'.sh':'shell','.bash':'shell'},
  sql:         {'.sql':'sql'},
  asm:         {'.asm':'asm','.s':'asm'},
};

async function _csStartScan() {
  const path = $('#cs-path').value.trim();
  if (!path) { alert('Enter a path'); return; }
  const body = {
    root_path: path,
    label: $('#cs-label').value.trim() || null,
    max_chunk_lines: parseInt($('#cs-chunk-lines').value) || 200,
    max_files: parseInt($('#cs-max-files').value) || 0,
    persist_low: $('#cs-persist-low').checked,
  };
  const lang = $('#cs-language').value;
  if (lang && lang !== 'all' && _CS_LANG_GROUPS[lang]) {
    body.extensions = _CS_LANG_GROUPS[lang];
  }
  if ($('#cs-diff-only').checked) {
    const since = $('#cs-git-since').value.trim();
    if (!since) { alert('Provide a git ref (e.g. HEAD~1) or uncheck diff-only.'); return; }
    body.git_since = since;
  }
  try {
    const r = await api('/api/scan/start', {
      method:'POST',
      headers:{'content-type':'application/json'},
      body: JSON.stringify(body),
    });
    await _csRefreshList();
    if (r.scan_id) _csOpenDetail(r.scan_id);
  } catch (e) {
    alert('scan start failed: ' + e.message);
  }
}

async function _csOpenDetail(id) {
  _csState.activeScanId = id;
  $('#cs-detail').classList.remove('hidden');
  $('#cs-detail-id').textContent = id;
  await _csRefreshDetail();
  if (_csState.pollTimer) clearInterval(_csState.pollTimer);
  _csState.pollTimer = setInterval(_csRefreshDetail, 5000);
}

async function _csRefreshDetail() {
  const id = _csState.activeScanId;
  if (!id) return;
  try {
    const d = await api(`/api/scan/${id}`);
    const s = d.scan, h = d.histogram, rt = d.runtime;
    $('#cs-detail-label').textContent = s.label || s.root_path;
    const pct = s.total_files ? Math.floor(100*s.scanned_files/s.total_files) : 0;
    $('#cs-detail-progress').innerHTML =
      `<strong>${escape(s.status)}</strong> · ${s.scanned_files}/${s.total_files} files (${pct}%)` +
      ` · ${s.findings_count} findings` +
      (rt && rt.active ? ` · scanning <code>${escape(rt.current_file||'')}</code>#${rt.current_chunk} (elapsed ${rt.elapsed_s}s)` : '') +
      (s.error ? ` · <span style="color:#b91c1c">error: ${escape(s.error)}</span>` : '');
    $('#cs-histogram').innerHTML = ['critical','high','medium','low','info']
      .map(k => `<span class="cs-hist-bucket cs-hist-${k}" style="background:${_SEV_COLORS[k]}">${k}: ${h[k]||0}</span>`)
      .join('');
    if (['done','error','cancelled'].includes(s.status) && _csState.pollTimer) {
      clearInterval(_csState.pollTimer);
      _csState.pollTimer = null;
      _csRefreshList();
    }
    await _csRefreshFindings();
  } catch (e) {
    console.warn('detail refresh', e);
  }
}

async function _csRefreshFindings() {
  const id = _csState.activeScanId;
  if (!id) return;
  const sev = $('#cs-filter-sev').value;
  const kind = $('#cs-filter-kind').value;
  const params = new URLSearchParams();
  if (sev) params.set('min_severity', sev);
  if (kind) params.set('kind', kind);
  params.set('limit', '500');
  try {
    const r = await api(`/api/scan/${id}/findings?${params}`);
    const findings = r.findings || [];
    if (!findings.length) {
      $('#cs-findings').innerHTML = `<p class="muted">No findings match the current filter.</p>`;
      return;
    }
    $('#cs-findings').innerHTML = findings.map(f => {
      const lineRef = f.line_hint
        ? `${f.file_path}:${f.chunk_start_line + (f.line_hint - 1)}`
        : `${f.file_path}:${f.chunk_start_line}-${f.chunk_end_line}`;
      return `
        <div class="cs-finding">
          <div class="cs-finding-head">
            ${_csSevPill(f.severity)}
            <span class="cs-kind">${escape(f.kind)}</span>
            <code class="cs-loc">${escape(lineRef)}</code>
            <span class="muted">${escape(f.language||'')}</span>
          </div>
          <pre class="cs-excerpt"><code>${escape(f.excerpt)}</code></pre>
          <div class="cs-explain">${escape(f.explanation || '')}</div>
          ${f.chunk_summary ? `<div class="cs-chunk-summary muted">chunk: ${escape(f.chunk_summary)}</div>` : ''}
        </div>`;
    }).join('');
  } catch (e) {
    $('#cs-findings').innerHTML = `<p class="muted">findings query failed: ${escape(e.message)}</p>`;
  }
}

$('#cs-start')         .addEventListener('click', _csStartScan);
$('#cs-refresh-list')  .addEventListener('click', _csRefreshList);
$('#cs-apply-filter')  .addEventListener('click', _csRefreshFindings);


// =============================================================
// v1.10 UI overhaul — toasts, empty states, drag-and-drop ingest,
// Detonations panel.
// =============================================================

// --- Toast notifications -------------------------------------------
// `toast(kind, title, body?)` -- kind ∈ {good, warn, bad}. Auto-removes
// after `ttlMs` (default 4.5s). Used by drag-and-drop ingest, error
// handlers, etc. Keeps user-visible feedback off the page chrome.
function toast(kind, title, body, ttlMs = 4500) {
  const root = document.getElementById('toast-root');
  if (!root) return;
  const icon = ({good: 'i-flame', bad: 'i-shield-alert', warn: 'i-info'})[kind] || 'i-info';
  const el = document.createElement('div');
  el.className = `toast ${escape(kind)}`;
  el.innerHTML = `
    <svg class="icon"><use href="#${icon}"/></svg>
    <div>
      <div class="toast-title">${escape(title)}</div>
      ${body ? `<div class="toast-body">${escape(body)}</div>` : ''}
    </div>`;
  root.appendChild(el);
  setTimeout(() => {
    el.style.transition = 'opacity 220ms';
    el.style.opacity = '0';
    setTimeout(() => el.remove(), 240);
  }, ttlMs);
}

// --- Empty-state renderer ------------------------------------------
// Returns an HTML string for the centered "nothing here yet" card.
// Pass `icon` (sprite symbol id without leading #), `title`, `body`
// (HTML allowed -- caller is responsible for escaping), and optional
// `actions` array of {label, onclick, panel}.
function emptyState({icon = 'i-inbox', title = 'Nothing here yet', body = '', actions = []}) {
  const acts = actions.length
    ? `<div class="empty-state-actions">${actions.map((a, i) =>
        `<button data-empty-act="${i}">${escape(a.label)}</button>`).join('')}</div>`
    : '';
  return `
    <div class="empty-state">
      <svg class="icon icon-xl"><use href="#${escape(icon)}"/></svg>
      <div class="empty-state-title">${escape(title)}</div>
      <div class="empty-state-body">${body}</div>
      ${acts}
    </div>`;
}
// Wire any [data-empty-act] buttons inside `root` to their handler.
function _bindEmptyActions(root, actions) {
  if (!actions || !actions.length) return;
  $$('[data-empty-act]', root).forEach(btn => {
    const a = actions[parseInt(btn.dataset.emptyAct, 10)];
    if (!a) return;
    btn.addEventListener('click', () => {
      if (a.panel) showPanel(a.panel);
      if (typeof a.onclick === 'function') a.onclick();
    });
  });
}

// --- Drag-and-drop ingest ------------------------------------------
// Each `.dropzone` element accepts a single JSON file. We try to
// auto-detect whether it's a SARIF v2.1.0 document or a sandbox report
// (Cuckoo / generic envelope) by sniffing the top-level keys, then
// POST it to the matching ingest endpoint. The dropzone's
// `data-mode` attribute can force a particular endpoint:
//   data-mode="auto"    -- detect SARIF vs sandbox (default)
//   data-mode="sandbox" -- always send to /api/scan/external/sandbox
//   data-mode="sarif"   -- always send to /api/scan/external/sarif
function _detectKind(payload) {
  if (!payload || typeof payload !== 'object') return null;
  // SARIF: `version` looks like "2.1.0" AND a `runs` array exists.
  if (Array.isArray(payload.runs)
      && typeof (payload.version || '') === 'string'
      && (payload.version || '').startsWith('2.')) return 'sarif';
  // Cuckoo: has `signatures` + at least one of info/target/behavior.
  if (Array.isArray(payload.signatures)
      && (payload.info || payload.target || payload.behavior)) return 'sandbox';
  // Generic sandbox envelope: top-level `signals` array.
  if (Array.isArray(payload.signals)) return 'sandbox';
  return null;
}

async function _ingestPayload(payload, kind, filename) {
  // Use the file name as a label hint so the user can recognise scans
  // back in the list view.
  const label = (filename || '').replace(/\.[^.]+$/, '') || null;
  if (kind === 'sarif') {
    const body = {root_path: '/dropped', sarif: payload, label};
    return {kind, result: await api('/api/scan/external/sarif', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    })};
  }
  if (kind === 'sandbox') {
    const body = {root_path: 'sandbox/dropped', report: payload, label};
    return {kind, result: await api('/api/scan/external/sandbox', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    })};
  }
  throw new Error('unrecognized JSON: not SARIF (no `runs`) and not a sandbox report (no `signatures` / `signals`)');
}

async function _handleDroppedFile(file, zone) {
  const status = zone.querySelector('.dropzone-status');
  const mode   = zone.dataset.mode || 'auto';
  if (!file) return;
  const setStatus = (msg, cls) => {
    if (!status) return;
    status.textContent = msg;
    status.className = 'dropzone-status ' + (cls || '');
  };
  setStatus(`reading ${file.name}…`, '');
  let text;
  try { text = await file.text(); }
  catch (e) { setStatus(`read failed: ${e.message || e}`, 'bad'); return; }
  let payload;
  try { payload = JSON.parse(text); }
  catch (e) {
    setStatus(`not valid JSON: ${e.message}`, 'bad');
    toast('bad', 'Could not ingest file', `${file.name} is not valid JSON.`);
    return;
  }
  let kind;
  if (mode === 'sarif' || mode === 'sandbox') kind = mode;
  else kind = _detectKind(payload);
  if (!kind) {
    setStatus(`unrecognized JSON shape`, 'bad');
    toast('bad', 'Could not detect format',
      'Expected SARIF v2.1 or a sandbox report (Cuckoo / generic envelope).');
    return;
  }
  setStatus(`ingesting as ${kind}…`, 'warn');
  try {
    const {result} = await _ingestPayload(payload, kind, file.name);
    let summary;
    if (kind === 'sarif') {
      const n = (result.runs || []).reduce((a, r) => a + (r.n_findings || 0), 0);
      summary = `${result.n_runs} run(s), ${n} findings`;
    } else {
      summary = `${result.tool} · ${result.n_findings} signature(s)`;
    }
    setStatus(`OK · ${summary}`, 'good');
    toast('good', `Ingested ${kind.toUpperCase()}`, summary);
    // Refresh the panel the user is looking at.
    const visible = $$('.panel').find(p => !p.classList.contains('hidden'));
    const pid = visible && visible.id || '';
    if (pid === 'panel-codescan')    _csRefreshList();
    if (pid === 'panel-scanobs')     loadScanObs();
    if (pid === 'panel-detonations') loadDetonations();
  } catch (e) {
    setStatus(`ingest failed: ${e.message || e}`, 'bad');
    toast('bad', 'Ingest failed', e.message || String(e));
  }
}

function _wireDropzone(zone) {
  if (!zone || zone._wired) return;
  zone._wired = true;
  const fileInput = zone.querySelector('input[type=file]');
  // The label.link-btn wraps the hidden file input. Avoid double-fire:
  // click-on-zone triggers the input only if the user didn't click the
  // <label> directly (the label already opens the picker via its for=).
  zone.addEventListener('click', (e) => {
    if (e.target.closest('label')) return;
    if (fileInput) fileInput.click();
  });
  if (fileInput) {
    fileInput.addEventListener('change', () => {
      if (fileInput.files && fileInput.files[0]) {
        _handleDroppedFile(fileInput.files[0], zone);
        fileInput.value = '';
      }
    });
  }
  ['dragenter', 'dragover'].forEach(ev =>
    zone.addEventListener(ev, e => {
      e.preventDefault();
      zone.classList.add('dragover');
    }));
  ['dragleave', 'drop'].forEach(ev =>
    zone.addEventListener(ev, e => {
      e.preventDefault();
      zone.classList.remove('dragover');
    }));
  zone.addEventListener('drop', e => {
    const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) _handleDroppedFile(f, zone);
  });
}
// Prevent browser navigation when files are dropped *outside* a zone.
['dragover','drop'].forEach(ev =>
  window.addEventListener(ev, e => { e.preventDefault(); }));
$$('.dropzone').forEach(_wireDropzone);


// --- Detonations panel ---------------------------------------------
// Reads from /api/sandbox/detonations and renders one card per scan.
// Each card is clickable -> loads its signatures via the existing
// /api/scan/{scan_id}/findings endpoint.
function _detSampleShort(sample) {
  if (!sample) return '(no sample)';
  if (sample.startsWith('sha256:')) {
    // Show 12+12 hex window for human-readable comparison.
    const hex = sample.slice('sha256:'.length);
    return hex.length > 28 ? `sha256:${hex.slice(0,12)}…${hex.slice(-12)}` : sample;
  }
  return sample;
}

function _detSevRow(sev) {
  const order = ['critical','high','medium','low','info'];
  return `<div class="det-sev-row">${order.map(k => {
    const n = sev[k] || 0;
    return `<span class="det-sev-chip ${k}${n ? ' has' : ''}" title="${k}">${k}: ${n}</span>`;
  }).join('')}</div>`;
}

async function loadDetonations() {
  try {
    const r = await api('/api/sandbox/detonations?limit=100');
    const list = r.detonations || [];
    $('#det-count').textContent = list.length
      ? `${list.length} detonation${list.length === 1 ? '' : 's'}`
      : '';
    const wrap = $('#det-list');
    if (!list.length) {
      wrap.innerHTML = emptyState({
        icon: 'i-file-search',
        title: 'No sandbox detonations yet',
        body: `Drop a Cuckoo / Joe / VMRay JSON above, or POST to
          <code>/api/scan/external/sandbox</code>. Each report becomes one
          detonation; each signature it fired becomes one finding.`,
      });
      $('#det-detail').classList.add('hidden');
      return;
    }
    wrap.innerHTML = list.map(d => `
      <div class="det-card" data-id="${d.id}">
        <div class="det-icon"><svg class="icon"><use href="#i-flame"/></svg></div>
        <div class="det-main">
          <div class="det-tool">${escape(d.tool)}${d.scanner_version ? ` · v${escape(d.scanner_version)}` : ''}</div>
          <div class="det-sample" title="${escape(d.sample_id || '')}">${escape(_detSampleShort(d.sample_id))}</div>
          <div class="det-meta">
            <span>scan #${d.id}</span>
            <span>${escape((d.started_at || '').replace('T',' ').slice(0,19))}</span>
            <span>${d.findings_count || 0} signature${(d.findings_count||0) === 1 ? '' : 's'}</span>
          </div>
        </div>
        ${_detSevRow(d.severity || {})}
      </div>
    `).join('');
    $$('#det-list .det-card').forEach(c =>
      c.addEventListener('click', () => _detOpenDetail(parseInt(c.dataset.id, 10), list)));
  } catch (e) {
    $('#det-list').innerHTML = `<p class="muted">load failed: ${escape(e.message || String(e))}</p>`;
  }
}

async function _detOpenDetail(scanId, list) {
  const det = (list || []).find(d => d.id === scanId);
  $('#det-detail').classList.remove('hidden');
  $('#det-detail-id').textContent = `#${scanId}`;
  $('#det-detail-sample').textContent = det ? ` · ${_detSampleShort(det.sample_id)}` : '';
  $('#det-detail-meta').textContent = det
    ? `${det.tool}${det.scanner_version ? ' · v' + det.scanner_version : ''} · ${det.findings_count} signatures · ${(det.started_at||'').replace('T',' ').slice(0,19)}`
    : '';
  $('#det-detail-signatures').innerHTML = `<p class="muted">loading signatures…</p>`;
  try {
    const r = await api(`/api/scan/${scanId}/findings?limit=500`);
    const findings = r.findings || [];
    if (!findings.length) {
      $('#det-detail-signatures').innerHTML = `<p class="muted">no signatures persisted for this scan</p>`;
      return;
    }
    $('#det-detail-signatures').innerHTML = findings.map(f => `
      <div class="det-sig sev-${escape(f.severity || 'info')}">
        <div class="det-sig-head">
          <span class="det-sev-chip ${escape(f.severity)} has">${escape(f.severity || 'info')}</span>
          <span class="det-sig-rule">${escape(f.rule_id || f.kind)}</span>
          <span class="det-sig-cat">${escape(f.kind)}</span>
        </div>
        <div class="det-sig-msg">${escape(f.explanation || '(no description)')}</div>
        ${f.excerpt ? `<div class="det-sig-ev">${escape(f.excerpt)}</div>` : ''}
      </div>
    `).join('');
    // Scroll the detail into view so the user sees the result of clicking.
    $('#det-detail').scrollIntoView({behavior: 'smooth', block: 'start'});
  } catch (e) {
    $('#det-detail-signatures').innerHTML =
      `<p class="muted">load failed: ${escape(e.message || String(e))}</p>`;
  }
}
$('#det-refresh') && $('#det-refresh').addEventListener('click', loadDetonations);


// --- Empty-state hooks for the most-visible panels -----------------
// We wrap (don't replace) the existing loaders so users landing on
// Agents / Runs / Code Scan / Scanner Obs / Detonations with an empty
// DB get an inviting card instead of a blank table or two-word muted
// note. The wrappers run *after* the original loader, then patch the
// DOM if it ended up empty.
function _maybeEmptyAgents() {
  const root = $('#agents-list');
  if (!root) return;
  if (root.querySelector('table tbody tr')) return;
  root.innerHTML = emptyState({
    icon: 'i-users',
    title: 'No agents have run yet',
    body: `An agent is anything you instrument with an AgentMonitor adapter
      (OpenAI, Anthropic, LangChain, AutoGen, Smolagents, Ollama, …).
      Run <code>python -m agent_monitor.smoke --with-agent</code> to see
      one materialize here.`,
  });
}
function _maybeEmptyRuns() {
  const root = $('#runs-table');
  if (!root) return;
  if (root.querySelector('table tbody tr')) return;
  root.innerHTML = emptyState({
    icon: 'i-play',
    title: 'No runs match the current filter',
    body: `Either nothing has been recorded yet, or your filter is too
      narrow. Try resetting the agent / runtime filters to <em>all</em>.`,
  });
}
const _origLoadAgents = loadAgents;
loadAgents = async function() { await _origLoadAgents.apply(this, arguments); _maybeEmptyAgents(); };
const _origLoadRuns = loadRuns;
loadRuns = async function() { await _origLoadRuns.apply(this, arguments); _maybeEmptyRuns(); };

// Code Scan: replace the "Refresh" / blank table state with a richer card.
const _origCsRefresh = _csRefreshList;
_csRefreshList = async function() {
  await _origCsRefresh.apply(this, arguments);
  const tbody = $('#cs-scans tbody');
  if (tbody && !tbody.querySelector('tr')) {
    // We insert the empty state in a slot above the table without
    // hiding the toolbar/dropzone -- so the user can still start a
    // scan or drop a SARIF file.
    let host = $('#cs-empty-host');
    if (!host) {
      host = document.createElement('div');
      host.id = 'cs-empty-host';
      $('#cs-scans').parentNode.insertBefore(host, $('#cs-scans'));
    }
    host.innerHTML = emptyState({
      icon: 'i-code',
      title: 'No code scans yet',
      body: `Type a folder path above and click <strong>Start scan</strong>
        to run the built-in LLM screening pass — or drop a SARIF
        document (Semgrep, CodeQL, Bandit, …) above to ingest external
        findings into the same dashboard.`,
    });
  } else {
    const host = $('#cs-empty-host'); if (host) host.remove();
  }
};

// Scanner Obs: when the per-tool table is empty, replace the muted
// hint with an empty state pointing at all three ingest endpoints.
const _origLoadScanObs = loadScanObs;
loadScanObs = async function() {
  await _origLoadScanObs.apply(this, arguments);
  const tt = $('#so-tools');
  if (tt && tt.textContent.trim().startsWith('No scans')) {
    tt.innerHTML = emptyState({
      icon: 'i-gauge',
      title: 'No scanner data yet',
      body: `Ingest some scans to see fleet KPIs.<br/>
        <code>POST /api/scan/external</code> for ad-hoc finding lists,
        <code>/api/scan/external/sarif</code> for any SARIF-emitting tool,
        or <code>/api/scan/external/sandbox</code> for Cuckoo / Joe /
        VMRay reports — or just drop a JSON file on the zone above.`,
      actions: [
        {label: 'Go to Detonations', panel: 'detonations'},
        {label: 'Go to Code Scan',   panel: 'codescan'},
      ],
    });
    _bindEmptyActions(tt, [
      {panel: 'detonations'}, {panel: 'codescan'},
    ]);
  }
};


// =============================================================
// v1.10 Welcome panel — default landing for the Flight-Recorder
// positioning. If the local DB has no runs yet we render an inline
// quick-start with `pip install cogniguardai` and a copy-pasteable
// `MonitoredRun` snippet so the user can record their first run
// without leaving the dashboard. (Sandbox-backed demo agent is
// deferred until the external scanner panels ship.)
// =============================================================
function _humanAgo(iso) {
  if (!iso) return '';
  const t = Date.parse(iso); if (isNaN(t)) return '';
  const s = Math.max(1, Math.floor((Date.now() - t) / 1000));
  if (s < 60)     return `${s}s ago`;
  if (s < 3600)   return `${Math.floor(s/60)}m ago`;
  if (s < 86400)  return `${Math.floor(s/3600)}h ago`;
  return `${Math.floor(s/86400)}d ago`;
}

async function loadWelcome() {
  const state = document.getElementById('welcome-state');
  if (!state) return;
  let nRuns = 0, nAgents = 0, lastRun = null;
  try {
    const [a, r] = await Promise.all([
      api('/api/agents').catch(() => ({agents: []})),
      api('/api/runs?limit=200').catch(() => ({runs: []})),
    ]);
    nAgents = (a.agents || []).length;
    const runs = r.runs || [];
    nRuns = runs.length;
    if (runs.length) {
      // Runs are returned newest-first; pick the most recent started_at.
      lastRun = runs[0].started_at || runs[0].created_at || null;
    }
  } catch (_) { /* leave at zeros */ }

  if (nRuns > 0) {
    state.innerHTML = `
      <div class="welcome-status good">
        <svg class="icon"><use href="#i-activity"/></svg>
        <div>
          <div class="welcome-status-title">AgentMonitor is recording.</div>
          <div class="welcome-status-body">
            <strong>${nRuns}</strong> run${nRuns === 1 ? '' : 's'} captured
            across <strong>${nAgents}</strong> agent${nAgents === 1 ? '' : 's'}.
            ${lastRun ? `Last activity <strong>${escape(_humanAgo(lastRun))}</strong>.` : ''}
          </div>
        </div>
        <div class="welcome-status-actions">
          <button data-welcome-act="live">Open Live View →</button>
          <button data-welcome-act="runs">Browse runs</button>
        </div>
      </div>`;
    state.querySelector('[data-welcome-act="live"]')
      .addEventListener('click', () => showPanel('live'));
    state.querySelector('[data-welcome-act="runs"]')
      .addEventListener('click', () => showPanel('runs'));
  } else {
    state.innerHTML = `
      <div class="welcome-quickstart">
        <div class="welcome-qs-title">No runs yet — let's record your first one.</div>
        <ol class="welcome-qs-steps">
          <li>Install:
            <pre class="welcome-qs-code"><code>pip install cogniguardai</code></pre>
          </li>
          <li>Wrap your agent call (works with OpenAI, Anthropic, LangChain, AutoGen, Ollama, …):
            <pre class="welcome-qs-code"><code>from agent_monitor.runner import MonitoredRun

with MonitoredRun(agent_name="my-agent", input_text="hello"):
    run_your_agent()</code></pre>
          </li>
          <li>Refresh this page — your run will appear in <strong>Live View</strong> and <strong>Runs</strong>.</li>
        </ol>
      </div>`;
  }
}

(function _wireWelcomeButtons() {
  const qs = document.getElementById('welcome-quickstart');
  if (qs) qs.addEventListener('click', () => {
    // Scroll the inline quick-start into view; render it first if the
    // user happens to be on the populated-state view.
    const state = document.getElementById('welcome-state');
    if (state && !state.querySelector('.welcome-quickstart')) {
      // Force-render the empty-state quickstart, even if runs exist,
      // so the install snippet is always one click away.
      state.innerHTML = `
        <div class="welcome-quickstart">
          <div class="welcome-qs-title">Record your first agent run</div>
          <ol class="welcome-qs-steps">
            <li>Install:
              <pre class="welcome-qs-code"><code>pip install cogniguardai</code></pre>
            </li>
            <li>Wrap your agent call:
              <pre class="welcome-qs-code"><code>from agent_monitor.runner import MonitoredRun

with MonitoredRun(agent_name="my-agent", input_text="hello"):
    run_your_agent()</code></pre>
            </li>
            <li>Refresh — your run will appear in <strong>Live View</strong>.</li>
          </ol>
        </div>`;
    }
    state && state.scrollIntoView({behavior: 'smooth', block: 'start'});
  });
})();


// ---------- boot ----------
showPanel('welcome');
refreshStatus();
setInterval(refreshStatus, 8000);
