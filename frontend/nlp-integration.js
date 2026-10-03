/* Live-only backend integration for the Marathi NLP dashboard.
   Every value shown comes from the FastAPI backend
   (multipart file upload via FormData, JSON analyze, dataset stats,
   model metrics, export). Empty states stay empty until data arrives. */
(function () {
  'use strict';

  var API_BASE = null;
  try {
    API_BASE = window.localStorage.getItem('nlp_api_base') || null;
  } catch (e) { API_BASE = null; }
  if (!API_BASE) {
    API_BASE = (window.location.port === '8000')
      ? window.location.origin
      : 'http://127.0.0.1:8000';
  }

  var activeTab = 'upload';

  // Must match BUILD in app/main.py. Shown in the header; on mismatch the
  // dashboard tells the user to restart + hard-refresh instead of silently
  // running stale backend/frontend code.
  var EXPECTED_BUILD = '1.4';

  function $(id) {
    if (typeof document.getElementById !== 'function') return null;
    return document.getElementById(id);
  }

  function setStatus(msg, isError) {
    var el = $('nlp-analyze-status');
    if (!el) {
      var btn = $('analyze-action-btn');
      if (!btn || !btn.parentElement) return;
      el = document.createElement('p');
      el.id = 'nlp-analyze-status';
      el.className = 'text-sm';
      btn.parentElement.appendChild(el);
    }
    el.textContent = msg;
    el.style.color = isError ? '#ba1a1a' : '';
  }

  async function api(path, options) {
    var res = await fetch(API_BASE + path, options);
    var data = null;
    try { data = await res.json(); } catch (e) { data = null; }
    if (!res.ok) {
      var detail = (data && (data.detail || data.reason)) || ('HTTP ' + res.status);
      throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
    }
    return data;
  }

  function postJSON(path, payload) {
    return api(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
  }

  async function refreshConnection() {
    var nodes = [];
    try {
      nodes = document.querySelectorAll('[data-backend-status]') || [];
    } catch (e) { nodes = []; }
    var online = false;
    try {
      var res = await fetch(API_BASE + '/health');
      online = !!(res && res.ok);
    } catch (e) { online = false; }
    try {
      Array.prototype.forEach.call(nodes, function (n) {
        n.textContent = online ? 'Connected' : 'Backend offline — run: powershell -ExecutionPolicy Bypass -File start-dev.ps1';
      });
    } catch (e) {}
    var tag = $('build-tag');
    if (tag) {
      if (!online) tag.textContent = 'build ? (backend offline)';
      else {
        try {
          var vr = await fetch(API_BASE + '/version');
          if (!vr.ok) tag.textContent = 'build old — restart backend!';
          else {
            var v = await vr.json();
            tag.textContent = 'build ' + (v.build || '?');
            if (v.build && v.build !== EXPECTED_BUILD) {
              tag.textContent += ' — refresh page!';
              setStatus('This page is outdated for the running backend (page v' +
                EXPECTED_BUILD + ' vs backend v' + v.build +
                '). Hard-refresh: Ctrl+Shift+R.', true);
            }
          }
        } catch (e) { tag.textContent = 'build old — restart backend!'; }
      }
    }
    return online;
  }

  function currentInput() {
    if (activeTab === 'upload') {
      var input = $('file-uploader');
      if (input && input.files && input.files[0]) return { kind: 'file', file: input.files[0] };
      return { kind: 'none' };
    }
    var ta = $('marathi-input-area');
    var text = ta ? ta.value : '';
    if (!text || !text.trim()) return { kind: 'none' };
    return { kind: 'text', text: text };
  }

  var CLAMP_LEN = 280;

  function setClampedText(bodyId, toggleId, countId, text, countLabel) {
    var body = $(bodyId), toggle = $(toggleId), count = $(countId);
    var full = text || '—';
    if (count) count.textContent = countLabel || '—';
    if (!body) return;
    if (full.length <= CLAMP_LEN) {
      body.textContent = full;
      if (toggle) toggle.style.display = 'none';
      return;
    }
    var expanded = false;
    body.textContent = full.slice(0, CLAMP_LEN) + '…';
    if (toggle) {
      toggle.style.display = '';
      toggle.textContent = 'Show more ▾';
      toggle.onclick = function () {
        expanded = !expanded;
        body.textContent = expanded ? full : (full.slice(0, CLAMP_LEN) + '…');
        toggle.textContent = expanded ? 'Show less ▴' : 'Show more ▾';
      };
    }
  }

  // Full original text of the last analysis (panes show clamped text only).
  var lastFullText = '';
  function renderAnalysis(data, sourceName) {
    lastFullText = data.original_text || '';
    var origText = data.original_text || '—';
    var cleanText = data.processed_text || data.clean_text || '—';
    setClampedText('original-text-body', 'original-toggle', 'original-count',
      origText, origText === '—' ? '—' : (origText.length + ' chars'));
    setClampedText('cleaned-text-body', 'cleaned-toggle', 'cleaned-count',
      cleanText, cleanText === '—' ? '—' : (cleanText.length + ' chars'));
    var wrap = $('tokens-body');
    if (!wrap) wrap = document.querySelector('#tab-pane-tokens div');
    var tokensToggle = $('tokens-toggle'), tokensCount = $('tokens-count');
    if (wrap) {
      wrap.textContent = '';
      var tokens = data.tokens || [];
      if (tokensCount) tokensCount.textContent = tokens.length ? (tokens.length + ' tokens') : '—';
      if (!tokens.length) {
        wrap.textContent = 'No tokens (empty after cleaning).';
        if (tokensToggle) tokensToggle.style.display = 'none';
      } else {
        var showAll = false, LIMIT = 40;
        var draw = function () {
          wrap.textContent = '';
          var list = showAll ? tokens : tokens.slice(0, LIMIT);
          list.forEach(function (t) {
            var s = document.createElement('span');
            s.className = 'tok-chip';
            s.textContent = t;
            s.title = t;
            wrap.appendChild(s);
          });
        };
        draw();
        if (tokensToggle) {
          if (tokens.length > LIMIT) {
            tokensToggle.style.display = '';
            tokensToggle.textContent = 'Show all (' + tokens.length + ') ▾';
            tokensToggle.onclick = function () {
              showAll = !showAll;
              draw();
              tokensToggle.textContent = showAll ? 'Show less ▴' : ('Show all (' + tokens.length + ') ▾');
            };
          } else tokensToggle.style.display = 'none';
        }
      }
    }
    var cards = document.querySelectorAll('#tab-pane-statistics .font-headline-md');
    var st = data.statistics || {};
    if (cards[0]) cards[0].textContent = String(st.word_count != null ? st.word_count : '—');
    if (cards[1]) cards[1].textContent = String(st.vocab_size != null ? st.vocab_size : '—');
    if (cards[2]) {
      var ttr = (st.word_count && st.vocab_size) ? (100 * st.vocab_size / st.word_count) : 0;
      cards[2].textContent = ttr.toFixed(1) + '%';
    }
    if (cards[3]) {
      var avg = (st.sentence_count && st.word_count) ? (st.word_count / st.sentence_count) : 0;
      cards[3].textContent = avg.toFixed(1);
    }
    var intent = data.intent;
    var nameEl = $('live-intent-name'), pctEl = $('live-intent-pct'),
        barEl = $('live-intent-bar'), bandEl = $('live-intent-band'),
        confEl = $('live-intent-confidence'), notesEl = $('live-intent-notes');
    if (intent && intent.intent) {
      if (nameEl) nameEl.textContent = intent.intent;
      var confText = intent.confidence_type === 'probability' && intent.confidence != null
        ? (100 * intent.confidence).toFixed(1) + '% probability'
        : (intent.confidence_type + (intent.confidence != null ? ' ' + intent.confidence.toFixed(3) : ''));
      if (pctEl) pctEl.textContent = confText;
      if (confEl) confEl.textContent = confText;
      if (barEl && intent.confidence_type === 'probability' && intent.confidence != null) {
        barEl.style.width = Math.max(0, Math.min(100, 100 * intent.confidence)) + '%';
      } else if (barEl) { barEl.style.width = '0%'; }
      if (bandEl) bandEl.textContent = intent.low_confidence ? 'Low confidence' : 'Predicted intent';
      if (notesEl) notesEl.textContent = 'Model: ' + intent.model_name + '. Labels: ' + (intent.labels || []).join(', ');
    } else {
      if (nameEl) nameEl.textContent = 'unavailable (no trained model)';
      if (pctEl) pctEl.textContent = '—';
      if (confEl) confEl.textContent = 'model unavailable';
      if (barEl) barEl.style.width = '0%';
      if (bandEl) bandEl.textContent = 'No prediction';
      if (notesEl) notesEl.textContent = (data.warnings || []).join('; ') || 'Intent model unavailable.';
    }
    renderEntities(data.entities || []);
    renderClauses(data.clauses || []);
    prependHistoryRow(sourceName || 'pasted text', data);
    return requestTranslation(data.original_text || '');
  }

  function renderClauses(clauses) {
    var tbody = $('clause-table-body');
    if (!tbody) return;
    tbody.textContent = '';
    if (!clauses.length) {
      var tr = document.createElement('tr');
      var td = document.createElement('td');
      td.setAttribute('colspan', '4');
      td.textContent = 'No clauses detected.';
      tr.appendChild(td);
      tbody.appendChild(tr);
      return;
    }
    clauses.forEach(function (c) {
      var row = document.createElement('tr');
      function cell(text, title) {
        var td = document.createElement('td');
        td.textContent = text != null ? String(text) : '—';
        if (title) td.title = title;
        return td;
      }
      row.appendChild(cell(c.index + 1));
      var full = c.original || '';
      row.appendChild(cell(full.length > 90 ? (full.slice(0, 90) + '…') : (full || '—'), full || ''));
      var intentTd = document.createElement('td');
      if (c.intent && c.intent.intent) {
        var pill = document.createElement('span');
        pill.className = 'ent-pill ent-SECTION';
        pill.textContent = c.intent.intent;
        intentTd.appendChild(pill);
      } else intentTd.textContent = '—';
      row.appendChild(intentTd);
      var n = (c.entities || []).length;
      row.appendChild(cell(n ? (n + (n === 1 ? ' entity' : ' entities')) : '—'));
      tbody.appendChild(row);
    });
  }

  var lastTranslatedText = '';
  async function requestTranslation(text) {
    var pane = $('english-text-body');
    if (!pane) pane = document.querySelector('#tab-pane-english p');
    var count = $('english-count'), retry = $('english-retry');
    if (!pane) return;
    if (retry) retry.style.display = 'none';
    if (!text || !text.trim() || text === '—') {
      pane.textContent = 'Nothing to translate.';
      if (count) count.textContent = '—';
      return;
    }
    lastTranslatedText = text;
    pane.textContent = 'Translating to English… (long documents take longer)';
    if (count) count.textContent = '…';
    setProgress(86, 'Translating to English…');
    var failed = function (msg) {
      pane.textContent = msg;
      if (count) count.textContent = 'off';
      var tg = $('english-toggle');
      if (tg) tg.style.display = 'none';
      if (retry) {
        retry.style.display = '';
        retry.onclick = function () { requestTranslation(lastTranslatedText); };
      }
    };
    try {
      var raw = await fetch(API_BASE + '/api/v1/translate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text })
      });
      if (raw.status === 404) {
        failed('Backend too old (no /translate route) — restart it: stop uvicorn, run start-dev.ps1, then Ctrl+Shift+R here.');
        return;
      }
      var out = await raw.json().catch(function () { return null; });
      if (!raw.ok || !out) throw new Error('HTTP ' + raw.status);
      if (out.available) {
        if (retry) retry.style.display = 'none';
        setClampedText('english-text-body', 'english-toggle', 'english-count',
          out.text, out.text.length + ' chars · ' + (out.model || 'groq'));
      } else failed('Translation unavailable: ' + (out.reason || 'no API key'));
    } catch (err) {
      failed('Translation unavailable (backend unreachable).');
    }
  }

  function renderEntities(entities) {
    var tbody = $('entity-table-body');
    if (!tbody) return;
    tbody.textContent = '';
    if (!entities.length) {
      var tr = document.createElement('tr');
      var td = document.createElement('td');
      td.setAttribute('colspan', '3');
      td.textContent = 'No entities detected.';
      tr.appendChild(td);
      tbody.appendChild(tr);
      return;
    }
    entities.forEach(function (e) {
      var row = document.createElement('tr');
      var textCell = document.createElement('td');
      var shown = e.text != null ? String(e.text) : '—';
      textCell.textContent = shown.length > 60 ? (shown.slice(0, 60) + '…') : shown;
      textCell.title = shown;
      row.appendChild(textCell);
      var labelCell = document.createElement('td');
      var pill = document.createElement('span');
      pill.className = 'ent-pill ent-' + (e.label || 'DATE');
      pill.textContent = e.label != null ? String(e.label) : '—';
      labelCell.appendChild(pill);
      row.appendChild(labelCell);
      var mCell = document.createElement('td');
      mCell.textContent = e.method != null ? String(e.method) : '—';
      row.appendChild(mCell);
      tbody.appendChild(row);
    });
  }

  var progressTimer = null;
  function showProgress() {
    var wrap = $('analyze-progress-wrap');
    if (wrap) wrap.style.display = '';
    setProgress(3, 'Starting…');
    if (progressTimer) { try { clearInterval(progressTimer); } catch (e) {} }
    progressTimer = setInterval(function () {
      var bar = $('analyze-progress-bar'), pct = $('analyze-progress-pct');
      if (!bar || !pct) return;
      var cur = parseInt((pct.textContent || '0').replace('%', ''), 10) || 0;
      if (cur < 88) setProgress(cur + 2, null);
    }, 450);
  }
  function setProgress(pct, stage) {
    var bar = $('analyze-progress-bar'), pctEl = $('analyze-progress-pct'),
        stageEl = $('analyze-progress-stage');
    pct = Math.max(0, Math.min(100, pct));
    if (bar) bar.style.width = pct + '%';
    if (pctEl) pctEl.textContent = pct + '%';
    if (stageEl && stage) stageEl.textContent = stage;
  }
  function stopProgress(finalPct, stage, hideAfterMs) {
    if (progressTimer) { try { clearInterval(progressTimer); } catch (e) {} progressTimer = null; }
    setProgress(finalPct, stage);
    var wrap = $('analyze-progress-wrap');
    if (wrap && hideAfterMs) {
      setTimeout(function () {
        try {
          var pct = $('analyze-progress-pct');
          var done = pct && pct.textContent === '100%';
          if (done) wrap.style.display = 'none';
        } catch (e) {}
      }, hideAfterMs);
    }
  }

  function prependHistoryRow(sourceName, data) {
    var tbody = $('history-table-body');
    if (!tbody) return;
    var placeholder = tbody.querySelector('td[colspan]');
    if (placeholder) tbody.textContent = '';
    var tr = document.createElement('tr');
    function cell(text) {
      var td = document.createElement('td');
      td.textContent = text;
      return td;
    }
    tr.appendChild(cell(String(sourceName).slice(0, 40)));
    tr.appendChild(cell(data.intent && data.intent.intent ? data.intent.intent : '— (no model)'));
    tr.appendChild(cell(String((data.entities || []).length) + ' entities'));
    tr.appendChild(cell(new Date().toLocaleDateString()));
    tr.appendChild(cell('Completed'));
    tbody.insertBefore(tr, tbody.firstChild);
    while (tbody.rows && tbody.rows.length > 12) tbody.deleteRow(-1);
  }

  var running = false;
  async function runAnalysis() {
    if (running) return;
    var input = currentInput();
    if (input.kind === 'none') {
      setStatus('Enter Marathi text or choose a .txt/.pdf/.docx file first.', true);
      return;
    }
    running = true;
    var btn = $('analyze-action-btn');
    var original = btn ? btn.innerHTML : '';
    if (btn) { btn.innerHTML = 'Analyzing…'; btn.disabled = true; }
    showProgress();
    setStatus('Contacting backend…', false);
    try {
      var data;
      var sourceName;
      if (input.kind === 'file') {
        // multipart upload expected by POST /api/v1/analyze-file
        setProgress(14, 'Uploading document…');
        var form = new FormData();
        form.append('file', input.file, input.file.name);
        var res = await fetch(API_BASE + '/api/v1/analyze-file', { method: 'POST', body: form });
        data = await res.json().catch(function () { return null; });
        if (!res.ok) throw new Error((data && (data.detail || data.reason)) || ('HTTP ' + res.status));
        sourceName = input.file.name;
      } else {
        setProgress(18, 'Sending text…');
        var pre = $('preprocess-toggle');
        data = await postJSON('/api/v1/analyze', {
          text: input.text,
          options: { remove_stopwords: !!(pre && pre.checked) }
        });
        sourceName = 'pasted text';
      }
      setProgress(62, 'Intent + entities…');
      await renderAnalysis(data, sourceName);
      var warns = (data.warnings || []).join('; ');
      stopProgress(100, 'Done ✓', 1200);
      setStatus('Analysis complete.' + (warns ? ' Notes: ' + warns : ''), false);
      if (window.location.hash !== '#/results') window.location.hash = '#/results';
    } catch (err) {
      stopProgress(100, 'Failed', 2500);
      setStatus('Analysis failed: ' + (err && err.message ? err.message : err) +
        ' — is the backend running at ' + API_BASE + '?', true);
    } finally {
      running = false;
      if (btn) { btn.innerHTML = original; btn.disabled = false; }
    }
  }

  function download(filename, mime, content) {
    var blob = new Blob([content], { type: mime + ';charset=utf-8' });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 500);
  }

  async function exportCurrent(format) {
    // Prefer the full analyzed text (result panes are clamped for display).
    var text = lastFullText || null;
    if (!text) {
      var input = currentInput();
      text = input.kind === 'text' ? input.text : null;
    }
    if (!text) {
      var orig = document.querySelector('#tab-pane-original p');
      text = orig ? orig.textContent : '';
    }
    if (!text || !text.trim() || text.indexOf('Awaiting analysis') === 0) {
      setStatus('Nothing to export yet — run an analysis first.', true);
      return;
    }
    setStatus('Preparing ' + format.toUpperCase() + ' export…', false);
    try {
      var payload = await postJSON('/api/v1/export', {
        format: format, text: text, include_translation: true });
      download(payload.filename, payload.mime_type, payload.content);
      setStatus('Exported ' + payload.filename + '.', false);
    } catch (err) {
      setStatus('Export failed: ' + (err && err.message ? err.message : err), true);
    }
  }

  function noticeBox(text) {
    var d = document.createElement('div');
    d.textContent = text;
    return d;
  }

  function widgetShell(id, title, subtitle) {
    var box = $(id);
    if (!box) return null;
    box.textContent = '';
    var h = document.createElement('h3');
    h.textContent = title;
    box.appendChild(h);
    if (subtitle) {
      var s = document.createElement('p');
      s.textContent = subtitle;
      box.appendChild(s);
    }
    return box;
  }

  function barRows(box, items) {
    items.forEach(function (it) {
      var row = document.createElement('div');
      row.style.cssText = 'display:flex;align-items:center;gap:0.5rem;margin-top:0.5rem;';
      var lab = document.createElement('span');
      lab.style.cssText = 'flex:0 0 45%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;';
      lab.textContent = it[0];
      var track = document.createElement('div');
      track.style.cssText = 'flex:1;background:rgba(0,0,0,0.08);border-radius:9999px;height:8px;overflow:hidden;';
      var fill = document.createElement('div');
      fill.style.cssText = 'background:#0051d5;height:100%;width:' + Math.max(0, Math.min(100, it[2])) + '%;';
      track.appendChild(fill);
      var val = document.createElement('span');
      val.style.cssText = 'flex:0 0 3rem;text-align:right;';
      val.textContent = String(it[1]);
      row.appendChild(lab); row.appendChild(track); row.appendChild(val);
      box.appendChild(row);
    });
  }

  function distItems(dist, limit) {
    var out = [];
    if (!dist || !dist.labels) return out;
    var max = Math.max.apply(null, dist.counts.concat([1]));
    for (var i = 0; i < dist.labels.length && i < (limit || 8); i++) {
      out.push([dist.labels[i], dist.counts[i], 100 * dist.counts[i] / max]);
    }
    return out;
  }

  function renderDatasetWidgets(stats) {
    var ok = !!(stats && stats.available);
    var reason = ok ? '' : 'No dataset connected — showing unavailable state. ' +
      ((stats && stats.reason) ? ('Backend says: ' + stats.reason) : 'Backend unreachable.');
    var box = widgetShell('live-intent-dist', 'Intent Distribution',
      ok ? 'Live label counts from the connected dataset' : 'Live intent distribution');
    if (box) {
      if (ok) barRows(box, distItems(stats.intent_distribution));
      else box.appendChild(noticeBox(reason));
    }
    box = widgetShell('live-entity-dist', 'Named Entity Recognition (NER) Density',
      ok ? 'Live entity-label counts' : 'Live entity density');
    if (box) {
      if (ok && stats.entity_distribution) barRows(box, distItems(stats.entity_distribution));
      else box.appendChild(noticeBox(ok
        ? ('Entity annotations: ' + (stats.entity_distribution_reason || 'none'))
        : reason));
    }
    var ta = ok ? stats.text_analytics : null;
    box = widgetShell('live-length-dist', 'Length Distribution',
      ok ? 'Characters per record' : 'Live length distribution');
    if (box) {
      if (ok && ta && ta.char_length_histogram) {
        var h = ta.char_length_histogram, items = [], m = 1, i;
        for (i = 0; i < h.bins.length; i++) { m = Math.max(m, h.counts[i]); }
        for (i = 0; i < h.bins.length; i++) items.push([h.bins[i] + ' chars', h.counts[i], 100 * h.counts[i] / m]);
        barRows(box, items);
      } else box.appendChild(noticeBox(reason));
    }
    box = widgetShell('live-top-tokens', 'Top Lexical Tokens',
      ok ? 'Most frequent tokens' : 'Live top tokens');
    if (box) {
      if (ok && ta) {
        var fw = (ta.frequent_words || []).slice(0, 8), mx = 1, rows = [];
        fw.forEach(function (w) { mx = Math.max(mx, w.count); });
        fw.forEach(function (w) { rows.push([w.token, w.count, 100 * w.count / mx]); });
        barRows(box, rows.length ? rows : [['—', 0, 0]]);
      } else box.appendChild(noticeBox(reason));
    }
    box = widgetShell('live-ngrams', 'N-Gram Collocations',
      ok ? 'Most frequent bigrams' : 'Live bigram collocations');
    if (box) {
      if (ok && ta) {
        var fb = (ta.frequent_bigrams || []).slice(0, 8), m2 = 1, r2 = [];
        fb.forEach(function (w) { m2 = Math.max(m2, w.count); });
        fb.forEach(function (w) { r2.push([w.bigram, w.count, 100 * w.count / m2]); });
        barRows(box, r2.length ? r2 : [['—', 0, 0]]);
      } else box.appendChild(noticeBox(reason));
    }
    box = widgetShell('live-domain', 'Domain Ratio',
      ok ? 'Top intent classes in the connected dataset' : 'Live domain split');
    if (box) {
      if (ok) barRows(box, distItems(stats.intent_distribution, 4));
      else box.appendChild(noticeBox(reason));
    }
    if (ok) {
      var dc = $('stat-doc-count');
      if (dc) dc.textContent = String(stats.records.used);
      var order = (stats.intent_distribution.labels || []).map(function (l, idx) {
        return [l, stats.intent_distribution.counts[idx]];
      });
      var sl = $('stat-legal'), sf = $('stat-financial');
      if (sl) sl.textContent = order[0] ? order[0][0] + ': ' + order[0][1] : '—';
      if (sf) sf.textContent = order[1] ? order[1][0] + ': ' + order[1][1] : '—';
    }
  }

  async function loadDataset() {
    var h1 = $('dataset');
    var banner = null;
    try {
      if (h1 && h1.parentElement) {
        banner = document.createElement('div');
        banner.id = 'live-dataset-banner';
        banner.textContent = 'Loading dataset status…';
        h1.parentElement.appendChild(banner);
      }
    } catch (e) { banner = null; }
    var stats = null;
    try {
      var res = await fetch(API_BASE + '/api/v1/dataset/stats');
      stats = await res.json();
    } catch (e) { stats = null; }
    try { renderDatasetWidgets(stats); } catch (e) {}
    if (banner) {
      if (stats && stats.available) {
        banner.textContent = 'Live dataset: ' + stats.records.used + ' usable of ' +
          stats.records.total + ' records.';
        var dc = $('stat-doc-count');
        if (dc) dc.textContent = String(stats.records.used);
      } else {
        banner.textContent = 'No dataset connected yet — place the annotated CSV at ' +
          'data/raw/intents.csv, then this panel shows live statistics. ' +
          ((stats && stats.reason) ? ('Backend says: ' + stats.reason) : 'Backend unreachable.');
      }
    }
    var tbody = $('dataset-table-body');
    var rows = [];
    try {
      var r2 = await fetch(API_BASE + '/api/v1/dataset/records?page=1&page_size=8');
      var rec = await r2.json();
      if (rec && rec.available) rows = rec.records || [];
    } catch (e) { rows = []; }
    if (tbody) {
      tbody.textContent = '';
      if (!rows.length) {
        var tr = document.createElement('tr');
        var td = document.createElement('td');
        td.setAttribute('colspan', '2');
        td.textContent = 'No dataset records available.';
        tr.appendChild(td);
        tbody.appendChild(tr);
      } else {
        rows.forEach(function (row) {
          var tr2 = document.createElement('tr');
          ['text', 'intent'].forEach(function (k) {
            var td2 = document.createElement('td');
            var v = row[k] != null ? String(row[k]) : '—';
            td2.textContent = k === 'text' ? v.slice(0, 60) : v;
            tr2.appendChild(td2);
          });
          tbody.appendChild(tr2);
        });
      }
    }
  }

  async function loadMetrics() {
    var data = null;
    try {
      var res = await fetch(API_BASE + '/api/v1/model/metrics');
      data = await res.json();
    } catch (e) { data = null; }
    function setText(id, text) {
      var el = $(id);
      if (el) el.textContent = text;
    }
    if (data && data.available) {
      var sel = data.selected_model;
      var r = (data.results && data.results[sel]) || {};
      setText('live-macro-f1', (100 * (r.macro_f1 || 0)).toFixed(1) + '%');
      setText('live-model-f1', 'F1: ' + (100 * (r.macro_f1 || 0)).toFixed(1) + '% (' + sel + ')');
    } else {
      setText('live-macro-f1', '—');
      setText('live-model-f1', 'F1: — (no trained model)');
    }
  }

  var pageMains = {};
  var currentPage = null;
  var PAGE_TITLES = {
    home: 'Dashboard', analyzer: 'Document Analyzer',
    results: 'Analysis Results', dataset: 'Dataset Analytics'
  };

  function detectPages() {
    var mains = [];
    try { mains = Array.prototype.slice.call(document.querySelectorAll('main')); }
    catch (e) { mains = []; }
    mains.forEach(function (m) {
      var page = 'home';
      try {
        if (m.querySelector('#marathi-input-area')) page = 'analyzer';
        else if (m.querySelector('#tab-pane-original')) page = 'results';
        else if (m.querySelector('#dataset')) page = 'dataset';
      } catch (e) {}
      m.dataset.page = page;
      if (!pageMains[page]) pageMains[page] = m;
    });
    if (!pageMains.home && mains[0]) {
      mains[0].dataset.page = 'home';
      pageMains.home = mains[0];
    }
  }

  function showPage(page) {
    if (!pageMains[page]) page = 'home';
    Object.keys(pageMains).forEach(function (p) {
      pageMains[p].style.display = (p === page) ? '' : 'none';
    });
    currentPage = page;
    document.title = 'Marathi NLP — ' + (PAGE_TITLES[page] || page);
    var links = [];
    try { links = document.querySelectorAll('aside nav a'); } catch (e) { links = []; }
    Array.prototype.forEach.call(links, function (a) {
      var href = '';
      try { href = a.getAttribute('href') || ''; } catch (e) {}
      var active = href === '#/' + page;
      a.style.fontWeight = active ? '700' : '';
      try {
        if (active) a.setAttribute('aria-current', 'page');
        else a.removeAttribute('aria-current');
      } catch (e) {}
    });
    try { window.scrollTo(0, 0); } catch (e) {}
  }

  function routeFromHash() {
    var h = '';
    try { h = (window.location.hash || '').replace(/^#\/?/, ''); } catch (e) {}
    return PAGE_TITLES[h] ? h : 'home';
  }

  function initRouter() {
    detectPages();
    var asides = [];
    try { asides = document.querySelectorAll('aside'); } catch (e) {}
    for (var i = 1; i < asides.length; i++) {
      try { asides[i].style.display = 'none'; } catch (e) {}
    }
    var headers = [];
    try { headers = document.querySelectorAll('body > div > header, body header, header'); }
    catch (e) { headers = []; }
    var seenHeader = false;
    Array.prototype.forEach.call(headers, function (h) {
      if (!seenHeader) { seenHeader = true; return; }
      try { h.style.display = 'none'; } catch (e) {}
    });
    var links = [];
    try { links = document.querySelectorAll('aside nav a'); } catch (e) {}
    Array.prototype.forEach.call(links, function (a) {
      var href = '';
      try { href = a.getAttribute('href') || ''; } catch (e) {}
      var navMap = { '#analyzer': '#/analyzer', '#results': '#/results', '#dataset': '#/dataset', '#top': '#/home', '#dashboard': '#/home' };
      if (navMap[href]) { try { a.setAttribute('href', navMap[href]); } catch (e) {} }
    });
    try { window.addEventListener('hashchange', function () { showPage(routeFromHash()); }); }
    catch (e) {}
    try {
      document.addEventListener('click', function (e) {
        var t = e.target && e.target.closest ? e.target.closest('a[href^="#"]') : null;
        if (!t || !pageMains) return;
        var href = '';
        try { href = t.getAttribute('href'); } catch (err) { return; }
        if (!href || href.indexOf('#/') === 0) return;
        var el = null;
        try { el = document.getElementById(href.slice(1)); } catch (err) { return; }
        if (!el) return;
        var main = el.closest ? el.closest('main') : null;
        if (main && main.dataset.page && main.dataset.page !== currentPage) {
          e.preventDefault();
          showPage(main.dataset.page);
          if (history.replaceState) history.replaceState(null, '', '#/' + main.dataset.page);
          setTimeout(function () { el.scrollIntoView(); }, 60);
        }
      });
    } catch (e) {}
    showPage(routeFromHash());
  }

  function switchTabUI(name) {
    activeTab = name;
    var up = $('upload-panel'), pp = $('paste-panel');
    var ub = $('tab-upload-btn'), pb = $('tab-paste-btn');
    if (up) up.style.display = name === 'upload' ? '' : 'none';
    if (pp) pp.style.display = name === 'paste' ? '' : 'none';
    if (ub) ub.style.fontWeight = name === 'upload' ? '700' : '';
    if (pb) pb.style.fontWeight = name === 'paste' ? '700' : '';
  }

  function updateWordCount() {
    var ta = $('marathi-input-area');
    var cc = $('char-counter');
    if (!ta || !cc) return;
    var n = ta.value.trim() ? ta.value.trim().split(/\s+/).length : 0;
    cc.textContent = n + ' words';
  }

  document.addEventListener('DOMContentLoaded', function () {
    initRouter();
    switchTabUI('upload');
    var ub = $('tab-upload-btn');
    if (ub) ub.addEventListener('click', function () { switchTabUI('upload'); });
    var pb = $('tab-paste-btn');
    if (pb) pb.addEventListener('click', function () { switchTabUI('paste'); });
    var ta = $('marathi-input-area');
    if (ta) ta.addEventListener('input', updateWordCount);
    var up = $('file-uploader');
    if (up) up.addEventListener('change', function () {
      var nm = $('staged-file-name'), st = $('staged-file-status');
      if (up.files && up.files[0]) {
        if (nm) nm.textContent = up.files[0].name;
        if (st) st.textContent = 'Ready for analysis';
      }
    });
    var clearBtn = $('clear-file-btn');
    if (clearBtn) clearBtn.addEventListener('click', function () {
      if (up) up.value = '';
      var nm = $('staged-file-name');
      if (nm) nm.textContent = 'No file staged';
      var st = $('staged-file-status');
      if (st) st.textContent = 'Please choose a document';
    });
    var btn = $('analyze-action-btn');
    if (btn) btn.addEventListener('click', function () { runAnalysis(); });
    var ej = $('exportJsonBtn');
    if (ej) ej.addEventListener('click', function () { exportCurrent('json'); });
    var pdf = $('downloadPdfBtn');
    if (pdf) pdf.addEventListener('click', function () { exportCurrent('csv'); });
    var fi = $('filter-intent');
    if (fi) fi.addEventListener('change', function () {
      var q = (fi.value || 'all').toLowerCase();
      Array.prototype.forEach.call(document.querySelectorAll('tbody tr'), function (row) {
        row.style.display = (!q || q === 'all' || row.innerText.toLowerCase().indexOf(q) >= 0) ? '' : 'none';
      });
    });
    var mains = [];
    try { mains = document.querySelectorAll('main'); } catch (e) {}
    if (mains && mains[0]) {
      try {
        var note = document.createElement('p');
        note.style.cssText = 'margin:1.5rem 0;';
        note.textContent = 'Experimental NLP research tool — does not provide legal advice. ' +
          'Verify all outputs with source documents and qualified professionals.';
        mains[0].appendChild(note);
      } catch (e) {}
    }
    refreshConnection();
    loadDataset();
    loadMetrics();
  });
})();
