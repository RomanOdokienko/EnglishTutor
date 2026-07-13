(function () {
  'use strict';

  // ---- API base (same behavior as app.js) ----
  function normalizeApiBase(raw) {
    const v = String(raw || '').trim();
    return v ? v.replace(/\/+$/, '') : '';
  }
  function apiBase() {
    const params = new URLSearchParams(window.location.search);
    const q = normalizeApiBase(params.get('api_base'));
    if (q) { try { localStorage.setItem('ENGLISH_TUTOR_API_BASE_URL', q); } catch (e) {} return q; }
    const inline = normalizeApiBase(window.ENGLISH_TUTOR_API_BASE_URL);
    if (inline) return inline;
    try { return normalizeApiBase(localStorage.getItem('ENGLISH_TUTOR_API_BASE_URL')); } catch (e) { return ''; }
  }
  function apiUrl(path) {
    if (/^https?:\/\//i.test(path)) return path;
    const base = apiBase();
    if (!base) return path;
    return base + (path.startsWith('/') ? path : '/' + path);
  }

  // ---- Config ----
  var PALETTE = ['#0ea5e9', '#f97316', '#10b981', '#a855f7']; // speaker 1, 2, ...
  var LOW_SAMPLE = 120; // english words below this = low-confidence point
  var CATEGORIES = [
    ['ARTICLE', 'Articles'], ['TENSE', 'Verb Tense'], ['VERB', 'Verb Form'],
    ['PREP', 'Prepositions'], ['ORDER', 'Word Order'], ['WORD', 'Wrong Word'],
    ['COLLOC', 'Collocation'],
  ];
  var FLUENCY = [
    { key: 'avg_words_per_turn', label: 'Words per turn', better: 'up' },
    { key: 'filler_per_100w', label: 'Fillers / 100 words', better: 'down' },
    { key: 'l1_fallback_pct', label: 'Russian fallback %', better: 'down' },
    { key: 'lexical_diversity_mattr', label: 'Lexical diversity (MATTR)', better: 'up' },
  ];

  var MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  function shortDate(d) {
    var m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(d || '');
    if (!m) return d || '';
    return parseInt(m[3], 10) + ' ' + MONTHS[parseInt(m[2], 10) - 1];
  }
  function niceMax(v) {
    if (!(v > 0)) return 1;
    var p = Math.pow(10, Math.floor(Math.log10(v)));
    var n = v / p;
    var m = n <= 1 ? 1 : n <= 2 ? 2 : n <= 2.5 ? 2.5 : n <= 5 ? 5 : 10;
    return m * p;
  }
  var SVGNS = 'http://www.w3.org/2000/svg';
  function el(tag, attrs) {
    var node = document.createElementNS(SVGNS, tag);
    for (var k in attrs) node.setAttribute(k, attrs[k]);
    return node;
  }

  var state = { history: null, speakers: [] };
  var tooltip = document.getElementById('pg-tooltip');

  function speakerColor(name) {
    var i = state.speakers.indexOf(name);
    return PALETTE[(i < 0 ? 0 : i) % PALETTE.length];
  }

  // Build [{name,color,values:[{v,low}|null]}] for a metric extractor.
  function buildSeries(extract) {
    var sessions = state.history.sessions;
    return state.speakers.map(function (name) {
      var values = sessions.map(function (s) {
        var p = (s.participants || []).find(function (x) { return x.name === name; });
        if (!p || !p.derived) return null;
        var v = extract(p.derived);
        if (v === null || v === undefined || isNaN(v)) return null;
        var words = (p.derived.metrics || {}).english_word_count || 0;
        return { v: v, low: words < LOW_SAMPLE };
      });
      return { name: name, color: speakerColor(name), values: values };
    });
  }

  // ---- Chart renderer (SVG line chart with hover) ----
  function makeChart(series, dates, opts) {
    opts = opts || {};
    var compact = !!opts.compact;
    var W = 720, H = compact ? 150 : 240;
    var pad = compact ? { t: 12, r: 14, b: 24, l: 30 } : { t: 16, r: 74, b: 26, l: 34 };
    var plotW = W - pad.l - pad.r, plotH = H - pad.t - pad.b;

    var all = [];
    series.forEach(function (s) { s.values.forEach(function (pt) { if (pt) all.push(pt.v); }); });
    var max = niceMax(all.length ? Math.max.apply(null, all) : 1);
    var n = dates.length;
    function xAt(i) { return pad.l + (n > 1 ? (i / (n - 1)) * plotW : plotW / 2); }
    function yAt(v) { return pad.t + (1 - v / (max || 1)) * plotH; }

    var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img' });

    // gridlines + y ticks (0, mid, max)
    [0, max / 2, max].forEach(function (t) {
      var y = yAt(t);
      svg.appendChild(el('line', { x1: pad.l, y1: y, x2: W - pad.r, y2: y, stroke: '#eef2f7', 'stroke-width': 1 }));
      var lbl = el('text', { x: pad.l - 6, y: y + 3.5, 'text-anchor': 'end', 'font-size': 10, fill: '#94a3b8' });
      lbl.textContent = (max <= 5 ? t.toFixed(t % 1 ? 1 : 0) : Math.round(t));
      svg.appendChild(lbl);
    });

    // x labels (first & last always; middle only on hero)
    dates.forEach(function (d, i) {
      if (compact && i !== 0 && i !== n - 1) return;
      var t = el('text', { x: xAt(i), y: H - 8, 'text-anchor': i === 0 ? 'start' : i === n - 1 ? 'end' : 'middle', 'font-size': 10, fill: '#94a3b8' });
      t.textContent = shortDate(d);
      svg.appendChild(t);
    });

    // series polylines + points
    series.forEach(function (s) {
      var segPts = [];
      var d = '';
      s.values.forEach(function (pt, i) {
        if (!pt) { return; }
        var x = xAt(i), y = yAt(pt.v);
        d += (d ? ' L' : 'M') + x + ' ' + y;
        segPts.push({ x: x, y: y, pt: pt });
      });
      if (d) svg.appendChild(el('path', { d: d, fill: 'none', stroke: s.color, 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }));
      segPts.forEach(function (sp) {
        svg.appendChild(el('circle', {
          cx: sp.x, cy: sp.y, r: 4,
          fill: sp.pt.low ? '#ffffff' : s.color,
          stroke: s.color, 'stroke-width': 2,
        }));
      });
      // direct end-label on hero charts
      if (!compact && segPts.length) {
        var last = segPts[segPts.length - 1];
        var t = el('text', { x: last.x + 8, y: last.y + 3.5, 'font-size': 11, 'font-weight': 600, fill: s.color });
        t.textContent = s.name;
        svg.appendChild(t);
      }
    });

    // hover overlay
    var crosshair = el('line', { x1: 0, y1: pad.t, x2: 0, y2: H - pad.b, stroke: '#cbd5e1', 'stroke-width': 1, 'stroke-dasharray': '3 3', opacity: 0 });
    svg.appendChild(crosshair);
    var overlay = el('rect', { x: pad.l, y: pad.t, width: plotW, height: plotH, fill: 'transparent', style: 'cursor:crosshair' });
    svg.appendChild(overlay);

    function onMove(evt) {
      var rect = svg.getBoundingClientRect();
      var relX = (evt.clientX - rect.left) / rect.width * W;
      var idx = Math.max(0, Math.min(n - 1, Math.round((relX - pad.l) / (plotW / (n > 1 ? n - 1 : 1)))));
      crosshair.setAttribute('x1', xAt(idx)); crosshair.setAttribute('x2', xAt(idx)); crosshair.setAttribute('opacity', 1);
      var rows = series.map(function (s) {
        var pt = s.values[idx];
        var val = pt ? (Math.round(pt.v * 100) / 100) : '—';
        return '<div class="pg-tt-row"><span class="pg-tt-dot" style="background:' + s.color + '"></span>' + s.name + ': <b>' + val + '</b>' + (pt && pt.low ? ' <span style="opacity:.6">(low sample)</span>' : '') + '</div>';
      }).join('');
      tooltip.innerHTML = '<b>' + shortDate(dates[idx]) + '</b>' + rows;
      tooltip.style.opacity = 1;
      tooltip.style.left = (evt.clientX + 14) + 'px';
      tooltip.style.top = (evt.clientY + 14) + 'px';
    }
    overlay.addEventListener('mousemove', onMove);
    overlay.addEventListener('mouseleave', function () { crosshair.setAttribute('opacity', 0); tooltip.style.opacity = 0; });
    return svg;
  }

  function card(title, hint, hintDir, svg, hero) {
    var c = document.createElement('div');
    c.className = 'pg-card' + (hero ? ' pg-hero' : '');
    var h = document.createElement('h3'); h.textContent = title; c.appendChild(h);
    if (hint) {
      var p = document.createElement('p'); p.className = 'pg-hint';
      var arrow = hintDir === 'up' ? '↑' : hintDir === 'down' ? '↓' : '';
      p.innerHTML = (arrow ? '<span class="arrow pg-good">' + arrow + '</span>' : '') + '<span>' + hint + '</span>';
      c.appendChild(p);
    }
    c.appendChild(svg);
    return c;
  }

  function section(title, sub) {
    var s = document.createElement('div'); s.className = 'pg-section';
    var h = document.createElement('h2'); h.textContent = title; s.appendChild(h);
    if (sub) { var p = document.createElement('p'); p.className = 'pg-sub'; p.textContent = sub; s.appendChild(p); }
    return s;
  }

  function render() {
    var root = document.getElementById('pg-root');
    root.innerHTML = '';
    var sessions = (state.history && state.history.sessions) || [];
    if (!sessions.length) { root.innerHTML = '<p class="pg-empty">No sessions yet. Record or upload one to see progress.</p>'; return; }
    var dates = sessions.map(function (s) { return s.date; });

    // legend
    var legend = document.getElementById('pg-legend');
    legend.innerHTML = state.speakers.map(function (name) {
      return '<span class="pg-leg-item"><span class="pg-swatch" style="background:' + speakerColor(name) + '"></span>' + name + '</span>';
    }).join('');

    // version note
    var last = sessions[sessions.length - 1];
    var ver = (last && last.analysis_version) || {};
    document.getElementById('pg-version').textContent = ver.metrics
      ? 'metrics v' + ver.metrics + ' · taxonomy v' + ver.taxonomy + (ver.annotation_model ? ' · ' + ver.annotation_model : '')
      : '';

    // 1) Grammar accuracy (hero)
    var s1 = section('Grammar accuracy', 'Errors per 100 words spoken. Lower is better. Hollow points = short session (low confidence).');
    var g1 = document.createElement('div'); g1.className = 'pg-grid';
    g1.appendChild(card('Overall error density', 'lower is better', 'down',
      makeChart(buildSeries(function (d) { return (d.grammar || {}).error_density_per_100w; }), dates, {}), true));
    s1.appendChild(g1);
    root.appendChild(s1);

    // 2) By category (small multiples)
    var s2 = section('By grammar category', 'Which specific patterns are closing. Each mini-chart is errors per 100 words for that category.');
    var g2 = document.createElement('div'); g2.className = 'pg-grid';
    CATEGORIES.forEach(function (pair) {
      var code = pair[0], label = pair[1];
      var series = buildSeries(function (d) {
        var by = (d.grammar || {}).by_category_density || {};
        return by[code];
      });
      g2.appendChild(card(label, 'lower is better', 'down', makeChart(series, dates, { compact: true })));
    });
    s2.appendChild(g2);
    root.appendChild(s2);

    // 3) Fluency & delivery (deterministic)
    var s3 = section('Fluency & delivery', 'Reproducible, computed from the transcript with no AI — the most trustworthy signals.');
    var g3 = document.createElement('div'); g3.className = 'pg-grid';
    FLUENCY.forEach(function (f) {
      var series = buildSeries(function (d) { return (d.metrics || {})[f.key]; });
      var hint = f.better === 'up' ? 'higher is better' : 'lower is better';
      g3.appendChild(card(f.label, hint, f.better, makeChart(series, dates, { compact: true })));
    });
    s3.appendChild(g3);
    root.appendChild(s3);

    // 4) Data table (accessibility + contrast relief)
    root.appendChild(buildTable(sessions));
  }

  function buildTable(sessions) {
    var box = document.createElement('details'); box.className = 'pg-table-box';
    var sum = document.createElement('summary'); sum.textContent = 'Show data table'; box.appendChild(sum);
    var wrap = document.createElement('div'); wrap.className = 'pg-table-wrap';
    var rows = [];
    rows.push('<tr><th>Speaker / date</th>' + sessions.map(function (s) { return '<th>' + shortDate(s.date) + '</th>'; }).join('') + '</tr>');
    var metrics = [
      ['Error density /100w', function (d) { return (d.grammar || {}).error_density_per_100w; }],
      ['Words / turn', function (d) { return (d.metrics || {}).avg_words_per_turn; }],
      ['Fillers /100w', function (d) { return (d.metrics || {}).filler_per_100w; }],
      ['Russian %', function (d) { return (d.metrics || {}).l1_fallback_pct; }],
      ['MATTR', function (d) { return (d.metrics || {}).lexical_diversity_mattr; }],
    ];
    state.speakers.forEach(function (name) {
      rows.push('<tr><td colspan="' + (sessions.length + 1) + '" style="background:#f8fafc;font-weight:600">' + name + '</td></tr>');
      metrics.forEach(function (m) {
        var cells = sessions.map(function (s) {
          var p = (s.participants || []).find(function (x) { return x.name === name; });
          var d = p && p.derived;
          var v = d ? m[1](d) : null;
          return '<td>' + (v === null || v === undefined ? '—' : v) + '</td>';
        }).join('');
        rows.push('<tr><td>' + m[0] + '</td>' + cells + '</tr>');
      });
    });
    wrap.innerHTML = '<table class="pg-table"><thead>' + rows[0] + '</thead><tbody>' + rows.slice(1).join('') + '</tbody></table>';
    box.appendChild(wrap);
    return box;
  }

  function collectSpeakers(sessions) {
    var seen = [];
    sessions.forEach(function (s) {
      (s.participants || []).forEach(function (p) { if (p.name && seen.indexOf(p.name) < 0) seen.push(p.name); });
    });
    return seen;
  }

  async function load() {
    var res = await fetch(apiUrl('/history.json'), { cache: 'no-store' });
    if (!res.ok) throw new Error('Unable to load history.json');
    state.history = await res.json();
    state.history.sessions = (state.history.sessions || []).slice().sort(function (a, b) { return (a.date || '').localeCompare(b.date || ''); });
    state.speakers = collectSpeakers(state.history.sessions);
  }

  function attachReanalyze() {
    var btn = document.getElementById('pg-reanalyze');
    btn.addEventListener('click', async function () {
      btn.disabled = true; var old = btn.textContent; btn.textContent = 'Re-analyzing…';
      try {
        var res = await fetch(apiUrl('/api/reanalyze'), { method: 'POST' });
        if (!res.ok) throw new Error(await res.text());
        await load(); render();
        btn.textContent = 'Done ✓';
        setTimeout(function () { btn.textContent = old; btn.disabled = false; }, 1500);
      } catch (e) {
        btn.textContent = 'Failed'; btn.disabled = false;
        console.error(e);
      }
    });
  }

  (async function () {
    attachReanalyze();
    try { await load(); render(); }
    catch (e) {
      document.getElementById('pg-root').innerHTML = '<p class="pg-empty">Could not load progress data: ' + (e && e.message ? e.message : e) + '</p>';
    }
  })();
})();
