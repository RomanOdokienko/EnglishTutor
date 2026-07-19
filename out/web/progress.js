(function () {
  'use strict';

  function normalizeApiBase(raw) {
    var value = String(raw || '').trim();
    return value ? value.replace(/\/+$/, '') : '';
  }

  function apiBase() {
    var params = new URLSearchParams(window.location.search);
    var queryBase = normalizeApiBase(params.get('api_base'));
    if (queryBase) {
      try { localStorage.setItem('ENGLISH_TUTOR_API_BASE_URL', queryBase); } catch (error) {}
      return queryBase;
    }
    var inlineBase = normalizeApiBase(window.ENGLISH_TUTOR_API_BASE_URL);
    if (inlineBase) return inlineBase;
    try { return normalizeApiBase(localStorage.getItem('ENGLISH_TUTOR_API_BASE_URL')); } catch (error) { return ''; }
  }

  function apiUrl(path) {
    if (/^https?:\/\//i.test(path)) return path;
    var base = apiBase();
    if (!base) return path;
    return base + (path.startsWith('/') ? path : '/' + path);
  }

  var PALETTE = ['#0ea5e9', '#f97316', '#10b981', '#a855f7'];
  var LOW_SAMPLE = 120;
  var CATEGORIES = [
    { code: 'ARTICLE', label: 'Articles' },
    { code: 'TENSE', label: 'Verb tense' },
    { code: 'VERB', label: 'Verb form' },
    { code: 'PREP', label: 'Prepositions' },
    { code: 'ORDER', label: 'Word order' },
    { code: 'WORD', label: 'Wrong word' },
    { code: 'COLLOC', label: 'Collocation' },
  ];
  var SECONDARY_METRICS = [
    {
      key: 'avg_words_per_turn',
      label: 'Words per turn',
      description: 'Average length of a speaking turn. This is descriptive, not a score.',
      help: 'Longer is not automatically better: the topic, role and conversation style strongly affect turn length.',
      decimals: 1,
    },
    {
      key: 'lexical_diversity_mattr',
      label: 'Lexical diversity (MATTR)',
      description: 'Vocabulary variety adjusted for transcript length.',
      help: 'MATTR averages type-token ratio over a 50-word moving window. It is useful for context, but small changes are hard to interpret.',
      decimals: 2,
    },
  ];
  var MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  var state = { history: null, speakers: [], focus: { focuses: [] } };
  var tooltip = document.getElementById('pg-tooltip');

  function escapeHtml(value) {
    return String(value === null || value === undefined ? '' : value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function shortDate(date) {
    var match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(date || '');
    if (!match) return date || '';
    return parseInt(match[3], 10) + ' ' + MONTHS[parseInt(match[2], 10) - 1];
  }

  function formatValue(value, decimals) {
    if (value === null || value === undefined || !isFinite(value)) return '—';
    return Number(value).toFixed(decimals === undefined ? 1 : decimals);
  }

  function niceMax(value) {
    if (!(value > 0)) return 1;
    var power = Math.pow(10, Math.floor(Math.log10(value)));
    var normalized = value / power;
    var rounded = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 2.5 ? 2.5 : normalized <= 5 ? 5 : 10;
    return rounded * power;
  }

  var SVGNS = 'http://www.w3.org/2000/svg';
  function svgElement(tag, attrs) {
    var node = document.createElementNS(SVGNS, tag);
    Object.keys(attrs || {}).forEach(function (key) { node.setAttribute(key, attrs[key]); });
    return node;
  }

  function speakerColor(name) {
    var index = state.speakers.indexOf(name);
    return PALETTE[(index < 0 ? 0 : index) % PALETTE.length];
  }

  function exclusionReason(session, words, requiresAnnotations) {
    if (words < LOW_SAMPLE) return 'Only ' + words + ' English words';
    var status = String((session.analysis_version || {}).annotations_status || '').toLowerCase();
    if (requiresAnnotations && status && status !== 'ok') return 'Annotations ' + status;
    return '';
  }

  function buildSeries(extract, options) {
    options = options || {};
    var sessions = state.history.sessions;
    return state.speakers.map(function (name) {
      var values = sessions.map(function (session) {
        var participant = (session.participants || []).find(function (item) { return item.name === name; });
        if (!participant || !participant.derived) return null;
        var rawValue = extract(participant.derived);
        if (rawValue === null || rawValue === undefined || rawValue === '') return null;
        var value = Number(rawValue);
        if (!Number.isFinite(value)) return null;
        var words = Number((participant.derived.metrics || {}).english_word_count || 0);
        var reason = exclusionReason(session, words, !!options.requiresAnnotations);
        return { v: value, words: words, excluded: !!reason, reason: reason };
      });
      return { name: name, color: speakerColor(name), values: values };
    });
  }

  function validPoints(series) {
    var points = [];
    series.forEach(function (speaker) {
      speaker.values.forEach(function (point) {
        if (point && !point.excluded) points.push(point);
      });
    });
    return points;
  }

  function hasMeasuredValues(series) {
    return validPoints(series).some(function (point) { return point.v > 0; });
  }

  function makeChart(series, dates, options) {
    options = options || {};
    var compact = !!options.compact;
    var decimals = options.decimals === undefined ? 1 : options.decimals;
    var width = 720;
    var height = compact ? 196 : 286;
    var pad = compact ? { t: 16, r: 20, b: 30, l: 38 } : { t: 20, r: 82, b: 32, l: 42 };
    var plotWidth = width - pad.l - pad.r;
    var plotHeight = height - pad.t - pad.b;
    var points = validPoints(series);
    var rawMax = points.length ? Math.max.apply(null, points.map(function (point) { return point.v; })) : 1;
    var max = niceMax(rawMax * 1.12);
    // Do not reserve an x-axis position for a date excluded for every speaker.
    // It remains visible in the exclusion notice below the chart.
    var visibleIndexes = dates.map(function (_date, index) { return index; }).filter(function (index) {
      return series.some(function (speaker) {
        var point = speaker.values[index];
        return point && !point.excluded;
      });
    });
    var count = visibleIndexes.length;

    function xAt(index) { return pad.l + (count > 1 ? (index / (count - 1)) * plotWidth : plotWidth / 2); }
    function yAt(value) { return pad.t + (1 - value / max) * plotHeight; }

    var svg = svgElement('svg', {
      viewBox: '0 0 ' + width + ' ' + height,
      role: 'img',
      'aria-label': options.ariaLabel || 'Progress chart',
    });

    [0, max / 2, max].forEach(function (tick) {
      var y = yAt(tick);
      svg.appendChild(svgElement('line', { x1: pad.l, y1: y, x2: width - pad.r, y2: y, stroke: '#e8eef5', 'stroke-width': 1 }));
      var label = svgElement('text', { x: pad.l - 8, y: y + 4, 'text-anchor': 'end', 'font-size': 11, fill: '#8492aa' });
      label.textContent = formatValue(tick, decimals);
      svg.appendChild(label);
    });

    visibleIndexes.forEach(function (sourceIndex, displayIndex) {
      if (compact && displayIndex !== 0 && displayIndex !== count - 1) return;
      var label = svgElement('text', {
        x: xAt(displayIndex),
        y: height - 8,
        'text-anchor': displayIndex === 0 ? 'start' : displayIndex === count - 1 ? 'end' : 'middle',
        'font-size': 11,
        fill: '#8492aa',
      });
      label.textContent = shortDate(dates[sourceIndex]);
      svg.appendChild(label);
    });

    series.forEach(function (speaker) {
      var path = '';
      var segmentOpen = false;
      var renderedPoints = [];
      visibleIndexes.forEach(function (sourceIndex, displayIndex) {
        var point = speaker.values[sourceIndex];
        if (!point || point.excluded) {
          segmentOpen = false;
          return;
        }
        var x = xAt(displayIndex);
        var y = yAt(point.v);
        path += (segmentOpen ? ' L' : ' M') + x + ' ' + y;
        segmentOpen = true;
        renderedPoints.push({ x: x, y: y, point: point });
      });
      if (path) {
        svg.appendChild(svgElement('path', {
          d: path,
          fill: 'none',
          stroke: speaker.color,
          'stroke-width': compact ? 2.25 : 2.75,
          'stroke-linejoin': 'round',
          'stroke-linecap': 'round',
        }));
      }
      renderedPoints.forEach(function (rendered) {
        svg.appendChild(svgElement('circle', {
          cx: rendered.x,
          cy: rendered.y,
          r: compact ? 4 : 4.5,
          fill: speaker.color,
          stroke: '#ffffff',
          'stroke-width': 1.5,
        }));
      });
      if (!compact && renderedPoints.length) {
        var last = renderedPoints[renderedPoints.length - 1];
        var endLabel = svgElement('text', { x: last.x + 9, y: last.y + 4, 'font-size': 12, 'font-weight': 700, fill: speaker.color });
        endLabel.textContent = speaker.name;
        svg.appendChild(endLabel);
      }
    });

    var crosshair = svgElement('line', {
      x1: 0,
      y1: pad.t,
      x2: 0,
      y2: height - pad.b,
      stroke: '#94a3b8',
      'stroke-width': 1,
      'stroke-dasharray': '3 3',
      opacity: 0,
    });
    svg.appendChild(crosshair);
    var overlay = svgElement('rect', {
      x: pad.l,
      y: pad.t,
      width: plotWidth,
      height: plotHeight,
      fill: 'transparent',
      style: 'cursor:crosshair',
    });
    svg.appendChild(overlay);

    overlay.addEventListener('mousemove', function (event) {
      var rect = svg.getBoundingClientRect();
      var relativeX = (event.clientX - rect.left) / rect.width * width;
      var divisor = plotWidth / (count > 1 ? count - 1 : 1);
      var displayIndex = Math.max(0, Math.min(count - 1, Math.round((relativeX - pad.l) / divisor)));
      var sourceIndex = visibleIndexes[displayIndex];
      crosshair.setAttribute('x1', xAt(displayIndex));
      crosshair.setAttribute('x2', xAt(displayIndex));
      crosshair.setAttribute('opacity', 1);
      var rows = series.map(function (speaker) {
        var point = speaker.values[sourceIndex];
        var content = 'No data';
        if (point && point.excluded) content = 'Excluded · ' + escapeHtml(point.reason.toLowerCase());
        if (point && !point.excluded) content = '<b>' + formatValue(point.v, decimals) + '</b> · ' + point.words.toLocaleString('en-US') + ' words';
        return '<div class="pg-tt-row"><span class="pg-tt-dot" style="background:' + speaker.color + '"></span><span>'
          + escapeHtml(speaker.name) + ': ' + content + '</span></div>';
      }).join('');
      tooltip.innerHTML = '<b>' + escapeHtml(shortDate(dates[sourceIndex])) + '</b>' + rows;
      tooltip.style.opacity = 1;
      tooltip.style.left = Math.min(event.clientX + 14, window.innerWidth - 250) + 'px';
      tooltip.style.top = Math.min(event.clientY + 14, window.innerHeight - 120) + 'px';
    });
    overlay.addEventListener('mouseleave', function () {
      crosshair.setAttribute('opacity', 0);
      tooltip.style.opacity = 0;
    });
    return svg;
  }

  // Speed-vs-accuracy trajectory. Unlike makeChart (time on X), this plots one
  // point per recorded, annotated call in tempo (X) x error-density (Y) space
  // and joins them in time. It answers Andrey's question — does accuracy hold,
  // improve or slip as speaking speed changes — without collapsing the two into
  // one opaque score (the same "named lines, not composites" rule as the rest
  // of Progress). Recorded-only: text uploads carry no tempo, so a point needs
  // both a speech rate and a trustworthy error density (enough words, annotated).
  function trajectorySeries() {
    var sessions = state.history.sessions;
    return state.speakers.map(function (name) {
      var points = [];
      sessions.forEach(function (session) {
        var participant = (session.participants || []).find(function (item) { return item.name === name; });
        if (!participant || !participant.derived) return;
        var metrics = participant.derived.metrics || {};
        var grammar = participant.derived.grammar || {};
        var tempo = Number(metrics.speech_rate_wpm);
        var errors = Number(grammar.error_density_per_100w);
        if (!Number.isFinite(tempo) || tempo <= 0 || !Number.isFinite(errors)) return;
        var words = Number(metrics.english_word_count || 0);
        if (exclusionReason(session, words, true)) return;
        points.push({ x: tempo, y: errors, date: session.date, words: words });
      });
      return { name: name, color: speakerColor(name), points: points };
    });
  }

  function attachScatterHover(node, speaker, point) {
    node.style.cursor = 'pointer';
    node.addEventListener('mousemove', function (event) {
      tooltip.innerHTML = '<b>' + escapeHtml(shortDate(point.date)) + '</b>'
        + '<div class="pg-tt-row"><span class="pg-tt-dot" style="background:' + speaker.color + '"></span><span>'
        + escapeHtml(speaker.name) + ': <b>' + Math.round(point.x) + '</b> wpm · <b>' + formatValue(point.y, 1) + '</b> err/100w</span></div>';
      tooltip.style.opacity = 1;
      tooltip.style.left = Math.min(event.clientX + 14, window.innerWidth - 250) + 'px';
      tooltip.style.top = Math.min(event.clientY + 14, window.innerHeight - 120) + 'px';
    });
    node.addEventListener('mouseleave', function () { tooltip.style.opacity = 0; });
  }

  function makeScatter(series, options) {
    options = options || {};
    var width = 720;
    var height = 300;
    var pad = { t: 18, r: 86, b: 48, l: 56 };
    var plotWidth = width - pad.l - pad.r;
    var plotHeight = height - pad.t - pad.b;

    var all = [];
    series.forEach(function (speaker) { speaker.points.forEach(function (point) { all.push(point); }); });
    var xs = all.map(function (point) { return point.x; });
    var ys = all.map(function (point) { return point.y; });
    var xLo = Math.min.apply(null, xs);
    var xHi = Math.max.apply(null, xs);
    var xSpan = Math.max(20, xHi - xLo);
    var xMin = Math.max(0, Math.floor((xLo - xSpan * 0.18) / 10) * 10);
    var xMax = Math.ceil((xHi + xSpan * 0.18) / 10) * 10;
    var yMax = niceMax(Math.max.apply(null, ys) * 1.12);

    function xAt(value) { return pad.l + (xMax > xMin ? (value - xMin) / (xMax - xMin) : 0.5) * plotWidth; }
    function yAt(value) { return pad.t + (1 - value / yMax) * plotHeight; }

    var svg = svgElement('svg', {
      viewBox: '0 0 ' + width + ' ' + height,
      role: 'img',
      'aria-label': options.ariaLabel || 'Speaking speed versus accuracy trajectory',
    });

    // Horizontal gridlines carry the error-density scale (Y).
    [0, yMax / 2, yMax].forEach(function (tick) {
      var y = yAt(tick);
      svg.appendChild(svgElement('line', { x1: pad.l, y1: y, x2: width - pad.r, y2: y, stroke: '#e8eef5', 'stroke-width': 1 }));
      var label = svgElement('text', { x: pad.l - 8, y: y + 4, 'text-anchor': 'end', 'font-size': 11, fill: '#8492aa' });
      label.textContent = formatValue(tick, 1);
      svg.appendChild(label);
    });
    // Vertical ticks carry the tempo scale (X).
    var xTickCount = 4;
    for (var i = 0; i <= xTickCount; i++) {
      var xValue = xMin + (xMax - xMin) * (i / xTickCount);
      var x = xAt(xValue);
      svg.appendChild(svgElement('line', { x1: x, y1: pad.t, x2: x, y2: height - pad.b, stroke: '#f1f5f9', 'stroke-width': 1 }));
      var xLabel = svgElement('text', { x: x, y: height - pad.b + 18, 'text-anchor': 'middle', 'font-size': 11, fill: '#8492aa' });
      xLabel.textContent = String(Math.round(xValue));
      svg.appendChild(xLabel);
    }

    var xTitle = svgElement('text', { x: pad.l + plotWidth / 2, y: height - 6, 'text-anchor': 'middle', 'font-size': 11, 'font-weight': 700, fill: '#64748b' });
    xTitle.textContent = 'words / minute  (faster →)';
    svg.appendChild(xTitle);
    var yMid = pad.t + plotHeight / 2;
    var yTitle = svgElement('text', { x: 15, y: yMid, 'text-anchor': 'middle', 'font-size': 11, 'font-weight': 700, fill: '#64748b', transform: 'rotate(-90 15 ' + yMid + ')' });
    yTitle.textContent = 'errors / 100w  (fewer ↓)';
    svg.appendChild(yTitle);

    series.forEach(function (speaker) {
      if (!speaker.points.length) return;
      if (speaker.points.length > 1) {
        var d = speaker.points.map(function (point, index) {
          return (index ? 'L' : 'M') + xAt(point.x) + ' ' + yAt(point.y);
        }).join(' ');
        svg.appendChild(svgElement('path', {
          d: d, fill: 'none', stroke: speaker.color, 'stroke-width': 2.5,
          'stroke-linejoin': 'round', 'stroke-linecap': 'round', opacity: 0.45,
        }));
      }
      speaker.points.forEach(function (point, index) {
        var isLast = index === speaker.points.length - 1;
        var isFirst = index === 0 && !isLast;
        var circle = svgElement('circle', {
          cx: xAt(point.x), cy: yAt(point.y),
          r: isLast ? 6 : 4.5,
          fill: isFirst ? '#ffffff' : speaker.color,
          stroke: isFirst ? speaker.color : '#ffffff',
          'stroke-width': isLast || isFirst ? 2 : 1.5,
        });
        attachScatterHover(circle, speaker, point);
        svg.appendChild(circle);
      });
      var latest = speaker.points[speaker.points.length - 1];
      var nameLabel = svgElement('text', { x: xAt(latest.x) + 10, y: yAt(latest.y) + 4, 'font-size': 12, 'font-weight': 700, fill: speaker.color });
      nameLabel.textContent = speaker.name;
      svg.appendChild(nameLabel);
    });
    return svg;
  }

  function metricHelp(text) {
    var details = document.createElement('details');
    details.className = 'pg-info';
    var summary = document.createElement('summary');
    summary.setAttribute('aria-label', 'How this metric works');
    summary.textContent = '?';
    details.appendChild(summary);
    var body = document.createElement('div');
    body.textContent = text;
    details.appendChild(body);
    return details;
  }

  function card(config) {
    var container = document.createElement('article');
    container.className = 'pg-card' + (config.hero ? ' pg-hero' : '');
    var header = document.createElement('div');
    header.className = 'pg-card-head';
    var title = document.createElement('h3');
    title.textContent = config.title;
    header.appendChild(title);
    if (config.help) header.appendChild(metricHelp(config.help));
    container.appendChild(header);
    if (config.description) {
      var description = document.createElement('p');
      description.className = 'pg-hint';
      description.textContent = config.description;
      container.appendChild(description);
    }
    container.appendChild(config.chart);
    return container;
  }

  function section(title, subtitle) {
    var container = document.createElement('section');
    container.className = 'pg-section';
    var heading = document.createElement('h2');
    heading.textContent = title;
    container.appendChild(heading);
    if (subtitle) {
      var description = document.createElement('p');
      description.className = 'pg-sub';
      description.textContent = subtitle;
      container.appendChild(description);
    }
    return container;
  }

  function overallSeries() {
    return buildSeries(function (derived) {
      return (derived.grammar || {}).error_density_per_100w;
    }, { requiresAnnotations: true });
  }

  function categorySeries(code) {
    return buildSeries(function (derived) {
      return ((derived.grammar || {}).by_category_density || {})[code];
    }, { requiresAnnotations: true });
  }

  function trendSummary(series) {
    var grid = document.createElement('div');
    grid.className = 'pg-summary-grid';
    series.forEach(function (speaker) {
      var comparable = speaker.values.map(function (point, index) {
        return point && !point.excluded ? { point: point, index: index } : null;
      }).filter(Boolean);
      var item = document.createElement('article');
      item.className = 'pg-summary-card';
      var name = document.createElement('div');
      name.className = 'pg-summary-name';
      name.innerHTML = '<span class="pg-swatch" style="background:' + speaker.color + '"></span>' + escapeHtml(speaker.name);
      item.appendChild(name);
      if (!comparable.length) {
        item.innerHTML += '<strong>Not enough comparable sessions</strong><span>A completed annotated call with ' + LOW_SAMPLE + '+ English words is needed.</span>';
        grid.appendChild(item);
        return;
      }
      var last = comparable[comparable.length - 1];
      var session = state.history.sessions[last.index] || {};
      var participant = (session.participants || []).find(function (entry) { return entry.name === speaker.name; }) || {};
      var canonical = participant.comparison && participant.comparison.version === 1
        ? participant.comparison
        : null;
      var comparison = canonical && canonical.overall ? canonical.overall : {};
      var referenceDates = canonical ? canonical.reference_dates || [] : [];
      if (!canonical) {
        var fallbackReferences = comparable.slice(Math.max(0, comparable.length - 4), -1);
        if (fallbackReferences.length) {
          var referenceAverage = fallbackReferences.reduce(function (sum, entry) { return sum + entry.point.v; }, 0) / fallbackReferences.length;
          var delta = last.point.v - referenceAverage;
          var threshold = Math.max(0.15, Math.abs(referenceAverage) * 0.1);
          comparison = {
            reference_average: referenceAverage,
            delta: delta,
            status: delta <= -threshold ? 'improving' : delta >= threshold ? 'needs_attention' : 'steady',
          };
          referenceDates = fallbackReferences.map(function (entry) { return state.history.sessions[entry.index].date; });
        } else {
          comparison = { status: 'no_baseline' };
        }
      }
      var statusMap = {
        improving: ['Improving', 'is-improving'],
        needs_attention: ['Needs attention', 'is-attention'],
        steady: ['Steady', 'is-steady'],
        no_baseline: ['Baseline not ready', 'is-steady'],
      };
      var status = statusMap[comparison.status] || ['Not scored', 'is-steady'];
      var values = document.createElement('div');
      values.className = 'pg-summary-values';
      values.innerHTML = '<strong>' + formatValue(last.point.v, 1) + ' <small>/100w</small></strong>'
        + '<span class="pg-trend-status ' + status[1] + '">' + status[0] + '</span>';
      item.appendChild(values);
      var detail = document.createElement('span');
      detail.className = 'pg-summary-detail';
      if (typeof comparison.reference_average === 'number') {
        var deltaValue = Number(comparison.delta || 0);
        var relation = comparison.status === 'steady'
          ? 'about the same as'
          : Math.abs(deltaValue).toFixed(1) + (deltaValue < 0 ? ' below' : ' above');
        detail.textContent = shortDate(session.date) + ' · ' + relation + ' the average of previous ' + referenceDates.length + ' (' + formatValue(comparison.reference_average, 1) + ')';
      } else {
        detail.textContent = shortDate(session.date) + ' · first comparable result';
      }
      item.appendChild(detail);
      var context = document.createElement('span');
      context.textContent = comparable.length + ' comparable calls since ' + shortDate(state.history.sessions[comparable[0].index].date);
      item.appendChild(context);
      if (referenceDates.length) {
        var reference = document.createElement('span');
        reference.className = 'pg-summary-reference';
        reference.textContent = 'Reference: ' + referenceDates.map(shortDate).join(', ');
        item.appendChild(reference);
      }
      grid.appendChild(item);
    });
    return grid;
  }

  function exclusionNotice(series, dates) {
    var rows = [];
    dates.forEach(function (date, index) {
      var excluded = series.map(function (speaker) {
        var point = speaker.values[index];
        return point && point.excluded ? speaker.name + ' — ' + point.reason.toLowerCase() : '';
      }).filter(Boolean);
      if (excluded.length) rows.push('<b>' + escapeHtml(shortDate(date)) + '</b>: ' + excluded.map(escapeHtml).join('; '));
    });
    if (!rows.length) return null;
    var note = document.createElement('aside');
    note.className = 'pg-exclusion';
    note.innerHTML = '<strong>Not included in the trend</strong><span>' + rows.join('<br>') + '.</span>'
      + '<small>Minimum: ' + LOW_SAMPLE + ' English words and completed annotations. The session remains available on the Session page.</small>';
    return note;
  }

  function selectedFocusCodes() {
    var active = (state.focus.focuses || []).filter(function (focus) { return focus.status === 'active'; });
    var activeCodes = [];
    active.forEach(function (focus) {
      var code = String(focus.category_code || '').toUpperCase();
      if (code && activeCodes.indexOf(code) < 0) activeCodes.push(code);
    });
    if (activeCodes.length) return { codes: activeCodes, fallback: false };

    var ranked = CATEGORIES.map(function (category) {
      var values = validPoints(categorySeries(category.code));
      var average = values.length ? values.reduce(function (sum, point) { return sum + point.v; }, 0) / values.length : 0;
      return { code: category.code, average: average };
    }).filter(function (item) { return item.average > 0; });
    ranked.sort(function (left, right) { return right.average - left.average; });
    return { codes: ranked.slice(0, 2).map(function (item) { return item.code; }), fallback: true };
  }

  function categoryCard(code, dates) {
    var category = CATEGORIES.find(function (item) { return item.code === code; });
    var series = categorySeries(code);
    return card({
      title: category ? category.label : code,
      description: 'Errors per 100 English words. Lower is better.',
      help: 'Counts only annotations assigned to this grammar category. Sessions with fewer than ' + LOW_SAMPLE + ' English words or incomplete annotations do not affect the line.',
      chart: makeChart(series, dates, { compact: true, decimals: 1, ariaLabel: (category ? category.label : code) + ' trend' }),
    });
  }

  function buildClosedFocuses() {
    var closed = (state.focus.focuses || []).filter(function (focus) { return focus.status === 'closed'; });
    if (!closed.length) return null;
    var categoryLabels = {};
    CATEGORIES.forEach(function (category) { categoryLabels[category.code] = category.label; });
    var completed = section('Closed focuses', 'A compact record of patterns that were deliberately practised and closed.');
    var list = document.createElement('div');
    list.className = 'pg-victories';
    closed.sort(function (left, right) { return String(right.closed_date || '').localeCompare(String(left.closed_date || '')); });
    closed.forEach(function (focus) {
      var row = document.createElement('div');
      row.className = 'pg-victory';
      row.innerHTML = '<span class="pg-victory-mark">✓</span><b>' + escapeHtml(categoryLabels[focus.category_code] || focus.category_code) + '</b>'
        + ' — ' + escapeHtml(focus.participant)
        + ' <span class="pg-victory-dates">set ' + escapeHtml(shortDate(focus.set_date)) + ' → closed ' + escapeHtml(shortDate(focus.closed_date)) + '</span>';
      list.appendChild(row);
    });
    completed.appendChild(list);
    return completed;
  }

  function buildTable(sessions) {
    var box = document.createElement('details');
    box.className = 'pg-table-box';
    var summary = document.createElement('summary');
    summary.textContent = 'Raw data table';
    box.appendChild(summary);
    var wrap = document.createElement('div');
    wrap.className = 'pg-table-wrap';
    var rows = [];
    rows.push('<tr><th>Speaker / date</th>' + sessions.map(function (session) { return '<th>' + escapeHtml(shortDate(session.date)) + '</th>'; }).join('') + '</tr>');
    var metrics = [
      ['English words', function (derived) { return (derived.metrics || {}).english_word_count; }, 0],
      ['Grammar errors /100w', function (derived) { return (derived.grammar || {}).error_density_per_100w; }, 1],
      ['Fillers /100w', function (derived) { return (derived.metrics || {}).filler_per_100w; }, 1],
      ['Words / turn', function (derived) { return (derived.metrics || {}).avg_words_per_turn; }, 1],
      ['MATTR', function (derived) { return (derived.metrics || {}).lexical_diversity_mattr; }, 2],
    ];
    state.speakers.forEach(function (name) {
      rows.push('<tr><td colspan="' + (sessions.length + 1) + '" class="pg-table-speaker">' + escapeHtml(name) + '</td></tr>');
      metrics.forEach(function (metric) {
        var cells = sessions.map(function (session) {
          var participant = (session.participants || []).find(function (item) { return item.name === name; });
          var derived = participant && participant.derived;
          var rawValue = derived ? metric[1](derived) : null;
          var value = rawValue === null || rawValue === undefined || rawValue === '' ? null : Number(rawValue);
          var words = derived ? Number((derived.metrics || {}).english_word_count || 0) : 0;
          var excludedClass = words && words < LOW_SAMPLE ? ' class="is-excluded" title="Excluded from trends: short session"' : '';
          return '<td' + excludedClass + '>' + (Number.isFinite(value) ? formatValue(value, metric[2]) : '—') + '</td>';
        }).join('');
        rows.push('<tr><td>' + escapeHtml(metric[0]) + '</td>' + cells + '</tr>');
      });
    });
    wrap.innerHTML = '<table class="pg-table"><thead>' + rows[0] + '</thead><tbody>' + rows.slice(1).join('') + '</tbody></table>';
    box.appendChild(wrap);
    return box;
  }

  function versionText(sessions) {
    var last = sessions[sessions.length - 1] || {};
    var version = last.analysis_version || {};
    if (!version.metrics) return 'Version information unavailable';
    return 'metrics v' + version.metrics + ' · taxonomy v' + version.taxonomy + (version.annotation_model ? ' · ' + version.annotation_model : '');
  }

  function buildDiagnostics(sessions) {
    var details = document.createElement('details');
    details.className = 'pg-diagnostics';
    var summary = document.createElement('summary');
    summary.textContent = 'Data & diagnostics';
    details.appendChild(summary);
    var body = document.createElement('div');
    body.className = 'pg-diagnostics-body';
    var copy = document.createElement('div');
    copy.innerHTML = '<strong>Analysis version</strong><span>' + escapeHtml(versionText(sessions)) + '</span>';
    body.appendChild(copy);
    var button = document.createElement('button');
    button.className = 'ghost';
    button.id = 'pg-reanalyze';
    button.type = 'button';
    button.textContent = 'Re-analyze all';
    body.appendChild(button);
    details.appendChild(body);
    details.appendChild(buildTable(sessions));
    return details;
  }

  function render() {
    var root = document.getElementById('pg-root');
    root.innerHTML = '';
    var sessions = (state.history && state.history.sessions) || [];
    if (!sessions.length) {
      root.innerHTML = '<p class="pg-empty">No sessions yet. Record or upload one to see progress.</p>';
      return;
    }
    var dates = sessions.map(function (session) { return session.date; });
    var legend = document.getElementById('pg-legend');
    legend.innerHTML = state.speakers.map(function (name) {
      return '<span class="pg-leg-item"><span class="pg-swatch" style="background:' + speakerColor(name) + '"></span>' + escapeHtml(name) + '</span>';
    }).join('');

    var grammarSeries = overallSeries();
    var overview = section('Grammar trend', 'Latest comparable result against that person\'s previous three comparable calls. Lower is better; the chart keeps the full history.');
    overview.appendChild(trendSummary(grammarSeries));
    var overviewGrid = document.createElement('div');
    overviewGrid.className = 'pg-grid';
    overviewGrid.appendChild(card({
      title: 'Overall error density',
      description: 'Grammar annotations per 100 English words. Lower is better.',
      help: 'This rate makes sessions of different lengths comparable. Only sessions with at least ' + LOW_SAMPLE + ' English words and completed annotations affect the trend.',
      chart: makeChart(grammarSeries, dates, { decimals: 1, ariaLabel: 'Overall grammar error density trend' }),
      hero: true,
    }));
    var seriousSeries = buildSeries(function (derived) {
      return (derived.grammar || {}).serious_error_density_per_100w;
    }, { requiresAnnotations: true });
    if (hasMeasuredValues(seriousSeries)) {
      overviewGrid.appendChild(card({
        title: 'Serious errors only',
        description: 'Findings a listener clearly notices (noticeable) or that put the meaning at risk (blocking), per 100 English words.',
        help: 'Severity is judged per finding by impact on the listener, independently of the category and of how frequent it is. Minor slips are excluded here, and so are findings stored before severity existed. Lower is better.',
        chart: makeChart(seriousSeries, dates, { decimals: 1, ariaLabel: 'Serious error density trend' }),
        hero: true,
      }));
      // Two related hero charts read better side by side — and take a fraction
      // of the height of one full-width chart stacked on another.
      overviewGrid.className = 'pg-grid pg-grid-two';
    }
    overview.appendChild(overviewGrid);
    var excluded = exclusionNotice(grammarSeries, dates);
    if (excluded) overview.appendChild(excluded);
    root.appendChild(overview);

    // Current focus and Speaking habits each hold one compact chart, so they
    // pair into a single row instead of two half-empty full-width sections.
    var duo = document.createElement('div');
    duo.className = 'pg-duo';

    var focusSelection = selectedFocusCodes();
    if (focusSelection.codes.length) {
      var focusSubtitle = focusSelection.fallback
        ? 'No active focus is set, so these are the two most persistent grammar categories across comparable sessions.'
        : 'Only categories currently marked as active focuses are shown here.';
      var focusSection = section('Current focus', focusSubtitle);
      var focusGrid = document.createElement('div');
      focusGrid.className = 'pg-grid';
      focusSelection.codes.forEach(function (code) { focusGrid.appendChild(categoryCard(code, dates)); });
      focusSection.appendChild(focusGrid);
      duo.appendChild(focusSection);
    }

    var habits = section('Speaking habits', 'A transcript-based signal that is easy to interpret and useful to track.');
    var habitsGrid = document.createElement('div');
    habitsGrid.className = 'pg-grid';
    var fillerSeries = buildSeries(function (derived) { return (derived.metrics || {}).filler_per_100w; });
    habitsGrid.appendChild(card({
      title: 'Fillers / 100 words',
      description: 'Recognised fillers per 100 English words. Lower is usually better.',
      help: 'Counts um, uh, er, erm, hmm, like, you know, I mean, kind of and sort of. Very short sessions are excluded from the line.',
      chart: makeChart(fillerSeries, dates, { compact: true, decimals: 1, ariaLabel: 'Filler words trend' }),
    }));
    habits.appendChild(habitsGrid);
    duo.appendChild(habits);
    root.appendChild(duo);

    // Timing-based fluency (ADR-0006) exists only for recorded calls, so the
    // whole section stays hidden until at least one recorded session has a
    // measurable value — text uploads never produce these keys.
    var fluencyDefs = [
      {
        key: 'speech_rate_wpm',
        label: 'Speech rate',
        description: 'Words per minute of your own speaking time in the recording.',
        help: 'Measured from word timestamps in the recorded audio. Includes hesitation pauses inside your speech; higher usually reads as more fluent, but very fast speech is not a goal in itself.',
        decimals: 0,
      },
      {
        key: 'pauses_per_min',
        label: 'Hesitation pauses / min',
        description: 'Silences of 0.5s or longer inside your own utterances, per speaking minute.',
        help: 'Pauses between speakers are not counted — only hesitations inside your own speech. Lower usually means smoother delivery.',
        decimals: 1,
      },
    ];
    var fluencySeriesByKey = fluencyDefs.map(function (metric) {
      return {
        metric: metric,
        series: buildSeries(function (derived) { return (derived.metrics || {})[metric.key]; }),
      };
    }).filter(function (entry) { return hasMeasuredValues(entry.series); });
    if (fluencySeriesByKey.length) {
      var fluencySection = section('Fluency', 'Timing-based signals from recorded calls: how fast and how smoothly speech flows. Text uploads carry no timings and are not shown.');
      var fluencyGrid = document.createElement('div');
      fluencyGrid.className = 'pg-grid pg-grid-two';
      fluencySeriesByKey.forEach(function (entry) {
        fluencyGrid.appendChild(card({
          title: entry.metric.label,
          description: entry.metric.description,
          help: entry.metric.help + ' Sessions below ' + LOW_SAMPLE + ' English words are excluded from the line.',
          chart: makeChart(entry.series, dates, { compact: true, decimals: entry.metric.decimals, ariaLabel: entry.metric.label + ' trend' }),
        }));
      });
      fluencySection.appendChild(fluencyGrid);
      root.appendChild(fluencySection);
    }

    // Speed vs accuracy: the recorded-call tempo paired with grammar error
    // density as a trajectory. Shown from the first qualifying recorded call so
    // progress is visible as it builds — one point is a valid "where you are",
    // a second joins into a trajectory. Text uploads never qualify (no tempo).
    var trajectory = trajectorySeries();
    var maxTrajectoryPoints = trajectory.reduce(function (max, speaker) {
      return Math.max(max, speaker.points.length);
    }, 0);
    if (maxTrajectoryPoints >= 1) {
      var trajectorySubtitle = maxTrajectoryPoints >= 2
        ? 'Each point is one recorded call, joined in time. Right is faster speech; down is fewer errors — so speeding up while holding level or dropping is progress, and the two do not have to move together. Hollow dot is the earliest call, solid is the latest.'
        : 'Your first recorded call is plotted here — one more comparable recorded call joins into a trajectory. Right is faster speech; down is fewer errors.';
      var trajectorySection = section('Speed vs accuracy', trajectorySubtitle);
      var trajectoryGrid = document.createElement('div');
      trajectoryGrid.className = 'pg-grid';
      trajectoryGrid.appendChild(card({
        title: 'Trajectory',
        description: 'Speaking tempo (words per minute) against grammar errors per 100 English words, per recorded call.',
        help: 'Both axes come only from recorded calls with completed annotations and at least ' + LOW_SAMPLE + ' English words; text uploads carry no tempo. It shows how speed and accuracy move together over time — not that one causes the other, and a hard topic can raise both at once.',
        chart: makeScatter(trajectory, { ariaLabel: 'Speaking speed versus grammar accuracy trajectory' }),
        hero: true,
      }));
      trajectorySection.appendChild(trajectoryGrid);
      root.appendChild(trajectorySection);
    }

    var closedSection = buildClosedFocuses();
    if (closedSection) root.appendChild(closedSection);

    var more = document.createElement('details');
    more.className = 'pg-more';
    var moreSummary = document.createElement('summary');
    moreSummary.innerHTML = '<span><strong>More metrics</strong><small>Other grammar categories, turn length and vocabulary variety</small></span>';
    more.appendChild(moreSummary);
    var moreBody = document.createElement('div');
    moreBody.className = 'pg-more-body';

    var otherCategories = CATEGORIES.filter(function (category) {
      return focusSelection.codes.indexOf(category.code) < 0 && hasMeasuredValues(categorySeries(category.code));
    });
    if (otherCategories.length) {
      var otherSection = section('Other grammar categories', 'Available for diagnosis, but kept out of the main view to reduce noise.');
      var otherGrid = document.createElement('div');
      otherGrid.className = 'pg-grid pg-grid-two';
      otherCategories.forEach(function (category) { otherGrid.appendChild(categoryCard(category.code, dates)); });
      otherSection.appendChild(otherGrid);
      moreBody.appendChild(otherSection);
    }

    var secondarySection = section('Context metrics', 'Useful supporting context, not scores and not direct targets.');
    var secondaryGrid = document.createElement('div');
    secondaryGrid.className = 'pg-grid pg-grid-two';
    SECONDARY_METRICS.forEach(function (metric) {
      var series = buildSeries(function (derived) { return (derived.metrics || {})[metric.key]; });
      secondaryGrid.appendChild(card({
        title: metric.label,
        description: metric.description,
        help: metric.help + ' Sessions below ' + LOW_SAMPLE + ' English words are excluded from the line.',
        chart: makeChart(series, dates, { compact: true, decimals: metric.decimals, ariaLabel: metric.label + ' trend' }),
      }));
    });
    secondarySection.appendChild(secondaryGrid);
    moreBody.appendChild(secondarySection);
    more.appendChild(moreBody);
    root.appendChild(more);

    root.appendChild(buildDiagnostics(sessions));
    attachReanalyze();
  }

  function collectSpeakers(sessions) {
    var seen = [];
    sessions.forEach(function (session) {
      (session.participants || []).forEach(function (participant) {
        if (participant.name && seen.indexOf(participant.name) < 0) seen.push(participant.name);
      });
    });
    return seen;
  }

  async function load() {
    state.history = await window.ET.cachedJson('/history.json');
    state.history.sessions = (state.history.sessions || []).slice().sort(function (left, right) {
      return (left.date || '').localeCompare(right.date || '');
    });
    state.speakers = collectSpeakers(state.history.sessions);
    try {
      var focusResponse = await fetch(apiUrl('/api/focus'), { cache: 'no-store' });
      state.focus = focusResponse.ok ? await focusResponse.json() : { focuses: [] };
    } catch (error) {
      state.focus = { focuses: [] };
    }
  }

  function attachReanalyze() {
    var button = document.getElementById('pg-reanalyze');
    if (!button) return;
    button.addEventListener('click', async function () {
      button.disabled = true;
      button.textContent = 'Re-analyzing…';
      try {
        var response = await fetch(apiUrl('/api/reanalyze'), { method: 'POST' });
        if (!response.ok) throw new Error(await response.text());
        window.ET.bustCache();
        await load();
        render();
      } catch (error) {
        button.textContent = 'Failed — try again';
        button.disabled = false;
        console.error(error);
      }
    });
  }

  (async function () {
    try {
      await load();
      render();
    } catch (error) {
      document.getElementById('pg-root').innerHTML = '<p class="pg-empty">Could not load progress data: '
        + escapeHtml(error && error.message ? error.message : error) + '</p>';
    }
  })();
})();
