(function () {
  'use strict';

  var root = document.getElementById('briefing-root');
  var directionLabels = { improving: 'improving', worsening: 'needs attention', steady: 'steady' };

  function escapeHtml(value) {
    return String(value || '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function shortDate(value) {
    return window.ET.shortDate(value || '');
  }

  function density(value) {
    return typeof value === 'number' ? value.toFixed(2).replace(/\.00$/, '') : '—';
  }

  function trendCard(item) {
    var trend = item.trend;
    if (!trend) return '';
    var arrow = trend.direction === 'improving' ? '↗' : trend.direction === 'worsening' ? '↘' : '→';
    var values = (trend.values || []).map(function (point) { return density(point.value); }).join(' → ');
    return '<div class="briefing-trend briefing-trend-' + escapeHtml(trend.direction) + '">'
      + '<div><span class="briefing-trend-arrow">' + arrow + '</span><b>' + escapeHtml(item.label) + '</b></div>'
      + '<span>' + values + escapeHtml(item.unit || '') + ' · ' + escapeHtml(directionLabels[trend.direction] || 'steady') + '</span>'
      + '</div>';
  }

  function focusCard(focus) {
    var progress;
    if (typeof focus.latest_density !== 'number') {
      progress = 'Baseline ' + density(focus.baseline_density) + '/100w · waiting for a comparable annotated call';
    } else {
      var change = focus.change_pct > 0 ? '+' + focus.change_pct : String(focus.change_pct);
      progress = density(focus.baseline_density) + ' → ' + density(focus.latest_density) + '/100w (' + change + '%)';
    }
    return '<article class="briefing-focus">'
      + '<div class="briefing-focus-head"><b>' + escapeHtml(focus.title) + '</b>'
      + (focus.ready_to_close ? '<span class="briefing-ready">ready to close</span>' : '') + '</div>'
      + '<p>Set ' + escapeHtml(shortDate(focus.set_date)) + ' · ' + escapeHtml(progress) + '</p>'
      + '</article>';
  }

  function categoryRow(category) {
    var recurrence = category.seen_sessions + ' of ' + category.sessions_considered + ' recent comparable calls';
    return '<div class="briefing-category">'
      + '<div><b>' + escapeHtml(category.title) + '</b><span>' + escapeHtml(recurrence) + '</span></div>'
      + '<strong>' + density(category.average_density) + '<small>/100w</small></strong>'
      + '</div>';
  }

  function exampleRow(example) {
    return '<div class="briefing-example">'
      + '<span class="briefing-example-category">' + escapeHtml(example.category_title) + ' · ' + escapeHtml(shortDate(example.date)) + '</span>'
      + '<p><q>' + escapeHtml(example.error) + '</q><span>→</span><b>' + escapeHtml(example.correction) + '</b></p>'
      + '</div>';
  }

  function participantCard(person) {
    var focuses = person.active_focuses && person.active_focuses.length
      ? person.active_focuses.map(focusCard).join('')
      : '<p class="briefing-empty">No active focus yet. Choose one from the Error evidence cards after a call.</p>';
    var categories = person.top_categories && person.top_categories.length
      ? person.top_categories.map(categoryRow).join('')
      : '<p class="briefing-empty">No comparable annotated sessions yet.</p>';
    var examples = person.examples && person.examples.length
      ? person.examples.map(exampleRow).join('')
      : '<p class="briefing-empty">Examples will appear after annotations finish.</p>';
    var trends = person.trends && person.trends.length
      ? person.trends.map(trendCard).join('')
      : '<p class="briefing-empty">No trend yet.</p>';
    return '<article class="briefing-person">'
      + '<h2>' + escapeHtml(person.name) + '</h2>'
      + '<section><h3>Active focuses</h3>' + focuses + '</section>'
      + '<section><h3>Patterns to watch</h3>' + categories + '</section>'
      + '<section><h3>Useful examples</h3>' + examples + '</section>'
      + '<section><h3>Trend snapshot</h3>' + trends + '</section>'
      + '<a class="briefing-session-link" href="highlights.html">Open Session evidence →</a>'
      + '</article>';
  }

  async function load() {
    try {
      var response = await fetch(window.ET.apiUrl('/briefing.json'), { cache: 'no-store' });
      if (!response.ok) throw new Error('Unable to load the briefing.');
      var briefing = await response.json();
      var participants = briefing.participants || [];
      root.innerHTML = participants.length
        ? participants.map(participantCard).join('')
        : '<p class="metric-note">No sessions yet. Record or upload a call to create the first briefing.</p>';
    } catch (error) {
      root.innerHTML = '<p class="metric-note">' + escapeHtml(error.message || 'Unable to load the briefing.') + '</p>';
    }
  }

  load();
})();
