(function () {
  'use strict';

  var root = document.getElementById('briefing-root');

  function escapeHtml(value) {
    return String(value === null || value === undefined ? '' : value)
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

  function normalizePerson(person) {
    if (person.focus !== undefined || person.patterns || person.recent_direction) return person;
    var focus = (person.active_focuses || [])[0] || null;
    if (!focus && (person.top_categories || []).length) {
      focus = Object.assign({ kind: 'suggested' }, person.top_categories[0]);
    }
    if (focus && !focus.kind) focus.kind = 'active';
    var errorTrend = (person.trends || []).find(function (item) { return item.key === 'error_density'; });
    return {
      name: person.name,
      focus: focus,
      additional_focus_count: Math.max(0, (person.active_focuses || []).length - 1),
      patterns: (person.top_categories || []).slice(0, 2),
      examples: (person.examples || []).slice(0, 2),
      grossest: [],
      recent_direction: errorTrend ? {
        points: errorTrend.trend.values || [],
        comparison: { status: errorTrend.trend.direction === 'worsening' ? 'needs_attention' : errorTrend.trend.direction },
        reference_dates: [],
      } : null,
    };
  }

  function participantCell(person, content) {
    return '<article class="briefing-cell"><h3>' + escapeHtml(person.name) + '</h3>' + content + '</article>';
  }

  function band(people, title, subtitle, className, renderCell) {
    return '<section class="briefing-band ' + escapeHtml(className || '') + '">'
      + '<div class="briefing-band-heading"><h2>' + escapeHtml(title) + '</h2><p>' + escapeHtml(subtitle) + '</p></div>'
      + '<div class="briefing-band-grid">'
      + people.map(function (person) { return participantCell(person, renderCell(person)); }).join('')
      + '</div></section>';
  }

  function focusCell(person) {
    var focus = person.focus;
    if (!focus) return '<p class="briefing-empty">One more comparable annotated call is needed to suggest a focus.</p>';
    if (focus.kind === 'active') {
      var progress = typeof focus.latest_density === 'number'
        ? density(focus.baseline_density) + ' → ' + density(focus.latest_density) + '/100w'
        : 'Waiting for the next comparable call';
      var change = typeof focus.change_pct === 'number'
        ? ' · ' + (focus.change_pct > 0 ? '+' : '') + focus.change_pct + '%'
        : '';
      return '<div class="briefing-focus-title"><strong>' + escapeHtml(focus.title) + '</strong>'
        + '<span class="briefing-pill is-active">Active focus</span>'
        + (focus.ready_to_close ? '<span class="briefing-pill is-positive">Ready to close</span>' : '')
        + '</div><p class="briefing-focus-detail">Set ' + escapeHtml(shortDate(focus.set_date)) + ' · ' + escapeHtml(progress + change) + '</p>'
        + (person.additional_focus_count ? '<p class="briefing-muted">+' + person.additional_focus_count + ' more active focus</p>' : '');
    }
    return '<div class="briefing-focus-title"><strong>' + escapeHtml(focus.title) + '</strong>'
      + '<span class="briefing-pill is-suggested">Suggested</span></div>'
      + '<p class="briefing-focus-detail">Highest-density pattern across the latest comparable calls: '
      + density(focus.average_density) + '/100w.</p>';
  }

  function patternsCell(person) {
    var patterns = person.patterns || [];
    if (!patterns.length) return '<p class="briefing-empty">No comparable annotated calls yet.</p>';
    var dates = patterns[0].dates || [];
    return '<p class="briefing-reference">Latest ' + dates.length + ' comparable calls'
      + (dates.length ? ': ' + dates.map(shortDate).join(', ') : '') + '</p>'
      + patterns.map(function (pattern) {
        return '<div class="briefing-pattern"><div><strong>' + escapeHtml(pattern.title) + '</strong>'
          + '<span>Seen in ' + pattern.seen_sessions + ' of ' + pattern.sessions_considered + '</span></div>'
          + '<b>' + density(pattern.average_density) + '<small>/100w avg</small></b></div>';
      }).join('');
  }

  function examplesCell(person) {
    var examples = person.examples || [];
    if (!examples.length) return '<p class="briefing-empty">Examples appear after annotations finish.</p>';
    return examples.map(function (example) {
      return '<article class="briefing-rewrite"><p class="briefing-example-meta">'
        + escapeHtml(example.category_title) + ' · ' + escapeHtml(shortDate(example.date)) + '</p>'
        + '<div class="briefing-wrong"><span>You said</span><strong>' + escapeHtml(example.error) + '</strong></div>'
        + '<div class="briefing-fix"><span>Try</span><strong>' + escapeHtml(example.correction) + '</strong></div></article>';
    }).join('');
  }

  function grossestCell(person) {
    var items = person.grossest || [];
    if (!items.length) return '<p class="briefing-empty">No serious findings in the latest comparable call.</p>';
    return items.map(function (item) {
      return '<article class="briefing-rewrite"><p class="briefing-example-meta">'
        + '<span class="severity-chip is-' + escapeHtml(String(item.severity || '').toLowerCase()) + '">' + escapeHtml(item.severity || '') + '</span> '
        + escapeHtml(item.category_title || '') + ' · ' + escapeHtml(shortDate(item.date)) + '</p>'
        + '<div class="briefing-wrong"><span>You said</span><strong>' + escapeHtml(item.error) + '</strong></div>'
        + '<div class="briefing-fix"><span>Try</span><strong>' + escapeHtml(item.correction) + '</strong></div></article>';
    }).join('');
  }

  function directionCell(person) {
    var direction = person.recent_direction;
    if (!direction || !(direction.points || []).length) return '<p class="briefing-empty">No comparable grammar results yet.</p>';
    var comparison = direction.comparison || {};
    var statusMap = {
      improving: ['Improving', 'is-positive'],
      needs_attention: ['Needs attention', 'is-warning'],
      steady: ['Steady', 'is-neutral'],
      no_baseline: ['Baseline not ready', 'is-neutral'],
    };
    var status = statusMap[comparison.status] || ['Not scored', 'is-neutral'];
    var points = direction.points || [];
    var latest = points[points.length - 1];
    var detail = 'First comparable result.';
    if (typeof comparison.reference_average === 'number') {
      var delta = Number(comparison.delta || 0);
      detail = comparison.status === 'steady'
        ? 'About the same as the average of the previous ' + (direction.reference_dates || []).length + ' calls.'
        : density(Math.abs(delta)) + '/100w ' + (delta < 0 ? 'below' : 'above') + ' the previous-call average of ' + density(comparison.reference_average) + '.';
    }
    return '<div class="briefing-direction-head"><strong>' + density(latest.value) + '/100w</strong>'
      + '<span class="briefing-pill ' + status[1] + '">' + status[0] + '</span></div>'
      + '<div class="briefing-points">' + points.map(function (point) {
        return '<span><small>' + escapeHtml(shortDate(point.date)) + '</small><b>' + density(point.value) + '</b></span>';
      }).join('') + '</div><p class="briefing-focus-detail">' + escapeHtml(detail) + '</p>';
  }

  function render(briefing) {
    var people = (briefing.participants || []).map(normalizePerson);
    if (!people.length) {
      root.innerHTML = '<p class="metric-note">No sessions yet. Record or upload a call to create the first briefing.</p>';
      return;
    }
    var anyGrossest = people.some(function (person) { return (person.grossest || []).length; });
    root.innerHTML = band(people, 'Focus for the next call', 'One primary target per person. Active choices stay separate from automatic suggestions.', 'is-focus', focusCell)
      + band(people, 'Patterns from recent calls', 'The two densest patterns across the latest three comparable annotated calls.', 'is-patterns', patternsCell)
      + band(people, 'Rehearse these', 'Two corrections chosen from the focus and recent patterns.', 'is-examples', examplesCell)
      // Guaranteed severity slot (ADR-0007): rendered only when someone has
      // severity-rated findings, so older briefings look exactly as before.
      + (anyGrossest ? band(people, 'Most serious last call', 'The highest-severity findings of the latest comparable call — independent of how frequent the category is and of the focus.', 'is-grossest', grossestCell) : '')
      + band(people, 'Recent grammar direction', 'The latest three actual results; the status compares the newest call with the three calls before it.', 'is-direction', directionCell)
      + '<a class="briefing-session-link" href="highlights.html">Open Session evidence →</a>';
  }

  async function load() {
    try {
      var response = await fetch(window.ET.apiUrl('/briefing.json'), { cache: 'no-store' });
      if (!response.ok) throw new Error('Unable to load the briefing.');
      render(await response.json());
    } catch (error) {
      root.innerHTML = '<p class="metric-note">' + escapeHtml(error.message || 'Unable to load the briefing.') + '</p>';
    }
  }

  load();
})();
