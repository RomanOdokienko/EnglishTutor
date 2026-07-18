(function () {
  'use strict';

  var root = document.getElementById('briefing-root');
  var viewerMount = document.getElementById('home-viewer');

  // Shared with the Session page so "who am I looking at" carries across pages.
  var PARTICIPANT_PREFERENCE_KEY = 'ENGLISH_TUTOR_SESSION_PARTICIPANT';
  var state = { people: [], selected: null };

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

  function readPref() {
    try { return String(window.localStorage.getItem(PARTICIPANT_PREFERENCE_KEY) || '').trim(); }
    catch (e) { return ''; }
  }
  function writePref(value) {
    try { window.localStorage.setItem(PARTICIPANT_PREFERENCE_KEY, value); } catch (e) {}
  }

  // Backward compatibility: fold a pre-v2 briefing shape into the current one.
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

  var STATUS_MAP = {
    improving: ['Improving', 'is-positive'],
    needs_attention: ['Needs attention', 'is-warning'],
    steady: ['Steady', 'is-neutral'],
    no_baseline: ['Baseline not ready', 'is-neutral'],
  };

  // ---- hero pieces ----
  function focusPills(focus) {
    if (focus.kind === 'active') {
      return '<span class="briefing-pill is-active">Active focus</span>'
        + (focus.ready_to_close ? '<span class="briefing-pill is-positive">Ready to close</span>' : '');
    }
    return '<span class="briefing-pill is-suggested">Suggested</span>';
  }

  function focusMeasurement(focus) {
    if (focus.kind === 'active') {
      var progress = typeof focus.latest_density === 'number'
        ? density(focus.baseline_density) + ' → ' + density(focus.latest_density) + '<small>/100w</small>'
        : 'Waiting for the next comparable call';
      var change = '';
      if (typeof focus.change_pct === 'number') {
        var dir = focus.change_pct < 0 ? 'is-down' : (focus.change_pct > 0 ? 'is-up' : 'is-flat');
        change = '<span class="tw-change ' + dir + '">' + (focus.change_pct > 0 ? '+' : '') + focus.change_pct + '%</span>';
      }
      return '<div class="tw-meas"><b>' + progress + '</b>' + change + '</div>'
        + '<p class="tw-note">Set ' + escapeHtml(shortDate(focus.set_date)) + '</p>';
    }
    return '<div class="tw-meas"><b>' + density(focus.average_density) + '<small>/100w avg</small></b></div>'
      + '<p class="tw-note">Highest-density pattern across the latest comparable calls.</p>';
  }

  function rehearseList(person) {
    var examples = (person.examples || []).slice(0, 3);
    if (!examples.length) return '<p class="briefing-empty">Examples appear after annotations finish.</p>';
    return examples.map(function (ex) {
      return '<article class="briefing-rewrite"><p class="briefing-example-meta">'
        + escapeHtml(ex.category_title) + ' · ' + escapeHtml(shortDate(ex.date)) + '</p>'
        + '<div class="briefing-wrong"><span>You said</span><strong>' + escapeHtml(ex.error) + '</strong></div>'
        + '<div class="briefing-fix"><span>Try</span><strong>' + escapeHtml(ex.correction) + '</strong></div></article>';
    }).join('');
  }

  // ---- evidence pieces ----
  function patternsBlock(person) {
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

  function grossestBlock(person) {
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

  function sparkline(direction) {
    var points = (direction && direction.points) || [];
    if (points.length < 2) return '';
    var max = points.reduce(function (m, p) { return Math.max(m, typeof p.value === 'number' ? p.value : 0); }, 0) || 1;
    return '<span class="tw-spark" aria-hidden="true">' + points.map(function (p, i) {
      var h = Math.max(12, Math.round(((typeof p.value === 'number' ? p.value : 0) / max) * 100));
      return '<i class="' + (i === points.length - 1 ? 'is-last' : '') + '" style="height:' + h + '%"></i>';
    }).join('') + '</span>';
  }

  function directionBlock(person) {
    var direction = person.recent_direction;
    if (!direction || !(direction.points || []).length) return '<p class="briefing-empty">No comparable grammar results yet.</p>';
    var comparison = direction.comparison || {};
    var status = STATUS_MAP[comparison.status] || ['Not scored', 'is-neutral'];
    var points = direction.points || [];
    var latest = points[points.length - 1];
    var detail = 'First comparable result.';
    if (typeof comparison.reference_average === 'number') {
      var delta = Number(comparison.delta || 0);
      detail = comparison.status === 'steady'
        ? 'About the same as the average of the previous ' + (direction.reference_dates || []).length + ' calls.'
        : density(Math.abs(delta)) + '/100w ' + (delta < 0 ? 'below' : 'above') + ' the previous-call average of ' + density(comparison.reference_average) + '.';
    }
    return '<div class="briefing-direction-head"><strong>' + density(latest.value) + '<small>/100w</small></strong>'
      + '<span class="briefing-pill ' + status[1] + '">' + status[0] + '</span></div>'
      + sparkline(direction)
      + '<div class="briefing-points">' + points.map(function (point) {
        return '<span><small>' + escapeHtml(shortDate(point.date)) + '</small><b>' + density(point.value) + '</b></span>';
      }).join('') + '</div><p class="briefing-focus-detail">' + escapeHtml(detail) + '</p>';
  }

  // ---- assembled views ----
  function personHero(person, opts) {
    opts = opts || {};
    var focus = person.focus;
    var head;
    if (focus) {
      head = '<div class="tw-focus-head"><span class="tw-kicker">Focus for the next call</span>'
        + '<span class="tw-pills">' + focusPills(focus) + '</span></div>'
        + '<h2 class="tw-focus-title">' + escapeHtml(focus.title) + '</h2>'
        + focusMeasurement(focus)
        + (person.additional_focus_count ? '<p class="tw-note">+' + person.additional_focus_count + ' more active focus</p>' : '')
        + '<h3 class="tw-sub">Rehearse these</h3>' + rehearseList(person);
    } else {
      head = '<div class="tw-focus-head"><span class="tw-kicker">Focus for the next call</span></div>'
        + '<p class="briefing-empty">One more comparable annotated call is needed to suggest a focus.</p>';
    }

    // The set/close machinery lives on Session; the briefing routes you there.
    // Viewer identity is shared, so Session opens already scoped to this person.
    var actions = '';
    if (focus) {
      actions = '<div class="tw-actions"><a class="tw-cta is-primary" href="highlights.html">Practise this in Session →</a>'
        + (focus.kind === 'active' && focus.ready_to_close ? '<a class="tw-cta is-ghost" href="highlights.html">Close it on Session ✓</a>' : '')
        + (focus.kind === 'suggested' ? '<a class="tw-cta is-ghost" href="highlights.html">Make it a focus →</a>' : '')
        + '</div>';
    }

    var grossest = (person.grossest && person.grossest.length)
      ? '<div class="tw-ev-block"><h4>Most serious last call</h4>' + grossestBlock(person) + '</div>' : '';
    var evidence = '<details class="tw-evidence"' + (opts.openEvidence ? ' open' : '') + '>'
      + '<summary>Why this — patterns, most serious &amp; direction</summary>'
      + '<div class="tw-evidence-body">'
      + '<div class="tw-ev-block"><h4>Patterns from recent calls</h4>' + patternsBlock(person) + '</div>'
      + grossest
      + '<div class="tw-ev-block"><h4>Recent grammar direction</h4>' + directionBlock(person) + '</div>'
      + '</div></details>';

    return '<section class="tw-person' + (opts.primary ? ' is-primary' : '') + '">'
      + '<div class="tw-person-name">' + escapeHtml(person.name) + '</div>'
      + '<div class="tw-hero">' + head + actions + '</div>'
      + evidence + '</section>';
  }

  function partnerCard(person) {
    var focus = person.focus;
    var direction = person.recent_direction;
    var latest = direction && (direction.points || []).length
      ? density(direction.points[direction.points.length - 1].value) + '/100w' : '';
    var statusText = '';
    if (direction && direction.comparison && STATUS_MAP[direction.comparison.status]) {
      statusText = ' · ' + STATUS_MAP[direction.comparison.status][0];
    }
    return '<button class="tw-partner" type="button" data-view="' + escapeHtml(person.name) + '">'
      + '<span class="tw-partner-top">' + escapeHtml(person.name) + ' this week</span>'
      + (focus
        ? '<strong>' + escapeHtml(focus.title) + '</strong><span class="tw-partner-tag">'
          + (focus.kind === 'active' ? 'Active focus' : 'Suggested') + '</span>'
        : '<span class="briefing-empty">No focus yet</span>')
      + (latest ? '<span class="tw-partner-dir">' + latest + statusText + '</span>' : '')
      + '<span class="tw-partner-cta">View →</span></button>';
  }

  function render() {
    var people = state.people;
    if (!people.length) {
      root.innerHTML = '<p class="metric-note">No sessions yet. Record or upload a call to create the first briefing.</p>';
      return;
    }
    if (state.selected === 'both') {
      root.innerHTML = '<div class="tw-layout is-both">'
        + people.map(function (p) { return personHero(p, {}); }).join('') + '</div>';
      return;
    }
    var viewer = people.filter(function (p) { return p.name === state.selected; })[0] || people[0];
    var partners = people.filter(function (p) { return p.name !== viewer.name; });
    root.innerHTML = '<div class="tw-layout">'
      + personHero(viewer, { primary: true, openEvidence: false })
      + (partners.length ? '<div class="tw-partners">' + partners.map(partnerCard).join('') + '</div>' : '')
      + '</div>';
  }

  function renderViewer() {
    if (!viewerMount) return;
    var options = state.people.map(function (p) { return p.name; }).concat('both');
    viewerMount.innerHTML = options.map(function (value) {
      var label = value === 'both' ? 'Both' : value;
      var on = state.selected === value;
      return '<button class="home-viewer-btn' + (on ? ' is-active' : '') + '" type="button" '
        + 'data-view="' + escapeHtml(value) + '" aria-pressed="' + on + '">' + escapeHtml(label) + '</button>';
    }).join('');
  }

  function setSelected(value) {
    if (state.selected === value) return;
    state.selected = value;
    writePref(value);
    renderViewer();
    render();
  }

  function ensureSelected() {
    var names = state.people.map(function (p) { return p.name; });
    var saved = readPref();
    state.selected = (saved === 'both' || names.indexOf(saved) !== -1) ? saved : (names[0] || 'both');
  }

  // One delegated handler covers the viewer segments and the partner cards.
  document.addEventListener('click', function (event) {
    var trigger = event.target.closest('[data-view]');
    if (trigger) setSelected(trigger.getAttribute('data-view'));
  });

  async function load() {
    try {
      var response = await fetch(window.ET.apiUrl('/briefing.json'), { cache: 'no-store' });
      if (!response.ok) throw new Error('Unable to load the briefing.');
      var briefing = await response.json();
      state.people = (briefing.participants || []).map(normalizePerson);
      if (!state.people.length) {
        if (viewerMount) viewerMount.innerHTML = '';
        root.innerHTML = '<p class="metric-note">No sessions yet. Record or upload a call to create the first briefing.</p>';
        return;
      }
      ensureSelected();
      renderViewer();
      render();
    } catch (error) {
      root.innerHTML = '<p class="metric-note">' + escapeHtml(error.message || 'Unable to load the briefing.') + '</p>';
    }
  }

  load();
})();
