/* Shared app shell + utilities.
 * Single source of truth for navigation and API access, included by every page.
 * A page opts in with:  <body data-nav="progress">  and  <nav id="app-nav"></nav>
 */
(function (window, document) {
  'use strict';

  // ---- API base resolution (query param > inline global > localStorage) ----
  function normalizeApiBase(raw) {
    var v = String(raw || '').trim();
    return v ? v.replace(/\/+$/, '') : '';
  }
  function getConfiguredApiBase() {
    var params = new URLSearchParams(window.location.search);
    var q = normalizeApiBase(params.get('api_base'));
    if (q) {
      try { window.localStorage.setItem('ENGLISH_TUTOR_API_BASE_URL', q); } catch (e) {}
      return q;
    }
    var inline = normalizeApiBase(window.ENGLISH_TUTOR_API_BASE_URL);
    if (inline) return inline;
    try { return normalizeApiBase(window.localStorage.getItem('ENGLISH_TUTOR_API_BASE_URL')); }
    catch (e) { return ''; }
  }
  function apiUrl(path) {
    var raw = String(path || '');
    if (!raw || /^https?:\/\//i.test(raw)) return raw;
    var base = getConfiguredApiBase();
    if (!base) return raw;
    return base + (raw.startsWith('/') ? raw : '/' + raw);
  }

  // ---- Data loaders ----
  async function loadHistory() {
    var res = await fetch(apiUrl('/history.json'), { cache: 'no-store' });
    if (!res.ok) throw new Error('Unable to load history.json');
    var data = await res.json();
    data.sessions = (data.sessions || []).slice().sort(function (a, b) {
      return String(a.date || '').localeCompare(String(b.date || ''));
    });
    return data;
  }
  async function loadAnalysis(date) {
    var res = await fetch(apiUrl('/sessions/' + date + '/analysis.json'), { cache: 'no-store' });
    if (!res.ok) throw new Error('Unable to load analysis for ' + date);
    return res.json();
  }

  // ---- Formatting ----
  var MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  function shortDate(d) {
    var m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(d || '');
    return m ? parseInt(m[3], 10) + ' ' + MONTHS[parseInt(m[2], 10) - 1] : String(d || '');
  }

  // ---- Navigation (the ONE place nav is defined) ----
  var NAV = [
    { key: 'session',  label: 'Session',  href: 'highlights.html' },
    { key: 'progress', label: 'Progress', href: 'progress.html' },
    { key: 'method',   label: 'Как это работает', href: 'method.html' },
    { key: 'record',   label: 'Record',   href: 'record.html', accent: true },
  ];
  function renderNav(activeKey) {
    var mount = document.getElementById('app-nav');
    if (!mount) return;
    mount.className = 'app-nav';
    mount.innerHTML = NAV.map(function (item, i) {
      var cls = 'app-nav-link'
        + (item.key === activeKey ? ' is-active' : '')
        + (item.accent ? ' is-accent' : '');
      var sep = item.accent && i > 0 ? '<span class="app-nav-spacer"></span>' : '';
      return sep + '<a class="' + cls + '" href="' + item.href + '">' + item.label + '</a>';
    }).join('');
  }

  function initNav() {
    var active = document.body ? document.body.getAttribute('data-nav') : '';
    renderNav(active || '');
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initNav);
  } else {
    initNav();
  }

  window.ET = {
    apiUrl: apiUrl,
    getConfiguredApiBase: getConfiguredApiBase,
    loadHistory: loadHistory,
    loadAnalysis: loadAnalysis,
    shortDate: shortDate,
    renderNav: renderNav,
    NAV: NAV,
  };
})(window, document);
