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

  // ---- Session-scoped JSON cache -----------------------------------------
  // The static artifacts (history.json, analysis.json, briefing.json) change
  // only when a session is recorded / rebuilt / deleted, yet this multi-page
  // app re-fetches them on every tab switch. Cache them in sessionStorage so
  // switching tabs within a browsing session is instant. Any mutation must
  // call bustCache(); a short TTL bounds staleness if a path is ever missed.
  // Reads stay on the backend (never Vercel static) so a freshly recorded
  // session shows up immediately — only the redundant re-fetch is removed.
  var CACHE_PREFIX = 'et:cache:';
  var CACHE_TTL_MS = 5 * 60 * 1000;
  function cacheKey(url) { return CACHE_PREFIX + url; }
  function readCache(url) {
    try {
      var raw = window.sessionStorage.getItem(cacheKey(url));
      if (!raw) return null;
      var entry = JSON.parse(raw);
      if (!entry || (Date.now() - entry.t) > CACHE_TTL_MS) return null;
      return entry.body; // raw response text, re-parsed per call for a fresh object
    } catch (e) { return null; }
  }
  function writeCache(url, text) {
    try {
      window.sessionStorage.setItem(cacheKey(url), JSON.stringify({ t: Date.now(), body: text }));
    } catch (e) { /* quota / disabled — skip caching, never block the load */ }
  }
  // Cached GET for a static JSON artifact. opts.force bypasses the read (but
  // still refreshes the cache) — use it for an explicit "refresh" action.
  async function cachedJson(path, opts) {
    var url = apiUrl(path);
    if (!(opts && opts.force)) {
      var hit = readCache(url);
      if (hit != null) { try { return JSON.parse(hit); } catch (e) {} }
    }
    var res = await fetch(url, { cache: 'no-store' });
    if (!res.ok) throw new Error('Unable to load ' + path);
    var text = await res.text();
    writeCache(url, text);
    return JSON.parse(text);
  }
  // Drop one cached path, or (no arg) every cached artifact. Call after any
  // write so the next read re-fetches fresh data.
  function bustCache(path) {
    try {
      if (path) { window.sessionStorage.removeItem(cacheKey(apiUrl(path))); return; }
      var keys = [];
      for (var i = 0; i < window.sessionStorage.length; i++) {
        var k = window.sessionStorage.key(i);
        if (k && k.indexOf(CACHE_PREFIX) === 0) keys.push(k);
      }
      keys.forEach(function (k) { window.sessionStorage.removeItem(k); });
    } catch (e) { /* storage disabled — nothing to bust */ }
  }

  // ---- Data loaders ----
  async function loadHistory() {
    var data = await cachedJson('/history.json');
    data.sessions = (data.sessions || []).slice().sort(function (a, b) {
      return String(a.date || '').localeCompare(String(b.date || ''));
    });
    return data;
  }
  async function loadAnalysis(date) {
    return cachedJson('/sessions/' + date + '/analysis.json');
  }

  // ---- Formatting ----
  var MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  function shortDate(d) {
    var m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(d || '');
    return m ? parseInt(m[3], 10) + ' ' + MONTHS[parseInt(m[2], 10) - 1] : String(d || '');
  }

  // ---- Navigation (the ONE place nav is defined) ----
  var NAV = [
    { key: 'home',     label: 'This week', href: 'home.html' },
    { key: 'session',  label: 'Session',  href: 'highlights.html' },
    { key: 'progress', label: 'Progress', href: 'progress.html' },
    { key: 'record',   label: 'Record',   href: 'record.html', accent: true },
  ];
  function renderNav(activeKey) {
    var mount = document.getElementById('app-nav');
    if (!mount) return;
    mount.className = 'app-nav';
    if (!mount.children.length) {
      mount.innerHTML = NAV.map(function (item) {
        var cls = 'app-nav-link' + (item.accent ? ' is-accent' : '');
        return '<a class="' + cls + '" data-nav-key="' + item.key + '" href="' + item.href + '">' + item.label + '</a>';
      }).join('');
    }
    mount.querySelectorAll('.app-nav-link').forEach(function (link) {
      link.classList.toggle('is-active', link.getAttribute('data-nav-key') === activeKey);
    });
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
    cachedJson: cachedJson,
    bustCache: bustCache,
    loadHistory: loadHistory,
    loadAnalysis: loadAnalysis,
    shortDate: shortDate,
    renderNav: renderNav,
    NAV: NAV,
  };
})(window, document);
