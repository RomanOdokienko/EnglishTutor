const sessionSelect = document.getElementById('highlight-session-select');
const highlightsRoot = document.getElementById('highlights-root');

const state = {
  history: null,
  analysisCache: new Map(),
};

async function loadHistory() {
  const response = await fetch('history.json');
  if (!response.ok) {
    throw new Error('Unable to load history.json');
  }
  return response.json();
}

async function loadAnalysis(date) {
  if (state.analysisCache.has(date)) {
    return state.analysisCache.get(date);
  }
  const response = await fetch(`sessions/${date}/analysis.json`);
  if (!response.ok) {
    throw new Error(`Unable to load analysis for ${date}`);
  }
  const data = await response.json();
  state.analysisCache.set(date, data);
  return data;
}

function formatSessionDate(value) {
  const raw = String(value || '').trim();
  if (!raw) {
    return '';
  }
  const parsed = new Date(`${raw}T00:00:00`);
  if (Number.isNaN(parsed.getTime())) {
    return raw;
  }
  return parsed.toLocaleDateString('en-GB', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
  });
}

function renderDropdown() {
  sessionSelect.innerHTML = '';
  state.history.sessions.forEach((session) => {
    const option = document.createElement('option');
    option.value = session.date;
    option.textContent = formatSessionDate(session.date);
    sessionSelect.appendChild(option);
  });
}

function sortParticipants(participants) {
  const list = Array.isArray(participants) ? [...participants] : [];
  return list.sort((a, b) => {
    const aName = (a?.name || '').toLowerCase();
    const bName = (b?.name || '').toLowerCase();
    if (aName === 'roman' && bName !== 'roman') return -1;
    if (bName === 'roman' && aName !== 'roman') return 1;
    return aName.localeCompare(bName);
  });
}

function escapeHtml(value) {
  if (!value) {
    return '';
  }
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function getTopInsights(participant) {
  const insights = participant.llm?.top3_insights || [];
  if (insights.length) {
    return insights.slice(0, 3);
  }
  const practical = participant.llm?.practical_recommendations || [];
  return practical.slice(0, 3).map((item) => ({
    title: item.title,
    why: item.why || '',
    focus: item.guidance || '',
    examples: item.examples || [],
  }));
}

function buildLead(participant, top3) {
  const totalErrors = participant.llm?.annotation_grammar?.total_errors;
  const labels = top3.map((item) => item.title).filter(Boolean);
  if (typeof totalErrors === 'number' && labels.length) {
    return `${participant.name}: ${totalErrors} total grammar issues. Top recurring patterns are ${labels.join(', ')}.`;
  }
  if (typeof totalErrors === 'number') {
    return `${participant.name}: ${totalErrors} total grammar issues in this session.`;
  }
  return 'Focused patterns to work on before the next session.';
}

function renderHighlights(analysis) {
  highlightsRoot.innerHTML = '';
  const participants = sortParticipants(analysis.participants || []);
  participants.forEach((participant) => {
    const card = document.createElement('div');
    card.className = 'highlight-card';

    const top3 = getTopInsights(participant);
    const items = top3.map((rec, index) => {
      const examples = (rec.examples || [])
        .map(
          (example) =>
            `<li><span class="example-error">${escapeHtml(example.error)}</span><span class="example-arrow">&rarr;</span><span class="example-fix">${escapeHtml(example.correction)}</span></li>`
        )
        .join('');
      const examplesBlock = examples
        ? `<ul class="errors compact">${examples}</ul>`
        : '<p class="metric-note">No examples captured yet.</p>';
      const why = rec.why || 'Recurring pattern in this session.';
      const focus = rec.focus ? `<p class="highlight-focus">${escapeHtml(rec.focus)}</p>` : '';
      return `
        <div class="highlight-item">
          <div class="highlight-rank">${index + 1}</div>
          <div class="highlight-body">
            <h3>${escapeHtml(rec.title)}</h3>
            <p class="highlight-why">${escapeHtml(why)}</p>
            ${focus}
            ${examplesBlock}
          </div>
        </div>
      `;
    }).join('');
    const lead = buildLead(participant, top3);

    const summary = participant.llm?.annotation_grammar?.total_errors
      ? `${participant.llm.annotation_grammar.total_errors} total errors.`
      : 'Key focuses for next session.';

    card.innerHTML = `
      <div class="highlight-header">
        <h2>${escapeHtml(participant.name)}</h2>
        <span class="highlight-meta">${escapeHtml(summary)}</span>
      </div>
      <p class="highlight-lead">${escapeHtml(lead)}</p>
      ${items || '<p class="metric-note">No highlights yet. Run annotations to generate focus areas.</p>'}
    `;
    highlightsRoot.appendChild(card);
  });
}

async function handleSelection() {
  const date = sessionSelect.value;
  const analysis = await loadAnalysis(date);
  renderHighlights(analysis);
}

async function init() {
  try {
    state.history = await loadHistory();
    renderDropdown();
    if (state.history.sessions.length) {
      sessionSelect.value = state.history.sessions[state.history.sessions.length - 1].date;
    }
    sessionSelect.addEventListener('change', handleSelection);
    await handleSelection();
  } catch (error) {
    highlightsRoot.innerHTML = `<p>${error.message}</p>`;
  }
}

init();
