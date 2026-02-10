const sessionSelect = document.getElementById('session-select');
const sessionTitle = document.getElementById('session-title');
const sessionDetails = document.getElementById('session-details');
const recommendationsEl = document.getElementById('recommendations');
const transcriptEl = document.getElementById('transcript-text');
const proficiencyChart = document.getElementById('progress-proficiency');
const errorsChart = document.getElementById('progress-errors');
const rebuildButton = document.getElementById('rebuild-button');
const deleteButton = document.getElementById('delete-button');
const rebuildStatus = document.getElementById('rebuild-status');
const llmStatus = document.getElementById('llm-status');

const state = {
  history: null,
  analysisCache: new Map(),
};

async function loadHistory() {
  const response = await fetch('../history.json');
  if (!response.ok) {
    throw new Error('Unable to load history.json');
  }
  return response.json();
}

async function loadAnalysis(date) {
  if (state.analysisCache.has(date)) {
    return state.analysisCache.get(date);
  }
  const response = await fetch(`../sessions/${date}/analysis.json`);
  if (!response.ok) {
    throw new Error(`Unable to load analysis for ${date}`);
  }
  const data = await response.json();
  state.analysisCache.set(date, data);
  return data;
}

function renderDropdown() {
  sessionSelect.innerHTML = '';
  state.history.sessions.forEach((session) => {
    const option = document.createElement('option');
    option.value = session.date;
    option.textContent = `${session.date} · ${session.topic}`;
    sessionSelect.appendChild(option);
  });
}

function renderParticipants(analysis) {
  sessionTitle.textContent = `${analysis.date} · ${analysis.session.topic}`;
  sessionDetails.innerHTML = '';
  analysis.participants.forEach((participant) => {
    const card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = `
      <h3>${participant.name} <span>${participant.role}</span></h3>
      <p class="metric"><strong>Words:</strong> ${participant.metrics.word_count}<span class="metric-note">Count of A-Z words in this speaker's turns.</span></p>
      <p class="metric"><strong>Turns:</strong> ${participant.metrics.turn_count}<span class="metric-note">Number of blocks starting with "${participant.name}:".</span></p>
      <p class="metric"><strong>Avg words/turn:</strong> ${participant.metrics.avg_words_per_turn}<span class="metric-note">Words divided by turns.</span></p>
      <p class="metric"><strong>Lexical diversity:</strong> ${participant.metrics.lexical_diversity}<span class="metric-note">Unique words / total words.</span></p>
      ${participant.llm ? `<p class="metric"><strong>LLM Proficiency:</strong> ${participant.llm.fluency.score} (${participant.llm.fluency.level})<span class="metric-note">Model-based assessment.</span></p>` : ''}
      ${participant.llm?.grammar?.error_rate_per_100_words !== undefined ? `<p class="metric"><strong>Grammar error rate:</strong> ${participant.llm.grammar.error_rate_per_100_words} / 100 words<span class="metric-note">From LLM-detected grammar errors.</span></p>` : ''}
    `;
    sessionDetails.appendChild(card);
  });
}

function renderRecommendations(analysis) {
  recommendationsEl.innerHTML = '';
  analysis.participants.forEach((participant) => {
    const wrapper = document.createElement('div');
    wrapper.className = 'card';

    const topErrors = (participant.llm?.grammar?.top_errors || [])
      .slice()
      .sort((a, b) => (b.count || 0) - (a.count || 0))
      .slice(0, 3);

    const recommendations = topErrors.map((error, index) => {
      const examples = (error.examples || []).slice(0, 2).map((ex) => {
        const value = typeof ex === 'string' ? { text: ex, correction: '' } : ex || {};
        const text = value.text || '';
        const correction = value.correction || '';
        const correctionLine = correction
          ? `<div class="correction">Correct: ${correction}</div>`
          : `<div class="correction missing">Correct: (not provided)</div>`;
        return `<div class="example"><div>${text}</div>${correctionLine}</div>`;
      }).join('');

      return `
        <li>
          <strong>${index + 1}. ${error.title}</strong> <span class="metric-note">(${error.count || 0} times)</span>
          <div class="metric-note">Focus on this in the next lesson.</div>
          ${examples}
        </li>
      `;
    }).join('');

    const fallbackItems = participant.llm?.grammar?.top_recommendations
      ?.slice(0, 3)
      .map((item, index) => `<li><strong>${index + 1}.</strong> ${item}</li>`)
      .join('');

    const llmBlock = recommendations || fallbackItems
      ? `
        <h4 class="subheading">LLM: top-3 focus areas for next lesson</h4>
        ${participant.llm?.grammar?.error_count !== undefined ? `<p class="metric-note">Total grammar errors: ${participant.llm.grammar.error_count}</p>` : ''}
        <ul class="errors">${recommendations || fallbackItems}</ul>
      `
      : '<p class="metric-note">No recommendations yet.</p>';

    wrapper.innerHTML = `
      <h3>${participant.name}</h3>
      ${llmBlock}
    `;
    recommendationsEl.appendChild(wrapper);
  });
}

function renderTranscript(analysis) {
  if (!transcriptEl) {
    return;
  }
  transcriptEl.textContent = analysis.transcript || '';
}

function renderLlmStatus(analysis) {
  if (!llmStatus) {
    return;
  }
  const info = analysis.llm || { status: 'skipped' };
  if (info.status === 'ok') {
    llmStatus.textContent = `OpenAI: success (${info.model || 'default model'})`;
  } else if (info.status === 'error') {
    llmStatus.textContent = `OpenAI: error (${info.model || 'default model'})`;
  } else {
    llmStatus.textContent = 'OpenAI: skipped (no token)';
  }
}


function drawLineChart(canvas, series, labels, options) {
  if (!canvas) {
    return;
  }
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#f8fafc';
  ctx.fillRect(0, 0, width, height);

  const padding = { top: 28, right: 20, bottom: 28, left: 36 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;

  const allValues = series.flatMap((item) => item.values.filter((v) => v !== null));
  const minValue = options.min !== undefined ? options.min : 0;
  const maxValue = allValues.length
    ? Math.max(...allValues, options.maxFallback || 1)
    : options.maxFallback || 1;

  const count = labels.length;
  const stepX = count > 1 ? plotWidth / (count - 1) : 0;

  function xAt(index) {
    return padding.left + (count > 1 ? index * stepX : plotWidth / 2);
  }

  function yAt(value) {
    const normalized = (value - minValue) / (maxValue - minValue || 1);
    return padding.top + (1 - normalized) * plotHeight;
  }

  // Grid
  ctx.strokeStyle = '#e2e8f0';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top);
  ctx.lineTo(padding.left, height - padding.bottom);
  ctx.lineTo(width - padding.right, height - padding.bottom);
  ctx.stroke();

  const ticks = options.ticks || [minValue, (minValue + maxValue) / 2, maxValue];
  ctx.fillStyle = '#64748b';
  ctx.font = '11px sans-serif';
  ticks.forEach((tick) => {
    const y = yAt(tick);
    ctx.fillText(tick.toFixed(options.tickDecimals || 0), 6, y + 4);
    ctx.strokeStyle = '#f1f5f9';
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(width - padding.right, y);
    ctx.stroke();
  });

  // Series
  series.forEach((item) => {
    ctx.strokeStyle = item.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    item.values.forEach((value, index) => {
      if (value === null || value === undefined) {
        return;
      }
      const x = xAt(index);
      const y = yAt(value);
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();

    item.values.forEach((value, index) => {
      if (value === null || value === undefined) {
        return;
      }
      const x = xAt(index);
      const y = yAt(value);
      ctx.fillStyle = item.color;
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
    });
  });

  // Legend
  let legendX = padding.left;
  series.forEach((item) => {
    ctx.fillStyle = item.color;
    ctx.fillRect(legendX, padding.top - 18, 10, 10);
    ctx.fillStyle = '#334155';
    ctx.font = '12px sans-serif';
    ctx.fillText(item.label, legendX + 16, padding.top - 9);
    legendX += 120;
  });

  // X labels
  ctx.fillStyle = '#64748b';
  labels.forEach((label, index) => {
    const x = xAt(index);
    ctx.fillText(label, x - 18, height - 8);
  });
}

function renderChart() {
  const sessions = state.history.sessions;
  if (!sessions.length) {
    return;
  }

  const dates = sessions.map((session) => session.date);
  const speakers = new Set();
  sessions.forEach((session) => {
    session.participants.forEach((participant) => speakers.add(participant.name));
  });
  const speakerList = Array.from(speakers);
  const colors = ['#0ea5e9', '#f97316', '#10b981', '#a855f7'];

  const proficiencySeries = speakerList.map((name, index) => {
    const values = sessions.map((session) => {
      const participant = session.participants.find((p) => p.name === name);
      if (!participant) {
        return null;
      }
      return participant.llm_fluency_score !== null && participant.llm_fluency_score !== undefined
        ? participant.llm_fluency_score
        : participant.fluency_score;
    });
    return {
      label: name,
      color: colors[index % colors.length],
      values,
    };
  });

  const errorSeries = speakerList.map((name, index) => {
    const values = sessions.map((session) => {
      const participant = session.participants.find((p) => p.name === name);
      if (!participant) {
        return null;
      }
      return participant.llm_error_rate ?? null;
    });
    return {
      label: name,
      color: colors[index % colors.length],
      values,
    };
  });

  drawLineChart(proficiencyChart, proficiencySeries, dates, {
    min: 0,
    maxFallback: 10,
    ticks: [0, 5, 10],
    tickDecimals: 0,
  });

  drawLineChart(errorsChart, errorSeries, dates, {
    min: 0,
    maxFallback: 5,
    tickDecimals: 1,
  });
}


async function handleSelection() {
  const date = sessionSelect.value;
  const analysis = await loadAnalysis(date);
  renderParticipants(analysis);
  renderRecommendations(analysis);
  renderTranscript(analysis);
  renderLlmStatus(analysis);
}

function attachRebuild() {
  if (!rebuildButton) {
    return;
  }
  rebuildButton.addEventListener('click', async () => {
    rebuildButton.disabled = true;
    rebuildStatus.textContent = 'Rebuilding...';
    try {
      const date = sessionSelect.value;
      const response = await fetch('/api/rebuild', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ date }),
      });
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || 'Rebuild failed.');
      }
      const result = await response.json();
      rebuildStatus.textContent = result.date
        ? `Rebuilt ${result.date}.`
        : `Rebuilt ${result.sessions} session(s).`;
      state.analysisCache.delete(date);
      state.history = await loadHistory();
      renderDropdown();
      renderChart();
      await handleSelection();
    } catch (error) {
      rebuildStatus.textContent = error.message || 'Rebuild failed.';
    } finally {
      rebuildButton.disabled = false;
    }
  });
}

function attachDelete() {
  if (!deleteButton) {
    return;
  }
  deleteButton.addEventListener('click', async () => {
    const date = sessionSelect.value;
    if (!date) {
      return;
    }
    if (!confirm(`Delete session ${date}? This cannot be undone.`)) {
      return;
    }
    deleteButton.disabled = true;
    rebuildStatus.textContent = 'Deleting...';
    try {
      const response = await fetch('/api/delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ date }),
      });
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || 'Delete failed.');
      }
      const result = await response.json();
      rebuildStatus.textContent = `Deleted ${result.deleted}.`;
      state.analysisCache.delete(date);
      state.history = await loadHistory();
      renderDropdown();
      renderChart();
      if (state.history.sessions.length) {
        sessionSelect.value = state.history.sessions[state.history.sessions.length - 1].date;
        await handleSelection();
      } else {
        sessionTitle.textContent = 'No sessions.';
        sessionDetails.innerHTML = '';
        recommendationsEl.innerHTML = '';
        if (transcriptEl) {
          transcriptEl.textContent = '';
        }
      }
    } catch (error) {
      rebuildStatus.textContent = error.message || 'Delete failed.';
    } finally {
      deleteButton.disabled = false;
    }
  });
}

async function init() {
  try {
    state.history = await loadHistory();
    renderDropdown();
    renderChart();
    sessionSelect.addEventListener('change', handleSelection);
    attachRebuild();
    attachDelete();
    await handleSelection();
  } catch (error) {
    sessionTitle.textContent = 'Failed to load data.';
    sessionDetails.innerHTML = `<p>${error.message}</p>`;
  }
}

init();
