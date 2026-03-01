const sessionSelect = document.getElementById('session-select');
const sessionTitle = document.getElementById('session-title');
const sessionDetails = document.getElementById('session-details');
const recommendationsEl = document.getElementById('recommendations');
const transcriptRowsEl = document.getElementById('transcript-rows');
const transcriptNoteEl = document.getElementById('transcript-note');
const proficiencyChart = document.getElementById('progress-proficiency');
const errorsChart = document.getElementById('progress-errors');
const rebuildMetricsButton = document.getElementById('rebuild-metrics-button');
const rebuildAnnotationsButton = document.getElementById('rebuild-annotations-button');
const testModelButton = document.getElementById('test-model-button');
const deleteButton = document.getElementById('delete-button');
const rebuildStatus = document.getElementById('rebuild-status');
const rebuildMeta = document.getElementById('rebuild-meta');
const llmStatus = document.getElementById('llm-status');

function normalizeApiBase(rawValue) {
  const value = String(rawValue || '').trim();
  return value ? value.replace(/\/+$/, '') : '';
}

function getConfiguredApiBase() {
  const params = new URLSearchParams(window.location.search);
  const queryValue = normalizeApiBase(params.get('api_base'));
  if (queryValue) {
    try {
      window.localStorage.setItem('ENGLISH_TUTOR_API_BASE_URL', queryValue);
    } catch (error) {}
    return queryValue;
  }
  const inlineValue = normalizeApiBase(window.ENGLISH_TUTOR_API_BASE_URL);
  if (inlineValue) {
    return inlineValue;
  }
  try {
    return normalizeApiBase(window.localStorage.getItem('ENGLISH_TUTOR_API_BASE_URL'));
  } catch (error) {
    return '';
  }
}

function apiUrl(path) {
  const rawPath = String(path || '');
  if (!rawPath) {
    return rawPath;
  }
  if (/^https?:\/\//i.test(rawPath)) {
    return rawPath;
  }
  const base = getConfiguredApiBase();
  if (!base) {
    return rawPath;
  }
  const normalizedPath = rawPath.startsWith('/') ? rawPath : `/${rawPath}`;
  return `${base}${normalizedPath}`;
}

const state = {
  history: null,
  analysisCache: new Map(),
  pendingTarget: null,
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

function readLocationState() {
  const params = new URLSearchParams(window.location.search);
  const date = (params.get('date') || '').trim();
  const seekRaw = params.get('seek');
  const seekEndRaw = params.get('seek_end');
  const seek = seekRaw !== null && seekRaw !== '' && !Number.isNaN(Number(seekRaw))
    ? Number(seekRaw)
    : null;
  const seekEnd = seekEndRaw !== null && seekEndRaw !== '' && !Number.isNaN(Number(seekEndRaw))
    ? Number(seekEndRaw)
    : null;
  return { date, seek, seekEnd };
}

function focusTranscriptTarget(target) {
  if (!transcriptRowsEl || !target || target.start === null || target.start === undefined) {
    return;
  }
  const rows = [...transcriptRowsEl.querySelectorAll('.transcript-row[data-start]')];
  rows.forEach((row) => row.classList.remove('transcript-target'));
  const match = rows.find((row) => {
    const start = Number(row.dataset.start || 0);
    const end = Number(row.dataset.end || start);
    return target.start >= start && target.start < end;
  });
  if (!match) {
    return;
  }
  match.classList.add('transcript-target');
  match.scrollIntoView({ block: 'center', behavior: 'smooth' });
}

function renderDropdown() {
  sessionSelect.innerHTML = '';
  state.history.sessions.forEach((session) => {
    const option = document.createElement('option');
    option.value = session.date;
    option.textContent = `${session.date} - ${session.topic}`;
    sessionSelect.appendChild(option);
  });
}

function renderParticipants(analysis) {
  if (!sessionTitle || !sessionDetails) {
    return;
  }
  sessionTitle.textContent = `${analysis.date} - ${analysis.session.topic}`;
  sessionDetails.innerHTML = '';
  sortParticipants(analysis.participants).forEach((participant) => {
    const card = document.createElement('div');
    card.className = 'card';
    const grammar = participant.llm?.annotation_grammar || participant.llm?.chunked_grammar;
    card.innerHTML = `
      <h3>${participant.name} <span>${participant.role}</span></h3>
      <p class="metric"><strong>Words:</strong> ${participant.metrics.word_count}<span class="metric-note">Count of A-Z words in this speaker's turns.</span></p>
      <p class="metric"><strong>Turns:</strong> ${participant.metrics.turn_count}<span class="metric-note">Number of blocks starting with "${participant.name}:".</span></p>
      <p class="metric"><strong>Avg words/turn:</strong> ${participant.metrics.avg_words_per_turn}<span class="metric-note">Words divided by turns.</span></p>
      <p class="metric"><strong>Lexical diversity:</strong> ${participant.metrics.lexical_diversity}<span class="metric-note">Unique words / total words.</span></p>
      ${participant.llm ? `<p class="metric"><strong>LLM Proficiency:</strong> ${participant.llm.fluency.score} (${participant.llm.fluency.level})<span class="metric-note">Model-based assessment.</span></p>` : ''}
      ${grammar?.error_rate_per_100_words !== undefined ? `<p class="metric"><strong>Grammar error rate:</strong> ${grammar.error_rate_per_100_words} / 100 words<span class="metric-note">Derived from annotations when available.</span></p>` : ''}
      ${grammar?.error_types?.length ? `<ul class="errors">${grammar.error_types.map((item) => `<li><strong>${item.title}</strong> (${item.count})</li>`).join('')}</ul>` : ''}
    `;
    sessionDetails.appendChild(card);
  });
}

function renderRecommendations(analysis) {
  recommendationsEl.innerHTML = '';
  const annotationItems = analysis.llm?.annotation_items || [];
  const annotationByName = buildAnnotationMap(analysis.transcript, annotationItems, analysis.speaker_map || {});
  sortParticipants(analysis.participants).forEach((participant) => {
    const wrapper = document.createElement('div');
    wrapper.className = 'card';
    const grammar = participant.llm?.annotation_grammar || participant.llm?.chunked_grammar;
    const chunkedErrors = grammar?.error_types
      ?.map((error) => {
        const categoryCode = getCategoryCode(error);
        const examples = findAnnotationExamples(annotationByName, participant.name, categoryCode);
        const exampleItems = examples.length
          ? examples.map((item) => `<li><strong>${escapeHtml(item.text)}</strong> &rarr; ${escapeHtml(item.correction)}</li>`).join('')
          : '<li class="metric-note">No examples yet.</li>';
        return `
          <details class="error-group">
            <summary>${escapeHtml(error.title)} <span class="count">(${error.count})</span></summary>
            <ul class="errors compact">${exampleItems}</ul>
          </details>
        `;
      })
      .join('');
    const practicalList = participant.llm?.practical_recommendations || [];
    const practicalItems = practicalList
      .map((rec) => {
        const examples = (rec.examples || [])
          .map(
            (example) =>
              `<li><span class="example-error">${escapeHtml(example.error)}</span><span class="example-arrow">&rarr;</span><span class="example-fix">${escapeHtml(example.correction)}</span></li>`
          )
          .join('');
        const guidance = rec.guidance ? `<p class="metric-note">${escapeHtml(rec.guidance)}</p>` : '';
        const examplesBlock = examples ? `<ul class="errors compact">${examples}</ul>` : '';
        return `
          <li class="recommendation-item">
            <p class="recommendation-title">${escapeHtml(rec.title)}</p>
            ${guidance}
            ${examplesBlock}
          </li>
        `;
      })
      .join('');
    const chunkedBlock = grammar
      ? `
        <h4 class="subheading">Grammar breakdown</h4>
        <p class="metric-note">Total grammar errors: ${grammar.total_errors}</p>
        ${chunkedErrors ? `<div class="error-groups">${chunkedErrors}</div>` : ''}
      `
      : '';
    const summaryText = buildRecommendationSummary(participant, practicalList);
    const practicalBlock = practicalItems
      ? `
        <h4 class="subheading">Practical recommendations</h4>
        <p class="recommendations-summary">${escapeHtml(summaryText)}</p>
        <p class="recommendations-note">Top 3 most frequent patterns this session.</p>
        <ol class="recommendations-list">${practicalItems}</ol>
      `
      : '';
    wrapper.innerHTML = `
      <h3>${participant.name}</h3>
      ${practicalBlock}
      ${chunkedBlock}
    `;
    recommendationsEl.appendChild(wrapper);
  });

  const chunkedContainer = analysis.llm?.chunked_metrics;
  if (chunkedContainer?.status) {
    const wrapper = document.createElement('div');
    wrapper.className = 'card';
    wrapper.innerHTML = `
      <h3>Chunked grammar (${chunkedContainer.status})</h3>
      <p class="metric-note">Chunked metrics run per speaker and appear in the speaker cards.</p>
    `;
    recommendationsEl.appendChild(wrapper);
  }
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

function buildRecommendationSummary(participant, practicalList) {
  const name = participant?.name || 'This speaker';
  const totalErrors = participant?.llm?.annotation_grammar?.total_errors ?? participant?.llm?.chunked_grammar?.total_errors;
  const countText = typeof totalErrors === 'number' ? `${totalErrors} total grammar errors.` : 'Grammar errors detected.';
  const topTitles = practicalList.slice(0, 3).map((rec) => rec.title).filter(Boolean);
  if (!topTitles.length) {
    return `${name}: ${countText} Focus on the examples below to correct recurring patterns.`;
  }
  return `${name}: ${countText} The main recurring issues are ${topTitles.join(', ')}.`;
}

const CATEGORY_LABEL_TO_CODE = {
  'Verb Tense': 'TENSE',
  'Verb Form': 'VERB',
  Articles: 'ARTICLE',
  Prepositions: 'PREP',
  'Word Order': 'ORDER',
  'Wrong Word': 'WORD',
  Collocation: 'COLLOC',
};

function getCategoryCode(error) {
  if (!error) return '';
  if (error.code) return String(error.code);
  return CATEGORY_LABEL_TO_CODE[error.title] || '';
}

function parseTranscriptBlocks(text) {
  const pattern = /^\s*([^:]+):/gm;
  const matches = [...text.matchAll(pattern)];
  if (!matches.length) {
    return [];
  }
  return matches.map((match, index) => {
    const start = match.index ?? 0;
    const end = index + 1 < matches.length ? (matches[index + 1].index ?? text.length) : text.length;
    return { speaker: match[1].trim(), start, end };
  });
}

function buildAnnotationMap(transcriptText, items, speakerMap) {
  if (!items.length) {
    return {};
  }
  const blocks = parseTranscriptBlocks(transcriptText);
  const sortedItems = [...items].sort((a, b) => (a.start || 0) - (b.start || 0));
  const byName = {};
  let blockIndex = 0;
  sortedItems.forEach((item) => {
    const start = Number(item.start || 0);
    while (blockIndex < blocks.length && start >= blocks[blockIndex].end) {
      blockIndex += 1;
    }
    if (blockIndex >= blocks.length) {
      return;
    }
    const block = blocks[blockIndex];
    if (start < block.start || start >= block.end) {
      return;
    }
    const label = block.speaker || '';
    const name = speakerMap[label] || label;
    if (!name) {
      return;
    }
    byName[name] = byName[name] || [];
    byName[name].push(item);
  });
  return byName;
}

function findAnnotationExamples(annotationByName, name, categoryCode) {
  const items = annotationByName[name] || [];
  if (!items.length) {
    return [];
  }
  if (!categoryCode) {
    return [];
  }
  return items.filter((item) => String(item.category || '') === String(categoryCode));
}

function buildAnnotatedLine(lineText, items, lineStart) {
  if (!items.length) {
    return escapeHtml(lineText);
  }
  const parts = [];
  let cursor = 0;
  const sorted = [...items].sort((a, b) => a.start - b.start);
  sorted.forEach((item) => {
    const start = Math.max(item.start - lineStart, 0);
    const end = Math.min(item.end - lineStart, lineText.length);
    if (end <= start) {
      return;
    }
    if (start > cursor) {
      parts.push(escapeHtml(lineText.slice(cursor, start)));
    }
    const segment = escapeHtml(lineText.slice(start, end));
    const correction = item.correction ? `Correct: ${item.correction}` : '';
    const explanation = item.explanation || '';
    let title = correction || explanation || 'Grammar issue';
    if (correction && explanation) {
      title = `${correction} | ${explanation}`;
    }
    parts.push(`<mark class="grammar-error" title="${escapeHtml(title)}">${segment}</mark>`);
    cursor = end;
  });
  if (cursor < lineText.length) {
    parts.push(escapeHtml(lineText.slice(cursor)));
  }
  return parts.join('');
}

function buildTargetedLine(lineText, lineStart, target) {
  if (!target || target.start === null || target.start === undefined) {
    return escapeHtml(lineText);
  }
  const targetStart = Number(target.start);
  const targetEnd = target.end !== null && target.end !== undefined
    ? Math.max(targetStart + 1, Number(target.end))
    : targetStart + 1;
  const lineEnd = lineStart + lineText.length;
  if (targetEnd <= lineStart || targetStart >= lineEnd) {
    return escapeHtml(lineText);
  }
  const start = Math.max(0, targetStart - lineStart);
  const end = Math.min(lineText.length, targetEnd - lineStart);
  if (end <= start) {
    return escapeHtml(lineText);
  }
  return [
    escapeHtml(lineText.slice(0, start)),
    `<mark class="transcript-target-mark">${escapeHtml(lineText.slice(start, end))}</mark>`,
    escapeHtml(lineText.slice(end)),
  ].join('');
}

function renderTranscript(analysis) {
  if (!transcriptRowsEl) {
    return;
  }
  const transcript = analysis.transcript || '';
  const items = analysis.llm?.annotation_items || analysis.annotation_items || [];
  const target = state.pendingTarget;

  if (transcriptNoteEl) {
    const meta = analysis.llm?.annotations_meta;
    if (meta && meta.total_chunks !== undefined) {
      transcriptNoteEl.textContent = `Annotated ${meta.chunks_processed}/${meta.total_chunks} chunks (${meta.processed_chars} chars).`;
    } else {
      transcriptNoteEl.textContent = items.length
        ? 'Annotated transcript generated.'
        : 'Annotations will appear after the chunked LLM analysis is implemented.';
    }
  }

  const lines = transcript.split('\n');
  let offset = 0;
  transcriptRowsEl.innerHTML = '';

  lines.forEach((line, index) => {
    const lineStart = offset;
    const lineEnd = offset + line.length;
    offset += line.length + 1;

    const lineItems = items.filter((item) => item.start < lineEnd && item.end > lineStart);

    const row = document.createElement('div');
    row.className = 'transcript-row';
    row.dataset.start = String(lineStart);
    row.dataset.end = String(lineEnd);
    if (!lineItems.length) {
      row.classList.add('row-empty');
    }

    const originalCell = document.createElement('div');
    originalCell.className = 'transcript-cell transcript-text';
    originalCell.innerHTML = line ? buildTargetedLine(line, lineStart, target) : '&nbsp;';

    const annotatedCell = document.createElement('div');
    annotatedCell.className = 'transcript-cell transcript-text';
    annotatedCell.innerHTML = line ? buildAnnotatedLine(line, lineItems, lineStart) : '&nbsp;';

    const issuesCell = document.createElement('div');
    issuesCell.className = 'transcript-cell transcript-issues';
    if (!lineItems.length) {
      issuesCell.innerHTML = line ? '&nbsp;' : '&nbsp;';
    } else {
      const list = document.createElement('ul');
      list.className = 'issue-list compact';
      lineItems.forEach((item) => {
        const li = document.createElement('li');
        const explanation = item.explanation ? item.explanation : '—';
        const correction = item.correction ? ` (${item.correction})` : '';
        li.innerHTML = `<div class="issue-why">${explanation}${correction}</div>`;
        list.appendChild(li);
      });
      issuesCell.appendChild(list);
    }

    row.appendChild(originalCell);
    row.appendChild(annotatedCell);
    row.appendChild(issuesCell);
    transcriptRowsEl.appendChild(row);
  });
}

function renderLlmStatus(analysis) {
  if (!llmStatus) {
    return;
  }
  const info = analysis.llm || { status: 'skipped' };
  const metricsAt = info.metrics_updated_at || 'unknown';
  const annotationsAt = info.annotations_updated_at || 'unknown';

  let metricsStatus = 'skipped';
  if (info.status === 'ok') {
    metricsStatus = `success (${info.model || 'default model'})`;
  } else if (info.status === 'error') {
    metricsStatus = `error (${info.model || 'default model'})`;
  }

  let annotationsStatus = 'skipped';
  if (info.annotations_status === 'ok') {
    const attempted = info.annotations_attempted_model
      || info.annotations_model
      || info.annotations_meta?.attempted_model;
    const model = attempted ? ` (${attempted})` : '';
    const fallback = info.annotations_meta?.fallback_from
      ? ` via ${info.annotations_meta.fallback_to}`
      : '';
    annotationsStatus = `ok${model}${fallback}`;
  } else if (info.annotations_status === 'in_progress') {
    const attempted = info.annotations_attempted_model
      || info.annotations_model
      || info.annotations_meta?.attempted_model;
    const model = attempted ? ` (${attempted})` : '';
    const meta = info.annotations_meta;
    const progress = meta && meta.total_chunks
      ? ` ${meta.chunks_processed}/${meta.total_chunks}`
      : '';
    annotationsStatus = `running${model}${progress}`;
  } else if (info.annotations_status === 'error') {
    annotationsStatus = 'error';
  }

  llmStatus.textContent = `Metrics: ${metricsStatus} / updated ${metricsAt}`;
  if (rebuildMeta) {
    rebuildMeta.textContent = `Annotations: ${annotationsStatus} / updated ${annotationsAt}`;
  }
}

function renderRebuildMeta(analysis) {
  if (!rebuildMeta) {
    return;
  }
  // handled by renderLlmStatus
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
      return participant.chunked_error_rate ?? participant.llm_error_rate ?? null;
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
  if (state.pendingTarget) {
    focusTranscriptTarget(state.pendingTarget);
    state.pendingTarget = null;
  }
  renderLlmStatus(analysis);
  renderRebuildMeta(analysis);
}

function attachRebuildMetrics() {
  if (!rebuildMetricsButton) {
    return;
  }
  rebuildMetricsButton.addEventListener('click', async () => {
    rebuildMetricsButton.disabled = true;
    rebuildStatus.textContent = 'Rebuilding metrics...';
    try {
      const date = sessionSelect.value;
      const response = await fetch(apiUrl('/api/rebuild-metrics'), {
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
      const activeDate = date;
      state.history = await loadHistory();
      renderDropdown();
      if (sessionSelect && activeDate) {
        sessionSelect.value = activeDate;
      }
      renderChart();
      await handleSelection();
    } catch (error) {
      rebuildStatus.textContent = error.message || 'Rebuild failed.';
    } finally {
      rebuildMetricsButton.disabled = false;
    }
  });
}

function attachRebuildAnnotations() {
  if (!rebuildAnnotationsButton) {
    return;
  }
  rebuildAnnotationsButton.addEventListener('click', async () => {
    rebuildAnnotationsButton.disabled = true;
    rebuildStatus.textContent = 'Rebuilding annotations...';
    try {
      const date = sessionSelect.value;
      const response = await fetch(apiUrl('/api/rebuild-annotations'), {
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
      const activeDate = date;
      state.history = await loadHistory();
      renderDropdown();
      if (sessionSelect && activeDate) {
        sessionSelect.value = activeDate;
      }
      renderChart();
      await handleSelection();
    } catch (error) {
      rebuildStatus.textContent = error.message || 'Rebuild failed.';
    } finally {
      rebuildAnnotationsButton.disabled = false;
    }
  });
}

function attachTestModel() {
  if (!testModelButton) {
    return;
  }
  testModelButton.addEventListener('click', async () => {
    testModelButton.disabled = true;
    rebuildStatus.textContent = 'Testing gpt-5-mini...';
    try {
      const response = await fetch(apiUrl('/api/test-gpt5'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || 'Test failed.');
      }
      const result = await response.json();
      const output = result.output ? `Output: ${result.output}` : 'No output.';
      rebuildStatus.textContent = `Test ${result.model}: ${output}`;
    } catch (error) {
      rebuildStatus.textContent = error.message || 'Test failed.';
    } finally {
      testModelButton.disabled = false;
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
      const response = await fetch(apiUrl('/api/delete'), {
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
        if (transcriptRowsEl) {
          transcriptRowsEl.innerHTML = '';
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
    const locationState = readLocationState();
    state.pendingTarget = locationState.seek !== null
      ? { start: locationState.seek, end: locationState.seekEnd }
      : null;
    if (state.history.sessions.length) {
      const hasDate = locationState.date
        && state.history.sessions.some((session) => session.date === locationState.date);
      sessionSelect.value = hasDate
        ? locationState.date
        : state.history.sessions[state.history.sessions.length - 1].date;
    }
    renderChart();
    sessionSelect.addEventListener('change', handleSelection);
    attachRebuildMetrics();
    attachRebuildAnnotations();
    attachTestModel();
    attachDelete();
    await handleSelection();
  } catch (error) {
    sessionTitle.textContent = 'Failed to load data.';
    sessionDetails.innerHTML = `<p>${error.message}</p>`;
  }
}

init();
