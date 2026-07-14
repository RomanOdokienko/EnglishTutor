const sessionSelect = document.getElementById('highlight-session-select');
const rebuildMetricsButton = document.getElementById('rebuild-metrics-button');
const rebuildAnnotationsButton = document.getElementById('rebuild-annotations-button');
const deleteButton = document.getElementById('delete-button');
const sessionStatus = document.getElementById('session-status');
const sessionStatsEl = document.getElementById('session-stats');
const sessionTranscriptRows = document.getElementById('session-transcript-rows');
const sessionTranscriptNote = document.getElementById('session-transcript-note');
const highlightsRoot = document.getElementById('highlights-root');

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

const CATEGORY_LABEL_TO_CODE = {
  'Verb Tense': 'TENSE',
  'Verb Form': 'VERB',
  Articles: 'ARTICLE',
  Prepositions: 'PREP',
  'Word Order': 'ORDER',
  'Wrong Word': 'WORD',
  Collocation: 'COLLOC',
};

const EXACT_BLOCKED_EXAMPLES = new Set([
  'to invite four||to do this',
  "it's a talk||it's talked",
]);

const GENERIC_CORRECTION_TOKENS = new Set([
  'a',
  'an',
  'be',
  'did',
  'do',
  'does',
  'get',
  'go',
  'is',
  'it',
  "it's",
  'make',
  'that',
  'the',
  'thing',
  'things',
  'this',
  'to',
]);

const FUNCTION_WORD_TOKENS = new Set([
  'a',
  'an',
  'am',
  'are',
  "aren't",
  'be',
  'been',
  'being',
  'did',
  "didn't",
  'do',
  'does',
  "doesn't",
  "don't",
  'had',
  "hadn't",
  'has',
  "hasn't",
  'have',
  "haven't",
  'he',
  'her',
  'him',
  'i',
  "i'm",
  'is',
  "isn't",
  'it',
  "it's",
  'its',
  'me',
  'she',
  'that',
  'the',
  'their',
  'them',
  'they',
  'this',
  'those',
  'we',
  'were',
  "weren't",
  'was',
  "wasn't",
  'you',
]);

const state = {
  history: null,
  analysisCache: new Map(),
  currentBundle: null,
  exerciseCache: new Map(),
  exerciseLoading: new Set(),
  exerciseErrors: new Map(),
  exerciseAnswerState: new Map(),
  exerciseRequestData: new Map(),
  exerciseSequence: 0,
  focusData: { focuses: [] },
  focusSetData: new Map(),
  focusBusy: false,
  pinnedTranscriptAnnotationId: '',
};

async function loadHistory() {
  const response = await fetch(apiUrl('/history.json'), { cache: 'no-store' });
  if (!response.ok) {
    throw new Error('Unable to load history.json');
  }
  return response.json();
}

async function loadAnalysis(date) {
  if (state.analysisCache.has(date)) {
    return state.analysisCache.get(date);
  }
  const response = await fetch(apiUrl(`/sessions/${date}/analysis.json`), { cache: 'no-store' });
  if (!response.ok) {
    throw new Error(`Unable to load analysis for ${date}`);
  }
  const data = await response.json();
  state.analysisCache.set(date, data);
  return data;
}

async function loadFocusData() {
  try {
    const response = await fetch(apiUrl('/api/focus'), { cache: 'no-store' });
    if (!response.ok) {
      throw new Error('unavailable');
    }
    state.focusData = await response.json();
  } catch (error) {
    state.focusData = { focuses: [] };
  }
}

async function postFocus(payload) {
  const response = await fetch(apiUrl('/api/focus'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(cleanServerErrorMessage(await response.text(), 'Focus update failed.'));
  }
  state.focusData = await response.json();
}

function getExerciseCacheKey(date, participantName, categoryCode) {
  return [
    encodeURIComponent(String(date || '').trim()),
    encodeURIComponent(String(participantName || '').trim().toLowerCase()),
    encodeURIComponent(String(categoryCode || '').trim().toUpperCase()),
  ].join('::');
}

function registerExerciseContext(sessionDate, participantName, candidate) {
  const key = getExerciseCacheKey(sessionDate, participantName, candidate.code);
  state.exerciseRequestData.set(key, {
    participant_name: participantName,
    category_code: candidate.code,
    category_title: candidate.title,
    focus_text: candidate.focus,
    examples: (candidate.examples || []).map((example) => ({
      error: example.error,
      correction: example.correction,
    })),
  });
  return key;
}

function getExerciseEntries(exerciseKey) {
  const entries = state.exerciseCache.get(exerciseKey);
  return Array.isArray(entries) ? entries : [];
}

function createExerciseEntry(exerciseKey, exercise) {
  state.exerciseSequence += 1;
  return {
    id: `${exerciseKey}::${state.exerciseSequence}`,
    exercise,
  };
}

function cleanServerErrorMessage(rawValue, fallback = 'Exercise generation failed.') {
  const raw = String(rawValue || '').trim();
  if (!raw) {
    return fallback;
  }
  if (raw.startsWith('<!DOCTYPE') || raw.startsWith('<html')) {
    const messageMatch = raw.match(/<p>Message:\s*([^<]+)<\/p>/i);
    if (messageMatch) {
      return messageMatch[1].trim();
    }
    return fallback;
  }
  return raw;
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
  const roleOrder = {
    student: 0,
    partner: 1,
  };
  return list.sort((a, b) => {
    const aRole = roleOrder[(a?.role || '').toLowerCase()] ?? 99;
    const bRole = roleOrder[(b?.role || '').toLowerCase()] ?? 99;
    if (aRole !== bRole) {
      return aRole - bRole;
    }
    const aName = (a?.name || '').toLowerCase();
    const bName = (b?.name || '').toLowerCase();
    return aName.localeCompare(bName);
  });
}

function escapeHtml(value) {
  if (value === null || value === undefined) {
    return '';
  }
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
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
  const sortedItems = [...items].sort((a, b) => Number(a.start || 0) - Number(b.start || 0));
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

function normalizeText(value) {
  return String(value || '')
    .toLowerCase()
    .replace(/\s+/g, ' ')
    .trim();
}

function tokenize(value) {
  return normalizeText(value)
    .replace(/[^a-z0-9'\s]/g, ' ')
    .split(/\s+/)
    .filter(Boolean);
}

function lexicalOverlapRatio(left, right) {
  const leftTokens = new Set(tokenize(left));
  const rightTokens = new Set(tokenize(right));
  if (!leftTokens.size || !rightTokens.size) {
    return 0;
  }
  let shared = 0;
  leftTokens.forEach((token) => {
    if (rightTokens.has(token)) {
      shared += 1;
    }
  });
  return shared / Math.max(1, Math.min(leftTokens.size, rightTokens.size));
}

async function loadContextBundle(selectedDate) {
  const currentAnalysis = await loadAnalysis(selectedDate);
  return { currentAnalysis };
}

function buildSourceSentenceData(transcriptText, item) {
  if (!transcriptText || !item) {
    return { text: '', seek: null };
  }
  const start = Math.max(0, Number(item.start || 0));
  const end = Math.max(start, Number(item.end || start));
  const lineStart = transcriptText.lastIndexOf('\n', Math.max(0, start - 1)) + 1;
  const rawLineEnd = transcriptText.indexOf('\n', end);
  const lineEnd = rawLineEnd === -1 ? transcriptText.length : rawLineEnd;
  const lineText = transcriptText.slice(lineStart, lineEnd).trim();
  if (!lineText) {
    return { text: '', seek: start };
  }
  const relativeStart = Math.max(0, start - lineStart);
  const relativeEnd = Math.max(relativeStart, end - lineStart);
  let sentenceStart = 0;
  for (let index = relativeStart - 1; index >= 0; index -= 1) {
    const character = lineText[index];
    if (character === '.' || character === '?' || character === '!') {
      sentenceStart = index + 1;
      break;
    }
  }
  while (sentenceStart < lineText.length && /\s/.test(lineText[sentenceStart])) {
    sentenceStart += 1;
  }
  let sentenceEnd = lineText.length;
  for (let index = Math.min(lineText.length - 1, relativeEnd); index < lineText.length; index += 1) {
    const character = lineText[index];
    if (character === '.' || character === '?' || character === '!') {
      sentenceEnd = index + 1;
      break;
    }
  }
  let sentence = lineText.slice(sentenceStart, sentenceEnd).replace(/\s+/g, ' ').trim();
  if (!sentence) {
    sentence = lineText.replace(/\s+/g, ' ').trim();
  }
  if (sentence.length > 220) {
    sentence = `${sentence.slice(0, 217).trim()}...`;
  }
  return {
    text: sentence,
    seek: start,
    seekEnd: end,
  };
}

function isGenericCorrection(correction) {
  const tokens = tokenize(correction);
  if (!tokens.length || tokens.length > 3) {
    return false;
  }
  return tokens.every((token) => GENERIC_CORRECTION_TOKENS.has(token));
}

function isFunctionWordFragment(text) {
  const tokens = tokenize(text);
  if (!tokens.length || tokens.length > 3) {
    return false;
  }
  return tokens.every((token) => FUNCTION_WORD_TOKENS.has(token));
}

function hasRepeatedBigram(text) {
  const tokens = tokenize(text);
  if (tokens.length < 4) {
    return false;
  }
  const seen = new Set();
  for (let index = 0; index < tokens.length - 1; index += 1) {
    const bigram = `${tokens[index]} ${tokens[index + 1]}`;
    if (seen.has(bigram)) {
      return true;
    }
    seen.add(bigram);
  }
  return false;
}

function isTeachableExample(example, categoryCode) {
  if (!example) {
    return false;
  }
  const error = String(example.error || '').trim();
  const correction = String(example.correction || '').trim();
  if (!error || !correction) {
    return false;
  }
  const pairKey = `${normalizeText(error)}||${normalizeText(correction)}`;
  if (EXACT_BLOCKED_EXAMPLES.has(pairKey)) {
    return false;
  }
  if (!/[a-z]/i.test(error) || !/[a-z]/i.test(correction)) {
    return false;
  }
  if (error.includes('\n') || correction.includes('\n')) {
    return false;
  }
  if (error.length < 4 || correction.length < 4 || error.length > 96 || correction.length > 112) {
    return false;
  }
  const errorTokens = tokenize(error);
  const correctionTokens = tokenize(correction);
  if (!errorTokens.length || !correctionTokens.length) {
    return false;
  }
  if (errorTokens.length > 12 || correctionTokens.length > 14) {
    return false;
  }
  if (normalizeText(error) === normalizeText(correction)) {
    return false;
  }
  if (isFunctionWordFragment(error) && isFunctionWordFragment(correction)) {
    return false;
  }
  if (hasRepeatedBigram(error)) {
    return false;
  }
  const lengthRatio = correction.length / Math.max(1, error.length);
  if (lengthRatio < 0.45 || lengthRatio > 2.6) {
    return false;
  }
  const overlap = lexicalOverlapRatio(error, correction);
  if ((categoryCode === 'ARTICLE' || categoryCode === 'PREP') && overlap < 0.5) {
    return false;
  }
  if ((categoryCode === 'VERB' || categoryCode === 'TENSE' || categoryCode === 'ORDER') && overlap < 0.35) {
    return false;
  }
  if ((categoryCode === 'WORD' || categoryCode === 'COLLOC') && errorTokens.length >= 3 && correctionTokens.length >= 3 && overlap < 0.45) {
    return false;
  }
  if (categoryCode === 'ARTICLE' && !/\b(a|an|the)\b/i.test(correction)) {
    return false;
  }
  if (isGenericCorrection(correction) && overlap < 0.75) {
    return false;
  }
  return true;
}

function scoreExampleQuality(example, categoryCode) {
  const errorTokens = tokenize(example.error);
  const correctionTokens = tokenize(example.correction);
  const overlap = lexicalOverlapRatio(example.error, example.correction);
  let score = 0;
  score += Math.max(0, 18 - Math.max(errorTokens.length, correctionTokens.length));
  score += overlap * 10;
  if (example.context) {
    score += 2;
  }
  if (categoryCode === 'ARTICLE' && /\b(a|an|the)\b/i.test(example.correction)) {
    score += 3;
  }
  if (categoryCode === 'PREP' && overlap >= 0.6) {
    score += 2;
  }
  if (categoryCode === 'WORD' && overlap >= 0.6) {
    score += 1.5;
  }
  return score;
}

function collectBestExamples(items, categoryCode, transcriptText, fallbackExamples = [], limit = 2) {
  if (!categoryCode) {
    return [];
  }
  const deduped = new Map();
  const itemList = Array.isArray(items) ? items : [];
  itemList.forEach((item) => {
    if (String(item.category || '').toUpperCase() !== categoryCode) {
      return;
    }
    const source = buildSourceSentenceData(transcriptText, item);
    const example = {
      error: String(item.text || '').trim(),
      correction: String(item.correction || '').trim(),
      context: source.text,
      seek: source.seek,
      seekEnd: source.seekEnd,
    };
    const key = `${normalizeText(example.error)}||${normalizeText(example.correction)}`;
    if (!deduped.has(key)) {
      deduped.set(key, example);
    }
  });
  const fallbackList = Array.isArray(fallbackExamples) ? fallbackExamples : [];
  fallbackList.forEach((item) => {
    const example = {
      error: String(item?.error || '').trim(),
      correction: String(item?.correction || '').trim(),
      context: '',
      seek: null,
      seekEnd: null,
    };
    const key = `${normalizeText(example.error)}||${normalizeText(example.correction)}`;
    if (!deduped.has(key)) {
      deduped.set(key, example);
    }
  });
  return [...deduped.values()]
    .filter((example) => isTeachableExample(example, categoryCode))
    .sort((left, right) => scoreExampleQuality(right, categoryCode) - scoreExampleQuality(left, categoryCode))
    .slice(0, limit);
}

// ---- Evidence cards ----
// The Session page is a witness, not a coach: every card is built only from
// stored data (counts and densities from derived.grammar, trend from
// history.json, examples from the transcript). No hidden weights, no advice
// templates. Picking a focus is a human decision (plan task 2.3).

const CATEGORY_CODES = ['TENSE', 'VERB', 'ARTICLE', 'PREP', 'ORDER', 'WORD', 'COLLOC'];
const CATEGORY_CODE_TO_LABEL = Object.fromEntries(
  Object.entries(CATEGORY_LABEL_TO_CODE).map(([label, code]) => [code, label])
);
const TREND_LOOKBACK = 3;
// A category counts as present in a past session when its density clears this
// floor (about two errors in a 700-word session) on a big-enough sample.
const PERSISTENCE_DENSITY_FLOOR = 0.3;
const PERSISTENCE_MIN_WORDS = 120;
const LOW_SAMPLE_WORDS = 30;

function getPreviousGrammarTrail(selectedDate, participantName) {
  const sessions = state.history?.sessions || [];
  return sessions
    .filter((session) => String(session.date || '') < String(selectedDate || ''))
    .slice(-TREND_LOOKBACK)
    .map((session) => {
      const participant = (session.participants || []).find((item) => item.name === participantName);
      return {
        date: session.date,
        words: Number(participant?.derived?.metrics?.english_word_count || 0),
        density: participant?.derived?.grammar?.by_category_density || null,
      };
    });
}

function buildEvidenceCards(participant, itemsForSpeaker, transcriptText, selectedDate) {
  const grammar = participant.derived?.grammar;
  if (!grammar) {
    return [];
  }
  const counts = grammar.by_category_count || {};
  const densities = grammar.by_category_density || {};
  const totalErrors = Number(grammar.error_count || 0);
  const trail = getPreviousGrammarTrail(selectedDate, participant.name);
  const comparable = trail.filter((entry) => entry.density && entry.words >= PERSISTENCE_MIN_WORDS);

  return CATEGORY_CODES
    .map((code) => {
      const count = Number(counts[code] || 0);
      if (!count) {
        return null;
      }
      const share = totalErrors > 0 ? Math.round((count / totalErrors) * 100) : 0;
      const seenIn = comparable.filter((entry) => Number(entry.density[code] || 0) >= PERSISTENCE_DENSITY_FLOOR).length;
      return {
        code,
        title: CATEGORY_CODE_TO_LABEL[code] || code,
        count,
        share,
        density: Number(densities[code] || 0),
        seenIn,
        comparableCount: comparable.length,
        recurring: seenIn >= 2,
        densityTrail: comparable.map((entry) => Number(entry.density[code] || 0)),
        examples: collectBestExamples(itemsForSpeaker, code, transcriptText, [], 3),
      };
    })
    .filter(Boolean)
    .sort((a, b) =>
      (Number(b.recurring) - Number(a.recurring))
      || (b.count - a.count)
      || (b.density - a.density)
      || a.code.localeCompare(b.code));
}

function buildEvidenceMeta(card) {
  const cases = card.count === 1 ? '1 case' : `${card.count} cases`;
  return `${cases} · ${card.share}% of errors · ${card.density}/100w`;
}

function buildEvidenceTrendLine(card) {
  if (!card.comparableCount) {
    return 'No comparable history yet';
  }
  const sessionsWord = card.comparableCount === 1 ? 'session' : 'sessions';
  const persistence = card.seenIn
    ? `Seen in ${card.seenIn} of last ${card.comparableCount} comparable ${sessionsWord}`
    : `Not seen in last ${card.comparableCount} comparable ${sessionsWord}`;
  const trail = [...card.densityTrail, card.density].join(' → ');
  return `${persistence} · density ${trail}`;
}

// ---- Focus of the week ----
// Focuses are chosen by a human from the evidence cards and stored in
// data/focus.json on the backend. The closure verdict is deterministic
// (rule v1, see docs/adr/0005): the focus category's density in the viewed
// session fell by 40%+ against the session where the focus was set, on a
// sample of 120+ English words.
const FOCUS_CLOSE_RATIO = 0.6;

function activeFocusesFor(participantName) {
  return (state.focusData?.focuses || []).filter(
    (item) => item.participant === participantName && item.status === 'active'
  );
}

function getGrammarSnapshot(date, participantName) {
  const session = (state.history?.sessions || []).find((item) => item.date === date);
  const participant = (session?.participants || []).find((item) => item.name === participantName);
  if (!participant?.derived) {
    return null;
  }
  return {
    density: participant.derived.grammar?.by_category_density || {},
    words: Number(participant.derived.metrics?.english_word_count || 0),
  };
}

function buildFocusVerdict(focus, selectedDate) {
  const baseline = getGrammarSnapshot(focus.set_date, focus.participant);
  if (!baseline) {
    return { text: 'Baseline session is missing from history — no verdict.', ready: false };
  }
  const baselineDensity = Number(baseline.density[focus.category_code] || 0);
  if (String(selectedDate || '') <= String(focus.set_date)) {
    return { text: `Baseline ${baselineDensity}/100w. Open a newer session for a verdict.`, ready: false };
  }
  const current = getGrammarSnapshot(selectedDate, focus.participant);
  if (!current) {
    return { text: `Baseline ${baselineDensity}/100w. No data in the viewed session.`, ready: false };
  }
  const currentDensity = Number(current.density[focus.category_code] || 0);
  const trail = `${baselineDensity} → ${currentDensity}/100w`;
  if (current.words < PERSISTENCE_MIN_WORDS) {
    return { text: `${trail} · low sample (under ${PERSISTENCE_MIN_WORDS} words) — no verdict`, ready: false };
  }
  const ready = baselineDensity > 0
    ? currentDensity <= baselineDensity * FOCUS_CLOSE_RATIO
    : currentDensity === 0;
  const change = baselineDensity > 0
    ? ` (${Math.round(((currentDensity - baselineDensity) / baselineDensity) * 100)}%)`
    : '';
  return { text: `${trail}${change}${ready ? ' ✓ ready to close' : ''}`, ready };
}

function renderFocusBlock(participantName, selectedDate) {
  const focuses = activeFocusesFor(participantName);
  if (!focuses.length) {
    return '';
  }
  const rows = focuses.map((focus) => {
    const verdict = buildFocusVerdict(focus, selectedDate);
    const label = CATEGORY_CODE_TO_LABEL[focus.category_code] || focus.category_code;
    return `
      <div class="focus-entry${verdict.ready ? ' is-ready' : ''}">
        <div class="focus-entry-main">
          <span class="focus-entry-title">${escapeHtml(label)}</span>
          <span class="focus-entry-meta">set ${escapeHtml(formatSessionDate(focus.set_date))}</span>
        </div>
        <span class="focus-entry-verdict">${escapeHtml(verdict.text)}</span>
        <span class="focus-entry-actions">
          <button class="ghost focus-close-button" type="button" data-focus-id="${escapeHtml(focus.id)}" ${state.focusBusy ? 'disabled' : ''}>Close</button>
          <button class="ghost focus-remove-button" type="button" data-focus-id="${escapeHtml(focus.id)}" ${state.focusBusy ? 'disabled' : ''}>Remove</button>
        </span>
      </div>
    `;
  }).join('');
  return `
    <h3 class="highlight-section-title">Focus of the week</h3>
    <div class="focus-block">${rows}</div>
  `;
}

function renderEvidenceCard(card, sessionDate, participantName) {
  const exerciseKey = registerExerciseContext(sessionDate, participantName, {
    code: card.code,
    title: card.title,
    focus: '',
    examples: card.examples,
  });
  const examplesHtml = card.examples.length
    ? `<ul class="errors compact highlight-examples-list">${card.examples.map((example) => `
        <li class="highlight-example-row">
          <span class="example-error">${escapeHtml(example.error)}</span>
          <span class="example-arrow">&rarr;</span>
          <span class="example-fix">${escapeHtml(example.correction)}</span>
          ${renderExampleSource(example, sessionDate)}
        </li>
      `).join('')}</ul>`
    : '<p class="metric-note">No clean examples passed the quality filters; see the annotated transcript below for raw evidence.</p>';
  const focusKey = `${participantName}::${card.code}`;
  state.focusSetData.set(focusKey, {
    participant: participantName,
    category_code: card.code,
    examples: card.examples.map((example) => ({ error: example.error, correction: example.correction })),
  });
  const inFocus = activeFocusesFor(participantName).some((item) => item.category_code === card.code);
  const focusControl = inFocus
    ? '<span class="focus-chip">In focus</span>'
    : `<button class="ghost focus-set-button" type="button" data-focus-key="${escapeHtml(focusKey)}" ${state.focusBusy ? 'disabled' : ''}>Add to focus</button>`;
  return `
    <section class="highlight-item evidence-card">
      <div class="highlight-body">
        <div class="highlight-item-head">
          <h3>${escapeHtml(card.title)}</h3>
          <span class="highlight-item-meta">${escapeHtml(buildEvidenceMeta(card))}</span>
          ${focusControl}
        </div>
        <p class="highlight-status-line">${escapeHtml(buildEvidenceTrendLine(card))}</p>
        ${examplesHtml}
        <div class="highlight-action-card">
          <div class="highlight-action-head">
            <p class="highlight-action-title">Practice</p>
            ${renderExerciseTrigger(exerciseKey)}
          </div>
          ${renderExercisePanel(exerciseKey)}
        </div>
      </div>
    </section>
  `;
}

function buildSummaryMeta(participant) {
  const grammar = participant.derived?.grammar;
  if (!grammar) {
    return 'Run metrics to build evidence.';
  }
  const total = Number(grammar.error_count || 0);
  if (!total) {
    return 'No mapped grammar issues';
  }
  return `${total} mapped issues / ${grammar.error_density_per_100w} per 100 words`;
}

function renderExerciseTrigger(exerciseKey) {
  const loading = state.exerciseLoading.has(exerciseKey);
  const hasExercise = getExerciseEntries(exerciseKey).length > 0;
  if (hasExercise) {
    return '';
  }
  const label = loading ? 'Generating...' : 'Generate exercise';
  return `
    <button class="ghost exercise-trigger" type="button" data-exercise-key="${exerciseKey}" ${loading ? 'disabled' : ''}>
      ${label}
    </button>
  `;
}

function renderExercisePanel(exerciseKey) {
  const loading = state.exerciseLoading.has(exerciseKey);
  const error = state.exerciseErrors.get(exerciseKey);
  const entries = getExerciseEntries(exerciseKey);
  if (!entries.length) {
    const statusParts = [];
    if (loading) {
      statusParts.push('<p class="exercise-status">Generating a new exercise...</p>');
    }
    if (error) {
      statusParts.push(`<p class="exercise-status exercise-status-error">${escapeHtml(error)}</p>`);
    }
    return statusParts.join('');
  }

  const cards = entries.map((entry, entryIndex) => {
    const exercise = entry.exercise || {};
    const answerState = state.exerciseAnswerState.get(entry.id);
    const options = (exercise.options || []).map((option, optionIndex) => {
      const isCorrect = String(option) === String(exercise.answer);
      const isSelected = answerState && Number(answerState.selectedIndex) === optionIndex;
      let className = 'ghost exercise-option';
      if (answerState) {
        if (isCorrect) {
          className += ' is-correct';
        } else if (isSelected) {
          className += ' is-wrong';
        }
      }
      return `
        <button
          class="${className}"
          type="button"
          data-exercise-key="${exerciseKey}"
          data-exercise-id="${entry.id}"
          data-option-index="${optionIndex}"
          ${answerState ? 'disabled' : ''}
        >${escapeHtml(option)}</button>
      `;
    }).join('');

    const showGenerateAnother = entryIndex === entries.length - 1 && !!answerState;
    const feedbackHtml = answerState
      ? `
        <p class="exercise-feedback ${answerState.correct ? 'is-correct' : 'is-wrong'}">
          ${answerState.correct ? 'Correct.' : `Correct answer: ${escapeHtml(exercise.answer)}`}
        </p>
        <p class="exercise-explanation">${escapeHtml(exercise.explanation || '')}</p>
        ${showGenerateAnother ? `
          <div class="exercise-actions">
            <button class="ghost exercise-trigger exercise-trigger-secondary" type="button" data-exercise-key="${exerciseKey}" ${loading ? 'disabled' : ''}>
              ${loading ? 'Generating...' : 'Generate another one'}
            </button>
          </div>
        ` : ''}
      `
      : '<p class="exercise-status">Pick one option.</p>';

    return `
      <div class="highlight-exercise">
        <p class="exercise-kicker">${escapeHtml(exercise.title || 'Mini exercise')}</p>
        <p class="exercise-prompt">${escapeHtml(exercise.prompt || 'Choose the better option.')}</p>
        <p class="exercise-question">${escapeHtml(exercise.question || '')}</p>
        <div class="exercise-options">${options}</div>
        ${feedbackHtml}
      </div>
    `;
  }).join('');

  const statusParts = [];
  if (loading) {
    statusParts.push('<p class="exercise-status">Generating a new exercise...</p>');
  }
  if (error) {
    statusParts.push(`<p class="exercise-status exercise-status-error">${escapeHtml(error)}</p>`);
  }

  return `
    <div class="highlight-exercise-shell">
      ${cards}
      ${statusParts.join('')}
    </div>
  `;
}

async function requestHighlightExercise(exerciseKey) {
  if (!exerciseKey || state.exerciseLoading.has(exerciseKey)) {
    return;
  }
  const payload = state.exerciseRequestData.get(exerciseKey);
  if (!payload) {
    return;
  }
  state.exerciseLoading.add(exerciseKey);
  state.exerciseErrors.delete(exerciseKey);
  if (state.currentBundle) {
    renderHighlights(state.currentBundle);
  }
  try {
    const response = await fetch(apiUrl('/api/highlight-exercise'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(cleanServerErrorMessage(errorText));
    }
    const result = await response.json();
    const entries = getExerciseEntries(exerciseKey);
    state.exerciseCache.set(exerciseKey, entries.concat(createExerciseEntry(exerciseKey, result.exercise)));
  } catch (error) {
    state.exerciseErrors.set(exerciseKey, error.message || 'Exercise generation failed.');
  } finally {
    state.exerciseLoading.delete(exerciseKey);
    if (state.currentBundle) {
      renderHighlights(state.currentBundle);
    }
  }
}

function selectExerciseOption(exerciseKey, exerciseId, optionIndex) {
  const entry = getExerciseEntries(exerciseKey).find((item) => item.id === exerciseId);
  const exercise = entry?.exercise;
  if (!exercise) {
    return;
  }
  const selected = exercise.options?.[optionIndex];
  if (selected === undefined) {
    return;
  }
  state.exerciseAnswerState.set(exerciseId, {
    selectedIndex: optionIndex,
    correct: String(selected) === String(exercise.answer),
  });
  if (state.currentBundle) {
    renderHighlights(state.currentBundle);
  }
}

function renderExampleSource(example, sessionDate) {
  if (!example.context) {
    return '';
  }
  return `
    <div class="example-source-row">
      <div class="example-source-text">${escapeHtml(example.context)}</div>
      <a class="example-source" href="#session-transcript-rows">Jump to transcript</a>
    </div>
  `;
}

function renderHighlights(bundle) {
  highlightsRoot.innerHTML = '';
  state.currentBundle = bundle;
  state.exerciseRequestData.clear();
  const analysis = bundle.currentAnalysis;
  const participants = sortParticipants(analysis.participants || []);
  if (!participants.length) {
    highlightsRoot.innerHTML = '<p>No participant data found for this session.</p>';
    return;
  }
  const transcriptText = analysis.transcript || '';
  const annotationItems = analysis.llm?.annotation_items || [];
  const annotationByName = buildAnnotationMap(transcriptText, annotationItems, analysis.speaker_map || {});

  participants.forEach((participant) => {
    const card = document.createElement('div');
    card.className = 'highlight-card';
    const itemsForSpeaker = annotationByName[participant.name] || [];
    const evidenceCards = buildEvidenceCards(participant, itemsForSpeaker, transcriptText, analysis.date);
    const lowSample = Number(participant.derived?.metrics?.english_word_count || 0) < LOW_SAMPLE_WORDS;
    const cardsHtml = evidenceCards.length
      ? evidenceCards.map((item) => renderEvidenceCard(item, analysis.date, participant.name)).join('')
      : '<p class="metric-note">No categorized errors stored for this session. Run annotations to build the evidence.</p>';

    card.innerHTML = `
      <div class="highlight-header">
        <h2>${escapeHtml(participant.name)}</h2>
        <span class="highlight-meta">${escapeHtml(buildSummaryMeta(participant))}</span>
      </div>
      ${lowSample ? '<p class="metric-note">Low sample (under 30 English words) — treat these numbers as anecdotal.</p>' : ''}
      ${renderFocusBlock(participant.name, analysis.date)}
      <h3 class="highlight-section-title">Error evidence</h3>
      ${cardsHtml}
    `;
    highlightsRoot.appendChild(card);
  });
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
    const annotationId = item._transcriptAnnotationId || '';
    const annotationNumber = item._transcriptAnnotationNumber || '';
    parts.push(
      `<mark class="grammar-error" id="${escapeHtml(annotationId)}-text" data-transcript-annotation-id="${escapeHtml(annotationId)}" `
      + `data-transcript-annotation-number="${escapeHtml(annotationNumber)}" title="${escapeHtml(title)}" tabindex="0">${segment}</mark>`
    );
    cursor = end;
  });
  if (cursor < lineText.length) {
    parts.push(escapeHtml(lineText.slice(cursor)));
  }
  return parts.join('');
}

const SESSION_METRICS = [
  ['English words', (d) => d.metrics?.english_word_count, ''],
  ['Words / turn', (d) => d.metrics?.avg_words_per_turn, ''],
  ['Speaking share', (d) => d.metrics?.speaking_share_pct, '%'],
  ['Error density', (d) => d.grammar?.error_density_per_100w, ' /100w'],
  ['Fillers', (d) => d.metrics?.filler_per_100w, ' /100w'],
  ['Russian fallback', (d) => d.metrics?.l1_fallback_pct, '%'],
  ['Lexical diversity', (d) => d.metrics?.lexical_diversity_mattr, ''],
];

function renderSessionStats(analysis) {
  if (!sessionStatsEl) {
    return;
  }
  const participants = sortParticipants(analysis.participants || []);
  if (!participants.length) {
    sessionStatsEl.innerHTML = '<p class="metric-note">No participant data.</p>';
    return;
  }
  sessionStatsEl.innerHTML = participants.map((participant) => {
    const derived = participant.derived;
    if (!derived) {
      return `<div class="stat-card"><h3>${escapeHtml(participant.name)}</h3><p class="metric-note">No metrics yet — re-run metrics.</p></div>`;
    }
    const rows = SESSION_METRICS.map(([label, get, unit]) => {
      const value = get(derived);
      const shown = value === null || value === undefined ? '—' : `${value}${unit}`;
      return `<div class="stat-row"><span>${label}</span><b>${shown}</b></div>`;
    }).join('');
    return `<div class="stat-card"><h3>${escapeHtml(participant.name)}</h3>${rows}</div>`;
  }).join('');
}

function renderSessionTranscript(analysis) {
  if (!sessionTranscriptRows) {
    return;
  }
  state.pinnedTranscriptAnnotationId = '';
  const transcript = analysis.transcript || '';
  const items = analysis.llm?.annotation_items || analysis.annotation_items || [];
  const annotationIds = new Map(items.map((item, index) => [item, `transcript-annotation-${index + 1}`]));
  if (sessionTranscriptNote) {
    const meta = analysis.llm?.annotations_meta;
    if (meta && meta.total_chunks !== undefined) {
      sessionTranscriptNote.textContent = `Annotated ${meta.chunks_processed}/${meta.total_chunks} chunks.`;
    } else {
      sessionTranscriptNote.textContent = items.length
        ? 'Annotated transcript generated.'
        : 'No annotations yet — run annotations to highlight errors.';
    }
  }

  const lines = transcript.split('\n');
  let offset = 0;
  sessionTranscriptRows.innerHTML = '';
  lines.forEach((line) => {
    const lineStart = offset;
    const lineEnd = offset + line.length;
    offset += line.length + 1;
    const lineItems = items
      .filter((item) => item.start < lineEnd && item.end > lineStart)
      .map((item, index) => ({
        ...item,
        _transcriptAnnotationId: annotationIds.get(item),
        _transcriptAnnotationNumber: items.indexOf(item) + 1,
      }));

    const row = document.createElement('div');
    row.className = 'transcript-row' + (lineItems.length ? '' : ' row-empty');

    const annotatedCell = document.createElement('div');
    annotatedCell.className = 'transcript-cell transcript-text';
    annotatedCell.innerHTML = line ? buildAnnotatedLine(line, lineItems, lineStart) : '&nbsp;';

    const issuesCell = document.createElement('div');
    issuesCell.className = 'transcript-cell transcript-issues';
    if (!lineItems.length) {
      issuesCell.innerHTML = '&nbsp;';
    } else {
      const list = document.createElement('ul');
      list.className = 'issue-list compact';
      lineItems.forEach((item) => {
        const li = document.createElement('li');
        const annotationId = item._transcriptAnnotationId || '';
        const annotationNumber = item._transcriptAnnotationNumber || '';
        li.className = 'issue-item';
        li.tabIndex = 0;
        li.dataset.transcriptAnnotationId = annotationId;
        li.setAttribute('aria-controls', `${annotationId}-text`);
        li.setAttribute('aria-label', `Issue ${annotationNumber}: ${item.text || item.explanation || 'grammar issue'}`);
        const explanation = item.explanation ? item.explanation : '—';
        const correction = item.correction ? ` (${item.correction})` : '';
        li.innerHTML = `
          <div class="issue-reference"><span class="issue-number">${escapeHtml(annotationNumber)}</span><q>${escapeHtml(item.text || 'Marked phrase')}</q></div>
          <div class="issue-why">${escapeHtml(explanation)}${escapeHtml(correction)}</div>
        `;
        list.appendChild(li);
      });
      issuesCell.appendChild(list);
    }

    row.appendChild(annotatedCell);
    row.appendChild(issuesCell);
    sessionTranscriptRows.appendChild(row);
  });
}

function transcriptAnnotationIdFromTarget(target) {
  const source = target && target.closest ? target.closest('[data-transcript-annotation-id]') : null;
  return source?.dataset.transcriptAnnotationId || '';
}

function setActiveTranscriptAnnotation(annotationId, options = {}) {
  if (!sessionTranscriptRows) {
    return;
  }
  const selector = `[data-transcript-annotation-id="${CSS.escape(annotationId)}"]`;
  const matches = sessionTranscriptRows.querySelectorAll(selector);
  matches.forEach((element) => element.classList.toggle('is-active', Boolean(annotationId)));
  if (options.scroll && matches.length) {
    const mark = [...matches].find((element) => element.classList.contains('grammar-error'));
    mark?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }
}

function clearActiveTranscriptAnnotation(annotationId) {
  if (!annotationId || annotationId === state.pinnedTranscriptAnnotationId || !sessionTranscriptRows) {
    return;
  }
  const selector = `[data-transcript-annotation-id="${CSS.escape(annotationId)}"]`;
  sessionTranscriptRows.querySelectorAll(selector).forEach((element) => element.classList.remove('is-active'));
}

function attachTranscriptInteractions() {
  if (!sessionTranscriptRows) {
    return;
  }
  sessionTranscriptRows.addEventListener('pointerover', (event) => {
    const annotationId = transcriptAnnotationIdFromTarget(event.target);
    if (annotationId) {
      setActiveTranscriptAnnotation(annotationId);
    }
  });
  sessionTranscriptRows.addEventListener('pointerout', (event) => {
    const annotationId = transcriptAnnotationIdFromTarget(event.target);
    if (annotationId && annotationId !== transcriptAnnotationIdFromTarget(event.relatedTarget)) {
      clearActiveTranscriptAnnotation(annotationId);
    }
  });
  sessionTranscriptRows.addEventListener('focusin', (event) => {
    const annotationId = transcriptAnnotationIdFromTarget(event.target);
    if (annotationId) {
      setActiveTranscriptAnnotation(annotationId, { scroll: true });
    }
  });
  sessionTranscriptRows.addEventListener('focusout', (event) => {
    const annotationId = transcriptAnnotationIdFromTarget(event.target);
    if (annotationId && annotationId !== transcriptAnnotationIdFromTarget(event.relatedTarget)) {
      clearActiveTranscriptAnnotation(annotationId);
    }
  });
  sessionTranscriptRows.addEventListener('click', (event) => {
    const annotationId = transcriptAnnotationIdFromTarget(event.target);
    if (!annotationId) {
      return;
    }
    if (state.pinnedTranscriptAnnotationId === annotationId) {
      state.pinnedTranscriptAnnotationId = '';
      clearActiveTranscriptAnnotation(annotationId);
      return;
    }
    if (state.pinnedTranscriptAnnotationId) {
      const previous = state.pinnedTranscriptAnnotationId;
      state.pinnedTranscriptAnnotationId = '';
      clearActiveTranscriptAnnotation(previous);
    }
    state.pinnedTranscriptAnnotationId = annotationId;
    setActiveTranscriptAnnotation(annotationId, { scroll: true });
  });
}

function setSessionStatus(message) {
  if (sessionStatus) {
    sessionStatus.textContent = message || '';
  }
}

async function rerunSession(endpoint, label, button) {
  const date = sessionSelect.value;
  if (!date) {
    return;
  }
  if (button) button.disabled = true;
  setSessionStatus(`${label}…`);
  try {
    const response = await fetch(apiUrl(endpoint), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ date }),
    });
    if (!response.ok) {
      throw new Error((await response.text()) || 'Failed.');
    }
    let done = 'Done.';
    try {
      const result = await response.json();
      if (result && 'annotations_status' in result) {
        if (result.annotations_status === 'ok') {
          done = `Done — ${result.annotation_items} annotation${result.annotation_items === 1 ? '' : 's'}.`;
        } else if (result.annotations_status === 'skipped') {
          done = `Skipped: ${result.annotations_error || 'no OpenAI key on this server'}.`;
        } else if (result.annotations_status === 'error') {
          done = `Annotation error: ${result.annotations_error || 'see server logs'}.`;
        }
      }
    } catch (parseError) {
      /* keep the generic Done. */
    }
    setSessionStatus(done);
    // The rebuild rewrote this session server-side; drop the cached copy so
    // the transcript and evidence cards re-render from the fresh analysis.
    state.analysisCache.delete(date);
    state.history = await loadHistory();
    renderDropdown();
    sessionSelect.value = date;
    await handleSelection();
  } catch (error) {
    setSessionStatus(error.message || 'Failed.');
  } finally {
    if (button) button.disabled = false;
  }
}

function attachSessionActions() {
  if (rebuildMetricsButton) {
    rebuildMetricsButton.addEventListener('click', () =>
      rerunSession('/api/rebuild-metrics', 'Re-running metrics', rebuildMetricsButton));
  }
  if (rebuildAnnotationsButton) {
    rebuildAnnotationsButton.addEventListener('click', () =>
      rerunSession('/api/rebuild-annotations', 'Re-running annotations', rebuildAnnotationsButton));
  }
  if (deleteButton) {
    deleteButton.addEventListener('click', async () => {
      const date = sessionSelect.value;
      if (!date || !window.confirm(`Delete session ${date}? This cannot be undone.`)) {
        return;
      }
      deleteButton.disabled = true;
      setSessionStatus('Deleting…');
      try {
        const response = await fetch(apiUrl('/api/delete'), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ date }),
        });
        if (!response.ok) {
          throw new Error((await response.text()) || 'Delete failed.');
        }
        state.history = await loadHistory();
        renderDropdown();
        if (state.history.sessions.length) {
          sessionSelect.value = state.history.sessions[state.history.sessions.length - 1].date;
          await handleSelection();
        } else {
          highlightsRoot.innerHTML = '<p>No sessions left.</p>';
          if (sessionStatsEl) sessionStatsEl.innerHTML = '';
          if (sessionTranscriptRows) sessionTranscriptRows.innerHTML = '';
        }
        setSessionStatus('Deleted.');
      } catch (error) {
        setSessionStatus(error.message || 'Delete failed.');
      } finally {
        deleteButton.disabled = false;
      }
    });
  }
}

async function handleSelection() {
  const date = sessionSelect.value;
  if (!date) {
    highlightsRoot.innerHTML = '<p>No sessions yet.</p>';
    return;
  }
  highlightsRoot.innerHTML = '<p class="metric-note">Loading session…</p>';
  const bundle = await loadContextBundle(date);
  renderHighlights(bundle);
  renderSessionStats(bundle.currentAnalysis);
  renderSessionTranscript(bundle.currentAnalysis);
}

async function runFocusAction(payload, confirmText) {
  if (state.focusBusy) {
    return;
  }
  if (confirmText && !window.confirm(confirmText)) {
    return;
  }
  state.focusBusy = true;
  try {
    await postFocus(payload);
    setSessionStatus('');
  } catch (error) {
    setSessionStatus(error.message || 'Focus update failed.');
  } finally {
    state.focusBusy = false;
    if (state.currentBundle) {
      renderHighlights(state.currentBundle);
    }
  }
}

function attachHighlightsInteractions() {
  highlightsRoot.addEventListener('click', async (event) => {
    const trigger = event.target.closest('.exercise-trigger');
    if (trigger) {
      const exerciseKey = trigger.dataset.exerciseKey || '';
      await requestHighlightExercise(exerciseKey);
      return;
    }
    const option = event.target.closest('.exercise-option');
    if (option) {
      const exerciseKey = option.dataset.exerciseKey || '';
      const exerciseId = option.dataset.exerciseId || '';
      const optionIndex = Number(option.dataset.optionIndex || -1);
      if (optionIndex >= 0) {
        selectExerciseOption(exerciseKey, exerciseId, optionIndex);
      }
      return;
    }
    const setButton = event.target.closest('.focus-set-button');
    if (setButton) {
      const request = state.focusSetData.get(setButton.dataset.focusKey || '');
      if (request) {
        await runFocusAction({ action: 'set', session_date: sessionSelect.value, ...request });
      }
      return;
    }
    const closeButton = event.target.closest('.focus-close-button');
    if (closeButton) {
      await runFocusAction({ action: 'close', id: closeButton.dataset.focusId || '' });
      return;
    }
    const removeButton = event.target.closest('.focus-remove-button');
    if (removeButton) {
      await runFocusAction(
        { action: 'remove', id: removeButton.dataset.focusId || '' },
        'Remove this focus without closing it? Use Close if it is actually done.'
      );
    }
  });
}

async function init() {
  try {
    [state.history] = await Promise.all([loadHistory(), loadFocusData()]);
    if (!state.history.sessions.length) {
      highlightsRoot.innerHTML = '<p>No sessions yet. Upload a transcript first.</p>';
      return;
    }
    renderDropdown();
    sessionSelect.value = state.history.sessions[state.history.sessions.length - 1].date;
    sessionSelect.addEventListener('change', handleSelection);
    attachHighlightsInteractions();
    attachTranscriptInteractions();
    attachSessionActions();
    await handleSelection();
  } catch (error) {
    highlightsRoot.innerHTML = `<p>${error.message}</p>`;
  }
}

init();
