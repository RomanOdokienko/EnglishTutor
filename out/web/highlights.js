const sessionSelect = document.getElementById('highlight-session-select');
const rebuildMetricsButton = document.getElementById('rebuild-metrics-button');
const rebuildAnnotationsButton = document.getElementById('rebuild-annotations-button');
const deleteButton = document.getElementById('delete-button');
const sessionStatus = document.getElementById('session-status');
const sessionStatsEl = document.getElementById('session-stats');
const sessionTranscriptRows = document.getElementById('session-transcript-rows');
const sessionTranscriptNote = document.getElementById('session-transcript-note');
const transcriptLinkingHint = document.getElementById('transcript-linking-hint');
const participantSwitch = document.getElementById('session-participant-switch');
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
  selectedParticipant: '',
};

const PARTICIPANT_PREFERENCE_KEY = 'ENGLISH_TUTOR_SESSION_PARTICIPANT';

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

// ---- Personal session view model ----
// Every comparison is built from canonical derived metrics in history.json;
// examples still come from the stored transcript annotations. The view model
// keeps ranking and baseline rules separate from DOM rendering.

const CATEGORY_CODES = ['TENSE', 'VERB', 'ARTICLE', 'PREP', 'ORDER', 'WORD', 'COLLOC'];
const CATEGORY_CODE_TO_LABEL = Object.fromEntries(
  Object.entries(CATEGORY_LABEL_TO_CODE).map(([label, code]) => [code, label])
);
const TREND_LOOKBACK = 3;
// A category counts as present in a past session when its density clears this
// floor (about two errors in a 700-word session) on a big-enough sample.
const PERSISTENCE_DENSITY_FLOOR = 0.3;
const PERSISTENCE_MIN_WORDS = 120;

function annotationCategoryCode(item) {
  const raw = String(item?.category_code || item?.category || '').trim();
  return CATEGORY_LABEL_TO_CODE[raw] || raw.toUpperCase();
}

function transcriptAnnotationId(item, allItems) {
  const index = allItems.indexOf(item);
  if (index >= 0) {
    return `transcript-annotation-${index + 1}`;
  }
  return `transcript-annotation-${Number(item?.start || 0)}-${Number(item?.end || 0)}`;
}

function safeDomToken(value) {
  return String(value || '').toLowerCase().replace(/[^a-z0-9_-]+/g, '-').replace(/^-+|-+$/g, '') || 'item';
}

function getPreviousGrammarTrail(selectedDate, participantName) {
  const sessions = state.history?.sessions || [];
  return sessions
    .filter((session) => String(session.date || '') < String(selectedDate || ''))
    .map((session) => {
      const participant = (session.participants || []).find((item) => item.name === participantName);
      return {
        date: session.date,
        annotationsStatus: session.analysis_version?.annotations_status || '',
        words: Number(participant?.derived?.metrics?.english_word_count || 0),
        metrics: participant?.derived?.metrics || null,
        grammar: participant?.derived?.grammar || null,
        density: participant?.derived?.grammar?.by_category_density || null,
      };
    })
    .filter((entry) =>
      entry.grammar
      && entry.metrics
      && entry.words >= PERSISTENCE_MIN_WORDS
      && (!entry.annotationsStatus || entry.annotationsStatus === 'ok'))
    .slice(-TREND_LOOKBACK);
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
  const comparable = trail.filter((entry) => entry.density);

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
        baselineDensity: comparable.length
          ? comparable.reduce((sum, entry) => sum + Number(entry.density[code] || 0), 0) / comparable.length
          : null,
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

function average(values) {
  const numbers = values.filter((value) => Number.isFinite(value));
  return numbers.length ? numbers.reduce((sum, value) => sum + value, 0) / numbers.length : null;
}

function formatNumber(value, maximumFractionDigits = 2) {
  if (!Number.isFinite(Number(value))) {
    return '—';
  }
  return Number(value).toLocaleString('en-US', {
    minimumFractionDigits: 0,
    maximumFractionDigits,
  });
}

function compareLowerIsBetter(current, baseline) {
  if (!Number.isFinite(current) || !Number.isFinite(baseline)) {
    return { delta: null, tone: 'neutral', label: 'Personal baseline not ready' };
  }
  const delta = current - baseline;
  const threshold = Math.max(0.15, Math.abs(baseline) * 0.1);
  if (delta <= -threshold) {
    return { delta, tone: 'positive', label: `${formatNumber(Math.abs(delta))} below usual` };
  }
  if (delta >= threshold) {
    return { delta, tone: 'warning', label: `${formatNumber(delta)} above usual` };
  }
  return { delta, tone: 'neutral', label: 'About your usual' };
}

function buildParticipantViewModel(participant, itemsForSpeaker, transcriptText, selectedDate, allAnnotationItems = []) {
  const derived = participant.derived || {};
  const grammar = derived.grammar || {};
  const metrics = derived.metrics || {};
  const words = Number(metrics.english_word_count || 0);
  const currentIsComparable = words >= PERSISTENCE_MIN_WORDS;
  const history = getPreviousGrammarTrail(selectedDate, participant.name);
  const evidence = buildEvidenceCards(participant, itemsForSpeaker, transcriptText, selectedDate);
  const evidenceByCode = new Map(evidence.map((card) => [card.code, card]));
  const counts = grammar.by_category_count || {};
  const densities = grammar.by_category_density || {};
  const totalErrors = Number(grammar.error_count || 0);
  const activeCodes = new Set(activeFocusesFor(participant.name).map((focus) => focus.category_code));

  const errorRows = CATEGORY_CODES.map((code) => {
    const currentDensity = Number(densities[code] || 0);
    const baselineDensity = average(history.map((entry) => Number(entry.density?.[code] || 0)));
    const comparison = compareLowerIsBetter(currentDensity, currentIsComparable ? baselineDensity : null);
    const count = Number(counts[code] || 0);
    const seenIn = history.filter((entry) => Number(entry.density?.[code] || 0) >= PERSISTENCE_DENSITY_FLOOR).length;
    const findings = itemsForSpeaker
      .filter((item) => annotationCategoryCode(item) === code)
      .map((item) => ({
        ...item,
        _transcriptAnnotationId: transcriptAnnotationId(item, allAnnotationItems),
      }));
    return {
      code,
      title: CATEGORY_CODE_TO_LABEL[code] || code,
      count,
      share: totalErrors > 0 ? Math.round((count / totalErrors) * 100) : 0,
      density: currentDensity,
      baselineDensity,
      comparison,
      recurring: seenIn >= 2,
      inFocus: activeCodes.has(code),
      evidence: evidenceByCode.get(code) || null,
      findings,
    };
  });

  const priorities = errorRows
    .filter((row) => row.count > 0 && row.evidence)
    .sort((left, right) => {
      const leftRegression = Math.max(0, left.comparison.delta || 0);
      const rightRegression = Math.max(0, right.comparison.delta || 0);
      const leftScore = left.density + leftRegression + (left.recurring ? 0.25 : 0) + (left.inFocus ? 0.25 : 0);
      const rightScore = right.density + rightRegression + (right.recurring ? 0.25 : 0) + (right.inFocus ? 0.25 : 0);
      return (rightScore - leftScore) || (right.count - left.count) || left.code.localeCompare(right.code);
    })
    .slice(0, 3)
    .map((row, index) => ({ ...row, rank: index + 1 }));
  const priorityRanks = new Map(priorities.map((row) => [row.code, row.rank]));
  errorRows.forEach((row) => { row.priorityRank = priorityRanks.get(row.code) || null; });

  const overallDensity = Number(grammar.error_density_per_100w || 0);
  const overallBaseline = average(history.map((entry) => Number(entry.grammar?.error_density_per_100w || 0)));
  const fillerRate = Number(metrics.filler_per_100w || 0);
  const fillerBaseline = average(history.map((entry) => Number(entry.metrics?.filler_per_100w || 0)));
  const bestImprovement = currentIsComparable ? errorRows
    .filter((row) => Number.isFinite(row.comparison.delta) && row.comparison.delta < 0)
    .sort((left, right) => left.comparison.delta - right.comparison.delta)[0] || null : null;

  return {
    participant,
    history,
    errorRows,
    priorities,
    overallDensity,
    overallBaseline,
    overallComparison: compareLowerIsBetter(overallDensity, currentIsComparable ? overallBaseline : null),
    fillerRate,
    fillerBaseline,
    fillerComparison: compareLowerIsBetter(fillerRate, currentIsComparable ? fillerBaseline : null),
    bestImprovement,
    words,
  };
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

function renderPracticeItem(card, sessionDate, participantName, priorityRank) {
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
    <details class="practice-item" ${priorityRank === 1 ? 'open' : ''}>
      <summary class="practice-summary">
        <span class="priority-number">${priorityRank}</span>
        <span class="practice-summary-main">
          <strong>${escapeHtml(card.title)}</strong>
          <small>${card.count} ${card.count === 1 ? 'example' : 'examples'} in this session &middot; ${formatNumber(card.density)} errors / 100 words</small>
        </span>
        <span class="practice-summary-action">Practice</span>
      </summary>
      <div class="practice-body">
        <div class="practice-context">
          <span>${escapeHtml(buildEvidenceTrendLine(card))}</span>
          ${focusControl}
        </div>
        ${examplesHtml}
        <div class="highlight-action-card">
          <div class="highlight-action-head">
            <p class="highlight-action-title">Mini exercise</p>
            ${renderExerciseTrigger(exerciseKey)}
          </div>
          ${renderExercisePanel(exerciseKey)}
        </div>
      </div>
    </details>
  `;
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

function renderComparisonBadge(comparison) {
  const tone = comparison?.tone || 'neutral';
  return `<span class="comparison-badge is-${tone}">${escapeHtml(comparison?.label || 'Personal baseline not ready')}</span>`;
}

function renderSessionAnswer(viewModel) {
  const currentIsComparable = viewModel.words >= PERSISTENCE_MIN_WORDS;
  const baselineText = !currentIsComparable
    ? `This sample is below ${PERSISTENCE_MIN_WORDS} English words, so it is not scored against your history.`
    : viewModel.history.length
    ? `Your recent average: ${formatNumber(viewModel.overallBaseline)} / 100 words across ${viewModel.history.length} comparable ${viewModel.history.length === 1 ? 'session' : 'sessions'}.`
    : 'Complete more sessions with at least 120 English words to build your baseline.';
  const best = viewModel.bestImprovement;
  const priority = viewModel.priorities[0];
  const bestHtml = best
    ? `
      <p class="answer-card-kicker">Best progress signal</p>
      <strong class="answer-card-value">${escapeHtml(best.title)}</strong>
      <span class="answer-card-detail">${formatNumber(Math.abs(best.comparison.delta))} fewer errors / 100 words than your recent average.</span>
    `
    : currentIsComparable ? `
      <p class="answer-card-kicker">Best progress signal</p>
      <strong class="answer-card-value">No clear drop yet</strong>
      <span class="answer-card-detail">This session did not beat your recent category averages. That is a useful baseline, not a score.</span>
    ` : `
      <p class="answer-card-kicker">Best progress signal</p>
      <strong class="answer-card-value">Not scored</strong>
      <span class="answer-card-detail">The transcript is useful, but this sample is too short for a fair comparison.</span>
    `;
  const priorityHtml = priority
    ? `
      <p class="answer-card-kicker">What to do next</p>
      <strong class="answer-card-value">Practice ${escapeHtml(priority.title)}</strong>
      <span class="answer-card-detail">${priority.count} examples at ${formatNumber(priority.density)} / 100 words. The plan below starts here.</span>
    `
    : `
      <p class="answer-card-kicker">What to do next</p>
      <strong class="answer-card-value">Review the transcript</strong>
      <span class="answer-card-detail">There is not enough categorized evidence for a practice priority yet.</span>
    `;
  return `
    <div class="session-answer-grid">
      <article class="answer-card">
        <p class="answer-card-kicker">Grammar this session</p>
        <div class="answer-card-value-row">
          <strong class="answer-card-value">${formatNumber(viewModel.overallDensity)} / 100w</strong>
          ${renderComparisonBadge(viewModel.overallComparison)}
        </div>
        <span class="answer-card-detail">${escapeHtml(baselineText)}</span>
      </article>
      <article class="answer-card is-positive">${bestHtml}</article>
      <article class="answer-card is-action">${priorityHtml}</article>
    </div>
  `;
}

function renderProfileFindings(row, participantName) {
  if (!row.findings.length) {
    return '<p class="profile-findings-empty">No transcript findings are linked to this category.</p>';
  }
  const countMatches = row.findings.length === row.count;
  const summary = countMatches
    ? `${row.findings.length} ${row.findings.length === 1 ? 'finding' : 'findings'} counted in this metric`
    : `${row.count} counted cases · ${row.findings.length} transcript findings linked`;
  const findings = row.findings.map((finding, index) => {
    const annotationId = finding._transcriptAnnotationId || '';
    return `
      <li class="profile-finding">
        <div class="profile-finding-main">
          <span class="profile-finding-number">${index + 1}</span>
          <span class="profile-finding-error">${escapeHtml(finding.text || 'Marked phrase')}</span>
          <span class="profile-finding-arrow" aria-hidden="true">→</span>
          <span class="profile-finding-fix">${escapeHtml(finding.correction || 'No suggested rewrite')}</span>
        </div>
        <div class="profile-finding-actions">
          <details class="profile-finding-why">
            <summary>Why?</summary>
            <p>${escapeHtml(finding.explanation || 'No explanation stored.')}</p>
          </details>
          <button class="finding-transcript-link" type="button" data-transcript-annotation-id="${escapeHtml(annotationId)}">Find in transcript</button>
        </div>
      </li>
    `;
  }).join('');
  return `
    <div class="profile-findings-head">
      <strong>${escapeHtml(summary)}</strong>
      <span>These are the raw annotations behind the category total, not only the examples selected for practice.</span>
    </div>
    <ol class="profile-findings-list" aria-label="${escapeHtml(row.title)} findings for ${escapeHtml(participantName)}">${findings}</ol>
  `;
}

function renderErrorProfile(viewModel) {
  const orderedRows = [...viewModel.errorRows].sort((left, right) => {
    if (left.priorityRank && right.priorityRank) return left.priorityRank - right.priorityRank;
    if (left.priorityRank) return -1;
    if (right.priorityRank) return 1;
    return (right.density - left.density) || left.code.localeCompare(right.code);
  });
  const rows = orderedRows.map((row) => {
    const panelId = `profile-findings-${safeDomToken(viewModel.participant.name)}-${safeDomToken(row.code)}`;
    const hasFindings = row.findings.length > 0;
    return `
    <tr class="error-profile-data-row${row.priorityRank ? ' is-priority' : ''}">
      <th scope="row">
        <span class="error-category-title">
          ${row.priorityRank ? `<span class="priority-number">${row.priorityRank}</span>` : '<span class="priority-placeholder"></span>'}
          <span>${escapeHtml(row.title)}</span>
        </span>
        ${row.recurring ? '<small>Recurring in recent sessions</small>' : ''}
        ${hasFindings ? `
          <button class="category-findings-toggle" type="button" aria-expanded="false" aria-controls="${panelId}" data-findings-target="${panelId}">
            View ${row.findings.length} ${row.findings.length === 1 ? 'finding' : 'findings'}
          </button>
        ` : ''}
      </th>
      <td><strong>${row.count}</strong><small>${row.count === 1 ? 'finding' : 'findings'}</small></td>
      <td><strong>${formatNumber(row.density)}</strong><small>errors / 100w</small></td>
      <td><strong>${row.share}%</strong><small>of mapped errors</small></td>
      <td>${renderComparisonBadge(row.comparison)}</td>
    </tr>
    ${hasFindings ? `
      <tr class="category-findings-row" id="${panelId}" hidden>
        <td colspan="5">${renderProfileFindings(row, viewModel.participant.name)}</td>
      </tr>
    ` : ''}
  `;
  }).join('');
  return `
    <section class="error-profile">
      <div class="section-heading-row">
        <div>
          <h3>Your error profile</h3>
          <p>All mapped grammar categories. The numbered top three become the practice plan below.</p>
        </div>
        <span class="info-note" title="Priorities combine current frequency, change versus your own recent average, and whether the pattern recurs.">How priorities work</span>
      </div>
      <div class="error-profile-table-wrap">
        <table class="error-profile-table">
          <thead><tr><th>Category</th><th>Findings</th><th>Rate</th><th>Share</th><th>Vs your usual</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </section>
  `;
}

function renderParticipantReview(viewModel, analysisDate) {
  const participant = viewModel.participant;
  const lowSample = viewModel.words < PERSISTENCE_MIN_WORDS;
  const practiceHtml = viewModel.priorities.length
    ? viewModel.priorities.map((priority) =>
      renderPracticeItem(priority.evidence, analysisDate, participant.name, priority.rank)).join('')
    : '<p class="metric-note">No categorized examples are available for a practice plan.</p>';
  return `
    <article class="highlight-card participant-review">
      <div class="highlight-header participant-review-header">
        <div>
          <p class="participant-review-kicker">Personal session review</p>
          <h2>${escapeHtml(participant.name)}</h2>
        </div>
        <span class="highlight-meta">${viewModel.words} English words &middot; ${viewModel.history.length} comparable prior ${viewModel.history.length === 1 ? 'session' : 'sessions'}</span>
      </div>
      ${lowSample ? `<p class="sample-warning">Short sample: fewer than ${PERSISTENCE_MIN_WORDS} English words. Keep the transcript, but treat comparisons as directional.</p>` : ''}
      ${renderSessionAnswer(viewModel)}
      ${renderFocusBlock(participant.name, analysisDate)}
      ${renderErrorProfile(viewModel)}
      <section class="practice-plan">
        <div class="section-heading-row">
          <div>
            <h3>Practice plan</h3>
            <p>Start with the first category. Open the others when you are ready.</p>
          </div>
          <span class="practice-count">${viewModel.priorities.length} priorities</span>
        </div>
        <div class="practice-list">${practiceHtml}</div>
      </section>
    </article>
  `;
}

function renderHighlights(bundle) {
  highlightsRoot.innerHTML = '';
  state.currentBundle = bundle;
  state.exerciseRequestData.clear();
  state.focusSetData.clear();
  const analysis = bundle.currentAnalysis;
  const participants = selectedParticipantsFor(analysis);
  if (!participants.length) {
    highlightsRoot.innerHTML = '<p>No participant data found for this session.</p>';
    return;
  }
  const transcriptText = analysis.transcript || '';
  const annotationItems = analysis.llm?.annotation_items || [];
  const annotationByName = buildAnnotationMap(transcriptText, annotationItems, analysis.speaker_map || {});

  participants.forEach((participant) => {
    const itemsForSpeaker = annotationByName[participant.name] || [];
    const viewModel = buildParticipantViewModel(
      participant,
      itemsForSpeaker,
      transcriptText,
      analysis.date,
      annotationItems
    );
    highlightsRoot.insertAdjacentHTML('beforeend', renderParticipantReview(viewModel, analysis.date));
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
  {
    label: 'English words',
    get: (d) => Number(d.metrics?.english_word_count),
    unit: '',
    description: 'Sample size used for grammar comparisons.',
  },
  {
    label: 'Words / turn',
    get: (d) => Number(d.metrics?.avg_words_per_turn),
    unit: '',
    description: 'Average length of one speaking turn. Context, not a score.',
  },
  {
    label: 'Speaking share',
    get: (d) => Number(d.metrics?.speaking_share_pct),
    unit: '%',
    description: 'Share of English words in this conversation.',
  },
  {
    label: 'Error density',
    get: (d) => Number(d.grammar?.error_density_per_100w),
    unit: ' /100w',
    lowerIsBetter: true,
    description: 'Mapped grammar errors per 100 English words.',
  },
  {
    label: 'Fillers',
    get: (d) => Number(d.metrics?.filler_per_100w),
    unit: ' /100w',
    lowerIsBetter: true,
    description: 'Um, uh and similar fillers per 100 English words.',
  },
  {
    label: 'Lexical diversity',
    get: (d) => Number(d.metrics?.lexical_diversity_mattr),
    unit: '',
    description: 'Vocabulary variety adjusted for transcript length. Context, not a score.',
  },
];

function renderMetricContext(metric, current, history) {
  if (metric.label === 'English words') {
    return current >= PERSISTENCE_MIN_WORDS ? 'Reliable sample' : 'Short sample';
  }
  const baseline = average(history.map((entry) => metric.get({ metrics: entry.metrics, grammar: entry.grammar })));
  if (!Number.isFinite(baseline)) {
    return 'No personal baseline yet';
  }
  if (metric.lowerIsBetter) {
    return compareLowerIsBetter(current, baseline).label;
  }
  return `Recent avg ${formatNumber(baseline)}${metric.unit}`;
}

function renderSessionStats(analysis) {
  if (!sessionStatsEl) {
    return;
  }
  const participants = selectedParticipantsFor(analysis);
  if (!participants.length) {
    sessionStatsEl.innerHTML = '<p class="metric-note">No participant data.</p>';
    return;
  }
  sessionStatsEl.innerHTML = participants.map((participant) => {
    const derived = participant.derived;
    if (!derived) {
      return `<div class="stat-card"><h3>${escapeHtml(participant.name)}</h3><p class="metric-note">No metrics yet — re-run metrics.</p></div>`;
    }
    const history = getPreviousGrammarTrail(analysis.date, participant.name);
    const rows = SESSION_METRICS.map((metric) => {
      const value = metric.get(derived);
      const shown = Number.isFinite(value) ? `${formatNumber(value)}${metric.unit}` : '—';
      const context = Number.isFinite(value) ? renderMetricContext(metric, value, history) : 'Not available';
      return `
        <div class="stat-row" title="${escapeHtml(metric.description)}">
          <span class="stat-label">${escapeHtml(metric.label)}<small>${escapeHtml(metric.description)}</small></span>
          <span class="stat-value"><b>${shown}</b><small>${escapeHtml(context)}</small></span>
        </div>
      `;
    }).join('');
    return `<div class="stat-card"><h3>${escapeHtml(participant.name)}</h3>${rows}</div>`;
  }).join('');
}

function renderCompactTranscriptIssue(item) {
  const annotationId = item._transcriptAnnotationId || '';
  const annotationNumber = item._transcriptAnnotationNumber || '';
  const original = item.text || 'Marked phrase';
  const correction = item.correction || 'No suggested rewrite';
  const explanation = item.explanation || 'No explanation stored.';
  const categoryCode = annotationCategoryCode(item);
  const categoryLabel = CATEGORY_CODE_TO_LABEL[categoryCode] || item.category || '';
  return `
    <li class="issue-item transcript-compact-issue" tabindex="0" data-transcript-annotation-id="${escapeHtml(annotationId)}"
      aria-controls="${escapeHtml(annotationId)}-text" aria-label="Issue ${annotationNumber}: ${escapeHtml(original)}">
      <div class="transcript-compact-meta">
        <span class="issue-number">${escapeHtml(annotationNumber)}</span>
        ${categoryLabel ? `<span class="issue-category">${escapeHtml(categoryLabel)}</span>` : ''}
      </div>
      <div class="transcript-compact-rewrite">
        <span class="compact-error">${escapeHtml(original)}</span>
        <span class="compact-arrow" aria-hidden="true">→</span>
        <span class="compact-fix">${escapeHtml(correction)}</span>
      </div>
      <details class="issue-explanation-details">
        <summary>Why?</summary>
        <p>${escapeHtml(explanation)}</p>
      </details>
    </li>
  `;
}

function renderSessionTranscript(analysis) {
  if (!sessionTranscriptRows) {
    return;
  }
  state.pinnedTranscriptAnnotationId = '';
  const transcript = analysis.transcript || '';
  const allItems = analysis.llm?.annotation_items || analysis.annotation_items || [];
  const annotationByName = buildAnnotationMap(transcript, allItems, analysis.speaker_map || {});
  const items = state.selectedParticipant === 'both'
    ? allItems
    : (annotationByName[state.selectedParticipant] || []);
  const annotationIds = new Map(items.map((item) => [item, transcriptAnnotationId(item, allItems)]));
  const itemNumbers = new Map(items.map((item, index) => [item, index + 1]));
  const selectedLabel = state.selectedParticipant === 'both' ? 'both participants' : state.selectedParticipant;
  if (transcriptLinkingHint) {
    transcriptLinkingHint.textContent = `Showing issues for ${selectedLabel}. Each paragraph is matched with its corrections; open Why? only when you need the explanation.`;
  }
  if (sessionTranscriptNote) {
    const meta = analysis.llm?.annotations_meta;
    if (meta && meta.total_chunks !== undefined) {
      sessionTranscriptNote.textContent = `${items.length} issues shown · annotated ${meta.chunks_processed}/${meta.total_chunks} chunks.`;
    } else {
      sessionTranscriptNote.textContent = items.length
        ? `${items.length} issues shown.`
        : 'No annotations yet — run annotations to highlight errors.';
    }
  }

  const lines = transcript.split('\n');
  let offset = 0;
  const paragraphRows = [];
  lines.forEach((line) => {
    const lineStart = offset;
    const lineEnd = offset + line.length;
    offset += line.length + 1;
    if (!line.trim()) {
      return;
    }
    const lineItems = items
      .filter((item) => item.start < lineEnd && item.end > lineStart)
      .map((item) => ({
        ...item,
        _transcriptAnnotationId: annotationIds.get(item),
        _transcriptAnnotationNumber: itemNumbers.get(item),
      }));
    const speakerMatch = line.match(/^\s*([^:]+):/);
    const speakerLabel = speakerMatch?.[1]?.trim() || '';
    const speakerName = analysis.speaker_map?.[speakerLabel] || speakerLabel;
    const isContext = state.selectedParticipant !== 'both'
      && speakerName
      && speakerName !== state.selectedParticipant;
    const firstIssues = lineItems.slice(0, 5);
    const extraIssues = lineItems.slice(5);
    const issueList = firstIssues.length
      ? `<ol class="transcript-compact-list">${firstIssues.map(renderCompactTranscriptIssue).join('')}</ol>`
      : '<p class="transcript-no-issues">No mapped issues in this paragraph.</p>';
    const extraList = extraIssues.length
      ? `
        <details class="transcript-more-issues">
          <summary>Show ${extraIssues.length} more ${extraIssues.length === 1 ? 'correction' : 'corrections'}</summary>
          <ol class="transcript-compact-list">${extraIssues.map(renderCompactTranscriptIssue).join('')}</ol>
        </details>
      `
      : '';
    paragraphRows.push(`
      <article class="transcript-paragraph-row${isContext ? ' is-context' : ''}">
        <div class="transcript-paragraph-text">${buildAnnotatedLine(line, lineItems, lineStart)}</div>
        <div class="transcript-paragraph-review">${issueList}${extraList}</div>
      </article>
    `);
  });

  sessionTranscriptRows.innerHTML = `
    <div class="transcript-flow-header" aria-hidden="true">
      <span>Paragraph in context</span>
      <span>Corrections</span>
    </div>
    <div class="transcript-paragraphs">${paragraphRows.join('')}</div>
  `;
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
    const issue = [...matches].find((element) => element.classList.contains('issue-item'));
    const hiddenIssues = issue?.closest('details.transcript-more-issues');
    if (hiddenIssues) {
      hiddenIssues.open = true;
    }
    mark?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    issue?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
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

function readParticipantPreference() {
  try {
    return String(window.localStorage.getItem(PARTICIPANT_PREFERENCE_KEY) || '').trim();
  } catch (error) {
    return '';
  }
}

function writeParticipantPreference(value) {
  try {
    window.localStorage.setItem(PARTICIPANT_PREFERENCE_KEY, value);
  } catch (error) {}
}

function ensureSelectedParticipant(analysis) {
  const participants = sortParticipants(analysis?.participants || []);
  const names = participants.map((participant) => participant.name);
  if (state.selectedParticipant === 'both' || names.includes(state.selectedParticipant)) {
    return;
  }
  const saved = readParticipantPreference();
  state.selectedParticipant = saved === 'both' || names.includes(saved)
    ? saved
    : (names[0] || 'both');
}

function selectedParticipantsFor(analysis) {
  const participants = sortParticipants(analysis?.participants || []);
  ensureSelectedParticipant(analysis);
  if (state.selectedParticipant === 'both') {
    return participants;
  }
  return participants.filter((participant) => participant.name === state.selectedParticipant);
}

function renderParticipantSwitch(analysis) {
  if (!participantSwitch) {
    return;
  }
  const participants = sortParticipants(analysis?.participants || []);
  ensureSelectedParticipant(analysis);
  const options = participants.map((participant) => participant.name).concat('both');
  participantSwitch.innerHTML = options.map((value) => {
    const label = value === 'both' ? 'Both' : value;
    const selected = state.selectedParticipant === value;
    return `
      <button class="participant-switch-button${selected ? ' is-active' : ''}" type="button"
        data-participant-view="${escapeHtml(value)}" aria-pressed="${selected}">${escapeHtml(label)}</button>
    `;
  }).join('');
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
  state.currentBundle = bundle;
  renderParticipantSwitch(bundle.currentAnalysis);
  renderHighlights(bundle);
  renderSessionStats(bundle.currentAnalysis);
  renderSessionTranscript(bundle.currentAnalysis);
}

function attachParticipantSwitch() {
  if (!participantSwitch) {
    return;
  }
  participantSwitch.addEventListener('click', (event) => {
    const button = event.target.closest('[data-participant-view]');
    if (!button || !state.currentBundle) {
      return;
    }
    const next = button.dataset.participantView || '';
    if (!next || next === state.selectedParticipant) {
      return;
    }
    state.selectedParticipant = next;
    writeParticipantPreference(next);
    const analysis = state.currentBundle.currentAnalysis;
    renderParticipantSwitch(analysis);
    renderHighlights(state.currentBundle);
    renderSessionStats(analysis);
    renderSessionTranscript(analysis);
  });
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
    const findingsToggle = event.target.closest('.category-findings-toggle');
    if (findingsToggle) {
      const panel = document.getElementById(findingsToggle.dataset.findingsTarget || '');
      if (panel) {
        const willOpen = panel.hidden;
        panel.hidden = !willOpen;
        findingsToggle.setAttribute('aria-expanded', String(willOpen));
        findingsToggle.closest('.error-profile-data-row')?.classList.toggle('is-expanded', willOpen);
      }
      return;
    }
    const transcriptLink = event.target.closest('.finding-transcript-link');
    if (transcriptLink) {
      const annotationId = transcriptLink.dataset.transcriptAnnotationId || '';
      if (annotationId) {
        if (state.pinnedTranscriptAnnotationId) {
          const previous = state.pinnedTranscriptAnnotationId;
          state.pinnedTranscriptAnnotationId = '';
          clearActiveTranscriptAnnotation(previous);
        }
        state.pinnedTranscriptAnnotationId = annotationId;
        setActiveTranscriptAnnotation(annotationId, { scroll: true });
      }
      return;
    }
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
    attachParticipantSwitch();
    attachHighlightsInteractions();
    attachTranscriptInteractions();
    attachSessionActions();
    await handleSelection();
  } catch (error) {
    highlightsRoot.innerHTML = `<p>${error.message}</p>`;
  }
}

init();
