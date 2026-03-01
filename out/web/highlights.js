const sessionSelect = document.getElementById('highlight-session-select');
const highlightsRoot = document.getElementById('highlights-root');

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
  if (!value) {
    return '';
  }
  return value
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

function getCategoryCode(error) {
  if (!error) return '';
  if (error.code) return String(error.code).toUpperCase();
  return String(CATEGORY_LABEL_TO_CODE[error.title] || '').toUpperCase();
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

function getPreviousSessionDates(selectedDate, limit = 3) {
  const sessions = state.history?.sessions || [];
  const currentIndex = sessions.findIndex((session) => session.date === selectedDate);
  if (currentIndex <= 0) {
    return [];
  }
  return sessions
    .slice(Math.max(0, currentIndex - limit), currentIndex)
    .map((session) => session.date);
}

async function loadContextBundle(selectedDate) {
  const previousDates = getPreviousSessionDates(selectedDate, 3);
  const [currentAnalysis, ...previousAnalyses] = await Promise.all([
    loadAnalysis(selectedDate),
    ...previousDates.map((date) => loadAnalysis(date)),
  ]);
  return {
    currentAnalysis,
    previousAnalyses,
  };
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

function buildRecurrenceMap(previousAnalyses) {
  const recurrence = {};
  const list = Array.isArray(previousAnalyses) ? previousAnalyses : [];
  list.forEach((analysis) => {
    sortParticipants(analysis.participants || []).forEach((participant) => {
      const types = participant.llm?.annotation_grammar?.error_types || [];
      if (!types.length) {
        return;
      }
      const name = participant.name;
      recurrence[name] = recurrence[name] || {};
      const seenCodes = new Set();
      types.forEach((item) => {
        const code = getCategoryCode(item);
        if (code) {
          seenCodes.add(code);
        }
      });
      seenCodes.forEach((code) => {
        recurrence[name][code] = (recurrence[name][code] || 0) + 1;
      });
    });
  });
  return recurrence;
}

function severityByShare(share) {
  if (share >= 30) {
    return 'high';
  }
  if (share >= 15) {
    return 'medium';
  }
  return 'low';
}

function scoreImpact(code) {
  switch (code) {
    case 'WORD':
    case 'ORDER':
    case 'COLLOC':
      return 3;
    case 'VERB':
    case 'TENSE':
      return 2.5;
    case 'PREP':
    case 'ARTICLE':
      return 2;
    default:
      return 1.5;
  }
}

function scoreFixability(code) {
  switch (code) {
    case 'ARTICLE':
      return 3;
    case 'PREP':
      return 2.5;
    case 'VERB':
    case 'TENSE':
      return 2;
    case 'WORD':
    case 'COLLOC':
    case 'ORDER':
      return 1.5;
    default:
      return 1;
  }
}

function defaultFocusGuidance(code) {
  switch (code) {
    case 'ARTICLE':
      return 'Slow down before singular countable nouns and choose a/an/the on purpose.';
    case 'PREP':
      return 'Keep the phrase, but verify the preposition after common verbs and nouns.';
    case 'VERB':
      return 'Check the verb shape, especially after he/she/it and modal verbs.';
    case 'TENSE':
      return 'Anchor the time first, then match the verb tense to that timeline.';
    case 'WORD':
      return 'Prefer the clearest, most concrete word instead of an approximate phrase.';
    case 'ORDER':
      return 'Say the simpler sentence first, then add extra detail.';
    case 'COLLOC':
      return 'Use standard word pairings instead of literal translations.';
    default:
      return 'Use one simple correction pattern and repeat it until it feels automatic.';
  }
}

function getInsightMap(participant) {
  const map = new Map();
  const sources = [
    ...(participant.llm?.top3_insights || []),
    ...((participant.llm?.practical_recommendations || []).map((item) => ({
      code: getCategoryCode(item),
      title: item.title,
      why: item.why || '',
      focus: item.guidance || '',
      examples: item.examples || [],
      count: Number(item.count || 0),
    }))),
  ];
  sources.forEach((item) => {
    const code = getCategoryCode(item);
    if (!code || map.has(code)) {
      return;
    }
    map.set(code, {
      code,
      title: item.title || code,
      why: item.why || '',
      focus: item.focus || item.guidance || '',
      examples: Array.isArray(item.examples) ? item.examples : [],
    });
  });
  return map;
}

function getTrendBadge(recurrenceCount, lookbackCount) {
  if (!lookbackCount) {
    return null;
  }
  if (recurrenceCount >= Math.min(3, lookbackCount) && recurrenceCount >= 2) {
    return { label: `${recurrenceCount}-session streak`, tone: 'streak' };
  }
  if (recurrenceCount >= 2) {
    return { label: `Seen in ${recurrenceCount} recent sessions`, tone: 'recurring' };
  }
  if (recurrenceCount === 1) {
    return { label: 'Seen recently', tone: 'recurring' };
  }
  return { label: 'New this week', tone: 'new' };
}

function buildFocusWhy(item, share, recurrenceCount, lookbackCount) {
  const base = item.why
    ? String(item.why).trim()
    : `${item.count} cases (${share}% of this session's mapped issues).`;
  const trendBadge = getTrendBadge(recurrenceCount, lookbackCount);
  if (!trendBadge || trendBadge.tone === 'new') {
    return base;
  }
  return `${base} ${trendBadge.label}.`;
}

function buildPracticeTask(code, examples) {
  const model = examples[0]?.correction ? `"${examples[0].correction}"` : '';
  switch (code) {
    case 'ARTICLE':
      return model
        ? `Write 3 short work-related phrases that reuse article patterns like ${model}.`
        : 'Write 3 short work-related phrases and choose a/an/the before each singular noun.';
    case 'PREP':
      return 'Take 3 phrases from your work topics and repeat them with the correct preposition.';
    case 'VERB':
      return 'Say 3 sentences with he/she/it out loud and check the verb each time.';
    case 'TENSE':
      return 'Retell one event from this week in the past, then one plan for next week.';
    case 'WORD':
      return model
        ? `Replace 3 vague phrases with cleaner wording, using models like ${model}.`
        : 'Replace 3 vague phrases from the transcript with simpler, more direct wording.';
    case 'ORDER':
      return 'Answer one question in two short clauses before adding extra detail.';
    case 'COLLOC':
      return 'Repeat 3 fixed phrases from your domain until the pairing sounds automatic.';
    default:
      return 'Create 3 new sentences that reuse this correction pattern.';
  }
}

function buildNextCallTrigger(code) {
  switch (code) {
    case 'ARTICLE':
      return 'Pause briefly before singular countable nouns.';
    case 'PREP':
      return 'Double-check the preposition after common verbs and nouns.';
    case 'VERB':
      return 'Listen for verb agreement after he/she/it.';
    case 'TENSE':
      return 'Name the time first, then choose the tense.';
    case 'WORD':
      return 'If a phrase sounds vague, simplify it before you continue.';
    case 'ORDER':
      return 'Use a shorter sentence shape first, then expand.';
    case 'COLLOC':
      return 'Prefer the phrase you have heard before over a literal translation.';
    default:
      return 'Use one short pause to self-check the sentence before you finish it.';
  }
}

function buildCandidateBadges(candidate, lookbackCount) {
  const badges = [];
  const trendBadge = getTrendBadge(candidate.recurrenceCount, lookbackCount);
  if (trendBadge) {
    badges.push(trendBadge);
  }
  if (candidate.share >= 25) {
    badges.push({ label: 'High share', tone: 'impact' });
  } else if (candidate.fixabilityScore >= 2.5) {
    badges.push({ label: 'Easy to monitor', tone: 'coach' });
  }
  return badges;
}

function buildSummaryMeta(participant) {
  const grammar = participant.llm?.annotation_grammar;
  if (!grammar) {
    return 'Run annotations to build a weekly plan.';
  }
  const totalErrors = Number(grammar.total_errors || 0);
  const errorRate = grammar.error_rate_per_100_words;
  if (totalErrors && errorRate !== undefined) {
    return `${totalErrors} mapped issues / ${errorRate} per 100 words`;
  }
  if (totalErrors) {
    return `${totalErrors} mapped issues`;
  }
  return 'No mapped grammar issues';
}

function buildLead(participant, slots) {
  const primary = slots.find((slot) => slot.key === 'fix_first')?.candidate || slots[0]?.candidate;
  const easyWin = slots.find((slot) => slot.key === 'easy_win')?.candidate;
  if (!primary) {
    return 'Annotations are missing or too noisy for a weekly plan. Open Detailed statistics to inspect the raw analysis.';
  }
  const primaryTitle = String(primary.title || 'this pattern').toLowerCase();
  const easyWinTitle = easyWin && easyWin.code !== primary.code
    ? ` The quickest gain is ${String(easyWin.title || '').toLowerCase()}.`
    : '';
  const recurringText = primary.recurrenceCount
    ? ' It has repeated recently, so it needs deliberate practice.'
    : '';
  return `Correct ${primaryTitle} first before the next call.${easyWinTitle}${recurringText}`;
}

function buildFocusStatusLine(candidate, lookbackCount) {
  const parts = [];
  const trendBadge = getTrendBadge(candidate.recurrenceCount, lookbackCount);
  if (trendBadge) {
    if (trendBadge.tone === 'streak') {
      parts.push(`Recurring for ${candidate.recurrenceCount} sessions`);
    } else if (trendBadge.tone === 'recurring') {
      parts.push(trendBadge.label);
    } else if (trendBadge.tone === 'new') {
      parts.push('New this week');
    }
  }
  if (candidate.share >= 25) {
    parts.push('High share');
  } else if (candidate.fixabilityScore >= 2.5) {
    parts.push('Easy to monitor');
  }
  return parts.join(' / ');
}

function buildImpactText(candidate) {
  let text = String(candidate.why || '').trim();
  text = text.replace(/^\d+\s+cases?\s*\([^)]*\)\.\s*/i, '');
  text = text.replace(/\b\d+-session streak\.\s*$/i, '');
  text = text.replace(/\bSeen in \d+ recent sessions\.\s*$/i, '');
  text = text.replace(/\bSeen recently\.\s*$/i, '');
  text = text.replace(/\bNew this week\.\s*$/i, '');
  text = text.replace(/\s+/g, ' ').trim();
  return text || 'This pattern is worth deliberate practice before the next call.';
}

function buildFocusCandidates(participant, itemsForSpeaker, transcriptText, recurrenceMap, lookbackCount) {
  const grammar = participant.llm?.annotation_grammar;
  const types = grammar?.error_types || [];
  const totalFromTypes = types.reduce((sum, item) => sum + Number(item.count || 0), 0);
  const totalErrors = Number(grammar?.total_errors || totalFromTypes);
  const insightMap = getInsightMap(participant);
  const recurrenceByCode = recurrenceMap?.[participant.name] || {};
  return types
    .map((item) => {
      const code = getCategoryCode(item);
      if (!code) {
        return null;
      }
      const count = Number(item.count || 0);
      const share = totalErrors > 0 ? Math.round((count / totalErrors) * 100) : 0;
      const insight = insightMap.get(code) || {};
      const examples = collectBestExamples(itemsForSpeaker, code, transcriptText, insight.examples || [], 2);
      const confidenceScore = examples.length >= 2 ? 2 : examples.length === 1 ? 1 : 0;
      const recurrenceCount = Number(recurrenceByCode[code] || 0);
      const impactScore = scoreImpact(code);
      const fixabilityScore = scoreFixability(code);
      let priorityScore =
        count * 2
        + share * 0.15
        + recurrenceCount * 3
        + impactScore * 2
        + fixabilityScore * 1.5
        + confidenceScore * 2;
      if (!examples.length) {
        priorityScore -= 3;
      }
      return {
        code,
        title: String(item.title || insight.title || code),
        count,
        share,
        severity: severityByShare(share),
        recurrenceCount,
        impactScore,
        fixabilityScore,
        confidenceScore,
        priorityScore,
        why: buildFocusWhy({ ...item, ...insight, count }, share, recurrenceCount, lookbackCount),
        focus: String(insight.focus || defaultFocusGuidance(code)),
        practiceTask: buildPracticeTask(code, examples),
        nextCallTrigger: buildNextCallTrigger(code),
        examples,
      };
    })
    .filter(Boolean)
    .sort((left, right) => right.priorityScore - left.priorityScore);
}

function pickDistinctCandidate(candidates, usedCodes, scorer) {
  const ranked = [...candidates].sort((left, right) => scorer(right) - scorer(left));
  return ranked.find((candidate) => !usedCodes.has(candidate.code)) || null;
}

function selectFocusSlots(candidates) {
  if (!candidates.length) {
    return [];
  }
  const usedCodes = new Set();
  const fixFirst = candidates[0] || null;
  if (fixFirst) {
    usedCodes.add(fixFirst.code);
  }
  const easyWin = pickDistinctCandidate(
    candidates,
    usedCodes,
    (candidate) => candidate.fixabilityScore * 4 + candidate.confidenceScore * 2 + candidate.count
  );
  if (easyWin) {
    usedCodes.add(easyWin.code);
  }
  const watchNext = pickDistinctCandidate(
    candidates,
    usedCodes,
    (candidate) => candidate.recurrenceCount * 5 + candidate.confidenceScore * 2 + candidate.priorityScore * 0.1
  );
  const slots = [
    {
      key: 'fix_first',
      label: 'Fix First',
      note: 'Biggest drag on this week\'s speaking.',
      candidate: fixFirst,
    },
    {
      key: 'easy_win',
      label: 'Easy Win',
      note: 'Most realistic gain by the next call.',
      candidate: easyWin,
    },
    {
      key: 'watch_next',
      label: 'Watch Next Call',
      note: 'Keep this in mind during live speech.',
      candidate: watchNext,
    },
  ];
  return slots.filter((slot) => slot.candidate);
}

function renderFocusItem(slot, lookbackCount, sessionDate, participantName) {
  const candidate = slot.candidate;
  if (!candidate) {
    return '';
  }
  const slotClass = slot.key.replace(/_/g, '-');
  const exerciseKey = registerExerciseContext(sessionDate, participantName, candidate);
  const statusLine = buildFocusStatusLine(candidate, lookbackCount);
  const impactText = buildImpactText(candidate);
  const examplesHtml = candidate.examples.length
    ? `<ul class="errors compact highlight-examples-list">${candidate.examples.map((example) => `
        <li class="highlight-example-row">
          <span class="example-error">${escapeHtml(example.error)}</span>
          <span class="example-arrow">&rarr;</span>
          <span class="example-fix">${escapeHtml(example.correction)}</span>
          ${renderExampleSource(example, sessionDate)}
        </li>
      `).join('')}</ul>`
    : '<p class="metric-note">Raw examples looked noisy, so they were hidden here. Use Detailed statistics to inspect the full evidence.</p>';
  return `
    <section class="highlight-item slot-${slotClass}">
      <div class="highlight-slot">
        <span class="highlight-slot-label">${escapeHtml(slot.label)}</span>
        <span class="highlight-slot-note">${escapeHtml(slot.note)}</span>
        <span class="highlight-slot-meta">${candidate.count} issues / ${candidate.share}% share</span>
      </div>
      <div class="highlight-body">
        <div class="highlight-item-head">
          <h3>${escapeHtml(candidate.title)}</h3>
          <span class="highlight-item-meta">${escapeHtml(candidate.severity)} priority</span>
        </div>
        ${statusLine ? `<p class="highlight-status-line">${escapeHtml(statusLine)}</p>` : ''}
        <div class="highlight-note-card">
          <p class="highlight-note-kicker">Why it matters</p>
          <p class="highlight-note-copy">${escapeHtml(impactText)}</p>
        </div>
        <p class="highlight-focus-cue"><span class="highlight-focus-label">Focus cue</span>${escapeHtml(candidate.focus)}</p>
        ${examplesHtml}
        <div class="highlight-action-card">
          <div class="highlight-action-head">
            <p class="highlight-action-title">Practice this week</p>
            ${renderExerciseTrigger(exerciseKey)}
          </div>
          <p class="highlight-action-copy">${escapeHtml(candidate.practiceTask)}</p>
          ${renderExercisePanel(exerciseKey)}
        </div>
        <div class="highlight-trigger-strip">
          <span class="highlight-trigger-label">Next call trigger</span>
          <span class="highlight-trigger-copy">${escapeHtml(candidate.nextCallTrigger)}</span>
        </div>
      </div>
    </section>
  `;
}

function buildWeeklyPlan(slots) {
  return slots.map((slot) => {
    const title = String(slot.candidate?.title || 'this pattern').toLowerCase();
    if (slot.key === 'fix_first') {
      return `Write 3 fresh sentences that deliberately fix ${title}.`;
    }
    if (slot.key === 'easy_win') {
      return `Do a 2-minute self-check for ${title} right before the next call.`;
    }
    return `During one long answer next call, pause once and check ${title} live.`;
  });
}

function renderWeeklyPlan(slots) {
  const items = buildWeeklyPlan(slots);
  if (!items.length) {
    return '';
  }
  return `
    <section class="highlight-plan">
      <h3 class="highlight-section-title">This Week Plan</h3>
      <ul>${items.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ul>
    </section>
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
    const response = await fetch('/api/highlight-exercise', {
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
  const baseHref = example.seek !== null && example.seek !== undefined
    ? `index.html?date=${encodeURIComponent(sessionDate)}&seek=${example.seek}${example.seekEnd !== null && example.seekEnd !== undefined ? `&seek_end=${example.seekEnd}` : ''}#transcript-rows`
    : `index.html?date=${encodeURIComponent(sessionDate)}#transcript-rows`;
  return `
    <div class="example-source-row">
      <div class="example-source-text">${escapeHtml(example.context)}</div>
      <a class="example-source" href="${baseHref}">Open full sentence</a>
    </div>
  `;
}

function renderErrorMap(participant, itemsForSpeaker, transcriptText, candidates, sessionDate) {
  const grammar = participant.llm?.annotation_grammar;
  const types = grammar?.error_types || [];
  if (!types.length) {
    return '<p class="metric-note">No mapped grammar categories yet. Run annotations to build the error map.</p>';
  }
  const totalFromTypes = types.reduce((sum, item) => sum + Number(item.count || 0), 0);
  const totalErrors = Number(grammar?.total_errors || totalFromTypes);
  const candidateByCode = new Map((candidates || []).map((candidate) => [candidate.code, candidate]));
  const rows = types.slice(0, 6).map((item) => {
    const title = String(item.title || 'Other');
    const count = Number(item.count || 0);
    const code = getCategoryCode(item);
    const share = totalErrors > 0 ? Math.round((count / totalErrors) * 100) : 0;
    const severity = severityByShare(share);
    const barWidth = Math.max(8, Math.min(100, share));
    const fallbackExamples = candidateByCode.get(code)?.examples || [];
    const examples = collectBestExamples(itemsForSpeaker, code, transcriptText, fallbackExamples, 2);
    const examplesHtml = examples.length
      ? examples.map((example) => `
          <li>
            <span class="example-error">${escapeHtml(example.error)}</span>
            <span class="example-arrow">&rarr;</span>
            <span class="example-fix">${escapeHtml(example.correction)}</span>
            ${renderExampleSource(example, sessionDate)}
          </li>
        `).join('')
      : '<li class="metric-note">No trusted examples shown for this category.</li>';

    return `
      <article class="error-map-item severity-${severity}">
        <div class="error-map-row">
          <h4>${escapeHtml(title)}</h4>
          <div class="error-map-meta">
            <span>${count} issues</span>
            <span>${share}%</span>
          </div>
        </div>
        <div class="error-bar"><span style="width:${barWidth}%"></span></div>
        <ul class="errors compact error-map-examples">${examplesHtml}</ul>
      </article>
    `;
  }).join('');

  return `
    <details class="highlight-details">
      <summary>Full error map <span class="highlight-meta">${totalErrors} mapped issues</span></summary>
      <div class="highlight-details-body">
        <div class="error-map-list">${rows}</div>
        <p class="metric-note">Use Detailed statistics for trend charts, the full transcript, and raw annotation evidence.</p>
      </div>
    </details>
  `;
}

function renderHighlights(bundle) {
  highlightsRoot.innerHTML = '';
  state.currentBundle = bundle;
  state.exerciseRequestData.clear();
  const analysis = bundle.currentAnalysis;
  const previousAnalyses = bundle.previousAnalyses || [];
  const lookbackCount = previousAnalyses.length;
  const recurrenceMap = buildRecurrenceMap(previousAnalyses);
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
    const candidates = buildFocusCandidates(participant, itemsForSpeaker, transcriptText, recurrenceMap, lookbackCount);
    const slots = selectFocusSlots(candidates);
    const lead = buildLead(participant, slots);
    const focusHtml = slots.length
      ? slots.map((slot) => renderFocusItem(slot, lookbackCount, analysis.date, participant.name)).join('')
      : '<p class="metric-note">No strong focus blocks yet. Run annotations or inspect the detailed page for raw evidence.</p>';
    const weeklyPlanHtml = renderWeeklyPlan(slots);
    const errorMapHtml = renderErrorMap(participant, itemsForSpeaker, transcriptText, candidates, analysis.date);
    const summary = buildSummaryMeta(participant);

    card.innerHTML = `
      <div class="highlight-header">
        <h2>${escapeHtml(participant.name)}</h2>
        <span class="highlight-meta">${escapeHtml(summary)}</span>
      </div>
      <p class="highlight-lead">${escapeHtml(lead)}</p>
      <h3 class="highlight-section-title">This Week Focus</h3>
      ${focusHtml}
      ${weeklyPlanHtml}
      ${errorMapHtml}
      <a class="highlight-link" href="index.html?date=${encodeURIComponent(analysis.date)}">Open full analysis</a>
    `;
    highlightsRoot.appendChild(card);
  });
}

async function handleSelection() {
  const date = sessionSelect.value;
  if (!date) {
    highlightsRoot.innerHTML = '<p>No sessions yet.</p>';
    return;
  }
  highlightsRoot.innerHTML = '<p class="metric-note">Loading weekly highlights...</p>';
  const bundle = await loadContextBundle(date);
  renderHighlights(bundle);
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
    }
  });
}

async function init() {
  try {
    state.history = await loadHistory();
    if (!state.history.sessions.length) {
      highlightsRoot.innerHTML = '<p>No sessions yet. Upload a transcript first.</p>';
      return;
    }
    renderDropdown();
    sessionSelect.value = state.history.sessions[state.history.sessions.length - 1].date;
    sessionSelect.addEventListener('change', handleSelection);
    attachHighlightsInteractions();
    await handleSelection();
  } catch (error) {
    highlightsRoot.innerHTML = `<p>${error.message}</p>`;
  }
}

init();
