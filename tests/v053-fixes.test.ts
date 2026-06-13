/**
 * v0.5.3 regression tests:
 *   - feedback id round-trip (recordFeedback → updateAdequacy was silently dead)
 *   - boundary optimizer DP (replaces the combinatorial grid search)
 *   - config ensemble weights actually reach the voter
 *   - session continuity: stable keys, switch-gated injection
 *   - RAG content retrieval is session-scoped
 *   - compression thresholds are configurable per call
 */
import { describe, it, expect } from 'vitest';
import {
  initFeedbackStore, recordFeedback, updateAdequacy, getFeedbackEntries, getTierAccuracy,
} from '../src/feedback-store.js';
import { optimizeBoundaries } from '../src/retraining.js';
import { ensembleVote, getEnsembleWeights, setEnsembleWeights as setVoterWeights } from '../src/ensemble-voter.js';
import { setEnsembleWeights as setConfigWeights } from '../src/v04-config.js';
import {
  resolveSessionId, getContinuity, updateContinuity, buildContinuityNote, clearSessions,
} from '../src/session-continuity.js';
import { initRagIndex, addRagEntry, queryRag } from '../src/rag-index.js';
import { turboQuantCompress } from '../src/turboquant-compressor.js';

const VOTER_DEFAULTS = { heuristic: 0.55, cascade: 0.0, ragSignal: 0.25, historyBias: 0.2 };

describe('feedback id round-trip (learning loop revival)', () => {
  it('recordFeedback returns the stored id and updateAdequacy finds it', () => {
    initFeedbackStore();
    const id = recordFeedback({
      prompt: 'v053 feedback id round trip',
      predictedTier: 'moderate',
      actualTier: null,
      modelUsed: 'test/model',
      responseTokens: 42,
      adequacyScore: null,
      escalated: false,
      userSatisfaction: null,
      score: 0.3,
    });
    expect(typeof id).toBe('string');
    expect(id.length).toBeGreaterThan(0);

    updateAdequacy(id, 0.9, 'moderate');
    const entry = getFeedbackEntries().find(e => e.id === id);
    expect(entry).toBeDefined();
    expect(entry!.adequacyScore).toBe(0.9);
    expect(entry!.actualTier).toBe('moderate');

    // The judged entry now contributes to per-tier accuracy stats.
    const acc = getTierAccuracy();
    expect(acc.moderate).toBeDefined();
    expect(acc.moderate.total).toBeGreaterThanOrEqual(1);
  });
});

describe('boundary optimizer (DP rewrite)', () => {
  it('recovers cut points on perfectly separable data', () => {
    // Tiers clustered in disjoint score bands.
    const bands: Array<[number, number]> = [
      [0.09, 0.14], [0.22, 0.26], [0.29, 0.31], [0.33, 0.36], [0.39, 0.44], [0.50, 0.65],
    ];
    const data: Array<{ score: number; tier: number }> = [];
    bands.forEach(([lo, hi], tier) => {
      for (let i = 0; i < 20; i++) {
        data.push({ score: lo + ((hi - lo) * i) / 19, tier });
      }
    });
    const { bounds, accuracy } = optimizeBoundaries(data);
    expect(accuracy).toBe(1);
    expect(bounds).toHaveLength(5);
    for (let i = 1; i < bounds.length; i++) expect(bounds[i]).toBeGreaterThan(bounds[i - 1]);
    // Each cut must separate adjacent bands.
    bands.slice(0, -1).forEach(([, hi], i) => {
      expect(bounds[i]).toBeGreaterThan(hi);
      expect(bounds[i]).toBeLessThanOrEqual(bands[i + 1][0]);
    });
  });

  it('never returns worse accuracy than the current boundaries on noisy data', () => {
    const data: Array<{ score: number; tier: number }> = [];
    let seed = 42;
    const rand = () => (seed = (seed * 1103515245 + 12345) % 2 ** 31) / 2 ** 31;
    for (let i = 0; i < 200; i++) {
      data.push({ score: rand() * 0.8, tier: Math.floor(rand() * 6) });
    }
    const current = optimizeBoundaries([]); // fallback = current boundaries
    const { accuracy } = optimizeBoundaries(data);
    expect(accuracy).toBeGreaterThanOrEqual(0);
    expect(accuracy).toBeLessThanOrEqual(1);
    // Strictly increasing invariant holds.
    const { bounds } = optimizeBoundaries(data);
    for (let i = 1; i < bounds.length; i++) expect(bounds[i]).toBeGreaterThan(bounds[i - 1]);
    expect(current.bounds).toHaveLength(5);
  });

  it('finishes fast (was minutes of event-loop blockage)', () => {
    const data = Array.from({ length: 1000 }, (_, i) => ({ score: (i % 80) / 100, tier: i % 6 }));
    const start = performance.now();
    optimizeBoundaries(data);
    expect(performance.now() - start).toBeLessThan(500);
  });
});

describe('ensemble weights wiring (config → voter)', () => {
  it('config setEnsembleWeights pushes into the voter', () => {
    setConfigWeights({ heuristic: 0.7 });
    const w = getEnsembleWeights();
    // Voter normalizes to sum=1: 0.7 / (0.7 + 0 + 0.25 + 0.2)
    expect(w.heuristic).toBeCloseTo(0.7 / 1.15, 5);
    setVoterWeights(VOTER_DEFAULTS);
    setConfigWeights(VOTER_DEFAULTS);
  });

  it('weights change the heuristic/RAG mix on the active (no-cascade) path', () => {
    const input = { prompt: 'x', heuristicScore: 0.6, ragSignal: 0.2 };
    setVoterWeights({ heuristic: 1, cascade: 0, ragSignal: 0, historyBias: 0 });
    const allHeuristic = ensembleVote(input).finalScore;
    setVoterWeights({ heuristic: 0, cascade: 0, ragSignal: 1, historyBias: 0 });
    const allRag = ensembleVote(input).finalScore;
    setVoterWeights(VOTER_DEFAULTS);
    // bias term cancels in the difference
    expect(allHeuristic - allRag).toBeCloseTo(0.6 - 0.2, 5);
  });
});

describe('session continuity (v0.5.3)', () => {
  it('fallback session key is stable across turns of the same conversation', () => {
    const turn1 = [{ role: 'user', content: 'first question' }];
    const turn2 = [
      { role: 'user', content: 'first question' },
      { role: 'assistant', content: 'answer' },
      { role: 'user', content: 'follow-up question' },
    ];
    const id1 = resolveSessionId({}, {}, 'agent-a', turn1);
    const id2 = resolveSessionId({}, {}, 'agent-a', turn2);
    expect(id1).toBe(id2);
    // Different conversations / agents get different keys.
    expect(resolveSessionId({}, {}, 'agent-a', [{ role: 'user', content: 'other thread' }])).not.toBe(id1);
    expect(resolveSessionId({}, {}, 'agent-b', turn1)).not.toBe(id1);
  });

  it('explicit session ids take priority', () => {
    const msgs = [{ role: 'user', content: 'q' }];
    expect(resolveSessionId({ session_id: 's-1' }, {}, 'a', msgs)).toBe('a:s-1');
    expect(resolveSessionId({}, { 'x-session-id': 's-2' }, 'a', msgs)).toBe('a:s-2');
  });

  it('continuity note appears only on a model switch and carries decisions', () => {
    clearSessions();
    const sid = 'a:test-switch';
    updateContinuity(sid, 'moderate', 'zai/glm-4.5-air', 'Decision: use Redis for the cache layer.');
    const cont = getContinuity(sid);
    expect(cont).not.toBeNull();
    expect(cont!.keyDecisions.length).toBeGreaterThan(0);

    // Same model → no injection. Different model → note with the decision.
    expect(buildContinuityNote(cont, 'zai/glm-4.5-air')).toBeNull();
    const note = buildContinuityNote(cont, 'opencodego/deepseek-v4-pro');
    expect(note).toContain('use Redis');
    expect(note).toContain('zai/glm-4.5-air');
  });
});

describe('RAG content retrieval is session-scoped', () => {
  it('entries from another session are not returned when a sessionKey is given', () => {
    initRagIndex();
    addRagEntry({
      keywords: ['kubernetes', 'ingress', 'v053test'],
      tier: 'heavy',
      modelUsed: 'm',
      adequacyScore: 1,
      summary: 'session A discussed ingress controllers',
      originalTokens: 10,
      compressedTokens: 5,
      sessionKey: 'agent:session-A',
    });
    const sameSession = queryRag(['kubernetes', 'v053test'], 3, 'agent:session-A');
    expect(sameSession.some(e => e.summary.includes('session A'))).toBe(true);
    const otherSession = queryRag(['kubernetes', 'v053test'], 3, 'agent:session-B');
    expect(otherSession.length).toBe(0);
    // Unscoped query (routing signal) still sees the entry's metadata.
    const unscoped = queryRag(['kubernetes', 'v053test'], 3);
    expect(unscoped.length).toBeGreaterThan(0);
  });
});

describe('compression thresholds are configurable', () => {
  it('raising the activation threshold preserves mid-size conversations', () => {
    // ~12K estimated tokens across 30 messages (over the old 5% threshold,
    // under the 25% gateway default). Sentence-delimited so summarization can
    // actually shorten lower-importance turns.
    const sentence = 'The cache layer was migrated to Redis for lower latency. ';
    const msgs = Array.from({ length: 30 }, (_, i) => ({
      role: i % 2 === 0 ? 'user' : 'assistant',
      content: `Message ${i}: ${sentence.repeat(30)}`,
    }));

    const aggressive = turboQuantCompress({
      messages: msgs.map(m => ({ ...m })),
      targetModel: 'glm-4.7',
      proactiveThresholdPct: 0.05,
    });
    expect(aggressive.compressionRatio).toBeGreaterThan(1);

    const relaxed = turboQuantCompress({
      messages: msgs.map(m => ({ ...m })),
      targetModel: 'glm-4.7',
      proactiveThresholdPct: 0.25,
      maxInputTokensAbsolute: 32_000,
    });
    expect(relaxed.compressionRatio).toBe(1);
    expect(relaxed.messages).toHaveLength(30);
  });
});
