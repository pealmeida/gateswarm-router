/**
 * GateSwarm MoMA Router — Retraining Pipeline (v0.5.2 rewrite)
 *
 * The previous pipeline tried to grid-search ENSEMBLE WEIGHTS, but:
 *   - simulateAccuracy() ignored the candidate weights entirely (it just counted
 *     predicted===actual), so the search was a no-op that returned the first
 *     candidate after a stable sort;
 *   - the feedback store never persisted the prompt or component scores, so
 *     re-running the ensemble under different weights was impossible; and
 *   - cascade is permanently disabled, so perturbing its weight only corrupted
 *     the saved config.
 *
 * What we CAN learn from real feedback is the right place to put the TIER
 * BOUNDARIES: each judged interaction gives us (routing score, LLM-judged
 * actual tier). Recalibrating the 5 cut points against those pairs is a genuine,
 * data-driven self-improvement — exactly the manual calibration in eval/, run
 * continuously. Boundaries are config-driven (see v04-config.syncTierBoundaries),
 * so an update takes effect live.
 */

import type { EffortLevel } from './types.js';
import { getConfig, saveConfig, type EnsembleWeightsConfig } from './v04-config.js';
import { getFeedbackEntries } from './feedback-store.js';
import { setTierBoundaries, getTierBoundaries } from './intent-engine.js';

const TIERS: EffortLevel[] = ['trivial', 'light', 'moderate', 'heavy', 'intensive', 'extreme'];

// ─── Weights (kept for API compatibility; weights are no longer the train target) ──

let _activeWeights: EnsembleWeightsConfig | null = null;

export function getActiveWeights(): EnsembleWeightsConfig {
  if (_activeWeights) return _activeWeights;
  return getConfig().ensemble.weights;
}

export function setWeights(weights: EnsembleWeightsConfig): void {
  _activeWeights = weights;
}

// ─── Boundary optimisation ────────────────────────────────────────

interface LabeledScore { score: number; tier: number; }

/** Exact tier accuracy of a boundary set against labeled (score → tier) pairs. */
function accuracyFor(bounds: number[], data: LabeledScore[]): number {
  if (data.length === 0) return 0;
  let correct = 0;
  for (const { score, tier } of data) {
    let pred = 0;
    while (pred < bounds.length && score >= bounds[pred]) pred++;
    if (pred === tier) correct++;
  }
  return correct / data.length;
}

/**
 * Find 5 strictly-increasing cut points maximising exact accuracy, via dynamic
 * programming over the candidate grid. Exact-tier accuracy decomposes per
 * segment (samples of tier t whose score lands in [b_{t-1}, b_t)), so the
 * optimum is computable in O(tiers × grid²) instead of enumerating all 5-tuples
 * (the previous nested grid search was ~10⁷ combinations × N samples — minutes
 * of event-loop blockage when triggered).
 */
export function optimizeBoundaries(data: LabeledScore[]): { bounds: number[]; accuracy: number } {
  const fallback = { bounds: getTierBoundaries(), accuracy: accuracyFor(getTierBoundaries(), data) };
  if (data.length === 0) return fallback;

  // Candidate cut points: 0.01-step grid spanning the plausible score range.
  const grid: number[] = [];
  for (let v = 8; v <= 70; v++) grid.push(v / 100);
  const G = grid.length;
  const K = TIERS.length; // 6 tiers → 5 cuts

  // cnt[t][g] = number of samples with label t and score < grid[g]; cnt[t][G] = all.
  const cnt: number[][] = Array.from({ length: K }, () => new Array(G + 1).fill(0));
  for (const { score, tier } of data) {
    if (tier < 0 || tier >= K) continue;
    for (let g = 0; g < G; g++) {
      if (score < grid[g]) { for (let h = g; h < G; h++) cnt[tier][h]++; break; }
    }
    cnt[tier][G]++;
  }

  // dp[t][g] = best #correct over tiers 0..t when cut t+1 is placed at grid[g]
  // (tier t covers scores in [previous cut, grid[g])).
  const dp: number[][] = Array.from({ length: K - 1 }, () => new Array(G).fill(-1));
  const parent: number[][] = Array.from({ length: K - 1 }, () => new Array(G).fill(-1));
  for (let g = 0; g < G; g++) dp[0][g] = cnt[0][g]; // tier 0: scores < grid[g]
  for (let t = 1; t < K - 1; t++) {
    for (let g = t; g < G; g++) {
      for (let p = t - 1; p < g; p++) {
        if (dp[t - 1][p] < 0) continue;
        const correct = dp[t - 1][p] + (cnt[t][g] - cnt[t][p]);
        if (correct > dp[t][g]) { dp[t][g] = correct; parent[t][g] = p; }
      }
    }
  }

  // Close with the last tier (scores ≥ final cut).
  let bestCorrect = -1;
  let bestLast = -1;
  for (let g = K - 2; g < G; g++) {
    if (dp[K - 2][g] < 0) continue;
    const correct = dp[K - 2][g] + (cnt[K - 1][G] - cnt[K - 1][g]);
    if (correct > bestCorrect) { bestCorrect = correct; bestLast = g; }
  }
  if (bestLast < 0) return fallback;

  const cuts: number[] = new Array(K - 1);
  let g = bestLast;
  for (let t = K - 2; t >= 0; t--) {
    cuts[t] = grid[g];
    if (t > 0) g = parent[t][g];
  }

  const accuracy = bestCorrect / data.length;
  return accuracy > fallback.accuracy ? { bounds: cuts, accuracy } : fallback;
}

// ─── Retraining trigger ───────────────────────────────────────────

export interface RetrainResult {
  retrained: boolean;
  reason?: string;
  accuracyBefore?: number;
  accuracyAfter?: number;
  boundaries?: number[];
}

/**
 * Recalibrate tier boundaries if there is enough judged feedback with scores.
 * Only applies the new boundaries when they beat the current ones by a margin
 * (guards against noise / overfitting to a small sample).
 */
export async function retrainIfNeeded(): Promise<RetrainResult> {
  const config = getConfig();
  const minSamples = config.feedback_loop.minSamplesPerTier;

  // Need judged entries that carry BOTH a routing score and an actual tier.
  const data: LabeledScore[] = getFeedbackEntries()
    .filter(e => e.actualTier !== null && typeof e.score === 'number')
    .map(e => ({ score: e.score as number, tier: TIERS.indexOf(e.actualTier as EffortLevel) }))
    .filter(d => d.tier >= 0);

  // Require a reasonable global sample before touching boundaries.
  if (data.length < Math.max(30, minSamples * 3)) {
    return { retrained: false, reason: `insufficient labeled+scored feedback (${data.length})` };
  }

  const current = getTierBoundaries();
  const accuracyBefore = accuracyFor(current, data);
  const best = optimizeBoundaries(data);

  // Apply only on a meaningful improvement (≥2 percentage points).
  if (best.accuracy < accuracyBefore + 0.02) {
    return { retrained: false, reason: 'no significant improvement', accuracyBefore, accuracyAfter: best.accuracy };
  }

  if (!setTierBoundaries(best.bounds)) {
    return { retrained: false, reason: 'optimizer produced invalid boundaries' };
  }

  // Persist to config so it survives restarts and stays the canonical source.
  const cfg = getConfig();
  cfg.tier_boundaries = {
    trivial:   [0, best.bounds[0]],
    light:     [best.bounds[0], best.bounds[1]],
    moderate:  [best.bounds[1], best.bounds[2]],
    heavy:     [best.bounds[2], best.bounds[3]],
    intensive: [best.bounds[3], best.bounds[4]],
    extreme:   [best.bounds[4], 1],
  };
  await saveConfig(cfg);

  return {
    retrained: true,
    reason: `recalibrated boundaries on ${data.length} labeled samples`,
    accuracyBefore,
    accuracyAfter: best.accuracy,
    boundaries: best.bounds,
  };
}

// ─── Automatic trigger ────────────────────────────────────────────
// Called from the gateway after each recorded interaction. Fire-and-forget,
// with a re-entrancy guard so concurrent requests can't stack retrains.

let _retrainInFlight = false;

export function maybeAutoRetrain(interactionCount: number): void {
  const cfg = getConfig();
  const every = cfg.feedback_loop.retrainAfterInteractions;
  if (_retrainInFlight || every <= 0 || interactionCount === 0 || interactionCount % every !== 0) return;
  _retrainInFlight = true;
  retrainIfNeeded()
    .then(r => {
      if (r.retrained) {
        console.log(`🎓 Auto-retrain: ${r.reason} (${((r.accuracyBefore ?? 0) * 100).toFixed(0)}% → ${((r.accuracyAfter ?? 0) * 100).toFixed(0)}%)`);
      }
    })
    .catch(err => console.error(`❌ Auto-retrain failed: ${err.message}`))
    .finally(() => { _retrainInFlight = false; });
}
