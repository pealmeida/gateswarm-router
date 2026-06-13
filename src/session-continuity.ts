/**
 * GateSwarm MoMA Router — Session Continuity (v0.5.3)
 *
 * Tracks per-session summaries across model switches so that when the router
 * changes models between turns, the new model receives the key decisions the
 * previous model produced. Extracted from moma-gateway.ts and fixed:
 *
 *   - Session key: the old fallback was `${agent.id}:${prompt.slice(0,100)}`,
 *     which CHANGES on every turn (the latest user message differs each turn),
 *     so continuity never survived a single exchange unless the client sent an
 *     explicit session_id. The fallback is now keyed on the FIRST user message
 *     of the conversation, which is stable for the lifetime of a chat thread.
 *   - Bounded memory: sessions expire after 1h and the map is capped; the old
 *     map grew without bound (expiry was only checked on read).
 *   - Bounded summaries: the rolling summary is capped instead of growing by
 *     ~300 chars per turn forever.
 */

import { createHash } from 'crypto';

export interface SessionContinuity {
  summary: string;         // model-agnostic rolling summary of the conversation
  lastTier: string;        // tier of the last response
  lastModel: string;       // "provider/model" used for the last response
  keyDecisions: string[];  // important decisions/conclusions
  updatedAt: number;
}

const SESSION_TTL_MS = 3_600_000; // 1 hour of inactivity
const MAX_SESSIONS = 1_000;
const MAX_SUMMARY_CHARS = 2_000;
const MAX_KEY_DECISIONS = 10;

const sessions = new Map<string, SessionContinuity>();

/**
 * Resolve a stable session key for a request.
 * Priority: body.session_id / body.session / X-Session-Id header.
 * Fallback: agent id + hash of the first user message — stable across turns
 * of the same conversation (messages accumulate; the first one doesn't change).
 */
export function resolveSessionId(
  body: any,
  headers: Record<string, string | string[] | undefined>,
  agentId: string,
  messages: any[],
): string {
  const explicit = body?.session_id || body?.session || (headers?.['x-session-id'] as string | undefined);
  if (explicit && typeof explicit === 'string') return `${agentId}:${explicit}`;

  const firstUser = (messages ?? []).find((m: any) => m?.role === 'user');
  let anchor = '';
  if (firstUser) {
    anchor = typeof firstUser.content === 'string' ? firstUser.content : JSON.stringify(firstUser.content ?? '');
  }
  const hash = createHash('sha256').update(anchor).digest('hex').slice(0, 16);
  return `${agentId}:${hash}`;
}

export function getContinuity(sessionId: string): SessionContinuity | null {
  const entry = sessions.get(sessionId);
  if (entry && Date.now() - entry.updatedAt > SESSION_TTL_MS) {
    sessions.delete(sessionId);
    return null;
  }
  return entry ?? null;
}

export function updateContinuity(sessionId: string, tier: string, model: string, responseText: string): void {
  const existing = sessions.get(sessionId);
  const keyDecisions = extractKeyDecisions(responseText);
  const turnLine = `[Turn: ${tier}→${model}] ${responseText.slice(0, 300)}`;
  const summary = existing ? `${existing.summary}\n${turnLine}` : turnLine;

  sessions.set(sessionId, {
    summary: summary.length > MAX_SUMMARY_CHARS ? summary.slice(-MAX_SUMMARY_CHARS) : summary,
    lastTier: tier,
    lastModel: model,
    keyDecisions: existing
      ? [...existing.keyDecisions, ...keyDecisions].slice(-MAX_KEY_DECISIONS)
      : keyDecisions.slice(-MAX_KEY_DECISIONS),
    updatedAt: Date.now(),
  });

  if (sessions.size > MAX_SESSIONS) evictOldest();
}

/**
 * Continuity note to inject when the routed model differs from the one that
 * produced the previous turn. Returns null when there is nothing to carry over
 * or the model did not switch (injecting on every turn pollutes the context).
 */
export function buildContinuityNote(continuity: SessionContinuity | null, currentModel: string): string | null {
  if (!continuity || continuity.keyDecisions.length === 0) return null;
  if (continuity.lastModel === currentModel) return null;
  return (
    `Continuity from previous turn (${continuity.lastTier}→${continuity.lastModel}):\n` +
    continuity.keyDecisions.slice(-3).map(d => `- ${d}`).join('\n')
  );
}

export function extractKeyDecisions(text: string): string[] {
  const decisions: string[] = [];
  const patterns = [
    /(?:decision|conclusion|therefore|resolved|agreed|final)[:\s]*(.+?)(?:\n|$)/gi,
    /(?:the answer is|key point|important|note that)[:\s]*(.+?)(?:\n|$)/gi,
  ];
  for (const pattern of patterns) {
    let match;
    while ((match = pattern.exec(text)) !== null) {
      const d = match[1].trim().slice(0, 150);
      if (d && !decisions.includes(d)) decisions.push(d);
    }
  }
  return decisions;
}

function evictOldest(): void {
  const now = Date.now();
  let oldestKey: string | null = null;
  let oldestAt = Infinity;
  for (const [key, entry] of sessions) {
    if (now - entry.updatedAt > SESSION_TTL_MS) {
      sessions.delete(key);
      continue;
    }
    if (entry.updatedAt < oldestAt) {
      oldestAt = entry.updatedAt;
      oldestKey = key;
    }
  }
  while (sessions.size > MAX_SESSIONS && oldestKey) {
    sessions.delete(oldestKey);
    oldestKey = null;
    oldestAt = Infinity;
    for (const [key, entry] of sessions) {
      if (entry.updatedAt < oldestAt) {
        oldestAt = entry.updatedAt;
        oldestKey = key;
      }
    }
  }
}

/** Drop expired sessions. Called periodically by the gateway. */
export function sweepExpiredSessions(): number {
  const now = Date.now();
  let removed = 0;
  for (const [key, entry] of sessions) {
    if (now - entry.updatedAt > SESSION_TTL_MS) {
      sessions.delete(key);
      removed++;
    }
  }
  return removed;
}

let _sweepInterval: ReturnType<typeof setInterval> | null = null;
export function startContinuitySweep(intervalMs = 600_000): void {
  if (_sweepInterval) return;
  _sweepInterval = setInterval(sweepExpiredSessions, intervalMs);
  _sweepInterval.unref?.();
}

/** Test/diagnostics helpers. */
export function getSessionCount(): number {
  return sessions.size;
}

export function clearSessions(): void {
  sessions.clear();
}
