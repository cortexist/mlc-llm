/** Optional semantic evidence routing through a resident d1 decision worker. */
import { readFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { join } from 'node:path';

export const QUESTIONS = {
  route: {
    type: 'choice',
    instructions: 'Classify the user intent. Recorded local facts use workspace. Only current deployment metadata uses inventory. General explanation/specifications use general. Requests to do or change something use action.',
    criteria: {
      workspace: 'Look up recorded local facts, measurements, notes, past settings or project history.',
      inventory: 'Inspect the currently deployed local model list, configured context/token limits or serving task types.',
      general: 'Answer general knowledge, native model specifications, abstract capabilities or standard software how-to.',
      action: 'Perform an operation, write code, modify files/settings, install, restart or update something.',
    },
  },
};

export const COVERAGE_QUESTIONS = {
  coverage: {
    type: 'choice',
    instructions: 'Does current deployment data answer the request? context_length is the configured context window/input limit in tokens. Choose search for absent or historical facts.',
    criteria: {
      inventory: 'The request only asks for current model names, task types or configured token limits.',
      search: 'The request needs other facts or historical records absent from the inventory.',
    },
  },
};

function validateAnswer(answer, names) {
  const p = answer?.probabilities;
  if (!p || !names.includes(answer.choice) || Object.keys(p).length !== names.length ||
      names.some(k => !Number.isFinite(p[k]) || p[k] < 0 || p[k] > 1) ||
      Math.abs(names.reduce((sum,k) => sum+p[k],0)-1) > 0.001) {
    throw new Error('Invalid routing probabilities');
  }
  return p;
}

export function coverageAction(answer) {
  validateAnswer(answer, Object.keys(COVERAGE_QUESTIONS.coverage.criteria));
  return answer.choice === 'inventory' ? 'inventory' : 'workspace';
}

export function evidenceAction(answer) {
  const p = validateAnswer(answer, Object.keys(QUESTIONS.route.criteria));
  if (answer.choice === 'workspace' || answer.choice === 'inventory') return answer.choice;
  // Read-only evidence is preferable to missing a plausible local fact. This is
  // an engineering threshold, not a claim that the scores are calibrated accuracy.
  if (answer.choice === 'general' && p.workspace >= 0.15) return 'workspace';
  return 'agent';
}

export class EvidenceRouter {
  constructor({ configPath = process.env.PI_XAVIER_CONFIG || join(homedir(), '.config/pi-xavier.json'),
      fetchImpl = globalThis.fetch, ttlMs = 300000, maxEntries = 128 } = {}) {
    if (!Number.isFinite(ttlMs) || ttlMs < 0 || !Number.isInteger(maxEntries) || maxEntries < 0) {
      throw new RangeError('Router cache limits must be nonnegative');
    }
    this.configPath = configPath;
    this.fetchImpl = fetchImpl;
    this.ttlMs = ttlMs;
    this.maxEntries = maxEntries;
    this.cache = new Map();
  }

  async route(request, environment) {
    if (typeof request !== 'string' || !request.trim() || request.length > 1600 || request.trimStart().startsWith('/')) {
      return { action: 'agent', reason: 'outside_router_scope', elapsed_ms: 0 };
    }
    return this.decide({ environment, request }, QUESTIONS, 'route', evidenceAction, null);
  }

  // Only call after an inventory route. One extra decision, then at most one
  // caller-owned search. No recursion and no extra decision on ordinary turns.
  async inventoryCoverage(request, environment, models) {
    if (!Array.isArray(models) || !models.length || models.length > 16 ||
        models.some(m => typeof m.id !== 'string' || !Number.isInteger(m.context_length))) {
      return { action: 'workspace', reason: 'invalid_inventory', elapsed_ms: 0 };
    }
    const current_deployment = models.map(({id,task,context_length}) =>
      ({id,task:task || 'chat',context_length}));
    return this.decide({environment,request,current_deployment},
      COVERAGE_QUESTIONS, 'coverage', coverageAction, 'workspace');
  }

  async decide(state, questions, name, actionFor, fallback) {
    const start = performance.now();
    const cacheKey = JSON.stringify({name,state});
    const cached = this.cache.get(cacheKey);
    if (cached && Date.now()-cached.at < this.ttlMs) {
      this.cache.delete(cacheKey); this.cache.set(cacheKey,cached);
      return { ...cached.result, cached: true, elapsed_ms: performance.now()-start };
    }
    try {
      const cfg = JSON.parse(await readFile(this.configPath, 'utf8'));
      const base = new URL(cfg.baseUrl);
      if (base.protocol !== 'https:' || base.username || base.password) throw new Error('Invalid routing endpoint');
      const key = (await readFile(cfg.keyFile,'utf8')).trim();
      if (!key) throw new Error('Missing routing credentials');
      const response = await this.fetchImpl(base.href.replace(/\/$/,'')+'/decisions', {
        method: 'POST', headers: { Authorization: 'Bearer '+key, 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: 'd1-omni-2b', state, questions }),
        redirect: 'error', signal: AbortSignal.timeout(5000),
      });
      if (!response.ok) throw new Error('Routing endpoint unavailable');
      const data = await response.json();
      const answer = data.answers?.[name];
      const action = actionFor(answer);
      const result = { action, choice: answer.choice, probabilities: answer.probabilities,
        cached: false, elapsed_ms: performance.now()-start,
        input_tokens: data.usage?.input_tokens,
        bucket_tokens: Number(response.headers.get('X-Xavier-Decision-Bucket-Tokens')) || null };
      this.cache.delete(cacheKey);
      this.cache.set(cacheKey,{at:Date.now(),result});
      while (this.cache.size > this.maxEntries) this.cache.delete(this.cache.keys().next().value);
      return result;
    } catch {
      return { action: fallback, reason: 'router_unavailable', elapsed_ms: performance.now()-start };
    }
  }
}
