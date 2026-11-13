/** Optional semantic evidence routing: resident Nemotron names one of a question's options. */
import { readFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { join } from 'node:path';

export const QUESTIONS = {
  route: {
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
    instructions: 'Does current deployment data answer the request? context_length is the configured context window/input limit in tokens. Choose search for absent or historical facts.',
    criteria: {
      inventory: 'The request only asks for current model names, task types or configured token limits.',
      search: 'The request needs other facts or historical records absent from the inventory.',
    },
  },
};

// The wording the measurements in docs/deploy/xavier_pi_evidence_router.md were taken with.
function prompt(question, {environment, request, current_deployment}) {
  const options = Object.entries(question.criteria).map(([name,text]) => `- ${name}: ${text}`).join('\n');
  const deployment = current_deployment ? `Current deployment: ${JSON.stringify(current_deployment)}\n` : '';
  return `${question.instructions}\n\nOptions:\n${options}\n\nEnvironment: ${environment}\n${deployment}`+
    `Request: ${request}\n\nReply with one word: the option name.`;
}

function validateChoice(choice, question) {
  if (!Object.hasOwn(question.criteria, choice)) throw new Error('Invalid routing choice');
}

export function coverageAction(choice) {
  validateChoice(choice, COVERAGE_QUESTIONS.coverage);
  return choice === 'inventory' ? 'inventory' : 'workspace';
}

export function evidenceAction(choice) {
  validateChoice(choice, QUESTIONS.route);
  return choice === 'workspace' || choice === 'inventory' ? choice : 'agent';
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
      // A scratch request stays out of the two conversations the server keeps for reuse.
      const response = await this.fetchImpl(base.href.replace(/\/$/,'')+'/chat/completions', {
        method: 'POST', headers: { Authorization: 'Bearer '+key, 'Content-Type': 'application/json',
          'X-Xavier-Scratch': '1' },
        body: JSON.stringify({ model: 'nemotron', temperature: 0, max_tokens: 8,
          messages: [{ role: 'user', content: prompt(questions[name], state) }] }),
        redirect: 'error', signal: AbortSignal.timeout(5000),
      });
      if (!response.ok) throw new Error('Routing endpoint unavailable');
      const data = await response.json();
      const choice = String(data.choices?.[0]?.message?.content ?? '').replace(/[*`.\s]/g,'').toLowerCase();
      const action = actionFor(choice);
      const result = { action, choice, cached: false, elapsed_ms: performance.now()-start,
        input_tokens: data.usage?.prompt_tokens };
      this.cache.delete(cacheKey);
      this.cache.set(cacheKey,{at:Date.now(),result});
      while (this.cache.size > this.maxEntries) this.cache.delete(this.cache.keys().next().value);
      return result;
    } catch {
      return { action: fallback, reason: 'router_unavailable', elapsed_ms: performance.now()-start };
    }
  }
}
