/** Pi chat uses Nemotron; these tools call the resident decision/embedding APIs. */
import { readFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";
import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ExtensionAPI } from "@earendil-works/pi-coding-agent";

async function request(path: string, payload?: unknown, signal?: AbortSignal) {
  const configFile = process.env.PI_XAVIER_CONFIG || join(homedir(), ".config/pi-xavier.json");
  const config = JSON.parse(await readFile(configFile, "utf8"));
  const base = new URL(config.baseUrl);
  if (base.protocol !== "https:" || base.username || base.password) {
    throw new Error("Pi Xavier baseUrl must use HTTPS without embedded credentials");
  }
  const key = (await readFile(config.keyFile, "utf8")).trim();
  if (!key) throw new Error("Empty Xavier API key");
  const response = await fetch(base.href.replace(/\/$/, "") + path, {
    method: payload === undefined ? "GET" : "POST",
    headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
    body: payload === undefined ? undefined : JSON.stringify(payload),
    signal: AbortSignal.any([AbortSignal.timeout(90000), ...(signal ? [signal] : [])]),
    redirect: "error",
  });
  const body = await response.json();
  if (!response.ok) throw new Error(`Xavier HTTP ${response.status}: ${JSON.stringify(body).slice(0, 600)}`);
  return body;
}

export async function decide(state: string, question: string,
  options?: { name: string; description: string }[], signal?: AbortSignal) {
  if (!state.trim() || !question.trim()) throw new Error("State and question are required");
  if (options && (options.length < 2 || new Set(options.map(o => o.name)).size !== options.length)) {
    throw new Error("Use at least two uniquely named options");
  }
  const q = options ? { type: "choice", instructions: question,
    criteria: Object.fromEntries(options.map(o => [o.name, o.description])) }
    : { type: "noul", instructions: question };
  return request("/decisions", { model: "d1-omni-2b", state, questions: { result: q } }, signal);
}

export async function embed(text: string, signal?: AbortSignal) {
  return request("/embeddings", { model: "embeddinggemma-2", input: text, dimensions: 768 }, signal);
}

export async function rank(query: string, documents: string[], signal?: AbortSignal) {
  if (!query.trim() || documents.length < 1 || documents.length > 15 || documents.some(d => !d.trim())) {
    throw new Error("Provide a nonempty query and 1..15 nonempty short passages");
  }
  const body = await request("/embeddings", { model: "embeddinggemma-2", dimensions: 768,
    input: [`task: search result | query: ${query}`, ...documents.map(d => `title: none | text: ${d}`)] }, signal);
  const data = [...body.data].sort((a, b) => a.index - b.index);
  if (data.length !== documents.length + 1 || data.some(d => d.embedding.length !== 768)) {
    throw new Error("Unexpected embedding response shape");
  }
  const norm = (v: number[]) => Math.sqrt(v.reduce((s, n) => s + n*n, 0));
  const q = data[0].embedding as number[];
  const results = documents.map((text, index) => {
    const v = data[index+1].embedding as number[];
    const score = q.reduce((s, n, i) => s+n*v[i], 0) / (norm(q)*norm(v));
    if (!Number.isFinite(score)) throw new Error("Invalid embedding vector");
    return { index, score, text };
  }).sort((a, b) => b.score-a.score);
  return { model: "embeddinggemma-2", results, usage: body.usage };
}

function result(data: unknown) {
  return { content: [{ type: "text" as const, text: JSON.stringify(data) }], details: data };
}

export default function (pi: ExtensionAPI) {
  pi.registerTool(defineTool({
    name: "xavier_decide", label: "d1 decision",
    description: "Ask resident d1 for a yes/no probability or choose a supplied option. Provide short text evidence and a question; entire formatted input must fit the server capacity reported by /local-models (2048 with the current specialist profile). No audio/images. A probability is not proof.",
    parameters: Type.Object({ state: Type.String(), question: Type.String(),
      options: Type.Optional(Type.Array(Type.Object({ name: Type.String(), description: Type.String() }), { minItems: 2, maxItems: 8 })) }),
    annotations: { readOnlyHint: true, destructiveHint: false },
    async execute(_id, args, signal) {
      return result(await decide(args.state, args.question, args.options, signal));
    },
  }));
  pi.registerTool(defineTool({
    name: "xavier_rank", label: "EmbeddingGemma semantic ranking",
    description: "Rank 1..15 short supplied passages for a query using resident EmbeddingGemma 2 embeddings and cosine similarity. Each input must fit 128 tokens including task prefix. Read local notes first; this tool does not scan files. Similarity is not factual verification.",
    parameters: Type.Object({ query: Type.String(), documents: Type.Array(Type.String(), { minItems: 1, maxItems: 15 }) }),
    annotations: { readOnlyHint: true, destructiveHint: false },
    async execute(_id, args, signal) { return result(await rank(args.query, args.documents, signal)); },
  }));

  function command(name: string, description: string, handler: (args: string) => Promise<unknown>) {
    pi.registerCommand(name, { description, handler: async (args, ctx) => {
      try {
        const data = await handler(args);
        pi.sendMessage({ customType: "xavier-specialist", content: JSON.stringify(data, null, 2), display: true });
      } catch (error) { ctx.ui.notify(error instanceof Error ? error.message : String(error), "error"); }
    } });
  }
  command("local-models", "List resident API models and their capabilities", async () => {
    const body = await request("/models");
    return body.data.map(m => ({ model: m.id, task: m.task || "chat", context: m.context_length,
      use: m.id === "d1-omni-2b" ? "/decide or xavier_decide" : m.id === "embeddinggemma-2" ? "/embed, /rank or xavier_rank" : "Pi chat (/model)" }));
  });
  command("decide", "d1 yes/no: /decide question | short evidence", async args => {
    const separator = args.indexOf("|");
    if (separator < 0) throw new Error("Usage: /decide Is this a network issue? | DNS lookup failed.");
    return decide(args.slice(separator+1).trim(), args.slice(0, separator).trim());
  });
  command("rank", 'Rank passages: /rank {"query":"...","documents":["...","..."]}', async args => {
    const input = JSON.parse(args);
    return rank(input.query, input.documents);
  });
  command("embed", "Embed short text; show vector dimensions, norm and a preview", async args => {
    if (!args.trim()) throw new Error("Usage: /embed short text");
    const body = await embed(args.trim());
    const vector = body.data[0].embedding as number[];
    return { model: body.model, dimensions: vector.length, norm: Math.sqrt(vector.reduce((s, n) => s+n*n, 0)), preview: vector.slice(0, 8), usage: body.usage };
  });
}
