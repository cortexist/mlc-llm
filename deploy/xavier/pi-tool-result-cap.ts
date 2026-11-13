/**
 * Pi extension: one tool result takes at most a share of the model's context window.
 *
 * Pi keeps up to 50 KB or 2,000 lines of a tool's output. A model with an 8,192-token
 * context cannot read that beside its prompt, and compaction does not help, because the
 * result is the newest message: the request is refused and the turn ends. Of a result
 * that is too long, its beginning and its end stay, around a line that says what was
 * left out. Results that fit are not touched, so models with a large context never
 * notice the extension.
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

/**
 * The part of the context window that one tool result may take. The prompt with the tool
 * declarations (three tenths of 8,192 tokens in a full Pi configuration), the summary of
 * a compacted conversation and the answer need the rest.
 */
export const SHARE = 0.45;
/** Tokens kept free for the line that says what was left out; it takes about 50. */
const NOTE = 64;

/**
 * Tokens of the pieces of `text`, estimated without a tokenizer and rather high than
 * low: a word counts a token for every four letters, a run of blanks inside a line none
 * for one blank and one otherwise, and every other character (a digit, a punctuation
 * mark, a line end, a character outside ASCII) one. Returns where each piece ends and
 * the tokens up to there.
 */
function measure(text: string) {
  const ends: number[] = [];
  const sums: number[] = [];
  let sum = 0;
  for (const piece of text.matchAll(/[A-Za-z]+|[ \t]+|[\s\S]/gu)) {
    const [run] = piece;
    const first = run.charCodeAt(0);
    if (first === 32 || first === 9) sum += run.length > 1 ? 1 : 0;
    else if ((first | 32) >= 97 && (first | 32) <= 122) sum += Math.ceil(run.length / 4);
    else sum += 1;
    ends.push(piece.index + run.length);
    sums.push(sum);
  }
  return { ends, sums, total: sum };
}

export function estimateTokens(text: string) {
  return measure(text).total;
}

/** `text` within about `budget` tokens: its beginning and its end around a note. */
export function capText(text: string, budget: number) {
  const { ends, sums, total } = measure(text);
  if (total <= budget) return text;
  const half = Math.max(0, budget - NOTE) / 2;
  // The pieces that fit the first half, and the first piece after which the rest fits
  // the second half.
  let first = 0;
  while (sums[first] <= half) first += 1;
  let last = sums.length - 1;
  while (last > 0 && total - sums[last - 1] <= half) last -= 1;
  let head = first > 0 ? ends[first - 1] : 0;
  let tail = ends[last];
  // Whole lines, where that costs less than half of either part.
  const line = text.lastIndexOf("\n", head - 1) + 1;
  if (head > 0 && line > head / 2) head = line;
  const next = text.indexOf("\n", tail) + 1;
  if (tail > 0 && text[tail - 1] !== "\n" && next > 0 && next - tail < (text.length - tail) / 2) {
    tail = next;
  }
  const left = text.slice(head, tail);
  const note =
    `[... ${left.split("\n").length - 1} lines (${left.length} characters) left out here: ` +
    "the result was too long for this model's context. Ask for a smaller part ...]";
  const gap = head > 0 && text[head - 1] !== "\n" ? "\n" : "";
  return `${text.slice(0, head)}${gap}${note}\n${text.slice(tail)}`;
}

type Block = { type: string; text?: string };

/**
 * The content of a tool result for a model with `contextWindow` tokens, or null when it
 * fits. Text blocks share the budget by their size; other blocks stay as they are.
 */
export function capToolResult<T extends Block>(content: T[], contextWindow: number): T[] | null {
  const budget = Math.floor(contextWindow * SHARE);
  const sizes = content.map((block) =>
    block.type === "text" && typeof block.text === "string" ? estimateTokens(block.text) : 0,
  );
  const total = sizes.reduce((sum, size) => sum + size, 0);
  if (!(budget > 0) || total <= budget) return null;
  return content.map((block, index) =>
    sizes[index]
      ? ({ ...block, text: capText(block.text as string, (budget * sizes[index]) / total) } as T)
      : block,
  );
}

export default function (pi: ExtensionAPI) {
  pi.on("tool_result", async (event, ctx) => {
    // A nested call answers to the tool that made it; that tool's result is cut.
    if (event.parentToolCallId || !ctx.model?.contextWindow) return;
    const content = capToolResult(event.content, ctx.model.contextWindow);
    if (content) return { content };
  });
}
