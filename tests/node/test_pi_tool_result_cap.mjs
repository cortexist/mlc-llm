import test from 'node:test';
import assert from 'node:assert/strict';
import extension, { SHARE, capText, capToolResult, estimateTokens }
  from '../../deploy/xavier/pi-tool-result-cap.ts';

const lines = (from, count) =>
  Array.from({ length: count }, (_, i) => `line ${from + i} of the output`).join('\n') + '\n';

test('the estimate counts digits and marks one by one and words by their length', () => {
  assert.equal(estimateTokens('7001\n'), 5);
  assert.equal(estimateTokens('hello world'), 4);
  assert.equal(estimateTokens('a    b\tc'), 4);
  assert.equal(estimateTokens('{"key": [1, 22]}'), 12);
  assert.equal(estimateTokens('上下文😀'), 4);
  assert.equal(estimateTokens(''), 0);
});

test('a text that fits is returned as it is', () => {
  const text = lines(1, 20);
  assert.equal(capText(text, estimateTokens(text)), text);
});

test('of a long text whole lines of its beginning and end stay around a note', () => {
  const text = lines(1, 2000);
  const capped = capText(text, 1000);
  assert.ok(estimateTokens(capped) <= 1000);
  const [head, rest] = capped.split('[... ');
  const [note, tail] = rest.split(' ...]\n');
  assert.ok(head.startsWith('line 1 of the output\n') && head.endsWith('\n'));
  assert.ok(tail.endsWith('line 2000 of the output\n') && tail.startsWith('line '));
  const [, left, characters] = note.match(/^(\d+) lines \((\d+) characters\) left out here/);
  const kept = head.split('\n').length - 1 + tail.split('\n').length - 1;
  assert.equal(kept + Number(left), 2000);
  assert.equal(head.length + tail.length + Number(characters), text.length);
  // Both parts have their half.
  assert.ok(estimateTokens(head) > 350 && estimateTokens(tail) > 350);
  assert.equal(text.slice(0, head.length), head);
  assert.equal(text.slice(-tail.length), tail);
});

test('a single long line is cut inside the line', () => {
  const text = '0123456789'.repeat(5000);
  const capped = capText(text, 500);
  assert.ok(estimateTokens(capped) <= 500);
  assert.ok(capped.startsWith('0123456789') && capped.endsWith('0123456789'));
  assert.match(capped, /\n\[\.\.\. 0 lines \(49\d\d\d characters\) left out here/);
});

test('a budget below the note leaves the note', () => {
  assert.match(capText(lines(1, 100), 10), /^\[\.\.\. 100 lines /);
});

test('a result is cut to its share of the context window, text blocks by their size', () => {
  const digits = Array.from({ length: 2000 }, (_, i) => String(7001 + i)).join('\n') + '\n';
  assert.equal(capToolResult([{ type: 'text', text: digits }], 200000), null);
  assert.equal(capToolResult([{ type: 'text', text: lines(1, 50) }], 8192), null);
  assert.equal(capToolResult([{ type: 'image', data: 'x'.repeat(100000) }], 8192), null);
  const [one] = capToolResult([{ type: 'text', text: digits }], 8192);
  assert.ok(estimateTokens(one.text) <= 8192 * SHARE);
  assert.ok(estimateTokens(one.text) > 8192 * SHARE - 100);
  assert.ok(one.text.startsWith('7001\n') && one.text.endsWith('9000\n'));
  const image = { type: 'image', data: 'abc', mimeType: 'image/png' };
  const [large, same, small] = capToolResult(
    [{ type: 'text', text: digits }, image, { type: 'text', text: lines(1, 200) }], 8192);
  assert.equal(same, image);
  assert.ok(estimateTokens(large.text) + estimateTokens(small.text) <= 8192 * SHARE);
  assert.ok(estimateTokens(large.text) > 3 * estimateTokens(small.text));
  assert.match(small.text, /left out here/);
});

test('the extension cuts results for the active model only when they are too long', async () => {
  let handler;
  extension({ on: (name, fn) => { assert.equal(name, 'tool_result'); handler = fn; } });
  const content = [{ type: 'text', text: lines(1, 2000) }];
  const small = { model: { contextWindow: 8192 } };
  const result = await handler({ content }, small);
  assert.ok(estimateTokens(result.content[0].text) <= 8192 * SHARE);
  assert.deepEqual(Object.keys(result), ['content']);
  assert.equal(await handler({ content }, { model: { contextWindow: 200000 } }), undefined);
  assert.equal(await handler({ content }, {}), undefined);
  assert.equal(await handler({ content, parentToolCallId: 'call-1' }, small), undefined);
  assert.equal(await handler({ content: [{ type: 'text', text: 'ok' }] }, small), undefined);
});
