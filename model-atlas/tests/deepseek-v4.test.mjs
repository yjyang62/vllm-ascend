import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import test from "node:test";
import katex from "katex";

function readDeepSeek() {
  const folder = new URL("../app/models/deepseek-v4/", import.meta.url);
  const files = readdirSync(folder).filter(file => /\.(ts|tsx)$/.test(file));
  return files.map(file => readFileSync(new URL(file, folder), "utf8")).join("\n");
}

test("DeepSeek-V4 registry entry and hybrid attention variants are wired", () => {
  const index = readFileSync(new URL("../app/models/index.ts", import.meta.url), "utf8");
  const source = readDeepSeek();
  assert.match(index, /deepseekV4/);
  assert.match(index, /createModelRegistry\(\[deepseekV4, minimaxM3\]\)/);
  assert.match(source, /id: "deepseek-v4"/);
  assert.match(source, /name: "DeepSeek-V4-Flash"/);
  for (const type of ["swa", "csa", "hca"]) {
    assert.match(source, new RegExp(`LayerType = "swa" \\| "csa" \\| "hca"|onChange\\(\"${type}\"\\)|\"${type}\"`));
  }
  assert.match(source, /compress_ratio = 0/);
  assert.match(source, /Compress-4 \+ Lightning Indexer/);
  assert.match(source, /Compress-128/);
  assert.match(source, /index_topk/);
  assert.match(source, /npu_quant_lightning_indexer_v2/);
  assert.match(source, /npu_hc_pre_v2/);
  assert.match(source, /Indexer KV 独立/);
});

test("DeepSeek-V4 decoder graph endpoints match rendered nodes", () => {
  const source = readDeepSeek();
  const graphNodes = new Set(
    [...source.matchAll(/(?:graphId|inputGraphId|weightGraphId|data-graph-id)="((?:main|moe|attn)-[^"]+)"/g)].map((match) => match[1]),
  );
  const graphEndpoints = new Set(
    [...source.matchAll(/(?:from|to):"((?:main|moe|attn)-[^"]+)"/g)].map((match) => match[1]),
  );
  assert.deepEqual([...graphNodes].filter((id) => !graphEndpoints.has(id)), [], "every rendered graph node needs an edge");
  assert.deepEqual([...graphEndpoints].filter((id) => !graphNodes.has(id)), [], "every graph edge needs rendered endpoints");
});

test("DeepSeek-V4 formulas are valid LaTeX", () => {
  const source = readDeepSeek();
  const equations = [...source.matchAll(/String\.raw`([^`]*)`/g)].map((match) => match[1]);
  assert.ok(equations.length >= 20, `expected DSV4 formulas, got ${equations.length}`);
  for (const equation of equations) {
    assert.doesNotThrow(() => katex.renderToString(equation, { throwOnError: true }));
  }
});
