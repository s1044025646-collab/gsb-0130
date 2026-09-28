import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import type { Server } from 'node:http';
import { openDatabase } from '../src/db/database.js';
import { closeDatabase } from '../src/db/database.js';
import { createApiServer, listenLocal } from '../src/api/server.js';
import { buildCorpus } from '../fixtures/corpus.js';

let server: Server;
let base: string;
let dbPath: string;
let dir: string;
let db: ReturnType<typeof openDatabase>;

async function call(
  method: string,
  path: string,
  body?: unknown
): Promise<{ status: number; json: any }> {
  const res = await fetch(base + path, {
    method,
    headers: { 'content-type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const json = await res.json();
  return { status: res.status, json };
}

test.before(async () => {
  dir = mkdtempSync(join(tmpdir(), 'zhpos-api-'));
  dbPath = join(dir, 'api.sqlite');
  db = openDatabase(dbPath);
  server = createApiServer(db);
  const { port } = await listenLocal(server);
  base = `http://127.0.0.1:${port}`;
});

test.after(async () => {
  await new Promise<void>((res, rej) => server.close((e) => (e ? rej(e) : res())));
  closeDatabase(db);
  rmSync(dir, { recursive: true, force: true });
});

test('健康检查', async () => {
  const { status, json } = await call('GET', '/health');
  assert.equal(status, 200);
  assert.equal(json.status, 'ok');
});

test('端到端：建库→导入→划分→训练→标注→评测→比较', async () => {
  const created = await call('POST', '/api/corpora', { name: 'api-c' });
  assert.equal(created.status, 201);
  const corpusId = created.json.id;

  const imp1 = await call('POST', `/api/corpora/${corpusId}/import`, { sentences: buildCorpus() });
  assert.ok([200, 201].includes(imp1.status));
  const versionId = imp1.json.versionId;

  // 幂等再导入
  const imp2 = await call('POST', `/api/corpora/${corpusId}/import`, { sentences: buildCorpus() });
  assert.equal(imp2.status, 200);
  assert.equal(imp2.json.reused, true);

  const split = await call('POST', `/api/versions/${versionId}/split`, { seed: 11, valRatio: 0.3 });
  assert.equal(split.status, 200);
  assert.ok(split.json.trainSentences > 0);
  assert.ok(split.json.valSentences > 0);

  const train = await call('POST', '/api/models/train', {
    versionId,
    seed: 11,
    valRatio: 0.3,
    alpha: 1,
    unknownStrategy: 'laplace',
  });
  assert.equal(train.status, 201);
  const modelId = train.json.modelId;

  const tag = await call('POST', '/api/tag', { modelId, tokens: ['三', '本', '书'] });
  assert.equal(tag.status, 200);
  assert.ok(Array.isArray(tag.json.tags));
  assert.equal(tag.json.tags.length, 3);

  const ev = await call('POST', `/api/models/${modelId}/evaluate`, {});
  assert.equal(ev.status, 200);
  assert.ok(ev.json.report.perTag.length > 0);
  assert.ok(ev.json.report.tags.includes('X'));
  assert.ok(typeof ev.json.report.macroF1 === 'number');

  // 第二个模型（closed 策略）用于比较
  const train2 = await call('POST', '/api/models/train', {
    versionId,
    seed: 11,
    valRatio: 0.3,
    alpha: 0.5,
    unknownStrategy: 'closed',
  });
  const cmp = await call('POST', '/api/models/compare', {
    modelIds: [modelId, train2.json.modelId],
  });
  assert.equal(cmp.status, 200);
  assert.equal(cmp.json.data.length, 2);

  // 报告分页
  const reports = await call('GET', `/api/models/${modelId}/reports?limit=1&offset=0`);
  assert.equal(reports.status, 200);
  assert.equal(reports.json.limit, 1);
  assert.ok(reports.json.total >= 1);

  // 句子分页
  const sentences = await call('GET', `/api/versions/${versionId}/sentences?limit=5&offset=0`);
  assert.equal(sentences.status, 200);
  assert.equal(sentences.json.data.length, 5);
  assert.ok(sentences.json.total >= sentences.json.data.length);
});

test('参数校验与明确错误码', async () => {
  const bad = await call('POST', '/api/corpora', { name: '' });
  assert.equal(bad.status, 400);
  assert.equal(bad.json.error.code, 'VALIDATION');

  const notFound = await call('GET', '/api/models/99999');
  assert.equal(notFound.status, 404);
  assert.equal(notFound.json.error.code, 'NOT_FOUND');

  const created = await call('POST', '/api/corpora', { name: 'err' });
  const badLen = await call('POST', `/api/corpora/${created.json.id}/import`, {
    sentences: [{ tokens: ['a'], tags: [] }],
  });
  assert.equal(badLen.status, 400);

  // 非法 JSON
  const res = await fetch(base + '/api/tag', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: '{not json',
  });
  const json = await res.json();
  assert.equal(res.status, 400);
  assert.equal(json.error.code, 'VALIDATION');
});

test('不允许的标签在 API 层返回 400 BAD_TAG', async () => {
  const created = await call('POST', '/api/corpora', { name: 'tag-err' });
  const imp = await call('POST', `/api/corpora/${created.json.id}/import`, {
    sentences: buildCorpus(),
  });
  const versionId = imp.json.versionId;
  await call('POST', `/api/versions/${versionId}/split`, { seed: 3, valRatio: 0.2 });
  const tr = await call('POST', '/api/models/train', {
    versionId,
    seed: 3,
    valRatio: 0.2,
  });
  const r = await call('POST', '/api/tag', {
    modelId: tr.json.modelId,
    tokens: ['他'],
    allowedTags: ['不存在'],
  });
  assert.equal(r.status, 400);
  assert.equal(r.json.error.code, 'BAD_TAG');
});
