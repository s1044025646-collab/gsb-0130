import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { openDatabase } from '../src/db/schema.js';
import { Repository } from '../src/db/repository.js';
import { createApp, type RunningServer, startServer } from '../src/api/server.js';
import { buildFixtureCorpus } from '../fixtures/corpus.js';

let dir: string;
let server: RunningServer;
let baseUrl: string;

async function request(method: string, path: string, body?: unknown): Promise<{ status: number; json: any }> {
  const response = await fetch(`${baseUrl}${path}`, {
    method,
    headers: { 'content-type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body)
  });
  return { status: response.status, json: await response.json() };
}

before(async () => {
  dir = mkdtempSync(join(tmpdir(), 'hmm-api-'));
  server = await startServer(join(dir, 'api.sqlite'));
  baseUrl = `http://127.0.0.1:${server.port}`;
});

after(async () => {
  await server.close();
  rmSync(dir, { recursive: true, force: true });
});

test('健康检查', async () => {
  const { status, json } = await request('GET', '/health');
  assert.equal(status, 200);
  assert.equal(json.status, 'ok');
});

test('端到端: 导入->划分->训练->标注->评测->比较数据', async () => {
  const corpus = buildFixtureCorpus();
  const imported = await request('POST', '/corpora', corpus);
  assert.equal(imported.status, 201);
  const corpusId = imported.json.corpus.id;

  const repeated = await request('POST', '/corpora', corpus);
  assert.equal(repeated.status, 200);
  assert.equal(repeated.json.reused, true);

  const split = await request('POST', `/corpora/${corpusId}/splits`, { seed: 42, trainRatio: 0.8 });
  assert.equal(split.status, 201);
  const splitId = split.json.splitId;

  const trained = await request('POST', '/models', { splitId, smoothing: 1, unknownStrategy: 'uniform' });
  assert.equal(trained.status, 201);
  const modelId = trained.json.modelId;

  const tagged = await request('POST', `/models/${modelId}/tag`, { tokens: ['我', '爱', '一个新地方'] });
  assert.equal(tagged.status, 200);
  assert.equal(tagged.json.tags.length, 3);
  assert.ok(Array.isArray(tagged.json.steps));

  const report = await request('POST', `/models/${modelId}/reports`, { part: 'validation' });
  assert.equal(report.status, 201);
  assert.ok(Array.isArray(report.json.tagMetrics));
  assert.ok(report.json.confusion);

  const reports = await request('GET', `/reports?modelId=${modelId}&limit=10&offset=0`);
  assert.equal(reports.status, 200);
  assert.ok(reports.json.total >= 1);
});

test('错误码: 非法 JSON、参数错误、未知标签、未知路由', async () => {
  const badJson = await fetch(`${baseUrl}/corpora`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: '{not json'
  });
  assert.equal(badJson.status, 400);
  assert.equal((await badJson.json()).error.code, 'INVALID_JSON');

  const invalidCorpus = await request('POST', '/corpora', { name: '', sentences: [] });
  assert.equal(invalidCorpus.status, 400);
  assert.equal(invalidCorpus.json.error.code, 'INVALID_CORPUS');

  const mismatch = await request('POST', '/corpora', {
    name: 'x',
    sentences: [{ tokens: ['a'], tags: [] }]
  });
  assert.equal(mismatch.json.error.code, 'INVALID_SENTENCE');

  const trained = await request('POST', '/corpora', buildFixtureCorpus());
  const corpusId = trained.json.corpus.id;
  const split = await request('POST', `/corpora/${corpusId}/splits`, { seed: 1 });
  const model = await request('POST', '/models', { splitId: split.json.splitId, smoothing: 1 });
  const badTag = await request('POST', `/models/${model.json.modelId}/tag`, {
    tokens: ['x'],
    allowedTags: ['不存在']
  });
  assert.equal(badTag.status, 422);
  assert.equal(badTag.json.error.code, 'TAG_NOT_IN_MODEL');

  const missing = await request('GET', '/corpora/99999');
  assert.equal(missing.status, 404);

  const route = await request('GET', '/nope');
  assert.equal(route.status, 404);
});

test('分页参数校验', async () => {
  const response = await request('GET', '/corpora?limit=0');
  assert.equal(response.status, 400);
  assert.equal(response.json.error.code, 'INVALID_QUERY');
});

test('createApp 可直接注入仓库', async () => {
  const db = openDatabase(':memory:');
  const app = createApp(new Repository(db));
  await new Promise<void>((resolve) => app.listen(0, '127.0.0.1', () => {
    const address = app.address();
    const port = typeof address === 'object' && address ? address.port : 0;
    fetch(`http://127.0.0.1:${port}/health`).then((response) => {
      assert.equal(response.status, 200);
      app.close(() => {
        db.close();
        resolve();
      });
    });
  }));
});
