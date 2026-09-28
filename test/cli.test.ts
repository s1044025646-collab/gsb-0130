import { test, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, writeFileSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { runCli } from '../src/cli/index.js';
import { buildFixtureCorpus } from '../fixtures/corpus.js';

let dir: string;
let dbPath: string;
let fixturePath: string;
const originalLog = console.log;
let logs: string[];

function run(argv: string[]): string[] {
  logs = [];
  console.log = (...args: unknown[]) => logs.push(args.join(' '));
  try {
    runCli(argv, dbPath);
  } finally {
    console.log = originalLog;
  }
  return logs;
}

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), 'hmm-cli-'));
  dbPath = join(dir, 'cli.sqlite');
  fixturePath = join(dir, 'corpus.json');
  writeFileSync(fixturePath, JSON.stringify(buildFixtureCorpus()), 'utf-8');
});

afterEach(() => rmSync(dir, { recursive: true, force: true }));

test('CLI 全流程与导出报告', () => {
  assert.match(run(['validate', fixturePath]).join('\n'), /校验通过/);
  assert.match(run(['import', fixturePath]).join('\n'), /已导入新语料版本/);
  assert.match(run(['import', fixturePath]).join('\n'), /复用版本/);
  const splitLog = run(['split', '1', '--seed', '42', '--ratio', '0.8']).join('\n');
  assert.match(splitLog, /训练=/);
  const trainLog = run(['train', '1', '--smoothing', '1', '--unknown', 'uniform']).join('\n');
  assert.match(trainLog, /已训练新模型/);
  const tagLog = run(['tag', '1', '--sentence', '我 爱 一个陌生词']).join('\n');
  assert.match(tagLog, /最佳标签序列/);
  assert.match(tagLog, /未知词位置/);
  const outPath = join(dir, 'report.json');
  const evalLog = run(['evaluate', '1', '--part', 'validation', '--out', outPath]).join('\n');
  assert.match(evalLog, /已导出/);
  run(['train', '1', '--smoothing', '0.1']);
  const compareLog = run(['compare', '1', '2']).join('\n');
  assert.match(compareLog, /1 \|/);
  assert.match(compareLog, /2 \|/);
  const exported = JSON.parse(readFileSync(outPath, 'utf-8'));
  assert.ok(exported.perTag);
  assert.ok(exported.confusion);
});

test('CLI 用法错误以错误码失败', () => {
  assert.throws(
    () => run(['train', 'x']),
    (error: unknown) => ['CLI_USAGE', 'NOT_FOUND'].includes((error as { code?: string }).code ?? '')
  );
});
