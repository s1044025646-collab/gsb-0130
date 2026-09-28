import { test, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { openDatabase } from '../src/db/schema.js';
import { Repository, type Clock } from '../src/db/repository.js';
import type { TaggedSentence } from '../src/core/types.js';
import { viterbi } from '../src/core/hmm.js';
import { evaluate } from '../src/core/evaluate.js';

let dir: string;
let dbPath: string;

function makeClock(): Clock & { tick: () => void; value: number } {
  const clock = {
    value: 1_700_000_000_000,
    now() {
      return this.value;
    },
    tick() {
      this.value += 1000;
    }
  };
  return clock;
}

const corpus = (): { name: string; sentences: TaggedSentence[] } => ({
  name: 'test-zh',
  sentences: [
    { tokens: ['我', '看', '书'], tags: ['r', 'v', 'n'] },
    { tokens: ['你', '读', '报'], tags: ['r', 'v', 'n'] },
    { tokens: ['猫', '追', '狗'], tags: ['n', 'v', 'n'] },
    { tokens: ['花', '漂亮'], tags: ['n', 'a'] },
    { tokens: ['天', '蓝'], tags: ['n', 'a'] },
    { tokens: ['他', '去', '学校'], tags: ['r', 'v', 'n'] },
    { tokens: ['锁'], tags: ['n'] },
    { tokens: ['锁'], tags: ['v'] },
    { tokens: [], tags: [] },
    { tokens: ['哎呀'], tags: ['e'] }
  ]
});

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), 'hmm-test-'));
  dbPath = join(dir, 'test.sqlite');
});

afterEach(() => rmSync(dir, { recursive: true, force: true }));

test('重复导入幂等且不重复计数', () => {
  const db = openDatabase(dbPath);
  const repo = new Repository(db, makeClock());
  const first = repo.importCorpus(corpus().name, corpus().sentences);
  const second = repo.importCorpus(corpus().name, corpus().sentences);
  assert.equal(first.reused, false);
  assert.equal(second.reused, true);
  assert.equal(second.corpus.id, first.corpus.id);
  assert.equal(repo.listCorpora(50, 0).total, 1);
  assert.equal(repo.getSentences(first.corpus.id).length, corpus().sentences.length);
  db.close();
});

test('语料修订产生新版本；旧模型仍可重现预测', () => {
  const db = openDatabase(dbPath);
  const repo = new Repository(db, makeClock());
  const first = repo.importCorpus('rev', corpus().sentences);
  const split1 = repo.createSplit(first.corpus.id, 42, 0.8);
  const model1 = repo.trainModel(split1.splitId, 1, 'uniform');
  const paramsV1 = repo.loadModel(model1.modelId);
  const predictionV1 = viterbi(paramsV1, ['我', '读', '书']);

  const revised = corpus();
  revised.sentences.push({ tokens: ['新增', '句子'], tags: ['v', 'n'] });
  const second = repo.importCorpus('rev', revised.sentences);
  assert.equal(second.corpus.version, 2);
  const split2 = repo.createSplit(second.corpus.id, 42, 0.8);
  repo.trainModel(split2.splitId, 1, 'uniform');

  const paramsV1Again = repo.loadModel(model1.modelId);
  assert.deepEqual(viterbi(paramsV1Again, ['我', '读', '书']).tags, predictionV1.tags);
  assert.equal(repo.listCorpora(50, 0).total, 2);
  db.close();
});

test('词典只来自训练集：验证集生词不会进入模型词汇', () => {
  const db = openDatabase(dbPath);
  const repo = new Repository(db, makeClock());
  const data = corpus();
  const imported = repo.importCorpus('leak', data.sentences);
  const split = repo.createSplit(imported.corpus.id, 7, 0.75);
  const train = repo.getSplitSentences(split.splitId, 'train');
  const validation = repo.getSplitSentences(split.splitId, 'validation');
  const model = repo.trainModel(split.splitId, 1, 'uniform');
  const params = repo.loadModel(model.modelId);
  const trainVocab = new Set(train.flatMap((sentence) => sentence.tokens));
  for (const token of params.vocab) assert.ok(trainVocab.has(token), `验证集词汇泄漏: ${token}`);
  const validationOnly = validation
    .flatMap((sentence) => sentence.tokens)
    .filter((token) => !trainVocab.has(token));
  if (validationOnly.length > 0) {
    const decoded = viterbi(params, validationOnly);
    assert.equal(decoded.tags.length, validationOnly.length);
    assert.ok(decoded.unknownPositions.length > 0);
  }
  db.close();
});

test('评测报告含逐标签指标与混淆矩阵且幂等', () => {
  const db = openDatabase(dbPath);
  const repo = new Repository(db, makeClock());
  const imported = repo.importCorpus('eval', corpus().sentences);
  const split = repo.createSplit(imported.corpus.id, 42, 0.8);
  const model = repo.trainModel(split.splitId, 1, 'uniform');
  const first = repo.evaluateModel(model.modelId, 'validation');
  const second = repo.evaluateModel(model.modelId, 'validation');
  assert.equal(first.reportId, second.reportId);
  assert.ok(first.tagMetrics.length > 0);
  assert.ok(first.summary.tokenTotal >= 0);
  const params = repo.loadModel(model.modelId);
  const direct = evaluate(params, repo.getSplitSentences(split.splitId, 'validation'));
  assert.ok(Math.abs(direct.summary.accuracy - first.summary.accuracy) < 1e-12);
  db.close();
});

test('事务在失败时回滚', () => {
  const db = openDatabase(dbPath);
  const repo = new Repository(db, makeClock());
  const imported = repo.importCorpus('rollback', corpus().sentences);
  const before = repo.listCorpora(50, 0).total;
  assert.throws(() =>
    repo.transaction(() => {
      db.prepare('INSERT INTO corpora(name, version, content_sha, sentence_count, token_count, tag_count, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)')
        .run('rollback', 99, 'x', 0, 0, 0, 0);
      throw new Error('故意失败');
    })
  );
  assert.equal(repo.listCorpora(50, 0).total, before);
  assert.equal(imported.corpus.id, 1);
  db.close();
});

test('重启后数据持久化，迁移可重复执行', () => {
  {
    const db = openDatabase(dbPath);
    const repo = new Repository(db);
    repo.importCorpus('persist', corpus().sentences);
    db.close();
  }
  {
    const db = openDatabase(dbPath);
    openDatabase(dbPath).close();
    const repo = new Repository(db);
    assert.equal(repo.listCorpora(50, 0).total, 1);
    db.close();
  }
});

test('相同训练参数复用模型，不同平滑生成不同模型', () => {
  const db = openDatabase(dbPath);
  const repo = new Repository(db, makeClock());
  const imported = repo.importCorpus('dup', corpus().sentences);
  const split = repo.createSplit(imported.corpus.id, 42, 0.8);
  const a = repo.trainModel(split.splitId, 1, 'uniform');
  const b = repo.trainModel(split.splitId, 1, 'uniform');
  const c = repo.trainModel(split.splitId, 0.1, 'uniform');
  assert.equal(a.modelId, b.modelId);
  assert.notEqual(a.modelId, c.modelId);
  db.close();
});
