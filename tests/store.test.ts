import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { openDatabase, closeDatabase, type DB } from '../src/db/database.js';
import { createAppContext, evaluateAndStore, tagTokens } from '../src/core/app.js';
import { decodeRow } from '../src/core/corpus.js';
import { AppError } from '../src/core/errors.js';
import { unitHash } from '../src/core/hash.js';
import { buildCorpus, validationOnly } from '../fixtures/corpus.js';

function tempDb(): { dir: string; path: string } {
  const dir = mkdtempSync(join(tmpdir(), 'zhpos-'));
  return { dir, path: join(dir, 'test.sqlite') };
}

test('重复导入幂等：版本与句子计数不增加', () => {
  const { dir, path } = tempDb();
  const db = openDatabase(path);
  const ctx = createAppContext(db);
  const corpus = ctx.corpora.createCorpus('c');
  const v1 = ctx.corpora.importVersion(corpus.id, buildCorpus());
  const v2 = ctx.corpora.importVersion(corpus.id, buildCorpus());
  assert.equal(v2.reused, true);
  assert.equal(v1.versionId, v2.versionId);
  const rows = db.prepare('SELECT COUNT(*) c FROM sentences WHERE version_id=?').get(v1.versionId) as {
    c: number;
  };
  assert.equal(rows.c, v1.sentenceCount);
  db.close();
  rmSync(dir, { recursive: true, force: true });
});

test('语料修订产生新版本，旧模型仍可重现预测', () => {
  const { dir, path } = tempDb();
  const db = openDatabase(path);
  const ctx = createAppContext(db);
  const corpus = ctx.corpora.createCorpus('c');
  const v1 = ctx.corpora.importVersion(corpus.id, buildCorpus());
  const split1 = ctx.corpora.getOrCreateSplit(v1.versionId, { seed: 7, valRatio: 0.25 });
  const m1 = ctx.models.train(v1.versionId, split1.splitId, {
    alpha: 1,
    unknownStrategy: 'laplace',
  });
  const before = tagTokens(ctx, m1.modelId, ['他', '吃', '苹果']);

  const revised = [...buildCorpus(), ...validationOnly];
  const v2 = ctx.corpora.importVersion(corpus.id, revised);
  assert.notEqual(v1.versionId, v2.versionId);
  assert.equal(v2.version, v1.version + 1);
  const split2 = ctx.corpora.getOrCreateSplit(v2.versionId, { seed: 7, valRatio: 0.25 });
  ctx.models.train(v2.versionId, split2.splitId, { alpha: 1, unknownStrategy: 'laplace' });

  const after = tagTokens(ctx, m1.modelId, ['他', '吃', '苹果']);
  assert.deepEqual(after.tags, before.tags);
  assert.ok(Math.abs(after.score - before.score) < 1e-12);
  db.close();
  rmSync(dir, { recursive: true, force: true });
});

test('固定种子划分：重复内容不会跨训练/验证集（泄漏防护）', () => {
  const { dir, path } = tempDb();
  const db = openDatabase(path);
  const ctx = createAppContext(db);
  const corpus = ctx.corpora.createCorpus('c');
  const v = ctx.corpora.importVersion(corpus.id, buildCorpus());
  const seed = 123;
  const ratio = 0.3;
  const { train, val } = ctx.corpora.getOrCreateSplit(v.versionId, { seed, valRatio: ratio });
  const trainFp = new Set(train.map((r) => r.fingerprint));
  for (const r of val) assert.ok(!trainFp.has(r.fingerprint), '验证句指纹不得出现在训练集');

  // 直接核对哈希决定性
  for (const r of [...train, ...val]) {
    const inVal = unitHash(seed, r.fingerprint) < ratio;
    assert.equal(inVal, val.includes(r));
  }
  db.close();
  rmSync(dir, { recursive: true, force: true });
});

test('词汇仅来自训练集：验证集专有词不在模型词表', () => {
  const { dir, path } = tempDb();
  const db = openDatabase(path);
  const ctx = createAppContext(db);
  const corpus = ctx.corpora.createCorpus('c');
  // 基表做训练，手工把 validationOnly 放进同版本以制造只落验证集的句
  const v = ctx.corpora.importVersion(corpus.id, [...buildCorpus(), ...validationOnly]);
  const split = ctx.corpora.getOrCreateSplit(v.versionId, { seed: 1, valRatio: 0.001 });
  // 显式把 validationOnly 的句子挪到验证集，模拟“训练时未见过”
  const allRows = db.prepare('SELECT id, tokens FROM sentences WHERE version_id=?').all(v.versionId) as
    | { id: number; tokens: string }[];
  const valIds = allRows
    .filter((r) => {
      const toks = JSON.parse(r.tokens) as string[];
      return validationOnly.some((x) => JSON.stringify(x.tokens) === JSON.stringify(toks));
    })
    .map((r) => r.id);
  db
    .prepare(
      'UPDATE split_assignments SET bucket=? WHERE split_id=? AND sentence_id IN (' +
        valIds.map(() => '?').join(',') +
        ')'
    )
    .run('val', split.splitId, ...valIds);
  const trained = ctx.models.train(v.versionId, split.splitId, {
    alpha: 1,
    unknownStrategy: 'laplace',
  });
  const { model } = ctx.models.loadModel(trained.modelId);
  // 关键：validationOnly 特有词必须不在训练词表
  assert.ok(!model.vocab.has('企鹅'));
  assert.ok(!model.vocab.has('磷虾'));
  assert.ok(!model.vocab.has('考拉'));
  db.close();
  rmSync(dir, { recursive: true, force: true });
});

test('事务回滚：非法模型训练不留下部分数据', () => {
  const { dir, path } = tempDb();
  const db = openDatabase(path);
  const ctx = createAppContext(db);
  const corpus = ctx.corpora.createCorpus('c');
  const v = ctx.corpora.importVersion(corpus.id, buildCorpus());
  // 全部句子都进验证集 -> 训练为空
  const split = ctx.corpora.getOrCreateSplit(v.versionId, { seed: 1, valRatio: 0.999 });
  assert.throws(
    () => ctx.models.train(v.versionId, split.splitId, { alpha: 1, unknownStrategy: 'laplace' }),
    (e: unknown) => e instanceof AppError && e.code === 'INSUFFICIENT_DATA'
  );
  const count = Number((db.prepare('SELECT COUNT(*) c FROM models').get() as { c: number }).c);
  assert.equal(count, 0);
  db.close();
  rmSync(dir, { recursive: true, force: true });
});

test('迁移可重复执行，且关闭重开后状态持久', () => {
  const { dir, path } = tempDb();
  let db: DB = openDatabase(path);
  const ctx = createAppContext(db);
  const corpus = ctx.corpora.createCorpus('persist');
  const v = ctx.corpora.importVersion(corpus.id, buildCorpus());
  const split = ctx.corpora.getOrCreateSplit(v.versionId, { seed: 5, valRatio: 0.2 });
  const m = ctx.models.train(v.versionId, split.splitId, { alpha: 1, unknownStrategy: 'laplace' });
  const pred = tagTokens(ctx, m.modelId, ['三', '本', '书']);
  const { reportId } = evaluateAndStore(ctx, m.modelId);
  db.close();

  db = openDatabase(path); // 重新迁移（幂等）并核对持久状态
  const ctx2 = createAppContext(db);
  assert.equal(ctx2.corpora.listCorpora().length, 1);
  const pred2 = tagTokens(ctx2, m.modelId, ['三', '本', '书']);
  assert.deepEqual(pred2.tags, pred.tags);
  const report = db.prepare('SELECT * FROM reports WHERE id=?').get(reportId);
  assert.ok(report);
  db.close();
  rmSync(dir, { recursive: true, force: true });
});

test('相同训练配置重复训练是幂等的（同一模型版本）', () => {
  const { dir, path } = tempDb();
  const db = openDatabase(path);
  const ctx = createAppContext(db);
  const corpus = ctx.corpora.createCorpus('c');
  const v = ctx.corpora.importVersion(corpus.id, buildCorpus());
  const split = ctx.corpora.getOrCreateSplit(v.versionId, { seed: 5, valRatio: 0.2 });
  const a = ctx.models.train(v.versionId, split.splitId, { alpha: 1, unknownStrategy: 'laplace' });
  const b = ctx.models.train(v.versionId, split.splitId, { alpha: 1, unknownStrategy: 'laplace' });
  assert.equal(a.modelId, b.modelId);
  assert.equal(a.modelVersion, b.modelVersion);
  db.close();
  rmSync(dir, { recursive: true, force: true });
});

test('所有数据都在项目/临时目录内（数据库文件存在）', () => {
  const { dir, path } = tempDb();
  const db = openDatabase(path);
  db.close();
  assert.ok(existsSync(path));
  rmSync(dir, { recursive: true, force: true });
});
