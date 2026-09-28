/**
 * 可重复端到端演示：使用独立的 .data/demo.sqlite，
 * 依次建库、导入、划分、训练两个模型、标注、解释、评测、比较，
 * 并在结束时关闭。重复运行结果一致（固定种子 + 幂等）。
 */
import { rmSync } from 'node:fs';
import { resolve } from 'node:path';
import { openDatabase } from '../src/db/database.js';
import { createAppContext, compare, evaluateAndStore, tagTokens } from '../src/core/app.js';
import { buildCorpus } from '../fixtures/corpus.js';

const dbPath = resolve(process.cwd(), '.data', 'demo.sqlite');
rmSync(dbPath, { force: true });

const db = openDatabase(dbPath);
const ctx = createAppContext(db);

console.log('>> 创建语料库');
const corpus = ctx.corpora.createCorpus('演示语料', '合成中文词性夹具');
console.log('corpusId =', corpus.id);

console.log('>> 导入（含 4 条重复，自动去重）');
const v = ctx.corpora.importVersion(corpus.id, buildCorpus());
console.log(`versionId=${v.versionId} version=${v.version} 唯一句=${v.sentenceCount}`);

console.log('>> 再次导入（幂等，应复用）');
const vAgain = ctx.corpora.importVersion(corpus.id, buildCorpus());
console.log('reused =', vAgain.reused);

console.log('>> 固定种子划分');
const split = ctx.corpora.getOrCreateSplit(v.versionId, { seed: 42, valRatio: 0.25 });
console.log(`splitId=${split.splitId} 训练=${split.train.length} 验证=${split.val.length}`);

console.log('>> 训练两个模型：laplace(alpha=1) 与 closed(alpha=1)');
const lap = ctx.models.train(v.versionId, split.splitId, {
  alpha: 1,
  unknownStrategy: 'laplace',
});
const closed = ctx.models.train(v.versionId, split.splitId, {
  alpha: 1,
  unknownStrategy: 'closed',
});
console.log('laplace =', lap.modelVersion);
console.log('closed  =', closed.modelVersion);

console.log('>> 标注含未知词的句子（“企鹅/考拉”训练时未见）');
const tagged = tagTokens(ctx, lap.modelId, ['企鹅', '在', '北京', '计划', '旅行']);
console.log('tokens:', tagged.tokens.join(' '));
console.log('tags  :', tagged.tags.join(' '));
console.log('score :', tagged.score.toFixed(4), 'usedUnknown =', tagged.usedUnknown);

console.log('>> 逐位置路径解释（laplace）');
for (const step of tagged.path) {
  console.log(
    `  #${step.position} ${step.token}/${step.tag} <- ${step.fromTag} ` +
      `trans=${step.transLogProb.toFixed(3)} emit=${step.emitLogProb.toFixed(3)} ` +
      `cum=${step.cumulative.toFixed(3)}`
  );
}

console.log('>> 验证集评测（laplace）');
const { report } = evaluateAndStore(ctx, lap.modelId);
console.log(
  `词准确率=${(report.tokenAccuracy * 100).toFixed(1)}% ` +
    `句准确率=${(report.sentenceAccuracy * 100).toFixed(1)}% ` +
    `macroF1=${report.macroF1.toFixed(3)}`
);

console.log('>> 模型比较');
const rows = compare(ctx, [lap.modelId, closed.modelId]);
for (const r of rows) {
  console.log(
    `  ${r.modelVersion} tokenAcc=${r.report.tokenAccuracy.toFixed(3)} ` +
      `macroF1=${r.report.macroF1.toFixed(3)}`
  );
}

db.close();
console.log('>> 演示完成，数据库位于', dbPath);
