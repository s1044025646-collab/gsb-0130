import test from 'node:test';
import assert from 'node:assert/strict';
import { countTraining, trainModel } from '../src/hmm/model.js';
import { viterbi } from '../src/hmm/viterbi.js';
import { AppError } from '../src/core/errors.js';
import type { TaggedSentence } from '../src/core/types.js';

function s(tokens: string[], tags: string[]): TaggedSentence {
  return { tokens, tags };
}

/** 对短序列做全枚举，核对 Viterbi 确实取到全局最优。 */
function bruteForce(
  model: ReturnType<typeof trainModel>,
  tokens: string[],
  allowed?: string[]
): { score: number; tags: string[] } {
  const tags = allowed ?? model.tags;
  const emit = (token: string, tag: string) => {
    const arr = model.logEmit.get(token);
    const i = model.tags.indexOf(tag);
    if (arr) return arr[i]!;
    if (model.params.unknownStrategy === 'laplace') return model.logEmit.get('__UNK__')![i]!;
    return Math.log(1 / model.tags.length);
  };
  let best = Number.NEGATIVE_INFINITY;
  let bestTags: string[] = [];
  const recurse = (pos: number, prev: string | null, score: number, chosen: string[]) => {
    if (pos === tokens.length) {
      const endI = model.tags.indexOf(chosen[chosen.length - 1]!);
      const total = score + (tokens.length === 0 ? 0 : model.logFinal[endI]!);
      if (total > best) {
        best = total;
        bestTags = [...chosen];
      }
      return;
    }
    for (const tag of tags) {
      const i = model.tags.indexOf(tag);
      const trans =
        prev === null ? model.logInitial[i]! : model.logTrans[model.tags.indexOf(prev) * tags.length + i]!;
      const transFix =
        prev === null
          ? model.logInitial[i]!
          : model.logTrans[model.tags.indexOf(prev) * model.tags.length + i]!;
      void trans;
      recurse(pos + 1, tag, score + transFix + emit(tokens[pos]!, tag), [...chosen, tag]);
    }
  };
  if (tokens.length === 0) return { score: 0, tags: [] };
  recurse(0, null, 0, []);
  return { score: best, tags: bestTags };
}

const train = [
  s(['我', '爱', '中国'], ['PRON', 'V', 'N']),
  s(['他', '吃', '苹果'], ['PRON', 'V', 'N']),
  s(['猫', '追', '老鼠'], ['N', 'V', 'N']),
  s(['花', '很', '香'], ['N', 'ADV', 'A']),
  s(['水', '非常', '深'], ['N', 'ADV', 'A']),
  s(['三', '本', '书'], ['NUM', 'M', 'N']),
  s(['五', '个', '苹果'], ['NUM', 'M', 'N']),
  s(['停'], ['V']),
];

test('各条件分布归一（句首/句尾/转移/发射）', () => {
  const model = trainModel(countTraining(train), { alpha: 1, unknownStrategy: 'laplace' });
  const T = model.tags.length;
  const sumExp = (arr: ArrayLike<number>) => {
    let total = 0;
    for (let i = 0; i < arr.length; i++) total += Math.exp(arr[i] as number);
    return total;
  };
  assert.ok(Math.abs(sumExp(model.logInitial) - 1) < 1e-9);
  assert.ok(Math.abs(sumExp(model.logFinal) - 1) < 1e-9);
  for (let i = 0; i < T; i++) {
    const row = model.logTrans.slice(i * T, (i + 1) * T);
    assert.ok(Math.abs(sumExp(row) - 1) < 1e-9, `转移行 ${model.tags[i]} 归一`);
  }
  // 发射分布在 |V|+1 个符号（含预留 UNK）上归一：把 UNK 列补到每个标签。
  const unk = model.logEmit.get('__UNK__')!;
  for (let j = 0; j < T; j++) {
    let total = Math.exp(unk[j]!);
    for (const [token, arr] of model.logEmit) {
      if (token === '__UNK__') continue;
      total += Math.exp(arr[j]!);
    }
    assert.ok(Math.abs(total - 1) < 1e-9, `标签 ${model.tags[j]} 发射归一`);
  }
});

test('训练句以高概率被还原', () => {
  const model = trainModel(countTraining(train), { alpha: 0.1, unknownStrategy: 'laplace' });
  const r = viterbi(model, ['他', '吃', '苹果']);
  assert.deepEqual(r.tags, ['PRON', 'V', 'N']);
});

test('未知词不导致整句失败，且标记 usedUnknown', () => {
  const model = trainModel(countTraining(train), { alpha: 1, unknownStrategy: 'laplace' });
  const r = viterbi(model, ['企鹅', '吃', '磷虾']);
  assert.equal(r.tags.length, 3);
  assert.equal(r.usedUnknown, true);
  assert.deepEqual(r.tags[1], 'V');
});

test('closed 未知词策略给出有限分数', () => {
  const model = trainModel(countTraining(train), { alpha: 0, unknownStrategy: 'closed' });
  const r = viterbi(model, ['未知词', '吃']);
  assert.ok(Number.isFinite(r.score));
  assert.equal(r.tags[1], 'V');
});

test('短序列穷举核对最优路径与分数', () => {
  const model = trainModel(countTraining(train), { alpha: 0.5, unknownStrategy: 'laplace' });
  for (const tokens of [['他', '吃'], ['花', '很', '香'], ['企鹅', '本', '书'], ['停']]) {
    const v = viterbi(model, tokens);
    const bf = bruteForce(model, tokens);
    assert.ok(Math.abs(v.score - bf.score) < 1e-9, `分数一致: ${tokens.join('/')}`);
    assert.deepEqual(v.tags, bf.tags, `标签一致: ${tokens.join('/')}`);
  }
});

test('同分时按字典序标签确定性决胜（重复调用一致）', () => {
  const sym: TaggedSentence[] = [
    s(['x'], ['A']),
    s(['y'], ['B']),
  ];
  const model = trainModel(countTraining(sym), { alpha: 1, unknownStrategy: 'laplace' });
  const r1 = viterbi(model, ['从没见过']);
  const r2 = viterbi(model, ['从没见过']);
  assert.deepEqual(r1.tags, r2.tags);
  // A 与 B 对未知词完全对称，决胜应取字典序 A
  assert.deepEqual(r1.tags, ['A']);
});

test('不允许的标签返回 BAD_TAG，alpha=0 无可行路径返回 NO_PATH', () => {
  const model = trainModel(countTraining(train), { alpha: 1, unknownStrategy: 'laplace' });
  assert.throws(
    () => viterbi(model, ['他'], { allowedTags: ['不存在'] }),
    (e: unknown) => e instanceof AppError && e.code === 'BAD_TAG'
  );
  const closed = trainModel(countTraining(train), { alpha: 0, unknownStrategy: 'laplace' });
  // 只允许从未作为句首的标签，alpha=0 时句首概率为 0
  const noInitial = 'M';
  assert.throws(
    () => viterbi(closed, ['他'], { allowedTags: [noInitial] }),
    (e: unknown) => e instanceof AppError && e.code === 'NO_PATH'
  );
});

test('空句返回空路径，不产生计数', () => {
  const model = trainModel(
    countTraining([s([], []), ...train, s([], [])]),
    { alpha: 1, unknownStrategy: 'laplace' }
  );
  const r = viterbi(model, []);
  assert.deepEqual(r.tags, []);
  assert.equal(r.score, 0);
});

test('数值稳定：长句分数有限（对数空间）', () => {
  const model = trainModel(countTraining(train), { alpha: 1, unknownStrategy: 'laplace' });
  const tokens = Array.from({ length: 200 }, (_, i) => (i % 2 === 0 ? '他' : '吃'));
  const r = viterbi(model, tokens);
  assert.ok(Number.isFinite(r.score));
  assert.equal(r.tags.length, 200);
});
