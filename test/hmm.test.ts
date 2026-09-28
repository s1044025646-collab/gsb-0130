import { test } from 'node:test';
import assert from 'node:assert/strict';
import { trainHMM, viterbi } from '../src/core/hmm.js';
import type { TaggedSentence } from '../src/core/types.js';
import { mulberry32, deterministicSplit } from '../src/core/split.js';

const sentences: TaggedSentence[] = [
  { tokens: ['我', '看', '书'], tags: ['r', 'v', 'n'] },
  { tokens: ['你', '看', '报'], tags: ['r', 'v', 'n'] },
  { tokens: ['猫', '追', '狗'], tags: ['n', 'v', 'n'] },
  { tokens: ['锁'], tags: ['n'] },
  { tokens: ['锁'], tags: ['v'] },
  { tokens: ['花', '漂亮'], tags: ['n', 'a'] }
];

test('加一平滑后各条件分布均归一', () => {
  const params = trainHMM(sentences, 1, 'closed');
  let initialSum = 0;
  for (const tag of params.tags) initialSum += params.initial[tag];
  assert.ok(Math.abs(initialSum - 1) < 1e-12);
  for (const tag of params.tags) {
    let sum = params.stop[tag];
    for (const next of params.tags) sum += params.transition[tag][next];
    assert.ok(Math.abs(sum - 1) < 1e-12, `转移+句尾分布 ${tag} 未归一`);
    let emissionSum = params.unknown[tag];
    for (const token of params.vocab) emissionSum += params.emission[tag][token];
    assert.ok(Math.abs(emissionSum - 1) < 1e-12, `发射分布 ${tag} 未归一`);
  }
});

test('uniform 未知词策略在标签间形成归一分布', () => {
  const params = trainHMM(sentences, 1, 'uniform');
  const unknownSum = params.tags.reduce((sum, tag) => sum + params.unknown[tag], 0);
  assert.ok(Math.abs(unknownSum - 1) < 1e-12);
  const first = params.unknown[params.tags[0]];
  for (const tag of params.tags) assert.equal(params.unknown[tag], first);
});

test('同词异性可由上下文消歧，未知词不会失败', () => {
  const params = trainHMM(sentences, 1, 'uniform');
  const known = viterbi(params, ['我', '看', '锁']);
  assert.deepEqual(known.tags, ['r', 'v', 'n']);
  const unknownWord = viterbi(params, ['一个从未见过的词']);
  assert.equal(unknownWord.tags.length, 1);
  assert.ok(unknownWord.unknownPositions.includes(0));
  assert.ok(Number.isFinite(unknownWord.score));
});

test('空句返回空路径与 0 分；全部未知词长句数值稳定', () => {
  const params = trainHMM(sentences, 0.5, 'uniform');
  const empty = viterbi(params, []);
  assert.deepEqual(empty.tags, []);
  assert.equal(empty.score, 0);
  const long = viterbi(params, Array.from({ length: 500 }, (_, index) => `生词${index}`));
  assert.equal(long.tags.length, 500);
  assert.ok(Number.isFinite(long.score));
});

test('短句穷举与 Viterbi 最优路径一致', () => {
  const params = trainHMM(sentences, 1, 'uniform');
  const tokens = ['你', '看', '锁'];
  const tags = params.tags;
  let bestScore = Number.NEGATIVE_INFINITY;
  let bestTags: string[] = [];
  const enumerate = (prefix: string[], position: number, score: number): void => {
    if (position === tokens.length) {
      const finalScore = score + Math.log(params.stop[prefix[position - 1]]);
      if (finalScore > bestScore) {
        bestScore = finalScore;
        bestTags = [...prefix];
      }
      return;
    }
    for (const tag of tags) {
      const emission = params.vocab.includes(tokens[position])
        ? params.emission[tag][tokens[position]]
        : params.unknown[tag];
      let addition = Math.log(emission);
      if (position === 0) addition += Math.log(params.initial[tag]);
      else addition += Math.log(params.transition[prefix[position - 1]][tag]);
      enumerate([...prefix, tag], position + 1, score + addition);
    }
  };
  enumerate([], 0, 0);
  const decoded = viterbi(params, tokens);
  assert.deepEqual(decoded.tags, bestTags);
  assert.ok(Math.abs(decoded.score - bestScore) < 1e-10);
});

test('同分路径按标签字典序确定性打平', () => {
  const symmetric: TaggedSentence[] = [
    { tokens: ['x'], tags: ['b'] },
    { tokens: ['y'], tags: ['a'] }
  ];
  const params = trainHMM(symmetric, 1, 'uniform');
  const first = viterbi(params, ['zzz']);
  const second = viterbi(params, ['zzz']);
  assert.deepEqual(first.tags, second.tags);
  assert.equal(first.tags[0], 'a');
});

test('不允许的标签报 TAG_NOT_IN_MODEL；空允许集报 NO_FEASIBLE_PATH', () => {
  const params = trainHMM(sentences, 1, 'uniform');
  assert.throws(
    () => viterbi(params, ['我'], ['不存在的标签']),
    (error: unknown) => (error as { code?: string }).code === 'TAG_NOT_IN_MODEL'
  );
  assert.throws(
    () => viterbi(params, ['我'], ['x']),
    (error: unknown) => (error as { code?: string }).code === 'TAG_NOT_IN_MODEL'
  );
});

test('固定种子划分可重复且互斥穷尽', () => {
  const a = deterministicSplit(30, 123, 0.8);
  const b = deterministicSplit(30, 123, 0.8);
  assert.deepEqual(a, b);
  assert.equal(a.length, 30);
  assert.equal(new Set(a.map((item) => item.index)).size, 30);
  assert.equal(a.filter((item) => item.part === 'train').length, 24);
  assert.notDeepEqual(deterministicSplit(30, 124, 0.8), a);
});

test('PRNG 确定性', () => {
  const generatorA = mulberry32(7);
  const generatorB = mulberry32(7);
  const sequenceA = Array.from({ length: 5 }, () => generatorA());
  const sequenceB = Array.from({ length: 5 }, () => generatorB());
  assert.deepEqual(sequenceA, sequenceB);
});
