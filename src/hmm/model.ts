import { AppError } from '../core/errors.js';
import type { TaggedSentence, TrainParams } from '../core/types.js';

const LOG_ZERO = Number.NEGATIVE_INFINITY;

export interface RawCounts {
  tags: string[];
  initial: Map<string, number>;
  final: Map<string, number>;
  tagCount: Map<string, number>;
  /** from -> (to -> count) */
  transitions: Map<string, Map<string, number>>;
  /** tag -> (token -> count) */
  emissions: Map<string, Map<string, number>>;
  vocab: Set<string>;
  trainSentences: number;
  trainTokens: number;
  nonEmptySentences: number;
}

/** 只基于训练句统计计数；词典同样只来自训练集。 */
export function countTraining(sentences: TaggedSentence[]): RawCounts {
  const initial = new Map<string, number>();
  const final = new Map<string, number>();
  const tagCount = new Map<string, number>();
  const transitions = new Map<string, Map<string, number>>();
  const emissions = new Map<string, Map<string, number>>();
  const tagSet = new Set<string>();
  const vocab = new Set<string>();
  let trainTokens = 0;
  let nonEmptySentences = 0;

  for (const s of sentences) {
    const n = s.tags.length;
    if (n === 0) continue; // 空句不贡献任何发射/转移计数
    nonEmptySentences++;
    for (let i = 0; i < n; i++) {
      const tag = s.tags[i] as string;
      const token = s.tokens[i] as string;
      tagSet.add(tag);
      vocab.add(token);
      tagCount.set(tag, (tagCount.get(tag) ?? 0) + 1);
      let em = emissions.get(tag);
      if (!em) {
        em = new Map();
        emissions.set(tag, em);
      }
      em.set(token, (em.get(token) ?? 0) + 1);
      trainTokens++;
    }
    const first = s.tags[0] as string;
    const last = s.tags[n - 1] as string;
    initial.set(first, (initial.get(first) ?? 0) + 1);
    final.set(last, (final.get(last) ?? 0) + 1);
    for (let i = 0; i < n - 1; i++) {
      const from = s.tags[i] as string;
      const to = s.tags[i + 1] as string;
      let row = transitions.get(from);
      if (!row) {
        row = new Map();
        transitions.set(from, row);
      }
      row.set(to, (row.get(to) ?? 0) + 1);
    }
  }

  return {
    tags: [...tagSet].sort(),
    initial,
    final,
    tagCount,
    transitions,
    emissions,
    vocab,
    trainSentences: sentences.length,
    trainTokens,
    nonEmptySentences,
  };
}

export interface TrainedModel {
  tags: string[];
  vocab: Set<string>;
  params: TrainParams;
  /** 对数概率：句首 / 句尾 / 转移[from*T+to] / 发射按 Map 查询。 */
  logInitial: Float64Array;
  logFinal: Float64Array;
  logTrans: Float64Array;
  logEmit: Map<string, Float64Array>; // token(含 __UNK__) -> 每个标签的对数概率
  trainSentences: number;
  trainTokens: number;
  trainLogLikelihood: number;
}

function safeLog(n: number): number {
  return n > 0 ? Math.log(n) : LOG_ZERO;
}

function sumValues(m: Map<string, number>): number {
  let total = 0;
  for (const v of m.values()) total += v;
  return total;
}

/**
 * 训练并归一化所有分布。
 * - 句首/句尾/转移：在 |T| 个标签上加 alpha 平滑，每个条件分布和为 1。
 * - 发射（laplace）：在 |V|+1 个符号（含预留 UNK）上加 alpha，每个标签分布和为 1。
 * - 发射（closed）：已知词 MLE；未知词在解码时按标签均匀处理，不污染词表分布。
 */
export function trainModel(counts: RawCounts, params: TrainParams): TrainedModel {
  const T = counts.tags.length;
  if (T === 0) {
    throw new AppError('INSUFFICIENT_DATA', '训练集没有任何标签，无法训练');
  }
  const alpha = params.alpha;
  const V = counts.vocab.size;
  const tagIndex = new Map(counts.tags.map((t, i) => [t, i]));

  const initialTotal = counts.nonEmptySentences;
  const logInitial = new Float64Array(T);
  const logFinal = new Float64Array(T);
  const finalTotal = sumValues(counts.final);
  for (let i = 0; i < T; i++) {
    const tag = counts.tags[i] as string;
    logInitial[i] = safeLog(((counts.initial.get(tag) ?? 0) + alpha) / (initialTotal + alpha * T));
    // 句尾标签分布（在标签集合上归一）：P(last tag = t)。
    logFinal[i] = safeLog(((counts.final.get(tag) ?? 0) + alpha) / (finalTotal + alpha * T));
  }

  const logTrans = new Float64Array(T * T).fill(LOG_ZERO);
  for (let i = 0; i < T; i++) {
    const from = counts.tags[i] as string;
    const row = counts.transitions.get(from);
    // from 作为“非句尾”出现次数 = 标签总数 - 句尾次数
    const denom = (counts.tagCount.get(from) ?? 0) - (counts.final.get(from) ?? 0);
    for (let j = 0; j < T; j++) {
      const to = counts.tags[j] as string;
      const c = row?.get(to) ?? 0;
      logTrans[i * T + j] = safeLog((c + alpha) / (denom + alpha * T));
    }
  }

  const logEmit = new Map<string, Float64Array>();
  if (params.unknownStrategy === 'laplace') {
    const symbols = V + 1;
    const unk = new Float64Array(T);
    for (let i = 0; i < T; i++) {
      const tag = counts.tags[i] as string;
      const tagTotal = counts.tagCount.get(tag) ?? 0;
      unk[i] = safeLog(alpha / (tagTotal + alpha * symbols));
    }
    logEmit.set('__UNK__', unk);
    for (const token of counts.vocab) {
      const arr = new Float64Array(T);
      for (let i = 0; i < T; i++) {
        const tag = counts.tags[i] as string;
        const tagTotal = counts.tagCount.get(tag) ?? 0;
        const c = counts.emissions.get(tag)?.get(token) ?? 0;
        arr[i] = safeLog((c + alpha) / (tagTotal + alpha * symbols));
      }
      logEmit.set(token, arr);
    }
  } else {
    for (const token of counts.vocab) {
      const arr = new Float64Array(T).fill(LOG_ZERO);
      for (let i = 0; i < T; i++) {
        const tag = counts.tags[i] as string;
        const c = counts.emissions.get(tag)?.get(token) ?? 0;
        const tagTotal = counts.tagCount.get(tag) ?? 0;
        arr[i] = tagTotal > 0 ? safeLog(c / tagTotal) : LOG_ZERO;
      }
      logEmit.set(token, arr);
    }
  }

  const trained: TrainedModel = {
    tags: counts.tags,
    vocab: counts.vocab,
    params,
    logInitial,
    logFinal,
    logTrans,
    logEmit,
    trainSentences: counts.trainSentences,
    trainTokens: counts.trainTokens,
    trainLogLikelihood: 0,
  };
  trained.trainLogLikelihood = trainingLogLikelihood(trained, counts);
  return trained;
}

function trainingLogLikelihood(model: TrainedModel, counts: RawCounts): number {
  let ll = 0;
  for (const [tag, c] of counts.initial) {
    const i = model.tags.indexOf(tag);
    ll += c * model.logInitial[i]!;
  }
  for (const [tag, c] of counts.final) {
    const i = model.tags.indexOf(tag);
    ll += c * model.logFinal[i]!;
  }
  for (const [from, row] of counts.transitions) {
    const i = model.tags.indexOf(from);
    for (const [to, c] of row) {
      const j = model.tags.indexOf(to);
      ll += c * model.logTrans[i * model.tags.length + j]!;
    }
  }
  for (const [tag, em] of counts.emissions) {
    const i = model.tags.indexOf(tag);
    for (const [token, c] of em) {
      ll += c * (model.logEmit.get(token)?.[i] ?? LOG_ZERO);
    }
  }
  return ll;
}
