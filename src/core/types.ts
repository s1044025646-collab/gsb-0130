/** 一条已分词并标注的句子：tokens 与 tags 等长，可同时为空串（空句）。 */
export interface TaggedSentence {
  /** 原始句子（可选，便于展示）。 */
  text?: string;
  /** 已分词的词序列。空数组表示空句。 */
  tokens: string[];
  /** 与 tokens 一一对应的词性标签。 */
  tags: string[];
}

export type UnknownWordStrategy = 'laplace' | 'closed';

export interface TrainParams {
  /** 拉普拉斯/加性平滑系数 alpha，必须 >= 0。 */
  alpha: number;
  /** 未知词策略。 */
  unknownStrategy: UnknownWordStrategy;
}

export interface SplitParams {
  /** 固定划分种子（整数）。 */
  seed: number;
  /** 验证集比例，范围 (0,1)。 */
  valRatio: number;
}

export interface PageParams {
  limit: number;
  offset: number;
}

export interface Page<T> {
  data: T[];
  total: number;
  limit: number;
  offset: number;
}

export interface PathStep {
  /** 词位（0 基）。 */
  position: number;
  token: string;
  /** 该位置选出的标签。 */
  tag: string;
  /** 发射对数概率（未知词按策略给出）。 */
  emitLogProb: number;
  /** 前一标签；首词为句首标记 "^"。 */
  fromTag: string;
  /** 转移对数概率。 */
  transLogProb: number;
  /** 到该状态的累计对数分数。 */
  cumulative: number;
}

export interface DecodeResult {
  modelId: number;
  modelVersion: string;
  tokens: string[];
  tags: string[];
  /** 完整最优路径的对数分数（含句尾转移）。 */
  score: number;
  /** 是否遇到训练词表之外的词。 */
  usedUnknown: boolean;
  /** 逐位置解释（不含虚拟句尾步）。 */
  path: PathStep[];
  /** 句尾（最后标签 -> $）对数概率。 */
  endLogProb: number;
}

export interface TagMetrics {
  tag: string;
  tp: number;
  fp: number;
  fn: number;
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface EvaluationReport {
  modelId: number;
  modelVersion: string;
  sentences: number;
  tokens: number;
  /** 逐标签准确率（全序列完全正确的句子占比）。 */
  sentenceAccuracy: number;
  /** 词级别准确率。 */
  tokenAccuracy: number;
  macroPrecision: number;
  macroRecall: number;
  macroF1: number;
  perTag: TagMetrics[];
  /** 混淆矩阵：行=金标，列=预测。 */
  confusion: { tag: string; predicted: string; count: number }[];
  tags: string[];
}

export interface Clock {
  now: () => Date;
}

export const systemClock: Clock = { now: () => new Date() };
