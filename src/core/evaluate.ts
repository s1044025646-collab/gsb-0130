import type { TaggedSentence, TagMetric, ReportSummary } from './types.js';
import { viterbi, type HMMParams } from './hmm.js';

export interface EvaluationResult {
  tagMetrics: TagMetric[];
  confusion: Record<string, Record<string, number>>;
  summary: ReportSummary;
}

export function evaluate(params: HMMParams, sentences: TaggedSentence[]): EvaluationResult {
  const tagList = [...params.tags].sort();
  const tp: Record<string, number> = {};
  const fp: Record<string, number> = {};
  const fn: Record<string, number> = {};
  const support: Record<string, number> = {};
  const confusion: Record<string, Record<string, number>> = {};
  for (const tag of tagList) {
    tp[tag] = 0;
    fp[tag] = 0;
    fn[tag] = 0;
    support[tag] = 0;
    confusion[tag] = {};
    for (const predicted of tagList) confusion[tag][predicted] = 0;
  }

  let tokenTotal = 0;
  let tokenCorrect = 0;
  let sentenceTotal = 0;
  let sentenceCorrect = 0;
  let logLikelihood = 0;

  for (const sentence of sentences) {
    if (sentence.tokens.length === 0) continue;
    sentenceTotal += 1;
    const decoded = viterbi(params, sentence.tokens);
    logLikelihood += decoded.score;
    let allCorrect = true;
    for (let index = 0; index < sentence.tags.length; index += 1) {
      const gold = sentence.tags[index];
      const predicted = decoded.tags[index];
      tokenTotal += 1;
      support[gold] ??= 0;
      support[gold] += 1;
      if (predicted === gold) {
        tokenCorrect += 1;
        tp[gold] += 1;
      } else {
        allCorrect = false;
        fp[predicted] = (fp[predicted] ?? 0) + 1;
        fn[gold] = (fn[gold] ?? 0) + 1;
      }
      confusion[gold] ??= {};
      confusion[gold][predicted] = (confusion[gold][predicted] ?? 0) + 1;
    }
    if (allCorrect) sentenceCorrect += 1;
  }

  const tagMetrics: TagMetric[] = tagList.map((tag) => {
    const precisionDenom = tp[tag] + fp[tag];
    const recallDenom = tp[tag] + fn[tag];
    const precision = precisionDenom === 0 ? 0 : tp[tag] / precisionDenom;
    const recall = recallDenom === 0 ? 0 : tp[tag] / recallDenom;
    const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
    return {
      tag,
      tp: tp[tag],
      fp: fp[tag],
      fn: fn[tag],
      precision,
      recall,
      f1,
      support: support[tag] ?? 0
    };
  });

  return {
    tagMetrics,
    confusion,
    summary: {
      tokenTotal,
      tokenCorrect,
      sentenceTotal,
      sentenceCorrect,
      accuracy: tokenTotal === 0 ? 0 : tokenCorrect / tokenTotal,
      logLikelihood
    }
  };
}
