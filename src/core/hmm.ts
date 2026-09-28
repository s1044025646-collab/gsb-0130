import type {
  TaggedSentence,
  UnknownStrategy,
  ViterbiResult,
  ViterbiStep
} from './types.js';
import { AppError } from './types.js';

const UNK = '\u0000UNK';
const LOG_ZERO = Number.NEGATIVE_INFINITY;

export interface HMMParams {
  tags: string[];
  vocab: string[];
  initial: Record<string, number>;
  stop: Record<string, number>;
  transition: Record<string, Record<string, number>>;
  emission: Record<string, Record<string, number>>;
  unknown: Record<string, number>;
  smoothing: number;
  unknownStrategy: UnknownStrategy;
}

export function trainHMM(
  sentences: TaggedSentence[],
  smoothing: number,
  unknownStrategy: UnknownStrategy
): HMMParams {
  if (!Number.isFinite(smoothing) || smoothing < 0) {
    throw new AppError('INVALID_PARAMS', 'smoothing 必须是非负有限数');
  }
  const tagSet = new Set<string>();
  const vocabSet = new Set<string>();
  const start: Record<string, number> = {};
  const end: Record<string, number> = {};
  const trans: Record<string, Record<string, number>> = {};
  const emit: Record<string, Record<string, number>> = {};
  const tagTotals: Record<string, number> = {};

  for (const sentence of sentences) {
    const { tokens, tags } = sentence;
    if (tokens.length === 0) continue;
    for (const tag of tags) {
      tagSet.add(tag);
      start[tag] ??= 0;
      end[tag] ??= 0;
      tagTotals[tag] ??= 0;
      trans[tag] ??= {};
      emit[tag] ??= {};
    }
    start[tags[0]] += 1;
    end[tags[tags.length - 1]] += 1;
    for (let index = 0; index < tokens.length; index += 1) {
      const tag = tags[index];
      const token = tokens[index];
      vocabSet.add(token);
      tagTotals[tag] += 1;
      emit[tag][token] = (emit[tag][token] ?? 0) + 1;
      if (index + 1 < tokens.length) {
        const nextTag = tags[index + 1];
        trans[tag][nextTag] = (trans[tag][nextTag] ?? 0) + 1;
      }
    }
  }

  const tags = [...tagSet].sort();
  const vocab = [...vocabSet].sort();
  const tagCount = Math.max(1, tags.length);
  let nonEmptySentences = 0;
  for (const sentence of sentences) if (sentence.tokens.length > 0) nonEmptySentences += 1;

  const initial: Record<string, number> = {};
  const stopProb: Record<string, number> = {};
  const transition: Record<string, Record<string, number>> = {};
  const emission: Record<string, Record<string, number>> = {};
  const unknown: Record<string, number> = {};

  const k = smoothing;
  for (const tag of tags) {
    initial[tag] = (start[tag] + k) / (nonEmptySentences + k * tagCount);
    let outgoing = 0;
    for (const nextTag of tags) outgoing += trans[tag][nextTag] ?? 0;
    const denom = outgoing + (end[tag] ?? 0) + k * (tagCount + 1);
    stopProb[tag] = ((end[tag] ?? 0) + k) / denom;
    transition[tag] = {};
    for (const nextTag of tags) {
      transition[tag][nextTag] = ((trans[tag][nextTag] ?? 0) + k) / denom;
    }
    const emitDenom = (tagTotals[tag] ?? 0) + k * (vocab.length + 1);
    emission[tag] = {};
    for (const token of vocab) {
      emission[tag][token] = ((emit[tag][token] ?? 0) + k) / emitDenom;
    }
    unknown[tag] =
      unknownStrategy === 'uniform' ? 1 / tagCount : k / emitDenom;
  }

  return {
    tags,
    vocab,
    initial,
    stop: stopProb,
    transition,
    emission,
    unknown,
    smoothing,
    unknownStrategy
  };
}

function emissionLogProb(params: HMMParams, tag: string, token: string): number {
  const probability = params.vocab.includes(token)
    ? params.emission[tag][token]
    : params.unknown[tag];
  return Math.log(probability);
}

export function viterbi(
  params: HMMParams,
  tokens: string[],
  allowedTags?: string[]
): ViterbiResult {
  const allowed = (allowedTags ?? params.tags)
    .filter((tag) => params.tags.includes(tag))
    .sort();
  const requested = allowedTags ? [...new Set(allowedTags)].sort() : null;
  if (requested) {
    const invalid = requested.filter((tag) => !params.tags.includes(tag));
    if (invalid.length > 0) {
      throw new AppError('TAG_NOT_IN_MODEL', `标签不在模型标签集中: ${invalid.join(', ')}`, {
        invalidTags: invalid
      });
    }
  }
  if (params.tags.length === 0) {
    throw new AppError('EMPTY_MODEL', '模型没有任何标签，无法解码');
  }
  if (allowed.length === 0) {
    throw new AppError('NO_FEASIBLE_PATH', '允许标签集合为空，不存在可行路径');
  }
  if (tokens.length === 0) {
    return { tags: [], score: 0, steps: [], unknownPositions: [] };
  }

  const length = tokens.length;
  const score: number[][] = Array.from({ length }, () => []);
  const back: (number | null)[][] = Array.from({ length }, () => []);
  const unknownPositions: number[] = [];

  for (let tagIndex = 0; tagIndex < allowed.length; tagIndex += 1) {
    const tag = allowed[tagIndex];
    if (!params.vocab.includes(tokens[0])) unknownPositions.push(0);
    score[0][tagIndex] = Math.log(params.initial[tag]) + emissionLogProb(params, tag, tokens[0]);
    back[0][tagIndex] = null;
  }

  for (let position = 1; position < length; position += 1) {
    if (!params.vocab.includes(tokens[position])) unknownPositions.push(position);
    for (let tagIndex = 0; tagIndex < allowed.length; tagIndex += 1) {
      const tag = allowed[tagIndex];
      let bestIndex = 0;
      let bestScore = LOG_ZERO;
      for (let prevIndex = 0; prevIndex < allowed.length; prevIndex += 1) {
        const candidate =
          score[position - 1][prevIndex] +
          Math.log(params.transition[allowed[prevIndex]][tag]);
        if (candidate > bestScore) {
          bestScore = candidate;
          bestIndex = prevIndex;
        }
      }
      score[position][tagIndex] = bestScore + emissionLogProb(params, tag, tokens[position]);
      back[position][tagIndex] = bestIndex;
    }
  }

  let bestFinalIndex = 0;
  let bestFinalScore = LOG_ZERO;
  for (let tagIndex = 0; tagIndex < allowed.length; tagIndex += 1) {
    const candidate = score[length - 1][tagIndex] + Math.log(params.stop[allowed[tagIndex]]);
    if (candidate > bestFinalScore) {
      bestFinalScore = candidate;
      bestFinalIndex = tagIndex;
    }
  }

  const tagIndices: number[] = new Array(length);
  tagIndices[length - 1] = bestFinalIndex;
  for (let position = length - 1; position > 0; position -= 1) {
    tagIndices[position - 1] = back[position][tagIndices[position]] as number;
  }
  const resultTags = tagIndices.map((index) => allowed[index]);

  const steps: ViterbiStep[] = [];
  for (let position = 0; position < length; position += 1) {
    const tag = resultTags[position];
    const prevTag = position === 0 ? null : resultTags[position - 1];
    let stepProb = emissionLogProb(params, tag, tokens[position]);
    if (position === 0) stepProb += Math.log(params.initial[tag]);
    else stepProb += Math.log(params.transition[prevTag as string][tag]);
    if (position === length - 1) stepProb += Math.log(params.stop[tag]);
    steps.push({
      position,
      token: tokens[position],
      chosen_tag: tag,
      prev_tag: prevTag,
      step_log_prob: stepProb,
      cumulative_log_prob: 0
    });
  }
  let cumulative = 0;
  for (const step of steps) {
    cumulative += step.step_log_prob;
    step.cumulative_log_prob = cumulative;
  }

  return { tags: resultTags, score: bestFinalScore, steps, unknownPositions: [...new Set(unknownPositions)] };
}

export { UNK };
