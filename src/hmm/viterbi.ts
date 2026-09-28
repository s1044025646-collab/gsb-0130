import { AppError } from '../core/errors.js';
import type { DecodeResult, PathStep } from '../core/types.js';
import type { TrainedModel } from './model.js';

const LOG_ZERO = Number.NEGATIVE_INFINITY;

export interface DecodeOptions {
  /** 允许使用的标签白名单；缺省使用全部训练标签。 */
  allowedTags?: string[];
  modelId?: number;
  modelVersion?: string;
}

/**
 * 一阶 HMM 的 Viterbi 解码，全程在对数空间进行。
 * 同分决胜规则（确定）：按标签序数组索引（训练时字典序）取较小者，
 * 因此同概率路径始终得到相同结果。
 */
export function viterbi(
  model: TrainedModel,
  tokens: string[],
  opts: DecodeOptions = {}
): DecodeResult {
  const T = model.tags.length;
  const tagIndex = new Map(model.tags.map((t, i) => [t, i]));

  let allowed: number[];
  if (opts.allowedTags) {
    const unknown = opts.allowedTags.filter((t) => !tagIndex.has(t));
    if (unknown.length > 0) {
      throw new AppError(
        'BAD_TAG',
        `存在模型不认识的标签：${unknown.join(', ')}`,
        { unknownTags: unknown, knownTags: model.tags }
      );
    }
    if (opts.allowedTags.length === 0) {
      throw new AppError('BAD_TAG', 'allowedTags 不能为空');
    }
    allowed = opts.allowedTags.map((t) => tagIndex.get(t)!).sort((a, b) => a - b);
  } else {
    allowed = model.tags.map((_, i) => i);
  }

  const n = tokens.length;
  const path: PathStep[] = [];
  let usedUnknown = false;

  if (n === 0) {
    return {
      modelId: opts.modelId ?? 0,
      modelVersion: opts.modelVersion ?? '',
      tokens: [],
      tags: [],
      score: 0,
      usedUnknown: false,
      path: [],
      endLogProb: 0,
    };
  }

  let score = new Float64Array(T).fill(LOG_ZERO);
  const back: Int32Array[] = [];
  const emitAt: Float64Array[] = [];

  for (let pos = 0; pos < n; pos++) {
    const token = tokens[pos] as string;
    let emit = model.logEmit.get(token);
    if (!emit) {
      usedUnknown = true;
      emit = unknownEmission(model, token);
    }
    emitAt.push(emit);

    const next = new Float64Array(T).fill(LOG_ZERO);
    const bp = new Int32Array(T).fill(-1);

    for (const cur of allowed) {
      let best: number;
      let bestFrom = -1;
      if (pos === 0) {
        best = model.logInitial[cur]! + emit[cur]!;
      } else {
        best = LOG_ZERO;
        for (const from of allowed) {
          const cand = score[from]! + model.logTrans[from * T + cur]!;
          if (cand > best) {
            best = cand;
            bestFrom = from;
          }
        }
        best += emit[cur]!;
      }
      next[cur] = best;
      bp[cur] = bestFrom;
    }
    back.push(bp);
    score = next;
  }

  let bestEnd = LOG_ZERO;
  let endState = -1;
  for (const cur of allowed) {
    const cand = score[cur]! + model.logFinal[cur]!;
    if (cand > bestEnd) {
      bestEnd = cand;
      endState = cur;
    }
  }
  if (endState === -1 || !Number.isFinite(bestEnd)) {
    throw new AppError(
      'NO_PATH',
      '在当前标签与平滑设置下不存在概率为正的可行路径',
      { tokens, allowedTags: allowed.map((i) => model.tags[i]) }
    );
  }

  const stateSeq = new Array<number>(n);
  stateSeq[n - 1] = endState;
  for (let pos = n - 1; pos > 0; pos--) {
    stateSeq[pos - 1] = back[pos]![stateSeq[pos]!]!;
  }

  let cumulative = 0;
  for (let pos = 0; pos < n; pos++) {
    const cur = stateSeq[pos]!;
    const from = pos === 0 ? -1 : stateSeq[pos - 1]!;
    const trans = pos === 0 ? model.logInitial[cur]! : model.logTrans[from * T + cur]!;
    cumulative += trans + emitAt[pos]![cur]!;
    path.push({
      position: pos,
      token: tokens[pos] as string,
      tag: model.tags[cur] as string,
      emitLogProb: emitAt[pos]![cur]!,
      fromTag: pos === 0 ? '^' : (model.tags[from] as string),
      transLogProb: trans,
      cumulative,
    });
  }
  const endLogProb = model.logFinal[endState]!;

  return {
    modelId: opts.modelId ?? 0,
    modelVersion: opts.modelVersion ?? '',
    tokens,
    tags: stateSeq.map((i) => model.tags[i] as string),
    score: bestEnd,
    usedUnknown,
    path,
    endLogProb,
  };
}

function unknownEmission(model: TrainedModel, _token: string): Float64Array {
  if (model.params.unknownStrategy === 'laplace') {
    return model.logEmit.get('__UNK__')!;
  }
  const T = model.tags.length;
  const arr = new Float64Array(T).fill(LOG_ZERO);
  const uniform = Math.log(1 / T);
  for (let i = 0; i < T; i++) arr[i] = uniform;
  return arr;
}
