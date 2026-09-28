import { AppError } from './errors.js';
import type { TaggedSentence, TrainParams } from './types.js';

export function requireString(value: unknown, field: string): string {
  if (typeof value !== 'string' || value.trim() === '') {
    throw new AppError('VALIDATION', `字段 ${field} 必须是非空字符串`, { field });
  }
  return value;
}

export function parseTaggedSentence(raw: unknown): TaggedSentence {
  if (typeof raw !== 'object' || raw === null) {
    throw new AppError('VALIDATION', '句子必须是对象');
  }
  const obj = raw as Record<string, unknown>;
  if (!Array.isArray(obj.tokens) || !Array.isArray(obj.tags)) {
    throw new AppError('VALIDATION', '句子必须包含 tokens 与 tags 数组');
  }
  const tokens = obj.tokens as unknown[];
  const tags = obj.tags as unknown[];
  if (tokens.length !== tags.length) {
    throw new AppError('VALIDATION', 'tokens 与 tags 长度必须一致', {
      tokens: tokens.length,
      tags: tags.length,
    });
  }
  for (const t of tokens) {
    if (typeof t !== 'string' || t.length === 0) {
      throw new AppError('VALIDATION', 'tokens 中每个元素必须是非空字符串');
    }
  }
  for (const g of tags) {
    if (typeof g !== 'string' || g.length === 0) {
      throw new AppError('VALIDATION', 'tags 中每个元素必须是非空字符串');
    }
  }
  return {
    text: typeof obj.text === 'string' ? obj.text : undefined,
    tokens: tokens as string[],
    tags: tags as string[],
  };
}

export function parseSentenceList(raw: unknown): TaggedSentence[] {
  if (!Array.isArray(raw)) {
    throw new AppError('VALIDATION', 'sentences 必须是数组');
  }
  return raw.map(parseTaggedSentence);
}

export function parseAlpha(value: unknown): number {
  const alpha = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(alpha) || alpha < 0) {
    throw new AppError('VALIDATION', 'alpha 必须是 >= 0 的有限数值');
  }
  return alpha;
}

export function parseTrainParams(raw: unknown): TrainParams {
  const obj = (raw ?? {}) as Record<string, unknown>;
  const strategy = obj.unknownStrategy ?? 'laplace';
  if (strategy !== 'laplace' && strategy !== 'closed') {
    throw new AppError('VALIDATION', 'unknownStrategy 仅支持 laplace 或 closed');
  }
  return { alpha: parseAlpha(obj.alpha ?? 1), unknownStrategy: strategy };
}

export function parseSeed(value: unknown, fallback = 42): number {
  if (value === undefined || value === null) return fallback;
  const seed = Number(value);
  if (!Number.isInteger(seed)) {
    throw new AppError('VALIDATION', 'seed 必须是整数');
  }
  return seed;
}

export function parseRatio(value: unknown, fallback = 0.2): number {
  const ratio = value === undefined || value === null ? fallback : Number(value);
  if (!Number.isFinite(ratio) || ratio <= 0 || ratio >= 1) {
    throw new AppError('VALIDATION', 'valRatio 必须在 (0,1) 区间');
  }
  return ratio;
}

export function parsePage(query: URLSearchParams): { limit: number; offset: number } {
  const limit = query.has('limit') ? Number(query.get('limit')) : 50;
  const offset = query.has('offset') ? Number(query.get('offset')) : 0;
  if (!Number.isInteger(limit) || limit < 1 || limit > 500) {
    throw new AppError('VALIDATION', 'limit 必须是 1..500 的整数');
  }
  if (!Number.isInteger(offset) || offset < 0) {
    throw new AppError('VALIDATION', 'offset 必须是非负整数');
  }
  return { limit, offset };
}

export function parseId(value: string | undefined, name: string): number {
  const id = Number(value);
  if (!Number.isInteger(id) || id < 1) {
    throw new AppError('VALIDATION', `${name} 必须是正整数`);
  }
  return id;
}
