import { createHash } from 'node:crypto';
import type { TaggedSentence } from './types.js';
import { AppError } from './types.js';

export function validateSentence(raw: unknown, index: number): TaggedSentence {
  if (typeof raw !== 'object' || raw === null) {
    throw new AppError('INVALID_SENTENCE', `第 ${index} 句不是对象`, { index });
  }
  const obj = raw as Record<string, unknown>;
  if (!Array.isArray(obj.tokens) || !Array.isArray(obj.tags)) {
    throw new AppError('INVALID_SENTENCE', `第 ${index} 句必须包含 tokens 与 tags 数组`, { index });
  }
  const tokens = obj.tokens;
  const tags = obj.tags;
  if (tokens.length !== tags.length) {
    throw new AppError(
      'INVALID_SENTENCE',
      `第 ${index} 句 tokens(${tokens.length}) 与 tags(${tags.length}) 长度不一致`,
      { index }
    );
  }
  for (const token of tokens) {
    if (typeof token !== 'string' || token.length === 0) {
      throw new AppError('INVALID_SENTENCE', `第 ${index} 句存在空或非字符串 token`, { index });
    }
  }
  for (const tag of tags) {
    if (typeof tag !== 'string' || tag.length === 0) {
      throw new AppError('INVALID_SENTENCE', `第 ${index} 句存在空或非字符串 tag`, { index });
    }
  }
  return { tokens: [...tokens], tags: [...tags] };
}

export function validateCorpusInput(raw: unknown): { name: string; sentences: TaggedSentence[] } {
  if (typeof raw !== 'object' || raw === null) {
    throw new AppError('INVALID_CORPUS', '语料必须是对象');
  }
  const obj = raw as Record<string, unknown>;
  if (typeof obj.name !== 'string' || obj.name.trim().length === 0) {
    throw new AppError('INVALID_CORPUS', 'name 必须是非空字符串');
  }
  if (obj.name.length > 100) {
    throw new AppError('INVALID_CORPUS', 'name 长度不能超过 100');
  }
  if (!Array.isArray(obj.sentences)) {
    throw new AppError('INVALID_CORPUS', 'sentences 必须是数组');
  }
  const sentences = obj.sentences.map((sentence, index) => validateSentence(sentence, index));
  return { name: obj.name, sentences };
}

export function sha256(text: string): string {
  return createHash('sha256').update(text).digest('hex');
}

export function sentenceFingerprint(sentence: TaggedSentence): string {
  return sha256(JSON.stringify([sentence.tokens, sentence.tags]));
}

export function corpusFingerprint(name: string, sentences: TaggedSentence[]): string {
  return sha256(JSON.stringify({ name, sentences }));
}
