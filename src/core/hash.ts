import { createHash } from 'node:crypto';

/** 稳定的内容哈希（与对象 key 顺序无关，基于结构序列化）。 */
export function sha256(input: string): string {
  return createHash('sha256').update(input, 'utf8').digest('hex');
}

/** 单句内容指纹：相同 token/tag 序列（含重复样本）得到相同值。 */
export function sentenceFingerprint(tokens: string[], tags: string[]): string {
  return sha256(JSON.stringify([tokens, tags]));
}

/** 语料整体指纹：按句子指纹排序，避免导入顺序造成差异。 */
export function corpusFingerprint(fingerprints: string[]): string {
  const sorted = [...fingerprints].sort();
  return sha256(sorted.join('\n'));
}

/** 把 (seed, 句子指纹) 映射到 [0,1) 的确定值，用于固定划分。 */
export function unitHash(seed: number, fingerprint: string): number {
  const hex = sha256(`${seed}::${fingerprint}`).slice(0, 12);
  return parseInt(hex, 16) / 0xffffffffffff;
}
