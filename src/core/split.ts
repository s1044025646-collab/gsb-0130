export function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let mixed = state;
    mixed = Math.imul(mixed ^ (mixed >>> 15), mixed | 1);
    mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
  };
}

export interface SplitAssignment {
  index: number;
  part: 'train' | 'validation';
}

export function deterministicSplit(
  count: number,
  seed: number,
  trainRatio: number
): SplitAssignment[] {
  if (!Number.isFinite(trainRatio) || trainRatio <= 0 || trainRatio > 1) {
    throw new Error('trainRatio 必须在 (0, 1] 范围内');
  }
  const order = Array.from({ length: count }, (_, index) => index);
  const random = mulberry32(seed);
  for (let index = order.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(random() * (index + 1));
    [order[index], order[swapIndex]] = [order[swapIndex], order[index]];
  }
  const trainCount = Math.round(count * trainRatio);
  const parts: SplitAssignment[] = count === 0
    ? []
    : order.map((originalIndex, position) => ({
        index: originalIndex,
        part: (position < Math.max(1, trainCount) ? 'train' : 'validation') as 'train' | 'validation'
      }));
  parts.sort((left, right) => left.index - right.index);
  return parts;
}
