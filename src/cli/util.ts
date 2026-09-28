import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

export function readJsonFile(path: string): unknown {
  return JSON.parse(readFileSync(resolve(path), 'utf8'));
}

export function fmt(n: number, digits = 4): string {
  if (!Number.isFinite(n)) return '-∞';
  return n.toFixed(digits);
}

export function header(title: string): void {
  console.log(`\n=== ${title} ===`);
}
