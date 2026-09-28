import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { buildFixtureCorpus } from '../fixtures/corpus.js';

mkdirSync(resolve('data'), { recursive: true });
writeFileSync(resolve('fixtures/synthetic-zh.json'), JSON.stringify(buildFixtureCorpus(), null, 2), 'utf-8');
console.log('已写出 fixtures/synthetic-zh.json');
