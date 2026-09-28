import { mkdirSync, rmSync, existsSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { openDatabase } from '../src/db/schema.js';
import { Repository } from '../src/db/repository.js';
import { runCli } from '../src/cli/index.js';
import { startServer } from '../src/api/server.js';
import { buildFixtureCorpus } from '../fixtures/corpus.js';

const dataDir = resolve('data/demo');
if (existsSync(dataDir)) rmSync(dataDir, { recursive: true, force: true });
mkdirSync(dataDir, { recursive: true });
const dbPath = resolve(dataDir, 'demo.sqlite');
const fixturePath = resolve(dataDir, 'corpus.json');
writeFileSync(fixturePath, JSON.stringify(buildFixtureCorpus(), null, 2), 'utf-8');

console.log('=== 1. CLI 幂等导入与校验 ===');
runCli(['validate', fixturePath], dbPath);
runCli(['import', fixturePath], dbPath);
runCli(['import', fixturePath], dbPath);

console.log('\n=== 2. 固定种子划分与训练两个模型 ===');
runCli(['split', '1', '--seed', '42', '--ratio', '0.8'], dbPath);
runCli(['train', '1', '--smoothing', '1', '--unknown', 'uniform'], dbPath);
runCli(['train', '1', '--smoothing', '0.1', '--unknown', 'uniform'], dbPath);

console.log('\n=== 3. 标注与路径解释（含未知词） ===');
runCli(['explain', '1', '--sentence', '我 爱 一个从未出现的地方'], dbPath);

console.log('\n=== 4. 评测、导出与模型比较 ===');
const reportPath = resolve(dataDir, 'report.json');
runCli(['evaluate', '1', '--part', 'validation', '--out', reportPath], dbPath);
runCli(['compare', '1', '2'], dbPath);

console.log('\n=== 5. 语料修订后的版本隔离 ===');
const db = openDatabase(dbPath);
const repo = new Repository(db);
const revised = buildFixtureCorpus();
revised.sentences.push({ tokens: ['全新', '词'], tags: ['a', 'n'] });
const revision = repo.importCorpus(revised.name, revised.sentences);
console.log(`修订导入为 v${revision.corpus.version}，旧模型 id=1 仍可加载并重现预测`);
db.close();

console.log('\n=== 6. HTTP API（仅 127.0.0.1 自动端口） ===');
const server = await startServer(dbPath);
try {
  const response = await fetch(`http://127.0.0.1:${server.port}/health`);
  console.log('GET /health ->', response.status, await response.json());
  const tagged = await fetch(`http://127.0.0.1:${server.port}/models/1/tag`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ tokens: ['你', '去', '陌生城市'] })
  });
  console.log('POST /models/1/tag ->', tagged.status, JSON.stringify(await tagged.json()));
} finally {
  await server.close();
}

console.log(`\n演示完成，全部数据保留在 ${dataDir}（重跑自动重建）`);
