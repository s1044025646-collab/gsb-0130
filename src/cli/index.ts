import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { openDatabase } from '../db/schema.js';
import { Repository } from '../db/repository.js';
import { validateCorpusInput } from '../core/validation.js';
import { viterbi } from '../core/hmm.js';
import { AppError, type UnknownStrategy } from '../core/types.js';

interface CommandContext {
  repo: Repository;
  args: string[];
  flags: Map<string, string>;
}

function parseFlags(args: string[]): { positional: string[]; flags: Map<string, string> } {
  const flags = new Map<string, string>();
  const positional: string[] = [];
  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (arg.startsWith('--')) {
      const key = arg.slice(2);
      const next = args[index + 1];
      if (next !== undefined && !next.startsWith('--')) {
        flags.set(key, next);
        index += 1;
      } else {
        flags.set(key, 'true');
      }
    } else {
      positional.push(arg);
    }
  }
  return { positional, flags };
}

function readJsonFile(path: string): unknown {
  return JSON.parse(readFileSync(resolve(path), 'utf-8'));
}

export const commands: Record<string, (context: CommandContext) => void> = {
  validate({ args }) {
    const [file] = args;
    if (!file) throw new AppError('CLI_USAGE', '用法: cli validate <corpus.json>');
    const raw = readJsonFile(file);
    const result = validateCorpusInput(raw);
    const tags = new Set<string>();
    let tokens = 0;
    for (const sentence of result.sentences) {
      tokens += sentence.tokens.length;
      sentence.tags.forEach((tag) => tags.add(tag));
    }
    console.log(`校验通过: ${result.sentences.length} 句, ${tokens} 词, ${tags.size} 个标签`);
  },

  import({ repo, args }) {
    const [file] = args;
    if (!file) throw new AppError('CLI_USAGE', '用法: cli import <corpus.json>');
    const before = repo.listCorpora(1, 0).total;
    const input = validateCorpusInput(readJsonFile(file));
    const { corpus, reused } = repo.importCorpus(input.name, input.sentences);
    console.log(`[前] 语料版本总数: ${before}`);
    console.log(
      reused
        ? `内容未变化，复用版本 v${corpus.version} (id=${corpus.id})，训练计数未重复增加`
        : `已导入新语料版本 v${corpus.version} (id=${corpus.id})`
    );
    console.log(`[后] ${corpus.sentence_count} 句 / ${corpus.token_count} 词 / ${corpus.tag_count} 标签`);
  },

  corpora({ repo, flags }) {
    const limit = Number(flags.get('limit') ?? '50');
    const offset = Number(flags.get('offset') ?? '0');
    const page = repo.listCorpora(limit, offset);
    console.log(`共 ${page.total} 个语料版本:`);
    for (const corpus of page.items) {
      console.log(
        `  id=${corpus.id} ${corpus.name} v${corpus.version} ${corpus.sentence_count}句 ${corpus.token_count}词`
      );
    }
  },

  split({ repo, args, flags }) {
    const corpusId = Number(args[0]);
    const seed = Number(flags.get('seed') ?? '42');
    const ratio = Number(flags.get('ratio') ?? '0.8');
    if (!Number.isInteger(corpusId)) throw new AppError('CLI_USAGE', '用法: cli split <corpusId> --seed 42 --ratio 0.8');
    const result = repo.createSplit(corpusId, seed, ratio);
    console.log(
      `${result.reused ? '复用已有划分' : '创建划分'} splitId=${result.splitId} 训练=${result.train} 验证=${result.validation}`
    );
  },

  train({ repo, args, flags }) {
    const splitId = Number(args[0]);
    const smoothing = Number(flags.get('smoothing') ?? '1');
    const strategy = (flags.get('unknown') ?? 'uniform') as UnknownStrategy;
    if (!Number.isInteger(splitId)) throw new AppError('CLI_USAGE', '用法: cli train <splitId> --smoothing 1 --unknown uniform');
    const before = repo.listModels(1, 0).total;
    console.log(`[前] 模型总数: ${before}`);
    const result = repo.trainModel(splitId, smoothing, strategy);
    console.log(`${result.reused ? '参数完全相同，复用' : '已训练新'}模型 id=${result.modelId}`);
    console.log(`[后] 模型总数: ${repo.listModels(1, 0).total}`);
  },

  models({ repo }) {
    const page = repo.listModels(50, 0);
    console.log(`共 ${page.total} 个模型:`);
    for (const model of page.items as Array<Record<string, number | string>>) {
      console.log(
        `  id=${model.id} split=${model.split_id} 语料v${model.corpus_version} smoothing=${model.smoothing} unknown=${model.unknown_strategy}`
      );
    }
  },

  tag({ repo, args, flags }) {
    const modelId = Number(args[0]);
    const sentence = flags.get('sentence');
    const allowed = flags.get('allowed-tags');
    if (!Number.isInteger(modelId) || sentence === undefined) {
      throw new AppError('CLI_USAGE', '用法: cli tag <modelId> --sentence "我 爱 北京" [--allowed-tags "n,v"]');
    }
    const tokens = sentence.split(/\s+/).filter(Boolean);
    const allowedTags = allowed === undefined ? undefined : allowed.split(',');
    const params = repo.loadModel(modelId);
    const result = viterbi(params, tokens, allowedTags);
    console.log(`未知词位置: [${result.unknownPositions.join(', ')}]`);
    for (const step of result.steps) {
      console.log(
        `  ${String(step.position).padStart(2)} ${step.token} -> ${step.chosen_tag}  Δ=${step.step_log_prob.toFixed(4)} 累计=${step.cumulative_log_prob.toFixed(4)}`
      );
    }
    console.log(`最佳标签序列: ${result.tags.join(' ')}`);
    console.log(`路径对数分数: ${result.score}`);
  },

  explain({ repo, args, flags }) {
    commands.tag({ repo, args, flags });
  },

  evaluate({ repo, args, flags }) {
    const modelId = Number(args[0]);
    const part = (flags.get('part') ?? 'validation') as 'train' | 'validation';
    if (!Number.isInteger(modelId)) throw new AppError('CLI_USAGE', '用法: cli evaluate <modelId> --part validation');
    const result = repo.evaluateModel(modelId, part);
    console.log(`${result.reused ? '复用已有' : '生成新'}报告 id=${result.reportId} (${part})`);
    console.log(
      `token准确率=${(result.summary.accuracy * 100).toFixed(2)}% 整句=${result.summary.sentenceCorrect}/${result.summary.sentenceTotal} 对数似然=${result.summary.logLikelihood.toFixed(4)}`
    );
    for (const metric of result.tagMetrics) {
      console.log(
        `  ${metric.tag.padEnd(6)} P=${metric.precision.toFixed(3)} R=${metric.recall.toFixed(3)} F1=${metric.f1.toFixed(3)} support=${metric.support}`
      );
    }
    if (flags.get('out')) {
      const payload = {
        summary: result.summary,
        perTag: result.tagMetrics,
        confusion: result.confusion
      };
      writeFileSync(resolve(flags.get('out') as string), JSON.stringify(payload, null, 2), 'utf-8');
      console.log(`已导出逐标签指标与混淆矩阵: ${flags.get('out')}`);
    }
  },

  compare({ repo, args }) {
    const modelIds = args.map(Number);
    if (modelIds.length < 2 || modelIds.some((id) => !Number.isInteger(id))) {
      throw new AppError('CLI_USAGE', '用法: cli compare <modelA> <modelB> [...]');
    }
    console.log('modelId | 准确率 | 整句正确率 | 对数似然 | smoothing | unknown');
    for (const modelId of modelIds) {
      const result = repo.evaluateModel(modelId, 'validation');
      const model = repo.listModels(50, 0).items.find((item) => (item as { id: number }).id === modelId) as
        | { smoothing: number; unknown_strategy: string }
        | undefined;
      console.log(
        `${modelId} | ${result.summary.accuracy.toFixed(4)} | ${result.summary.sentenceCorrect}/${result.summary.sentenceTotal} | ${result.summary.logLikelihood.toFixed(2)} | ${model?.smoothing} | ${model?.unknown_strategy}`
      );
    }
  }
};

export function runCli(argv: string[], dbPath = 'data/app.sqlite'): void {
  const [commandName, ...rest] = argv;
  const { positional, flags } = parseFlags(rest);
  const db = openDatabase(dbPath);
  try {
    const repo = new Repository(db);
    const handler = commandName ? commands[commandName] : undefined;
    if (!handler) {
      console.log('可用命令: validate, import, corpora, split, train, models, tag, explain, evaluate, compare');
      if (commandName) throw new AppError('CLI_USAGE', `未知命令: ${commandName}`);
      return;
    }
    handler({ repo, args: positional, flags });
  } finally {
    db.close();
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    runCli(process.argv.slice(2), process.env.HMM_DB_PATH ?? 'data/app.sqlite');
  } catch (error) {
    if (error instanceof AppError) {
      console.error(`错误[${error.code}]: ${error.message}`);
      process.exitCode = 1;
    } else {
      throw error;
    }
  }
}
