#!/usr/bin/env node
import { writeFileSync, readFileSync, existsSync, rmSync } from 'node:fs';
import { resolve } from 'node:path';
import { openDatabase } from './db/database.js';
import { createApiServer, listenLocal } from './api/server.js';
import { createAppContext, compare, evaluateAndStore, tagTokens } from './core/app.js';
import { parseSentenceList, parseTrainParams } from './core/validate.js';
import { readJsonFile, fmt, header } from './cli/util.js';
import { AppError } from './core/errors.js';

const DATA_DIR = resolve(process.cwd(), '.data');
const DB_PATH = resolve(DATA_DIR, 'app.sqlite');
const PID_PATH = resolve(DATA_DIR, 'server.pid');
const PORT_PATH = resolve(DATA_DIR, 'server.port');

function arg(name: string, fallback?: string): string | undefined {
  const i = process.argv.indexOf(`--${name}`);
  if (i >= 0 && process.argv[i + 1]) return process.argv[i + 1];
  const eq = process.argv.find((a) => a.startsWith(`--${name}=`));
  return eq ? eq.slice(name.length + 3) : fallback;
}

function numbersFromArg(name: string): number[] {
  const v = arg(name);
  if (!v) return [];
  return v.split(',').map((s) => {
    const n = Number(s.trim());
    if (!Number.isInteger(n) || n < 1) throw new AppError('VALIDATION', `${name} 需为正整数列表`);
    return n;
  });
}

function printHelp(): void {
  console.log(`zh-pos —— 中文词性标注 / 一阶 HMM 评测 CLI

用法:
  init-db                         初始化（或重复执行）数据库迁移
  corpus-add --name <名称>        创建语料库
  corpus-list                     列出语料库
  import --corpus <id> --file <json>
                                  导入并校验语料（幂等）
  versions --corpus <id>          列出版本
  sentences --version <id> [--limit N] [--offset N]
                                  分页查看句子
  split --version <id> [--seed N] [--ratio 0..1]
                                  固定种子划分训练/验证
  train --version <id> [--seed N] [--ratio R] [--alpha N] [--unknown laplace|closed]
  tag --model <id> --file <json>  标注（JSON: {"tokens":[...],"allowedTags":[...]}）
  explain --model <id> --file <json>
                                  标注并打印逐位置路径解释
  evaluate --model <id>           验证集评测（逐标签 P/R/F1 + 混淆矩阵）
  compare --models 1,2,3          比较多个模型
  serve                           本机 HTTP 服务（自动端口，写 .data/server.pid）
  stop                            停止本项目服务进程

全局:
  --db <path>                     指定 SQLite 路径（默认 .data/app.sqlite）
`);
}

async function main(): Promise<void> {
  const cmd = process.argv[2];
  const dbPath = arg('db', DB_PATH)!;
  const db = openDatabase(dbPath);
  const ctx = createAppContext(db);

  try {
    switch (cmd) {
      case undefined:
      case 'help':
      case '--help':
      case '-h':
        printHelp();
        break;

      case 'init-db': {
        header('数据库状态');
        console.log('迁移已应用，路径:', dbPath);
        const v = db.prepare("SELECT value FROM schema_meta WHERE key='schema_version'").get() as
          | { value: string }
          | undefined;
        console.log('schema_version =', v?.value);
        break;
      }

      case 'corpus-add': {
        const before = (db.prepare('SELECT COUNT(*) c FROM corpora').get() as { c: number }).c;
        const name = arg('name') ?? '';
        const created = ctx.corpora.createCorpus(name, arg('description') ?? '');
        const after = (db.prepare('SELECT COUNT(*) c FROM corpora').get() as { c: number }).c;
        header('corpus-add');
        console.log(`前: ${before} 个语料 -> 后: ${after} 个；新 id=${created.id}`);
        break;
      }

      case 'corpus-list':
        console.table(ctx.corpora.listCorpora());
        break;

      case 'import': {
        const corpusId = Number(arg('corpus')!);
        const file = arg('file')!;
        const sentences = parseSentenceList(readJsonFile(file));
        header('导入校验');
        console.log(`读取 ${sentences.length} 条（含可能重复），语料 id=${corpusId}`);
        const before = ctx.corpora.listVersions(corpusId).length;
        const v = ctx.corpora.importVersion(corpusId, sentences);
        const after = ctx.corpora.listVersions(corpusId).length;
        console.log(
          v.reused ? '内容指纹命中既有版本（幂等，不新增计数）' : '创建新版本'
        );
        console.log(
          `版本前:${before} -> 后:${after}；version=${v.version} 句子=${v.sentenceCount} 词次=${v.tokenCount}`
        );
        break;
      }

      case 'versions':
        console.table(ctx.corpora.listVersions(Number(arg('corpus')!)));
        break;

      case 'sentences': {
        const versionId = Number(arg('version')!);
        const limit = Number(arg('limit', '50'));
        const offset = Number(arg('offset', '0'));
        const page = ctx.corpora.listSentences(versionId, { limit, offset });
        header(`版本 ${versionId} 句子 (total=${page.total})`);
        console.table(
          page.data.map((r) => ({
            ord: r.ord,
            length: r.length,
            tokens: r.tokens,
            tags: r.tags,
          }))
        );
        break;
      }

      case 'split': {
        const versionId = Number(arg('version')!);
        const seed = Number(arg('seed', '42'));
        const ratio = Number(arg('ratio', '0.2'));
        const result = ctx.corpora.getOrCreateSplit(versionId, { seed, valRatio: ratio });
        header(`划分 seed=${seed} ratio=${ratio}`);
        console.log(`splitId=${result.splitId} 训练句=${result.train.length} 验证句=${result.val.length}`);
        const overlap = new Set(result.train.map((r) => r.fingerprint));
        const leaked = result.val.filter((r) => overlap.has(r.fingerprint));
        console.log(`跨集重复句（应为 0）: ${leaked.length}`);
        break;
      }

      case 'train': {
        const versionId = Number(arg('version')!);
        const seed = Number(arg('seed', '42'));
        const ratio = Number(arg('ratio', '0.2'));
        const params = parseTrainParams({
          alpha: arg('alpha', '1'),
          unknownStrategy: arg('unknown', 'laplace'),
        });
        const split = ctx.corpora.getOrCreateSplit(versionId, { seed, valRatio: ratio });
        const before = (db.prepare('SELECT COUNT(*) c FROM models').get() as { c: number }).c;
        const res = ctx.models.train(versionId, split.splitId, params);
        const after = (db.prepare('SELECT COUNT(*) c FROM models').get() as { c: number }).c;
        const meta = ctx.models.getMeta(res.modelId);
        header('训练');
        console.log(`模型数 前:${before} -> 后:${after}`);
        console.log(
          `id=${res.modelId} version=${res.modelVersion} 标签=${meta.tag_count} 词表=${meta.vocab_size} 训练句=${meta.train_sentences} 训练词次=${meta.train_tokens}`
        );
        console.log(`训练集对数似然=${fmt(Number(meta.log_likelihood_train), 2)}`);
        break;
      }

      case 'tag':
      case 'explain': {
        const modelId = Number(arg('model')!);
        const payload = readJsonFile(arg('file')!) as Record<string, unknown>;
        const tokens = payload.tokens as string[];
        const result = tagTokens(ctx, modelId, tokens, payload.allowedTags as string[] | undefined);
        header(cmd === 'explain' ? '标注 + 路径解释' : '标注');
        console.log('tokens :', result.tokens.join(' '));
        console.log('tags   :', result.tags.join(' '));
        console.log(`score(对数)=${fmt(result.score, 6)} 未知词=${result.usedUnknown}`);
        if (cmd === 'explain') {
          console.table(
            result.path.map((s) => ({
              pos: s.position,
              token: s.token,
              tag: s.tag,
              from: s.fromTag,
              trans: fmt(s.transLogProb, 4),
              emit: fmt(s.emitLogProb, 4),
              cum: fmt(s.cumulative, 4),
            }))
          );
          console.log(`句尾 logP(^->$)=${fmt(result.endLogProb, 4)}`);
        }
        break;
      }

      case 'evaluate': {
        const modelId = Number(arg('model')!);
        const { reportId, report } = evaluateAndStore(ctx, modelId);
        header(`模型 ${modelId} 验证集评测 (reportId=${reportId})`);
        console.log(
          `句子=${report.sentences} 词次=${report.tokens} 句准确率=${fmt(report.sentenceAccuracy)} 词准确率=${fmt(report.tokenAccuracy)}`
        );
        console.log(
          `macro P=${fmt(report.macroPrecision)} R=${fmt(report.macroRecall)} F1=${fmt(report.macroF1)}`
        );
        console.table(
          report.perTag.map((m) => ({
            tag: m.tag,
            tp: m.tp,
            fp: m.fp,
            fn: m.fn,
            P: fmt(m.precision),
            R: fmt(m.recall),
            F1: fmt(m.f1),
            support: m.support,
          }))
        );
        if (report.confusion.length > 0) {
          console.log('混淆矩阵（仅列错误，行=金标 列=预测）:');
          console.table(report.confusion);
        }
        break;
      }

      case 'compare': {
        const ids = numbersFromArg('models');
        const rows = compare(ctx, ids).map((r) => ({
          id: r.modelId,
          version: r.modelVersion,
          tokenAcc: fmt(r.report.tokenAccuracy),
          sentenceAcc: fmt(r.report.sentenceAccuracy),
          macroF1: fmt(r.report.macroF1),
        }));
        header('模型比较');
        console.table(rows);
        break;
      }

      case 'serve': {
        const server = createApiServer(db);
        const { port, close } = await listenLocal(server);
        writeFileSync(PID_PATH, String(process.pid));
        writeFileSync(PORT_PATH, String(port));
        header('服务已启动（仅 127.0.0.1，自动端口）');
        console.log(`port=${port} pid=${process.pid} db=${dbPath}`);
        console.log('健康检查: curl http://127.0.0.1:' + port + '/health');
        const shutdown = async () => {
          await close();
          db.close();
          rmSync(PID_PATH, { force: true });
          rmSync(PORT_PATH, { force: true });
          process.exit(0);
        };
        process.on('SIGINT', shutdown);
        process.on('SIGTERM', shutdown);
        return;
      }

      case 'stop': {
        if (!existsSync(PID_PATH)) {
          console.log('未找到运行中的本项目服务（无 PID 文件）。');
          break;
        }
        const pid = Number(readFileSync(PID_PATH, 'utf8').trim());
        console.log(`停止本项目服务进程 pid=${pid}`);
        try {
          process.kill(pid, 'SIGTERM');
          console.log('已发送 SIGTERM。');
          // Windows 下 SIGTERM 可能直接终止进程而来不及执行清理钩子，
          // 等待其退出后由本命令清理本项目的 pid/port 文件。
          for (let i = 0; i < 20; i++) {
            await new Promise((r) => setTimeout(r, 100));
            try {
              process.kill(pid, 0);
            } catch {
              break;
            }
          }
        } catch (e) {
          console.log('进程已不存在，清理 PID 文件。', (e as Error).message);
        }
        rmSync(PID_PATH, { force: true });
        rmSync(PORT_PATH, { force: true });
        console.log('已清理本项目 pid/port 文件。');
        break;
      }

      default:
        console.error(`未知命令: ${cmd}\n`);
        printHelp();
        process.exitCode = 2;
    }
  } catch (err) {
    if (err instanceof AppError) {
      console.error(`错误[${err.code}]: ${err.message}`);
      if (err.details) console.error('详情:', JSON.stringify(err.details));
      process.exitCode = 1;
    } else {
      throw err;
    }
  } finally {
    if (cmd !== 'serve') db.close();
  }
}

main();
