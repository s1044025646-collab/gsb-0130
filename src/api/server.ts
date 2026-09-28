import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import type { AddressInfo } from 'node:net';
import { AppError, httpStatusFor } from '../core/errors.js';
import {
  parseAlpha,
  parseId,
  parsePage,
  parseRatio,
  parseSeed,
  parseSentenceList,
  parseTrainParams,
  requireString,
} from '../core/validate.js';
import { compare, createAppContext, evaluateAndStore, tagTokens } from '../core/app.js';
import type { AppContext } from '../core/app.js';
import type { DB } from '../db/database.js';
import type { Clock } from '../core/types.js';

interface JsonRequest {
  method: string;
  url: URL;
  body: unknown;
}

function send(res: ServerResponse, status: number, payload: unknown): void {
  const json = JSON.stringify(payload);
  res.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'content-length': Buffer.byteLength(json),
  });
  res.end(json);
}

async function readJson(req: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) chunks.push(chunk as Buffer);
  const raw = Buffer.concat(chunks).toString('utf8');
  if (raw === '') return {};
  try {
    return JSON.parse(raw);
  } catch {
    throw new AppError('VALIDATION', '请求体必须是合法 JSON');
  }
}

export function createApiServer(db: DB, clock: Clock = { now: () => new Date() }): Server {
  const ctx: AppContext = createAppContext(db, clock);

  const server = createServer(async (req, res) => {
    try {
      const url = new URL(req.url ?? '/', 'http://127.0.0.1');
      if (url.pathname === '/health') {
        return send(res, 200, { status: 'ok', time: clock.now().toISOString() });
      }
      const body = req.method === 'GET' || req.method === 'HEAD' ? null : await readJson(req);
      await route(ctx, { method: req.method ?? 'GET', url, body }, res, send);
    } catch (err) {
      if (err instanceof AppError) {
        return send(res, httpStatusFor(err.code), {
          error: { code: err.code, message: err.message, details: err.details ?? null },
        });
      }
      return send(res, 500, {
        error: { code: 'INTERNAL', message: (err as Error).message },
      });
    }
  });

  return server;
}

async function route(
  ctx: AppContext,
  req: JsonRequest,
  res: ServerResponse,
  send: (res: ServerResponse, status: number, payload: unknown) => void
): Promise<void> {
  const p = req.url.pathname;
  const q = req.url.searchParams;
  const b = (req.body ?? {}) as Record<string, unknown>;

  // 语料库
  if (p === '/api/corpora' && req.method === 'POST') {
    const created = ctx.corpora.createCorpus(
      requireString(b.name, 'name'),
      typeof b.description === 'string' ? b.description : ''
    );
    return send(res, 201, { id: created.id, createdAt: created.createdAt });
  }
  if (p === '/api/corpora' && req.method === 'GET') {
    return send(res, 200, { data: ctx.corpora.listCorpora() });
  }

  const corpusMatch = /^\/api\/corpora\/(\d+)$/.exec(p);
  if (corpusMatch) {
    const id = parseId(corpusMatch[1], 'corpusId');
    if (req.method === 'GET') return send(res, 200, ctx.corpora.getCorpus(id));
  }

  // 导入版本
  const importMatch = /^\/api\/corpora\/(\d+)\/import$/.exec(p);
  if (importMatch && req.method === 'POST') {
    const id = parseId(importMatch[1], 'corpusId');
    const sentences = parseSentenceList(b.sentences);
    const version = ctx.corpora.importVersion(id, sentences);
    return send(res, version.reused ? 200 : 201, version);
  }

  const versionsMatch = /^\/api\/corpora\/(\d+)\/versions$/.exec(p);
  if (versionsMatch && req.method === 'GET') {
    const id = parseId(versionsMatch[1], 'corpusId');
    return send(res, 200, { data: ctx.corpora.listVersions(id) });
  }

  // 版本句子分页
  const sentencesMatch = /^\/api\/versions\/(\d+)\/sentences$/.exec(p);
  if (sentencesMatch && req.method === 'GET') {
    const id = parseId(sentencesMatch[1], 'versionId');
    const page = ctx.corpora.listSentences(id, parsePage(q));
    return send(res, 200, page);
  }

  // 划分
  const splitMatch = /^\/api\/versions\/(\d+)\/split$/.exec(p);
  if (splitMatch && req.method === 'POST') {
    const id = parseId(splitMatch[1], 'versionId');
    const seed = parseSeed(b.seed, 42);
    const valRatio = parseRatio(b.valRatio, 0.2);
    const result = ctx.corpora.getOrCreateSplit(id, { seed, valRatio });
    return send(res, 200, {
      splitId: result.splitId,
      trainSentences: result.train.length,
      valSentences: result.val.length,
    });
  }

  // 训练
  if (p === '/api/models/train' && req.method === 'POST') {
    const versionId = parseId(String(b.versionId), 'versionId');
    const seed = parseSeed(b.seed, 42);
    const valRatio = parseRatio(b.valRatio, 0.2);
    const params = parseTrainParams(b);
    const split = ctx.corpora.getOrCreateSplit(versionId, { seed, valRatio });
    const trained = ctx.models.train(versionId, split.splitId, params);
    return send(res, 201, trained);
  }

  if (p === '/api/models' && req.method === 'GET') {
    const corpusId = q.get('corpusId') ? parseId(q.get('corpusId')!, 'corpusId') : undefined;
    return send(res, 200, ctx.models.listModels(corpusId, parsePage(q)));
  }

  const modelMatch = /^\/api\/models\/(\d+)$/.exec(p);
  if (modelMatch && req.method === 'GET') {
    return send(res, 200, ctx.models.getMeta(parseId(modelMatch[1], 'modelId')));
  }

  // 标注
  if (p === '/api/tag' && req.method === 'POST') {
    const modelId = parseId(String(b.modelId), 'modelId');
    if (!Array.isArray(b.tokens)) {
      throw new AppError('VALIDATION', 'tokens 必须是字符串数组');
    }
    const tokens = b.tokens as unknown[];
    if (tokens.some((t) => typeof t !== 'string' || t.length === 0)) {
      throw new AppError('VALIDATION', 'tokens 中每个元素必须是非空字符串');
    }
    const allowedTags =
      b.allowedTags === undefined
        ? undefined
        : (() => {
            if (!Array.isArray(b.allowedTags)) {
              throw new AppError('VALIDATION', 'allowedTags 必须是字符串数组');
            }
            return b.allowedTags as string[];
          })();
    return send(res, 200, tagTokens(ctx, modelId, tokens as string[], allowedTags));
  }

  // 评测
  const evalMatch = /^\/api\/models\/(\d+)\/evaluate$/.exec(p);
  if (evalMatch && req.method === 'POST') {
    const id = parseId(evalMatch[1], 'modelId');
    return send(res, 200, evaluateAndStore(ctx, id));
  }

  const reportsMatch = /^\/api\/models\/(\d+)\/reports$/.exec(p);
  if (reportsMatch && req.method === 'GET') {
    const id = parseId(reportsMatch[1], 'modelId');
    ctx.models.getMeta(id);
    const { limit, offset } = parsePage(q);
    const total = Number(
      (ctx.db.prepare('SELECT COUNT(*) c FROM reports WHERE model_id=?').get(id) as {
        c: number;
      }).c
    );
    const data = ctx.db
      .prepare('SELECT * FROM reports WHERE model_id=? ORDER BY id LIMIT ? OFFSET ?')
      .all(id, limit, offset);
    return send(res, 200, { data, total, limit, offset });
  }

  // 比较
  if (p === '/api/models/compare' && req.method === 'POST') {
    if (!Array.isArray(b.modelIds) || b.modelIds.length < 2) {
      throw new AppError('VALIDATION', 'modelIds 必须是至少两个模型 id 的数组');
    }
    const ids = (b.modelIds as unknown[]).map((v) => parseId(String(v), 'modelId'));
    return send(res, 200, { data: compare(ctx, ids) });
  }

  // 显式消费 alpha 以保持校验集中（无操作端点不需要）
  void parseAlpha;
  send(res, 404, { error: { code: 'NOT_FOUND', message: `未知路径 ${p}` } });
}

export function listenLocal(server: Server): Promise<{ port: number; close: () => Promise<void> }> {
  return new Promise((resolve) => {
    server.listen(0, '127.0.0.1', () => {
      const port = (server.address() as AddressInfo).port;
      const close = () =>
        new Promise<void>((resolveClose, rejectClose) =>
          server.close((err) => (err ? rejectClose(err) : resolveClose()))
        );
      resolve({ port, close });
    });
  });
}
