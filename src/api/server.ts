import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { pathToFileURL } from 'node:url';
import { openDatabase } from '../db/schema.js';
import { Repository, type Clock, systemClock } from '../db/repository.js';
import { validateCorpusInput } from '../core/validation.js';
import { AppError, type UnknownStrategy } from '../core/types.js';
import { viterbi } from '../core/hmm.js';

interface ParsedRequest {
  body: unknown;
  query: URLSearchParams;
}

function sendJson(response: ServerResponse, status: number, payload: unknown): void {
  const text = JSON.stringify(payload, null, 2);
  response.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'content-length': Buffer.byteLength(text)
  });
  response.end(text);
}

function parsePagination(query: URLSearchParams): { limit: number; offset: number } {
  const rawLimit = query.get('limit') ?? '50';
  const rawOffset = query.get('offset') ?? '0';
  const limit = Number(rawLimit);
  const offset = Number(rawOffset);
  if (!Number.isInteger(limit) || limit < 1 || limit > 200) {
    throw new AppError('INVALID_QUERY', 'limit 必须是 1-200 的整数');
  }
  if (!Number.isInteger(offset) || offset < 0) {
    throw new AppError('INVALID_QUERY', 'offset 必须是非负整数');
  }
  return { limit, offset };
}

export function createApp(repo: Repository): Server {
  async function readRequest(request: IncomingMessage): Promise<ParsedRequest> {
    const chunks: Buffer[] = [];
    for await (const chunk of request) chunks.push(chunk as Buffer);
    const raw = Buffer.concat(chunks).toString('utf-8');
    let body: unknown = null;
    if (raw.length > 0) {
      try {
        body = JSON.parse(raw);
      } catch {
        throw new AppError('INVALID_JSON', '请求体不是合法 JSON');
      }
    }
    return { body, query: new URL(request.url ?? '/', 'http://localhost').searchParams };
  }

  const server = createServer(async (request, response) => {
    try {
      const method = request.method ?? 'GET';
      const url = new URL(request.url ?? '/', 'http://localhost');
      const path = url.pathname.replace(/\/+$/, '') || '/';
      const { body, query } = await readRequest(request);

      if (method === 'GET' && path === '/health') {
        sendJson(response, 200, { status: 'ok' });
        return;
      }

      if (method === 'POST' && path === '/corpora') {
        const input = validateCorpusInput(body);
        const result = repo.importCorpus(input.name, input.sentences);
        sendJson(response, result.reused ? 200 : 201, result);
        return;
      }
      if (method === 'GET' && path === '/corpora') {
        sendJson(response, 200, repo.listCorpora(...Object.values(parsePagination(query)) as [number, number]));
        return;
      }
      const corpusMatch = /^\/corpora\/(\d+)$/.exec(path);
      if (method === 'GET' && corpusMatch) {
        sendJson(response, 200, {
          corpus: repo.getCorpus(Number(corpusMatch[1])),
          sentences: repo.getSentences(Number(corpusMatch[1]))
        });
        return;
      }
      const splitMatch = /^\/corpora\/(\d+)\/splits$/.exec(path);
      if (method === 'POST' && splitMatch) {
        const payload = (body ?? {}) as { seed?: unknown; trainRatio?: unknown };
        const seed = payload.seed;
        const trainRatio = payload.trainRatio ?? 0.8;
        if (typeof seed !== 'number' || !Number.isInteger(seed)) {
          throw new AppError('INVALID_BODY', 'seed 必须是整数');
        }
        if (typeof trainRatio !== 'number' || trainRatio <= 0 || trainRatio > 1) {
          throw new AppError('INVALID_BODY', 'trainRatio 必须在 (0, 1] 范围内');
        }
        sendJson(response, 201, repo.createSplit(Number(splitMatch[1]), seed, trainRatio));
        return;
      }

      if (method === 'POST' && path === '/models') {
        const payload = (body ?? {}) as {
          splitId?: unknown;
          smoothing?: unknown;
          unknownStrategy?: unknown;
        };
        if (typeof payload.splitId !== 'number' || !Number.isInteger(payload.splitId)) {
          throw new AppError('INVALID_BODY', 'splitId 必须是整数');
        }
        const smoothing = payload.smoothing ?? 1;
        const strategy = (payload.unknownStrategy ?? 'uniform') as UnknownStrategy;
        if (typeof smoothing !== 'number' || smoothing < 0 || !Number.isFinite(smoothing)) {
          throw new AppError('INVALID_BODY', 'smoothing 必须是非负有限数');
        }
        if (strategy !== 'uniform' && strategy !== 'closed') {
          throw new AppError('INVALID_BODY', "unknownStrategy 必须是 'uniform' 或 'closed'");
        }
        sendJson(response, 201, repo.trainModel(payload.splitId, smoothing, strategy));
        return;
      }
      if (method === 'GET' && path === '/models') {
        sendJson(
          response,
          200,
          repo.listModels(
            ...Object.values(parsePagination(query)) as [number, number]
          )
        );
        return;
      }

      const tagMatch = /^\/models\/(\d+)\/tag$/.exec(path);
      if (method === 'POST' && tagMatch) {
        const payload = (body ?? {}) as { tokens?: unknown; allowedTags?: unknown };
        if (!Array.isArray(payload.tokens) || payload.tokens.some((token) => typeof token !== 'string')) {
          throw new AppError('INVALID_BODY', 'tokens 必须是字符串数组');
        }
        if (
          payload.allowedTags !== undefined &&
          (!Array.isArray(payload.allowedTags) ||
            payload.allowedTags.some((tag) => typeof tag !== 'string'))
        ) {
          throw new AppError('INVALID_BODY', 'allowedTags 必须是字符串数组');
        }
        const params = repo.loadModel(Number(tagMatch[1]));
        const result = viterbi(
          params,
          payload.tokens as string[],
          payload.allowedTags as string[] | undefined
        );
        sendJson(response, 200, result);
        return;
      }

      const reportMatch = /^\/models\/(\d+)\/reports$/.exec(path);
      if (method === 'POST' && reportMatch) {
        const payload = (body ?? {}) as { part?: unknown };
        const part = payload.part ?? 'validation';
        if (part !== 'validation' && part !== 'train') {
          throw new AppError('INVALID_BODY', "part 必须是 'validation' 或 'train'");
        }
        sendJson(response, 201, repo.evaluateModel(Number(reportMatch[1]), part));
        return;
      }
      if (method === 'GET' && path === '/reports') {
        const modelId = query.get('modelId');
        sendJson(
          response,
          200,
          repo.listReports(
            modelId === null ? undefined : Number(modelId),
            ...Object.values(parsePagination(query)) as [number, number]
          )
        );
        return;
      }
      const singleReport = /^\/reports\/(\d+)$/.exec(path);
      if (method === 'GET' && singleReport) {
        sendJson(response, 200, repo.getReport(Number(singleReport[1])));
        return;
      }

      sendJson(response, 404, { error: { code: 'NOT_FOUND', message: `路由不存在: ${method} ${path}` } });
    } catch (error) {
      if (error instanceof AppError) {
        const status =
          error.code === 'NOT_FOUND'
            ? 404
            : error.code === 'TAG_NOT_IN_MODEL' || error.code === 'NO_FEASIBLE_PATH'
              ? 422
              : 400;
        sendJson(response, status, { error: { code: error.code, message: error.message, details: error.details } });
        return;
      }
      sendJson(response, 500, { error: { code: 'INTERNAL', message: (error as Error).message } });
    }
  });

  return server;
}

export interface RunningServer {
  port: number;
  close: () => Promise<void>;
}

export function startServer(dbPath: string, clock: Clock = systemClock): Promise<RunningServer> {
  const db = openDatabase(dbPath);
  const repo = new Repository(db, clock);
  const server = createApp(repo);
  return new Promise((resolve) => {
    server.listen(0, '127.0.0.1', () => {
      const address = server.address();
      const port = typeof address === 'object' && address ? address.port : 0;
      resolve({
        port,
        close: () =>
          new Promise<void>((resolveClose, reject) => {
            server.close((error) => (error ? reject(error) : resolveClose()));
            db.close();
          })
      });
    });
  });
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const dbPath = process.env.HMM_DB_PATH ?? 'data/app.sqlite';
  startServer(dbPath).then(({ port }) => {
    console.log(`HMM 标注服务已启动: http://127.0.0.1:${port} (数据库 ${dbPath})`);
  });
}
