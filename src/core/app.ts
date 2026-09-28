import type { DB } from '../db/database.js';
import { CorpusService } from './corpus.js';
import { ModelStore } from './modelStore.js';
import type { Clock } from './types.js';
import { viterbi } from '../hmm/viterbi.js';
import { compareModels, evaluateModel, saveReport } from './evaluate.js';

export interface AppContext {
  db: DB;
  corpora: CorpusService;
  models: ModelStore;
  clock: Clock;
}

export function createAppContext(db: DB, clock: Clock = { now: () => new Date() }): AppContext {
  const corpora = new CorpusService(db, clock);
  const models = new ModelStore(db, corpora, clock);
  return { db, corpora, models, clock };
}

export function tagTokens(
  ctx: AppContext,
  modelId: number,
  tokens: string[],
  allowedTags?: string[]
) {
  const { model, meta } = ctx.models.loadModel(modelId);
  return viterbi(model, tokens, {
    modelId,
    modelVersion: String(meta.model_version),
    allowedTags,
  });
}

export function evaluateAndStore(ctx: AppContext, modelId: number) {
  const report = evaluateModel(ctx.models, ctx.db, modelId);
  const splitId = Number(ctx.models.getMeta(modelId).split_id);
  const reportId = saveReport(ctx.db, report, splitId, ctx.clock.now().toISOString());
  return { reportId, report };
}

export function compare(ctx: AppContext, ids: number[]) {
  return compareModels(ctx.models, ctx.db, ids);
}
