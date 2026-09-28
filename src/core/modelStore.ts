import type { DB } from '../db/database.js';
import { AppError } from './errors.js';
import { sha256 } from './hash.js';
import { decodeRow, CorpusService } from './corpus.js';
import type { Clock, TrainParams } from './types.js';
import { countTraining, trainModel, type TrainedModel } from '../hmm/model.js';

/**
 * 训练一个模型并完整持久化计数。预测时由计数重新归一化重建模型，
 * 因此即使代码/时间变化，给定同一模型行也能重现历史预测。
 */
export class ModelStore {
  constructor(
    private db: DB,
    private corpora: CorpusService,
    private clock: Clock = { now: () => new Date() }
  ) {}

  train(
    versionId: number,
    splitId: number,
    params: TrainParams
  ): { modelId: number; modelVersion: string } {
    this.corpora.getVersion(versionId);
    const split = this.db.prepare('SELECT * FROM splits WHERE id=?').get(splitId) as
      | Record<string, number>
      | undefined;
    if (!split) throw new AppError('NOT_FOUND', `划分 ${splitId} 不存在`);
    if (Number(split.version_id) !== versionId) {
      throw new AppError('CONFLICT', '划分不属于该语料版本');
    }

    const { train } = this.corpora.getOrCreateSplit(versionId, {
      seed: Number(split.seed),
      valRatio: Number(split.val_ratio),
    });
    if (train.length === 0) {
      throw new AppError('INSUFFICIENT_DATA', '训练集为空，无法训练');
    }
    const trainSentences = train.map(decodeRow);
    const counts = countTraining(trainSentences);
    const trained = trainModel(counts, params);

    const revisionRow = this.db
      .prepare('SELECT COALESCE(MAX(revision),0) m FROM models WHERE version_id=?')
      .get(versionId) as { m: number };
    const revision = revisionRow.m + 1;
    const corpusRow = this.db
      .prepare('SELECT corpus_id FROM corpus_versions WHERE id=?')
      .get(versionId) as { corpus_id: number };
    const corpusId = corpusRow.corpus_id;
    const base = `m-v${versionId}-r${revision}`;
    const modelVersion = `${base}-${sha256(
      JSON.stringify({
        params,
        split: splitId,
        fp: this.corpora.getVersion(versionId).fingerprint,
      })
    ).slice(0, 10)}`;

    const dup = this.db
      .prepare(
        `SELECT id FROM models
         WHERE version_id=? AND split_id=? AND alpha=? AND unknown_strategy=?`
      )
      .get(versionId, splitId, params.alpha, params.unknownStrategy) as
      | { id: number }
      | undefined;
    if (dup) {
      // 相同配置重复训练：幂等返回既有模型，不新增计数行。
      const row = this.db.prepare('SELECT model_version FROM models WHERE id=?').get(dup.id) as {
        model_version: string;
      };
      return { modelId: dup.id, modelVersion: row.model_version };
    }

    const createdAt = this.clock.now().toISOString();
    this.db.exec('BEGIN');
    try {
      const res = this.db
        .prepare(
          `INSERT INTO models(corpus_id,version_id,split_id,model_version,revision,alpha,
            unknown_strategy,vocab_size,tag_count,train_sentences,train_tokens,
            log_likelihood_train,created_at)
           VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)`
        )
        .run(
          corpusId,
          versionId,
          splitId,
          modelVersion,
          revision,
          params.alpha,
          params.unknownStrategy,
          trained.vocab.size,
          trained.tags.length,
          counts.trainSentences,
          counts.trainTokens,
          trained.trainLogLikelihood,
          createdAt
        );
      const modelId = Number(res.lastInsertRowid);

      const insTag = this.db.prepare(
        'INSERT INTO model_tags(model_id,tag,ord,initial_count,final_count,tag_count) VALUES(?,?,?,?,?,?)'
      );
      const insTrans = this.db.prepare(
        'INSERT INTO model_transitions(model_id,from_tag,to_tag,count) VALUES(?,?,?,?)'
      );
      const insEm = this.db.prepare(
        'INSERT INTO model_emissions(model_id,tag,token,count) VALUES(?,?,?,?)'
      );

      counts.tags.forEach((tag, ord) => {
        insTag.run(
          modelId,
          tag,
          ord,
          counts.initial.get(tag) ?? 0,
          counts.final.get(tag) ?? 0,
          counts.tagCount.get(tag) ?? 0
        );
      });
      for (const [from, row] of counts.transitions) {
        for (const [to, c] of row) insTrans.run(modelId, from, to, c);
      }
      for (const [tag, em] of counts.emissions) {
        for (const [token, c] of em) insEm.run(modelId, tag, token, c);
      }
      this.db.exec('COMMIT');
      return { modelId, modelVersion };
    } catch (err) {
      this.db.exec('ROLLBACK');
      throw err;
    }
  }

  getMeta(modelId: number) {
    const row = this.db.prepare('SELECT * FROM models WHERE id=?').get(modelId) as
      | Record<string, number | string>
      | undefined;
    if (!row) throw new AppError('NOT_FOUND', `模型 ${modelId} 不存在`);
    return row;
  }

  listModels(corpusId?: number, page?: { limit: number; offset: number }) {
    const limit = page?.limit ?? 500;
    const offset = page?.offset ?? 0;
    if (corpusId !== undefined) {
      const total = Number(
        (this.db.prepare('SELECT COUNT(*) c FROM models WHERE corpus_id=?').get(corpusId) as {
          c: number;
        }).c
      );
      const data = this.db
        .prepare('SELECT * FROM models WHERE corpus_id=? ORDER BY id LIMIT ? OFFSET ?')
        .all(corpusId, limit, offset);
      return { data, total, limit, offset };
    }
    const total = Number(
      (this.db.prepare('SELECT COUNT(*) c FROM models').get() as { c: number }).c
    );
    const data = this.db
      .prepare('SELECT * FROM models ORDER BY id LIMIT ? OFFSET ?')
      .all(limit, offset);
    return { data, total, limit, offset };
  }

  /** 从持久化计数重建内存模型（与训练时完全一致）。 */
  loadModel(modelId: number): { model: TrainedModel; meta: Record<string, number | string> } {
    const meta = this.getMeta(modelId);
    const tagRows = this.db
      .prepare('SELECT * FROM model_tags WHERE model_id=? ORDER BY ord')
      .all(modelId) as {
      tag: string;
      ord: number;
      initial_count: number;
      final_count: number;
      tag_count: number;
    }[];
    if (tagRows.length === 0) {
      throw new AppError('INSUFFICIENT_DATA', `模型 ${modelId} 没有标签数据`);
    }
    const params: TrainParams = {
      alpha: Number(meta.alpha),
      unknownStrategy: String(meta.unknown_strategy) as TrainParams['unknownStrategy'],
    };
    const sentences = this.loadTrainSentences(Number(meta.split_id));
    const counts = countTraining(sentences);
    // tagRows 权威决定标签顺序，确保与训练时一致。
    counts.tags = tagRows.map((r) => r.tag);
    const model = trainModel(counts, params);
    return { model, meta };
  }

  private loadTrainSentences(splitId: number) {
    const rows = this.db
      .prepare(
        `SELECT s.* FROM sentences s
         JOIN split_assignments a ON a.sentence_id=s.id
         WHERE a.split_id=? AND a.bucket='train' ORDER BY s.ord`
      )
      .all(splitId) as unknown as Parameters<typeof decodeRow>[0][];
    return rows.map(decodeRow);
  }
}
