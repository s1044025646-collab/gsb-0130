import type { DatabaseSync } from 'node:sqlite';
import { corpusFingerprint, sentenceFingerprint, sha256 } from '../core/validation.js';
import type { TaggedSentence, UnknownStrategy } from '../core/types.js';
import { AppError } from '../core/types.js';
import { deterministicSplit } from '../core/split.js';
import { trainHMM, type HMMParams } from '../core/hmm.js';
import { evaluate, type EvaluationResult } from '../core/evaluate.js';

export interface Clock {
  now(): number;
}

export const systemClock: Clock = { now: () => Date.now() };

export interface Page<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
}

export interface CorpusRecord {
  id: number;
  name: string;
  version: number;
  content_sha: string;
  sentence_count: number;
  token_count: number;
  tag_count: number;
  created_at: number;
}

export class Repository {
  constructor(
    private readonly db: DatabaseSync,
    private readonly clock: Clock = systemClock
  ) {}

  transaction<T>(work: () => T): T {
    this.db.exec('BEGIN IMMEDIATE');
    try {
      const result = work();
      this.db.exec('COMMIT');
      return result;
    } catch (error) {
      this.db.exec('ROLLBACK');
      throw error;
    }
  }

  importCorpus(name: string, sentences: TaggedSentence[]): { corpus: CorpusRecord; reused: boolean } {
    return this.transaction(() => {
      const contentSha = corpusFingerprint(name, sentences);
      const identical = this.db
        .prepare('SELECT * FROM corpora WHERE name = ? AND content_sha = ?')
        .get(name, contentSha) as unknown as CorpusRecord | undefined;
      if (identical) {
        return { corpus: identical, reused: true };
      }
      const latest = this.db
        .prepare('SELECT MAX(version) AS version FROM corpora WHERE name = ?')
        .get(name) as { version: number | null };
      const version = (latest.version ?? 0) + 1;
      const tagSet = new Set<string>();
      let tokenCount = 0;
      for (const sentence of sentences) {
        tokenCount += sentence.tokens.length;
        for (const tag of sentence.tags) tagSet.add(tag);
      }
      const info = this.db
        .prepare(
          `INSERT INTO corpora(name, version, content_sha, sentence_count, token_count, tag_count, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)`
        )
        .run(
          name,
          version,
          contentSha,
          sentences.length,
          tokenCount,
          tagSet.size,
          this.clock.now()
        );
      const corpusId = Number(info.lastInsertRowid);
      const insertSentence = this.db.prepare(
        `INSERT OR IGNORE INTO corpus_sentences(corpus_id, position, tokens, tags, content_sha)
         VALUES (?, ?, ?, ?, ?)`
      );
      sentences.forEach((sentence, position) => {
        insertSentence.run(
          corpusId,
          position,
          JSON.stringify(sentence.tokens),
          JSON.stringify(sentence.tags),
          sentenceFingerprint(sentence)
        );
      });
      const corpus = this.db.prepare('SELECT * FROM corpora WHERE id = ?').get(corpusId) as unknown as CorpusRecord;
      return { corpus, reused: false };
    });
  }

  listCorpora(limit: number, offset: number): Page<CorpusRecord> {
    const total = (this.db.prepare('SELECT COUNT(*) AS count FROM corpora').get() as { count: number }).count;
    const items = this.db
      .prepare('SELECT * FROM corpora ORDER BY id DESC LIMIT ? OFFSET ?')
      .all(limit, offset) as unknown as CorpusRecord[];
    return { items, total, limit, offset };
  }

  getCorpus(id: number): CorpusRecord {
    const corpus = this.db.prepare('SELECT * FROM corpora WHERE id = ?').get(id) as CorpusRecord | undefined;
    if (!corpus) throw new AppError('NOT_FOUND', `语料 ${id} 不存在`);
    return corpus;
  }

  getSentences(corpusId: number): TaggedSentence[] {
    this.getCorpus(corpusId);
    const rows = this.db
      .prepare('SELECT tokens, tags FROM corpus_sentences WHERE corpus_id = ? ORDER BY position')
      .all(corpusId) as Array<{ tokens: string; tags: string }>;
    return rows.map((row) => ({ tokens: JSON.parse(row.tokens), tags: JSON.parse(row.tags) }));
  }

  createSplit(corpusId: number, seed: number, trainRatio: number): { splitId: number; reused: boolean; train: number; validation: number } {
    return this.transaction(() => {
      const sentences = this.getSentences(corpusId);
      if (sentences.length === 0) throw new AppError('EMPTY_CORPUS', '语料为空，无法划分');
      const existing = this.db
        .prepare('SELECT id FROM splits WHERE corpus_id = ? AND seed = ? AND train_ratio = ?')
        .get(corpusId, seed, trainRatio) as { id: number } | undefined;
      if (existing) {
        const counts = this.splitCounts(existing.id);
        return { splitId: existing.id, reused: true, ...counts };
      }
      const info = this.db
        .prepare('INSERT INTO splits(corpus_id, seed, train_ratio, created_at) VALUES (?, ?, ?, ?)')
        .run(corpusId, seed, trainRatio, this.clock.now());
      const splitId = Number(info.lastInsertRowid);
      const assignments = deterministicSplit(sentences.length, seed, trainRatio);
      const insert = this.db.prepare(
        'INSERT INTO split_members(split_id, sentence_id, part) VALUES (?, ?, ?)'
      );
      const rows = this.db
        .prepare('SELECT id FROM corpus_sentences WHERE corpus_id = ? ORDER BY position')
        .all(corpusId) as Array<{ id: number }>;
      assignments.forEach((assignment) => {
        insert.run(splitId, rows[assignment.index].id, assignment.part);
      });
      return { splitId, reused: false, ...this.splitCounts(splitId) };
    });
  }

  private splitCounts(splitId: number): { train: number; validation: number } {
    const rows = this.db
      .prepare('SELECT part, COUNT(*) AS count FROM split_members WHERE split_id = ? GROUP BY part')
      .all(splitId) as Array<{ part: string; count: number }>;
    const counts = { train: 0, validation: 0 };
    for (const row of rows) counts[row.part as 'train' | 'validation'] = row.count;
    return counts;
  }

  getSplitSentences(splitId: number, part: 'train' | 'validation'): TaggedSentence[] {
    const rows = this.db
      .prepare(
        `SELECT s.tokens AS tokens, s.tags AS tags
         FROM split_members m JOIN corpus_sentences s ON s.id = m.sentence_id
         WHERE m.split_id = ? AND m.part = ? ORDER BY s.position`
      )
      .all(splitId, part) as Array<{ tokens: string; tags: string }>;
    return rows.map((row) => ({ tokens: JSON.parse(row.tokens), tags: JSON.parse(row.tags) }));
  }

  trainModel(
    splitId: number,
    smoothing: number,
    unknownStrategy: UnknownStrategy
  ): { modelId: number; reused: boolean } {
    return this.transaction(() => {
      const split = this.db.prepare('SELECT * FROM splits WHERE id = ?').get(splitId) as
        | { corpus_id: number }
        | undefined;
      if (!split) throw new AppError('NOT_FOUND', `划分 ${splitId} 不存在`);
      const corpus = this.getCorpus(split.corpus_id);
      const trainSentences = this.getSplitSentences(splitId, 'train');
      const params = trainHMM(trainSentences, smoothing, unknownStrategy);
      const fingerprint = [
        splitId,
        corpus.version,
        smoothing,
        unknownStrategy,
        JSON.stringify(params.initial),
        JSON.stringify(params.transition),
        JSON.stringify(params.emission),
        JSON.stringify(params.unknown)
      ].join('|');
      const paramsHash = sha256(fingerprint);
      const existing = this.db
        .prepare('SELECT id FROM models WHERE params_sha = ?')
        .get(paramsHash) as { id: number } | undefined;
      if (existing) return { modelId: existing.id, reused: true };

      let trainTokens = 0;
      for (const sentence of trainSentences) trainTokens += sentence.tokens.length;
      const info = this.db
        .prepare(
          `INSERT INTO models(split_id, corpus_version, smoothing, unknown_strategy, params_sha,
             train_sentences, train_tokens, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
        )
        .run(
          splitId,
          corpus.version,
          smoothing,
          unknownStrategy,
          paramsHash,
          trainSentences.length,
          trainTokens,
          this.clock.now()
        );
      const modelId = Number(info.lastInsertRowid);
      this.db
        .prepare(
          `INSERT INTO model_params(model_id, initial_json, final_json, transition_json, emission_json, vocab_json, tags_json, unknown_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
        )
        .run(
          modelId,
          JSON.stringify(params.initial),
          JSON.stringify(params.stop),
          JSON.stringify(params.transition),
          JSON.stringify(params.emission),
          JSON.stringify(params.vocab),
          JSON.stringify(params.tags),
          JSON.stringify(params.unknown)
        );
      return { modelId, reused: false };
    });
  }

  loadModel(modelId: number): HMMParams {
    const row = this.db
      .prepare(
        `SELECT m.smoothing AS smoothing, m.unknown_strategy AS unknown_strategy,
                p.initial_json AS initial_json, p.final_json AS final_json,
                p.transition_json AS transition_json, p.emission_json AS emission_json,
                p.vocab_json AS vocab_json, p.tags_json AS tags_json,
                p.unknown_json AS unknown_json
         FROM models m JOIN model_params p ON p.model_id = m.id WHERE m.id = ?`
      )
      .get(modelId) as
      | {
          smoothing: number;
          unknown_strategy: UnknownStrategy;
          initial_json: string;
          final_json: string;
          transition_json: string;
          emission_json: string;
          vocab_json: string;
          tags_json: string;
          unknown_json: string;
        }
      | undefined;
    if (!row) throw new AppError('NOT_FOUND', `模型 ${modelId} 不存在`);
    return {
      tags: JSON.parse(row.tags_json),
      vocab: JSON.parse(row.vocab_json),
      initial: JSON.parse(row.initial_json),
      stop: JSON.parse(row.final_json),
      transition: JSON.parse(row.transition_json),
      emission: JSON.parse(row.emission_json),
      unknown: JSON.parse(row.unknown_json),
      smoothing: row.smoothing,
      unknownStrategy: row.unknown_strategy
    };
  }

  listModels(limit: number, offset: number): Page<Record<string, unknown>> {
    const total = (this.db.prepare('SELECT COUNT(*) AS count FROM models').get() as { count: number }).count;
    const items = this.db
      .prepare('SELECT * FROM models ORDER BY id DESC LIMIT ? OFFSET ?')
      .all(limit, offset);
    return { items: items as Record<string, unknown>[], total, limit, offset };
  }

  evaluateModel(modelId: number, part: 'train' | 'validation'): EvaluationResult & { reportId: number; reused: boolean } {
    return this.transaction(() => {
      const model = this.db.prepare('SELECT split_id FROM models WHERE id = ?').get(modelId) as
        | { split_id: number }
        | undefined;
      if (!model) throw new AppError('NOT_FOUND', `模型 ${modelId} 不存在`);
      const existing = this.db
        .prepare('SELECT id FROM reports WHERE model_id = ? AND part = ?')
        .get(modelId, part) as { id: number } | undefined;
      const params = this.loadModel(modelId);
      const sentences = this.getSplitSentences(model.split_id, part);
      const result = evaluate(params, sentences);
      if (existing) {
        return { ...result, reportId: existing.id, reused: true };
      }
      const info = this.db
        .prepare(
          `INSERT INTO reports(model_id, split_id, part, tag_metrics_json, confusion_json,
             token_total, token_correct, sentence_total, sentence_correct, accuracy, log_likelihood, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`
        )
        .run(
          modelId,
          model.split_id,
          part,
          JSON.stringify(result.tagMetrics),
          JSON.stringify(result.confusion),
          result.summary.tokenTotal,
          result.summary.tokenCorrect,
          result.summary.sentenceTotal,
          result.summary.sentenceCorrect,
          result.summary.accuracy,
          result.summary.logLikelihood,
          this.clock.now()
        );
      return { ...result, reportId: Number(info.lastInsertRowid), reused: false };
    });
  }

  getReport(reportId: number): Record<string, unknown> {
    const row = this.db.prepare('SELECT * FROM reports WHERE id = ?').get(reportId);
    if (!row) throw new AppError('NOT_FOUND', `报告 ${reportId} 不存在`);
    return row as Record<string, unknown>;
  }

  listReports(modelId: number | undefined, limit: number, offset: number): Page<Record<string, unknown>> {
    const total = (
      modelId === undefined
        ? (this.db.prepare('SELECT COUNT(*) AS count FROM reports').get() as { count: number })
        : (this.db.prepare('SELECT COUNT(*) AS count FROM reports WHERE model_id = ?').get(modelId) as { count: number })
    ).count;
    const items =
      modelId === undefined
        ? (this.db.prepare('SELECT * FROM reports ORDER BY id DESC LIMIT ? OFFSET ?').all(limit, offset) as Record<string, unknown>[])
        : (this.db
            .prepare('SELECT * FROM reports WHERE model_id = ? ORDER BY id DESC LIMIT ? OFFSET ?')
            .all(modelId, limit, offset) as Record<string, unknown>[]);
    return { items, total, limit, offset };
  }
}
