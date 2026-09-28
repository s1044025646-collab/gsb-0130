import type { DB } from '../db/database.js';
import { AppError } from './errors.js';
import { corpusFingerprint, sentenceFingerprint, unitHash } from './hash.js';
import type {
  Clock,
  Page,
  PageParams,
  SplitParams,
  TaggedSentence,
} from './types.js';

interface SentenceRow {
  id: number;
  version_id: number;
  ord: number;
  fingerprint: string;
  tokens: string;
  tags: string;
  length: number;
}

export interface CorpusVersionInfo {
  corpusId: number;
  versionId: number;
  version: number;
  fingerprint: string;
  sentenceCount: number;
  tokenCount: number;
  createdAt: string;
  reused: boolean;
}

export class CorpusService {
  constructor(
    private db: DB,
    private clock: Clock = { now: () => new Date() }
  ) {}

  createCorpus(name: string, description = ''): { id: number; createdAt: string } {
    const createdAt = this.clock.now().toISOString();
    const info = this.db
      .prepare('INSERT INTO corpora(name,description,created_at) VALUES(?,?,?)')
      .run(name, description, createdAt);
    return { id: Number(info.lastInsertRowid), createdAt };
  }

  getCorpus(id: number) {
    const row = this.db.prepare('SELECT * FROM corpora WHERE id=?').get(id) as
      | Record<string, unknown>
      | undefined;
    if (!row) throw new AppError('NOT_FOUND', `语料 ${id} 不存在`);
    return row;
  }

  listCorpora(): unknown[] {
    return this.db
      .prepare(
        `SELECT c.*, COUNT(cv.id) AS version_count FROM corpora c
         LEFT JOIN corpus_versions cv ON cv.corpus_id=c.id
         GROUP BY c.id ORDER BY c.id`
      )
      .all();
  }

  /**
   * 导入（或幂等重复导入）一个语料版本。
   * 相同内容指纹直接复用既有版本，绝不增加训练计数。
   * 失败时整笔事务回滚。
   */
  importVersion(corpusId: number, sentences: TaggedSentence[]): CorpusVersionInfo {
    this.getCorpus(corpusId);
    if (sentences.length === 0) {
      throw new AppError('EMPTY_CORPUS', '语料至少需要一条句子（空句也可，但数组不能为空）');
    }

    // 按指纹去重（同 token+tag 序列只保留首次出现），记录稳定顺序。
    const unique = new Map<string, TaggedSentence & { ord: number }>();
    let tokenCount = 0;
    sentences.forEach((s, ord) => {
      const fp = sentenceFingerprint(s.tokens, s.tags);
      if (!unique.has(fp)) unique.set(fp, { ...s, ord });
      tokenCount += s.tokens.length;
    });
    const fingerprints = [...unique.keys()];
    const fingerprint = corpusFingerprint(fingerprints);

    const existing = this.db
      .prepare('SELECT id FROM corpus_versions WHERE corpus_id=? AND fingerprint=?')
      .get(corpusId, fingerprint) as { id: number } | undefined;
    if (existing) {
      const v = this.db
        .prepare('SELECT * FROM corpus_versions WHERE id=?')
        .get(existing.id) as Record<string, number | string>;
      return {
        corpusId,
        versionId: Number(v.id),
        version: Number(v.version),
        fingerprint: String(v.fingerprint),
        sentenceCount: Number(v.sentence_count),
        tokenCount: Number(v.token_count),
        createdAt: String(v.created_at),
        reused: true,
      };
    }

    const createdAt = this.clock.now().toISOString();
    const tx = this.db.prepare(
      `INSERT INTO corpus_versions
       (corpus_id,version,fingerprint,sentence_count,token_count,created_at)
       VALUES(?,?,?,?,?,?)`
    );
    const insertSentence = this.db.prepare(
      `INSERT INTO sentences(version_id,ord,fingerprint,tokens,tags,length)
       VALUES(?,?,?,?,?,?)`
    );

    this.db.exec('BEGIN');
    try {
      const maxRow = this.db
        .prepare('SELECT COALESCE(MAX(version),0) AS m FROM corpus_versions WHERE corpus_id=?')
        .get(corpusId) as { m: number };
      const version = maxRow.m + 1;
      const res = tx.run(
        corpusId,
        version,
        fingerprint,
        unique.size,
        tokenCount,
        createdAt
      );
      const versionId = Number(res.lastInsertRowid);
      for (const [fp, s] of unique) {
        insertSentence.run(
          versionId,
          s.ord,
          fp,
          JSON.stringify(s.tokens),
          JSON.stringify(s.tags),
          s.tokens.length
        );
      }
      this.db.exec('COMMIT');
      return {
        corpusId,
        versionId,
        version,
        fingerprint,
        sentenceCount: unique.size,
        tokenCount,
        createdAt,
        reused: false,
      };
    } catch (err) {
      this.db.exec('ROLLBACK');
      throw err;
    }
  }

  getVersion(versionId: number) {
    const row = this.db.prepare('SELECT * FROM corpus_versions WHERE id=?').get(versionId) as
      | Record<string, number | string>
      | undefined;
    if (!row) throw new AppError('NOT_FOUND', `语料版本 ${versionId} 不存在`);
    return row;
  }

  listVersions(corpusId: number): unknown[] {
    this.getCorpus(corpusId);
    return this.db
      .prepare('SELECT * FROM corpus_versions WHERE corpus_id=? ORDER BY version')
      .all(corpusId);
  }

  listSentences(versionId: number, page: PageParams): Page<SentenceRow> {
    this.getVersion(versionId);
    const total = Number(
      (this.db.prepare('SELECT COUNT(*) c FROM sentences WHERE version_id=?').get(versionId) as {
        c: number;
      }).c
    );
    const data = this.db
      .prepare('SELECT * FROM sentences WHERE version_id=? ORDER BY ord LIMIT ? OFFSET ?')
      .all(versionId, page.limit, page.offset) as unknown as SentenceRow[];
    return { data, total, limit: page.limit, offset: page.offset };
  }

  /**
   * 固定种子划分：相同 (版本,seed,ratio) 复用；每个句子指纹哈希到 [0,1)，
   * 小于 ratio 进验证集。内容相同的重复句哈希相同，因此绝不会跨训练/验证。
   */
  getOrCreateSplit(versionId: number, params: SplitParams): {
    splitId: number;
    train: SentenceRow[];
    val: SentenceRow[];
  } {
    this.getVersion(versionId);
    const existing = this.db
      .prepare(
        'SELECT id FROM splits WHERE version_id=? AND seed=? AND val_ratio=?'
      )
      .get(versionId, params.seed, params.valRatio) as { id: number } | undefined;

    let splitId: number;
    if (existing) {
      splitId = existing.id;
    } else {
      splitId = this.createSplit(versionId, params);
    }
    return this.readSplit(splitId);
  }

  private createSplit(versionId: number, params: SplitParams): number {
    const rows = this.db
      .prepare('SELECT * FROM sentences WHERE version_id=? ORDER BY ord')
      .all(versionId) as unknown as SentenceRow[];
    const createdAt = this.clock.now().toISOString();
    this.db.exec('BEGIN');
    try {
      const res = this.db
        .prepare(
          'INSERT INTO splits(version_id,seed,val_ratio,created_at) VALUES(?,?,?,?)'
        )
        .run(versionId, params.seed, params.valRatio, createdAt);
      const splitId = Number(res.lastInsertRowid);
      const assign = this.db.prepare(
        'INSERT INTO split_assignments(split_id,sentence_id,bucket) VALUES(?,?,?)'
      );
      for (const row of rows) {
        const bucket =
          unitHash(params.seed, row.fingerprint) < params.valRatio ? 'val' : 'train';
        assign.run(splitId, row.id, bucket);
      }
      this.db.exec('COMMIT');
      return splitId;
    } catch (err) {
      this.db.exec('ROLLBACK');
      throw err;
    }
  }

  private readSplit(splitId: number) {
    const rows = this.db
      .prepare(
        `SELECT s.* FROM sentences s
         JOIN split_assignments a ON a.sentence_id=s.id
         WHERE a.split_id=? ORDER BY s.ord`
      )
      .all(splitId) as unknown as (SentenceRow & { bucket: string })[];
    // bucket 通过第二个查询获取
    const bucketRows = this.db
      .prepare('SELECT sentence_id,bucket FROM split_assignments WHERE split_id=?')
      .all(splitId) as { sentence_id: number; bucket: string }[];
    const bucketOf = new Map(bucketRows.map((b) => [b.sentence_id, b.bucket]));
    const train: SentenceRow[] = [];
    const val: SentenceRow[] = [];
    for (const r of rows) {
      if (bucketOf.get(r.id) === 'val') val.push(r);
      else train.push(r);
    }
    return { splitId, train, val };
  }
}

export function decodeRow(row: SentenceRow): TaggedSentence {
  return {
    tokens: JSON.parse(row.tokens) as string[],
    tags: JSON.parse(row.tags) as string[],
  };
}
