export const SCHEMA_VERSION = 1;

export const SCHEMA_STATEMENTS = `
CREATE TABLE IF NOT EXISTS schema_meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

-- 语料库（一个库可有多个版本）
CREATE TABLE IF NOT EXISTS corpora (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL
);

-- 语料版本：内容变化即产生新版本
CREATE TABLE IF NOT EXISTS corpus_versions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  corpus_id INTEGER NOT NULL REFERENCES corpora(id),
  version INTEGER NOT NULL,
  fingerprint TEXT NOT NULL,
  sentence_count INTEGER NOT NULL,
  token_count INTEGER NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(corpus_id, version),
  UNIQUE(corpus_id, fingerprint)
);

-- 句子。(version_id, fingerprint, ord) 唯一保证重复导入不重复计数。
CREATE TABLE IF NOT EXISTS sentences (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  version_id INTEGER NOT NULL REFERENCES corpus_versions(id),
  ord INTEGER NOT NULL,
  fingerprint TEXT NOT NULL,
  tokens TEXT NOT NULL,
  tags TEXT NOT NULL,
  length INTEGER NOT NULL,
  UNIQUE(version_id, fingerprint, ord)
);

-- 划分结果：按 (版本, 种子, 比例) 缓存，重复句子始终同属一边
CREATE TABLE IF NOT EXISTS splits (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  version_id INTEGER NOT NULL REFERENCES corpus_versions(id),
  seed INTEGER NOT NULL,
  val_ratio REAL NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(version_id, seed, val_ratio)
);

CREATE TABLE IF NOT EXISTS split_assignments (
  split_id INTEGER NOT NULL REFERENCES splits(id),
  sentence_id INTEGER NOT NULL REFERENCES sentences(id),
  bucket TEXT NOT NULL CHECK(bucket IN ('train','val')),
  PRIMARY KEY(split_id, sentence_id)
);

-- 模型版本
CREATE TABLE IF NOT EXISTS models (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  corpus_id INTEGER NOT NULL REFERENCES corpora(id),
  version_id INTEGER NOT NULL REFERENCES corpus_versions(id),
  split_id INTEGER NOT NULL REFERENCES splits(id),
  model_version TEXT NOT NULL UNIQUE,
  revision INTEGER NOT NULL,
  alpha REAL NOT NULL,
  unknown_strategy TEXT NOT NULL,
  vocab_size INTEGER NOT NULL,
  tag_count INTEGER NOT NULL,
  train_sentences INTEGER NOT NULL,
  train_tokens INTEGER NOT NULL,
  log_likelihood_train REAL NOT NULL,
  created_at TEXT NOT NULL
);

-- 相同（语料版本, 划分, 平滑参数, 未知词策略）只允许一个模型：重复训练幂等。
CREATE UNIQUE INDEX IF NOT EXISTS idx_models_config
  ON models(version_id, split_id, alpha, unknown_strategy);

-- 标签集合及句首/句尾计数
CREATE TABLE IF NOT EXISTS model_tags (
  model_id INTEGER NOT NULL REFERENCES models(id),
  tag TEXT NOT NULL,
  ord INTEGER NOT NULL,
  initial_count INTEGER NOT NULL,
  final_count INTEGER NOT NULL,
  tag_count INTEGER NOT NULL,
  PRIMARY KEY(model_id, tag)
);

CREATE TABLE IF NOT EXISTS model_transitions (
  model_id INTEGER NOT NULL REFERENCES models(id),
  from_tag TEXT NOT NULL,
  to_tag TEXT NOT NULL,
  count INTEGER NOT NULL,
  PRIMARY KEY(model_id, from_tag, to_tag)
);

CREATE TABLE IF NOT EXISTS model_emissions (
  model_id INTEGER NOT NULL REFERENCES models(id),
  tag TEXT NOT NULL,
  token TEXT NOT NULL,
  count INTEGER NOT NULL,
  PRIMARY KEY(model_id, tag, token)
);

CREATE TABLE IF NOT EXISTS reports (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  model_id INTEGER NOT NULL REFERENCES models(id),
  split_id INTEGER NOT NULL REFERENCES splits(id),
  created_at TEXT NOT NULL,
  sentences INTEGER NOT NULL,
  tokens INTEGER NOT NULL,
  sentence_accuracy REAL NOT NULL,
  token_accuracy REAL NOT NULL,
  macro_precision REAL NOT NULL,
  macro_recall REAL NOT NULL,
  macro_f1 REAL NOT NULL,
  per_tag_json TEXT NOT NULL,
  confusion_json TEXT NOT NULL,
  tags_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sentences_version ON sentences(version_id);
CREATE INDEX IF NOT EXISTS idx_models_corpus ON models(corpus_id);
CREATE INDEX IF NOT EXISTS idx_reports_model ON reports(model_id, id);
`;
