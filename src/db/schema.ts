import { DatabaseSync } from 'node:sqlite';

export const SCHEMA_VERSION = 1;

export const SCHEMA_SQL = `
CREATE TABLE IF NOT EXISTS schema_meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS corpora (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  version INTEGER NOT NULL,
  content_sha TEXT NOT NULL,
  sentence_count INTEGER NOT NULL,
  token_count INTEGER NOT NULL,
  tag_count INTEGER NOT NULL,
  created_at INTEGER NOT NULL,
  UNIQUE(name, version)
);

CREATE TABLE IF NOT EXISTS corpus_sentences (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  corpus_id INTEGER NOT NULL REFERENCES corpora(id),
  position INTEGER NOT NULL,
  tokens TEXT NOT NULL,
  tags TEXT NOT NULL,
  content_sha TEXT NOT NULL,
  UNIQUE(corpus_id, content_sha)
);

CREATE TABLE IF NOT EXISTS splits (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  corpus_id INTEGER NOT NULL REFERENCES corpora(id),
  seed INTEGER NOT NULL,
  train_ratio REAL NOT NULL,
  created_at INTEGER NOT NULL,
  UNIQUE(corpus_id, seed, train_ratio)
);

CREATE TABLE IF NOT EXISTS split_members (
  split_id INTEGER NOT NULL REFERENCES splits(id),
  sentence_id INTEGER NOT NULL REFERENCES corpus_sentences(id),
  part TEXT NOT NULL CHECK (part IN ('train','validation')),
  PRIMARY KEY (split_id, sentence_id)
);

CREATE TABLE IF NOT EXISTS models (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  split_id INTEGER NOT NULL REFERENCES splits(id),
  corpus_version INTEGER NOT NULL,
  smoothing REAL NOT NULL,
  unknown_strategy TEXT NOT NULL CHECK (unknown_strategy IN ('uniform','closed')),
  params_sha TEXT NOT NULL UNIQUE,
  train_sentences INTEGER NOT NULL,
  train_tokens INTEGER NOT NULL,
  created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS model_params (
  model_id INTEGER PRIMARY KEY REFERENCES models(id),
  initial_json TEXT NOT NULL,
  final_json TEXT NOT NULL,
  transition_json TEXT NOT NULL,
  emission_json TEXT NOT NULL,
  vocab_json TEXT NOT NULL,
  tags_json TEXT NOT NULL,
  unknown_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reports (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  model_id INTEGER NOT NULL REFERENCES models(id),
  split_id INTEGER NOT NULL REFERENCES splits(id),
  part TEXT NOT NULL CHECK (part IN ('validation','train')),
  tag_metrics_json TEXT NOT NULL,
  confusion_json TEXT NOT NULL,
  token_total INTEGER NOT NULL,
  token_correct INTEGER NOT NULL,
  sentence_total INTEGER NOT NULL,
  sentence_correct INTEGER NOT NULL,
  accuracy REAL NOT NULL,
  log_likelihood REAL NOT NULL,
  created_at INTEGER NOT NULL,
  UNIQUE(model_id, part)
);

CREATE INDEX IF NOT EXISTS idx_sentences_corpus ON corpus_sentences(corpus_id);
CREATE INDEX IF NOT EXISTS idx_members_split ON split_members(split_id, part);
CREATE INDEX IF NOT EXISTS idx_models_split ON models(split_id);
`;

export function migrate(db: DatabaseSync): void {
  db.exec('PRAGMA foreign_keys = ON;');
  db.exec(SCHEMA_SQL);
  db.prepare(
    `INSERT INTO schema_meta(key, value) VALUES('version', ?)
     ON CONFLICT(key) DO UPDATE SET value = excluded.value`
  ).run(String(SCHEMA_VERSION));
}

export function openDatabase(path: string): DatabaseSync {
  const db = new DatabaseSync(path);
  migrate(db);
  return db;
}
