import { DatabaseSync } from 'node:sqlite';
import { mkdirSync } from 'node:fs';
import { dirname } from 'node:path';
import { SCHEMA_STATEMENTS, SCHEMA_VERSION } from './schema.js';

export type DB = DatabaseSync;

/** 打开数据库并可重复执行迁移（IF NOT EXISTS，幂等）。 */
export function openDatabase(path: string): DB {
  if (path !== ':memory:') {
    mkdirSync(dirname(path), { recursive: true });
  }
  const db = new DatabaseSync(path);
  db.exec('PRAGMA foreign_keys = ON;');
  db.exec('PRAGMA journal_mode = WAL;');
  migrate(db);
  return db;
}

export function migrate(db: DB): void {
  db.exec(SCHEMA_STATEMENTS);
  db.prepare(
    `INSERT INTO schema_meta(key,value) VALUES('schema_version',?)
     ON CONFLICT(key) DO UPDATE SET value=excluded.value`
  ).run(String(SCHEMA_VERSION));
}

export function closeDatabase(db: DB): void {
  db.close();
}
