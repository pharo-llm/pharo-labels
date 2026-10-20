"""Thin SQLite wrapper for the evidence database."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from .schemas import DDL, SCHEMA_VERSION


def dumps(v) -> str | None:
    return None if v is None else json.dumps(v, ensure_ascii=False, sort_keys=True, default=str)


def loads(v, default=None):
    if v is None or v == "":
        return default
    try:
        return json.loads(v)
    except (TypeError, json.JSONDecodeError):
        return default


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.executescript(DDL)
        self.conn.execute("INSERT OR REPLACE INTO schema_info VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
        self.conn.commit()

    def execute(self, sql: str, params=()):
        return self.conn.execute(sql, params)

    def executemany(self, sql: str, rows):
        return self.conn.executemany(sql, rows)

    def query(self, sql: str, params=()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchall()

    def scalar(self, sql: str, params=()):
        row = self.conn.execute(sql, params).fetchone()
        return row[0] if row else None

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.commit()
        self.conn.close()

    def upsert(self, table: str, row: dict, key: list[str]):
        cols = list(row)
        placeholders = ",".join("?" for _ in cols)
        updates = ",".join(f"{c}=excluded.{c}" for c in cols if c not in key)
        sql = f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders})"
        sql += f" ON CONFLICT({','.join(key)}) DO UPDATE SET {updates}" if updates else f" ON CONFLICT({','.join(key)}) DO NOTHING"
        self.conn.execute(sql, [row[c] for c in cols])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
