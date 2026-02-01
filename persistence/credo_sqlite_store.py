# persistence/credo_sqlite_store.py
# Simple local SQLite storage for Credo beliefs / entries
# Fixed version – no reserved words in column names

import sqlite3
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

class CredoSQLiteStore:
    def __init__(self, db_path: str = "./credo_data/credo.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._create_table()

    def _create_table(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS entries (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts          TEXT NOT NULL,
                    entry_type  TEXT NOT NULL,
                    payload     TEXT NOT NULL,
                    entry_hash  TEXT UNIQUE NOT NULL
                )
            """)
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_hash ON entries(entry_hash)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_ts   ON entries(ts)")
            self.conn.commit()

    def close(self):
        """Call this when your program ends – good habit"""
        self.conn.close()

    def persist(self, data: dict, entry_type: str = "belief") -> str:
        """
        Save a piece of data (dictionary) and return its hash.
        If the exact same data already exists → does nothing (idempotent).
        """
        now = datetime.now(timezone.utc).isoformat()
        payload_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
        h = hashlib.sha256(payload_str.encode('utf-8')).hexdigest()

        try:
            with self.conn:
                self.conn.execute(
                    """
                    INSERT INTO entries (ts, entry_type, payload, entry_hash)
                    VALUES (?, ?, ?, ?)
                    """,
                    (now, entry_type, payload_str, h)
                )
        except sqlite3.IntegrityError:
            # already exists → ignore (safe)
            pass

        return h

    def get_by_hash(self, h: str) -> dict | None:
        """Find one entry by its hash or return None"""
        row = self.conn.execute(
            "SELECT ts, entry_type, payload FROM entries WHERE entry_hash = ?",
            (h,)
        ).fetchone()

        if row is None:
            return None

        return {
            "ts":         row[0],
            "entry_type": row[1],
            "payload":    json.loads(row[2])
        }

    def recent(self, limit: int = 10, entry_type: str | None = None) -> list[dict]:
        """Get the most recent entries (newest first)"""
        sql = "SELECT ts, entry_type, payload, entry_hash FROM entries"
        params = []

        if entry_type:
            sql += " WHERE entry_type = ?"
            params.append(entry_type)

        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(sql, params).fetchall()

        return [
            {
                "ts":         r[0],
                "entry_type": r[1],
                "payload":    json.loads(r[2]),
                "hash":       r[3]
            }
            for r in rows
        ]

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]


# ────────────────────────────────────────────────
# Quick test – run this file directly to see it work
# ────────────────────────────────────────────────

if __name__ == "__main__":
    db = CredoSQLiteStore("./test_credo.db")

    print("Saving some test entries...")

    db.persist({"thought": "The universe is a big mirror"}, "reflection")
    db.persist({"thought": "Reflection is the key"}, "reflection")
    db.persist({"fact": 42, "source": "Deep Thought"}, "fact")

    print("\nMost recent entries:")
    for item in db.recent(limit=5):
        print(f"  {item['ts']}  |  {item['entry_type']:10}  |  {item['payload']}")

    print(f"\nTotal entries in database: {db.count()}")

    db.close()