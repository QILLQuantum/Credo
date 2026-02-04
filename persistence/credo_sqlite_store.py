import sqlite3
import os
import hashlib
import json

class CredoSQLiteStore:
    def __init__(self, db_path="credo_data/credo.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.c = self.conn.cursor()
        self.c.execute('''CREATE TABLE IF NOT EXISTS entries
                          (id INTEGER PRIMARY KEY AUTOINCREMENT,
                           hash TEXT UNIQUE,
                           type TEXT,
                           payload TEXT,
                           timestamp TEXT)''')
        self.conn.commit()

    def persist(self, payload: dict, entry_type: str) -> str:
        payload_str = json.dumps(payload)
        entry_hash = hashlib.sha256(payload_str.encode()).hexdigest()
        timestamp = datetime.datetime.now().isoformat()
        try:
            self.c.execute("INSERT INTO entries (hash, type, payload, timestamp) VALUES (?, ?, ?, ?)",
                           (entry_hash, entry_type, payload_str, timestamp))
            self.conn.commit()
        except sqlite3.IntegrityError:
            pass  # Duplicate hash
        return entry_hash

    def get_entry_by_hash(self, entry_hash: str):
        self.c.execute("SELECT payload FROM entries WHERE hash = ?", (entry_hash,))
        row = self.c.fetchone()
        return json.loads(row[0]) if row else None

    def list_recent_entries(self, limit=10):
        self.c.execute("SELECT payload FROM entries ORDER BY id DESC LIMIT ?", (limit,))
        return [json.loads(row[0]) for row in self.c.fetchall()]

    def close(self):
        self.conn.close()
