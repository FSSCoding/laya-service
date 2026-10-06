"""
Two-tier result cache (LRU in-memory + SQLite).

Laya results are deterministic for a given (state, questions) — the same request always yields the
same calibrated probabilities — so a content-addressed cache is safe. Key = sha256 of the canonical
request. In-memory LRU is the hot tier; SQLite is the durable tier that survives restarts.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional


class ResultCache:
    def __init__(self, db_path: str, max_memory_entries: int = 10000, ttl_days: int = 30):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_mem = max_memory_entries
        self.ttl = ttl_days * 86400
        self._mem: "OrderedDict[str, tuple[float, dict]]" = OrderedDict()
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS laya_result_cache "
                "(key TEXT PRIMARY KEY, ts REAL, value TEXT)"
            )
            self._conn.commit()

    @staticmethod
    def key(state: Any, questions: Dict[str, Any]) -> str:
        # Canonical, order-stable serialisation of the request.
        payload = json.dumps(
            {"state": state, "questions": questions}, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, k: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            hit = self._mem.get(k)
            if hit is not None:
                self._mem.move_to_end(k)
                return hit[1]
            row = self._conn.execute(
                "SELECT ts, value FROM laya_result_cache WHERE key=?", (k,)
            ).fetchone()
            if row is None:
                return None
            ts, value = row
            if time.time() - ts > self.ttl:
                self._conn.execute("DELETE FROM laya_result_cache WHERE key=?", (k,))
                self._conn.commit()
                return None
            self._mem[k] = (ts, json.loads(value))
            self._mem.move_to_end(k)
            while len(self._mem) > self.max_mem:
                self._mem.popitem(last=False)
            return json.loads(value)

    def put(self, k: str, value: Dict[str, Any]) -> None:
        payload = json.dumps(value)
        ts = time.time()
        with self._lock:
            self._mem[k] = (ts, value)
            self._mem.move_to_end(k)
            while len(self._mem) > self.max_mem:
                self._mem.popitem(last=False)
            self._conn.execute(
                "INSERT OR REPLACE INTO laya_result_cache "
                "(key, ts, value) VALUES (?,?,?)",
                (k, ts, payload),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
