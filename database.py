"""
database.py
===========
Storage engine + RAG retrieval layer.

Two interchangeable backends, selected by configuration
-------------------------------------------------------
* **libSQL / Turso (external, default for hosted runs)** - when
  ``TURSO_DATABASE_URL`` is configured the archive lives in a hosted
  libSQL database. We open a *direct remote* connection to it (no local
  replica file): every read/write goes straight to the Turso primary.
  Embedded-replica mode was dropped because its local cache file
  corrupts under this bot's cross-thread DB access. The bot can
  therefore run on any host (or several) and survive the local machine
  being powered off.
* **Local SQLite (fallback)** - if no Turso URL is set the original
  single-file behaviour is used unchanged (zero hosting cost, fully
  local archive).

Both backends use the identical SQLite schema, including the FTS5
full-text index, so retrieval semantics are the same either way.

All functions here are synchronous and thread-safe (a single global lock
guards the connection). The async callers in main.py / scraper.py invoke
them through ``asyncio.to_thread(...)`` so the Discord event loop never
blocks on disk or network I/O.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

import config

log = logging.getLogger("database")

# Optional libSQL driver. Prefer the official Turso SDK ("libsql"); fall
# back to the older "libsql-experimental" package if that is what's
# installed. Either exposes a sqlite3-compatible connect()/cursor() API.
try:  # pragma: no cover - import resolution is environment dependent
    import libsql as _libsql  # type: ignore
except ImportError:  # pragma: no cover
    try:
        import libsql_experimental as _libsql  # type: ignore
    except ImportError:
        _libsql = None  # type: ignore

_LOCK = threading.RLock()
_CONN: Any = None
_ENGINE: str = "sqlite"  # "sqlite" | "libsql"

# Bump when the schema changes incompatibly. v2: Discord snowflake ids are
# stored as TEXT (the libSQL remote driver mangled 64-bit ints via float64,
# so message/channel/guild ids lost their low bits -> dead jump-links). An
# integer surrogate key (rowid_pk) backs the FTS index instead.
_SCHEMA_VERSION = 2


# --------------------------------------------------------------------------
# Connection / schema
# --------------------------------------------------------------------------
def _connect() -> Any:
    global _CONN, _ENGINE
    if _CONN is not None:
        return _CONN

    if config.USE_TURSO:
        if _libsql is None:
            raise RuntimeError(
                "TURSO_DATABASE_URL is set but the libSQL driver is not "
                "installed. Run: pip install libsql"
            )
        # Remote (direct) connection to the hosted Turso primary - NO local
        # replica file. The embedded-replica mode corrupts its local cache
        # file under this bot's cross-thread DB access (the recurring
        # "ValueError: file is not a database"), so every read/write now
        # goes straight to Turso. Writes are durably committed remotely;
        # no .sync() is required (and .sync() does not exist remote-side).
        _CONN = _libsql.connect(
            config.TURSO_DATABASE_URL,
            auth_token=config.TURSO_AUTH_TOKEN,
        )
        # "libsql-remote" (not "libsql") so _maybe_sync() is a no-op.
        _ENGINE = "libsql-remote"
        log.info("Storage engine: libSQL (Turso remote, no local replica).")
    else:
        _CONN = sqlite3.connect(
            config.DATABASE_PATH,
            check_same_thread=False,
            timeout=30.0,
        )
        _ENGINE = "sqlite"
        # WAL/synchronous tuning only applies to the local-file engine.
        _CONN.execute("PRAGMA journal_mode=WAL;")
        _CONN.execute("PRAGMA synchronous=NORMAL;")
        log.info("Storage engine: local SQLite file (%s).", config.DATABASE_PATH)
    return _CONN


def _maybe_sync() -> None:
    """Legacy embedded-replica refresh hook.

    No-op for the local SQLite engine AND for the current Turso "libsql-
    remote" engine (remote writes commit directly to the primary, so no
    replica sync exists or is needed). Retained only so the historical
    embedded-replica path still works if ever re-enabled.
    """
    if _ENGINE == "libsql" and _CONN is not None:
        try:
            _CONN.sync()
        except Exception:
            log.warning(
                "libSQL sync failed; will retry on the next write.",
                exc_info=True,
            )


_DROP_SQL = """
    DROP TRIGGER IF EXISTS messages_ai;
    DROP TRIGGER IF EXISTS messages_ad;
    DROP TRIGGER IF EXISTS messages_au;
    DROP TABLE IF EXISTS messages_fts;
    DROP TABLE IF EXISTS messages;
    DROP TABLE IF EXISTS sync_state;
"""

# All Discord ids are TEXT (snowflakes exceed 2^53; stored as ints they
# get float64-mangled by the libSQL remote driver). rowid_pk is an integer
# surrogate so the FTS5 external-content index still has an integer rowid.
_CREATE_SQL = """
    CREATE TABLE IF NOT EXISTS messages (
        rowid_pk     INTEGER PRIMARY KEY,   -- surrogate (FTS rowid)
        message_id   TEXT UNIQUE NOT NULL,  -- Discord snowflake (string)
        channel_id   TEXT NOT NULL,
        channel_name TEXT,
        guild_id     TEXT,
        author_id    TEXT,
        author_name  TEXT,
        content      TEXT,
        created_at   TEXT NOT NULL,         -- ISO-8601 UTC
        created_ts   REAL NOT NULL,         -- epoch seconds (range queries)
        attachments  TEXT                   -- newline-joined URLs, or ''
    );

    CREATE INDEX IF NOT EXISTS idx_messages_channel
        ON messages(channel_id);
    CREATE INDEX IF NOT EXISTS idx_messages_ts
        ON messages(created_ts);
    CREATE INDEX IF NOT EXISTS idx_messages_author
        ON messages(author_name);

    CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
        content,
        author_name,
        content='messages',
        content_rowid='rowid_pk'
    );

    CREATE TRIGGER IF NOT EXISTS messages_ai
    AFTER INSERT ON messages BEGIN
        INSERT INTO messages_fts(rowid, content, author_name)
        VALUES (new.rowid_pk, new.content, new.author_name);
    END;

    CREATE TRIGGER IF NOT EXISTS messages_ad
    AFTER DELETE ON messages BEGIN
        INSERT INTO messages_fts(messages_fts, rowid, content, author_name)
        VALUES ('delete', old.rowid_pk, old.content, old.author_name);
    END;

    CREATE TRIGGER IF NOT EXISTS messages_au
    AFTER UPDATE ON messages BEGIN
        INSERT INTO messages_fts(messages_fts, rowid, content, author_name)
        VALUES ('delete', old.rowid_pk, old.content, old.author_name);
        INSERT INTO messages_fts(rowid, content, author_name)
        VALUES (new.rowid_pk, new.content, new.author_name);
    END;

    CREATE TABLE IF NOT EXISTS sync_state (
        channel_id       TEXT PRIMARY KEY,
        channel_name     TEXT,
        last_message_id  TEXT,
        last_synced_at   TEXT,
        message_count    INTEGER DEFAULT 0
    );
"""


def init_db() -> None:
    """Create the schema, migrating once if it is an older version.

    A version bump drops the old tables and recreates them (the data is
    rebuilt by re-running !sync). The drop happens exactly once - guarded
    by PRAGMA user_version - so normal restarts never wipe the archive.
    """
    with _LOCK:
        conn = _connect()
        # Version marker lives in a normal table: Turso/hrana rejects
        # writing `PRAGMA user_version`. schema_meta is never dropped by
        # _DROP_SQL, so the marker survives the rebuild.
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_meta "
            "(key TEXT PRIMARY KEY, value INTEGER)"
        )
        conn.commit()
        row = _query_one(
            "SELECT value AS v FROM schema_meta WHERE key = 'version'"
        )
        ver = int(row["v"]) if row and row.get("v") is not None else 0

        if ver < _SCHEMA_VERSION:
            log.warning(
                "Schema v%d < v%d - rebuilding tables (snowflake-id fix). "
                "The archive will be empty until !sync is re-run.",
                ver, _SCHEMA_VERSION,
            )
            conn.executescript(_DROP_SQL)

        conn.executescript(_CREATE_SQL)

        if ver < _SCHEMA_VERSION:
            conn.execute(
                "INSERT INTO schema_meta(key, value) VALUES('version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (_SCHEMA_VERSION,),
            )
        conn.commit()
        _maybe_sync()


# --------------------------------------------------------------------------
# Row helpers (engine-agnostic: build dicts from cursor.description so we
# don't depend on sqlite3.Row, which libSQL does not provide)
# --------------------------------------------------------------------------
def _query(sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    with _LOCK:
        cur = _connect().cursor()
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description] if cur.description else []
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def _query_one(sql: str, params: tuple = ()) -> Optional[dict[str, Any]]:
    rows = _query(sql, params)
    return rows[0] if rows else None


def _row_to_message(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "message_id": row["message_id"],
        "channel_id": row["channel_id"],
        "channel_name": row["channel_name"],
        "guild_id": row.get("guild_id"),
        "author_name": row["author_name"],
        "content": row["content"],
        "created_at": row["created_at"],
        "created_ts": row["created_ts"],
        "attachments": row["attachments"] or "",
    }


def _sanitize(text: Optional[str]) -> str:
    """Trim and strip null bytes; SQLite parameter binding handles the rest."""
    if not text:
        return ""
    return text.replace("\x00", "").strip()


# --------------------------------------------------------------------------
# Writes
# --------------------------------------------------------------------------
def insert_message(msg: dict[str, Any]) -> bool:
    """Insert one message. Returns False if it already existed (no-op)."""
    return insert_messages_batch([msg]) > 0


def insert_messages_batch(messages: Iterable[dict[str, Any]]) -> int:
    """
    Bulk insert messages, ignoring duplicates (idempotent re-sync).
    Returns the number of *new* rows actually written.
    """
    rows = []
    for m in messages:
        created_at = m["created_at"]
        if isinstance(created_at, datetime):
            dt = created_at.astimezone(timezone.utc)
            created_iso = dt.isoformat()
            created_ts = dt.timestamp()
        else:  # already a string / epoch
            created_iso = str(created_at)
            created_ts = float(m.get("created_ts", 0.0))
        rows.append(
            (
                # str() while the true value is still a Python int
                # (lossless); never let a Discord id reach the driver as
                # an int - the libSQL remote driver float64-mangles it.
                str(m["message_id"]),
                str(m["channel_id"]),
                _sanitize(m.get("channel_name")),
                str(m["guild_id"]) if m.get("guild_id") else None,
                str(m["author_id"]) if m.get("author_id") else None,
                _sanitize(m.get("author_name")),
                _sanitize(m.get("content")),
                created_iso,
                created_ts,
                _sanitize(m.get("attachments")),
            )
        )

    if not rows:
        return 0

    with _LOCK:
        conn = _connect()
        cur = conn.cursor()
        # Engine-agnostic new-row count: total_changes is not exposed by the
        # libSQL driver, so diff COUNT(*) under the lock (we are the only
        # writer, so this is exact).
        cur.execute("SELECT COUNT(*) AS c FROM messages")
        before = int(cur.fetchone()[0])
        conn.executemany(
            """
            INSERT OR IGNORE INTO messages (
                message_id, channel_id, channel_name, guild_id,
                author_id, author_name, content, created_at,
                created_ts, attachments
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        cur.execute("SELECT COUNT(*) AS c FROM messages")
        after = int(cur.fetchone()[0])
        _maybe_sync()
        return after - before


# --------------------------------------------------------------------------
# Sync-state tracking
# --------------------------------------------------------------------------
def get_sync_state(channel_id: int) -> Optional[dict[str, Any]]:
    return _query_one(
        "SELECT * FROM sync_state WHERE channel_id = ?", (str(channel_id),)
    )


def set_sync_state(
    channel_id: int,
    channel_name: str,
    last_message_id: Optional[int],
    message_count: int,
) -> None:
    with _LOCK:
        conn = _connect()
        conn.execute(
            """
            INSERT INTO sync_state
                (channel_id, channel_name, last_message_id,
                 last_synced_at, message_count)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(channel_id) DO UPDATE SET
                channel_name    = excluded.channel_name,
                last_message_id = excluded.last_message_id,
                last_synced_at  = excluded.last_synced_at,
                message_count   = excluded.message_count
            """,
            (
                str(channel_id),
                _sanitize(channel_name),
                str(last_message_id) if last_message_id is not None else None,
                datetime.now(timezone.utc).isoformat(),
                message_count,
            ),
        )
        conn.commit()
        _maybe_sync()


# --------------------------------------------------------------------------
# Stats
# --------------------------------------------------------------------------
def count_messages() -> int:
    row = _query_one("SELECT COUNT(*) AS c FROM messages")
    return int(row["c"]) if row else 0


def is_empty() -> bool:
    return count_messages() == 0


def stats() -> dict[str, Any]:
    with _LOCK:
        total = _query_one("SELECT COUNT(*) AS c FROM messages")["c"]
        channels = _query_one(
            "SELECT COUNT(DISTINCT channel_id) AS c FROM messages"
        )["c"]
        authors = _query_one(
            "SELECT COUNT(DISTINCT author_id) AS c FROM messages"
        )["c"]
        span = _query_one(
            "SELECT MIN(created_at) AS lo, MAX(created_at) AS hi FROM messages"
        )
    return {
        "total_messages": total,
        "channels": channels,
        "authors": authors,
        "oldest": span["lo"] if span else None,
        "newest": span["hi"] if span else None,
    }


# --------------------------------------------------------------------------
# Retrieval (RAG)
# --------------------------------------------------------------------------
def _fts_query(text: str) -> str:
    """
    Turn free text into a safe FTS5 OR-query.
    Each alphanumeric token becomes a prefix term; FTS5 syntax chars are
    dropped so user punctuation can't break the MATCH expression.
    """
    tokens = []
    for raw in text.split():
        cleaned = "".join(ch for ch in raw if ch.isalnum())
        if len(cleaned) >= 2:
            tokens.append(f'"{cleaned}"*')
    return " OR ".join(tokens)


def search_by_keyword(query: str, limit: int = 30) -> list[dict[str, Any]]:
    """Full-text keyword search, best matches first (BM25 ranking)."""
    match = _fts_query(query)
    if not match:
        return []
    rows = _query(
        """
        SELECT m.*
        FROM messages_fts f
        JOIN messages m ON m.rowid_pk = f.rowid
        WHERE messages_fts MATCH ?
        ORDER BY bm25(messages_fts), m.created_ts DESC
        LIMIT ?
        """,
        (match, limit),
    )
    return [_row_to_message(r) for r in rows]


def search_by_timeframe(
    start_ts: float, end_ts: float, limit: int = 200
) -> list[dict[str, Any]]:
    """All messages within [start_ts, end_ts] (epoch seconds), oldest first."""
    rows = _query(
        """
        SELECT * FROM messages
        WHERE created_ts BETWEEN ? AND ?
        ORDER BY created_ts ASC
        LIMIT ?
        """,
        (start_ts, end_ts, limit),
    )
    return [_row_to_message(r) for r in rows]


def get_recent_messages(limit: int = 30) -> list[dict[str, Any]]:
    rows = _query(
        "SELECT * FROM messages ORDER BY created_ts DESC LIMIT ?",
        (limit,),
    )
    return [_row_to_message(r) for r in rows]


def get_thread_around(
    message_id: int, window: int = 4
) -> list[dict[str, Any]]:
    """Fetch a small window of messages before/after a hit for context."""
    anchor = _query_one(
        "SELECT channel_id, created_ts FROM messages WHERE message_id = ?",
        (str(message_id),),
    )
    if anchor is None:
        return []
    before = _query(
        """
        SELECT * FROM messages
        WHERE channel_id = ? AND created_ts <= ?
        ORDER BY created_ts DESC LIMIT ?
        """,
        (anchor["channel_id"], anchor["created_ts"], window + 1),
    )
    after = _query(
        """
        SELECT * FROM messages
        WHERE channel_id = ? AND created_ts > ?
        ORDER BY created_ts ASC LIMIT ?
        """,
        (anchor["channel_id"], anchor["created_ts"], window),
    )
    combined = list(reversed(before)) + list(after)
    return [_row_to_message(r) for r in combined]


def close() -> None:
    global _CONN
    with _LOCK:
        if _CONN is not None:
            try:
                _CONN.commit()
                _maybe_sync()  # final push of pending writes to Turso
            except Exception:
                log.warning("Error during final commit/sync", exc_info=True)
            try:
                _CONN.close()
            except Exception:
                pass
            _CONN = None
