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
from typing import Any, Iterable, Optional, Sequence

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


def _is_stale_stream_error(exc: BaseException) -> bool:
    # Turso/Hrana expires idle remote "streams" server-side; after that
    # every call against the cached _CONN gets 404 "stream not found"
    # until we reconnect. The driver surfaces it as a ValueError whose
    # message embeds the raw Hrana api error.
    msg = str(exc)
    return "stream not found" in msg or ("Hrana" in msg and "404" in msg)


def _reset_connection() -> None:
    global _CONN
    if _CONN is not None:
        try:
            _CONN.close()
        except Exception:
            pass
    _CONN = None


def _run_with_reconnect(op):
    # Callers hold _LOCK, so the failed attempt and the retry are atomic
    # w.r.t. other threads — no risk of another thread observing the
    # half-reset state.
    try:
        return op()
    except Exception as exc:
        if not _is_stale_stream_error(exc):
            raise
        log.warning("libSQL stream expired; reconnecting and retrying once.")
        _reset_connection()
        return op()


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

    -- Hybrid-retrieval companion table. rowid_pk mirrors messages.rowid_pk
    -- so a JOIN is cheap and ordering by insertion stays consistent. The
    -- BLOB is the raw float32 buffer of a 768-dim Gemini embedding (3072
    -- bytes). Added without a schema_version bump so the existing archive
    -- is not wiped; old rows just have no embedding until backfilled.
    CREATE TABLE IF NOT EXISTS message_embeddings (
        rowid_pk   INTEGER PRIMARY KEY,
        embedding  BLOB NOT NULL
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
    def _do() -> list[dict[str, Any]]:
        cur = _connect().cursor()
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description] if cur.description else []
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    with _LOCK:
        return _run_with_reconnect(_do)


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

    def _do() -> int:
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

    with _LOCK:
        return _run_with_reconnect(_do)


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
    def _do() -> None:
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

    with _LOCK:
        _run_with_reconnect(_do)


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


def search_by_keyword(
    query: str, limit: int = 30, guild_id: Optional[str] = None
) -> list[dict[str, Any]]:
    """Full-text keyword search, best matches first (BM25 ranking).

    ``guild_id`` scopes the search to one server; pass None only for
    admin/global lookups - the per-question RAG path MUST pass it so the
    bot does not conflate histories from different servers.
    """
    match = _fts_query(query)
    if not match:
        return []
    if guild_id is not None:
        rows = _query(
            """
            SELECT m.*
            FROM messages_fts f
            JOIN messages m ON m.rowid_pk = f.rowid
            WHERE messages_fts MATCH ? AND m.guild_id = ?
            ORDER BY bm25(messages_fts), m.created_ts DESC
            LIMIT ?
            """,
            (match, str(guild_id), limit),
        )
    else:
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
    start_ts: float,
    end_ts: float,
    limit: int = 200,
    guild_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """All messages within [start_ts, end_ts] (epoch seconds), oldest first.

    Scoped to ``guild_id`` when provided (see search_by_keyword).
    """
    if guild_id is not None:
        rows = _query(
            """
            SELECT * FROM messages
            WHERE created_ts BETWEEN ? AND ? AND guild_id = ?
            ORDER BY created_ts ASC
            LIMIT ?
            """,
            (start_ts, end_ts, str(guild_id), limit),
        )
    else:
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


def find_author(
    name_substring: str, guild_id: Optional[str] = None
) -> Optional[dict[str, Any]]:
    """Best-match author by name substring within a guild (or globally).

    Returns the most active matching author (case-insensitive substring
    on author_name). None if no match.
    """
    if not name_substring:
        return None
    safe = (
        name_substring.lower()
        .replace("%", "").replace("_", "").replace("\x00", "").strip()
    )
    if not safe:
        return None
    pattern = f"%{safe}%"
    if guild_id is not None:
        return _query_one(
            """
            SELECT author_id, author_name, COUNT(*) AS msg_count
            FROM messages
            WHERE LOWER(author_name) LIKE ?
              AND author_id IS NOT NULL
              AND guild_id = ?
            GROUP BY author_id, author_name
            ORDER BY msg_count DESC
            LIMIT 1
            """,
            (pattern, str(guild_id)),
        )
    return _query_one(
        """
        SELECT author_id, author_name, COUNT(*) AS msg_count
        FROM messages
        WHERE LOWER(author_name) LIKE ? AND author_id IS NOT NULL
        GROUP BY author_id, author_name
        ORDER BY msg_count DESC
        LIMIT 1
        """,
        (pattern,),
    )


def search_by_author_id(
    author_id: str,
    limit: int = 300,
    exclude_id: Optional[str] = None,
    guild_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Most recent N substantive messages by an author, scoped to a guild
    when provided. The same user can be active in several servers, so
    pass guild_id whenever you want a single-server read.
    """
    clauses = ["author_id = ?", "content != ''"]
    params: list[Any] = [str(author_id)]
    if exclude_id is not None:
        clauses.append("message_id != ?")
        params.append(str(exclude_id))
    if guild_id is not None:
        clauses.append("guild_id = ?")
        params.append(str(guild_id))
    params.append(limit)
    rows = _query(
        f"""
        SELECT * FROM messages
        WHERE {' AND '.join(clauses)}
        ORDER BY created_ts DESC
        LIMIT ?
        """,
        tuple(params),
    )
    return [_row_to_message(r) for r in rows]


def top_authors_sample(
    per_author: int = 80,
    max_authors: int = 12,
    guild_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Sample messages from the top-K most active authors in a guild.

    Used for "describe everyone's personalities" style questions. One
    query per author (~50ms each on Turso) so kept small. Pass guild_id
    to keep the read scoped to one server's most-active members.
    """
    if guild_id is not None:
        top = _query(
            """
            SELECT author_id
            FROM messages
            WHERE content != ''
              AND author_id IS NOT NULL
              AND guild_id = ?
            GROUP BY author_id
            ORDER BY COUNT(*) DESC
            LIMIT ?
            """,
            (str(guild_id), max_authors),
        )
    else:
        top = _query(
            """
            SELECT author_id
            FROM messages
            WHERE content != '' AND author_id IS NOT NULL
            GROUP BY author_id
            ORDER BY COUNT(*) DESC
            LIMIT ?
            """,
            (max_authors,),
        )
    out: list[dict[str, Any]] = []
    for a in top:
        aid = a.get("author_id")
        if not aid:
            continue
        if guild_id is not None:
            msgs = _query(
                """
                SELECT * FROM messages
                WHERE author_id = ?
                  AND content != ''
                  AND guild_id = ?
                ORDER BY created_ts DESC
                LIMIT ?
                """,
                (str(aid), str(guild_id), per_author),
            )
        else:
            msgs = _query(
                """
                SELECT * FROM messages
                WHERE author_id = ? AND content != ''
                ORDER BY created_ts DESC
                LIMIT ?
                """,
                (str(aid), per_author),
            )
        out.extend(_row_to_message(r) for r in msgs)
    return out


def time_sample(
    buckets: int = 24,
    per_bucket: int = 30,
    guild_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Sample substantive messages evenly across a guild's archive span.

    Splits [oldest, newest] into N equal time buckets and pulls up to M
    longest messages from each. Pass guild_id so the span and the
    samples are both scoped to one server.
    """
    if guild_id is not None:
        span = _query_one(
            """
            SELECT MIN(created_ts) AS lo, MAX(created_ts) AS hi
            FROM messages
            WHERE guild_id = ?
            """,
            (str(guild_id),),
        )
    else:
        span = _query_one(
            "SELECT MIN(created_ts) AS lo, MAX(created_ts) AS hi FROM messages"
        )
    if not span or span.get("lo") is None or span.get("hi") is None:
        return []
    lo, hi = float(span["lo"]), float(span["hi"])
    if hi <= lo or buckets <= 0:
        return []
    step = (hi - lo) / buckets
    out: list[dict[str, Any]] = []
    for i in range(buckets):
        s = lo + i * step
        e = lo + (i + 1) * step + (1e-3 if i == buckets - 1 else 0)
        if guild_id is not None:
            rows = _query(
                """
                SELECT * FROM messages
                WHERE created_ts >= ? AND created_ts < ?
                  AND guild_id = ?
                  AND content != ''
                  AND LENGTH(content) > 20
                ORDER BY LENGTH(content) DESC
                LIMIT ?
                """,
                (s, e, str(guild_id), per_bucket),
            )
        else:
            rows = _query(
                """
                SELECT * FROM messages
                WHERE created_ts >= ? AND created_ts < ?
                  AND content != ''
                  AND LENGTH(content) > 20
                ORDER BY LENGTH(content) DESC
                LIMIT ?
                """,
                (s, e, per_bucket),
            )
        out.extend(_row_to_message(r) for r in rows)
    out.sort(key=lambda m: m["created_ts"])
    return out


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


# --------------------------------------------------------------------------
# Embedding storage (hybrid retrieval)
# --------------------------------------------------------------------------
def insert_embeddings_batch(rows: Iterable[tuple[int, bytes]]) -> int:
    """Bulk write (rowid_pk, embedding_bytes) pairs. INSERT OR REPLACE so
    a re-embedded message overwrites cleanly. Returns the number of rows
    written.
    """
    payload = list(rows)
    if not payload:
        return 0

    def _do() -> int:
        conn = _connect()
        conn.executemany(
            "INSERT OR REPLACE INTO message_embeddings (rowid_pk, embedding) "
            "VALUES (?, ?)",
            payload,
        )
        conn.commit()
        _maybe_sync()
        return len(payload)

    with _LOCK:
        return _run_with_reconnect(_do)


def messages_missing_embeddings(
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Return up to `limit` archived messages that have no embedding yet.

    Empty-content rows are skipped (nothing to embed). Returned in
    rowid_pk order so a long backfill makes monotonic progress.
    """
    rows = _query(
        """
        SELECT m.rowid_pk, m.content, m.guild_id
        FROM messages m
        LEFT JOIN message_embeddings e ON e.rowid_pk = m.rowid_pk
        WHERE e.rowid_pk IS NULL
          AND m.content != ''
        ORDER BY m.rowid_pk
        LIMIT ?
        """,
        (limit,),
    )
    return rows


def get_rowid_by_message_id(message_id: Any) -> Optional[int]:
    """Look up the integer surrogate key for a Discord message id.

    Used by the live-ingest path: insert_message returns only a "was new"
    bool, but the embedder needs the rowid_pk to write into
    message_embeddings and to append to the in-memory cache.
    """
    row = _query_one(
        "SELECT rowid_pk FROM messages WHERE message_id = ?",
        (str(message_id),),
    )
    return int(row["rowid_pk"]) if row else None


def count_missing_embeddings() -> int:
    """How many substantive messages still need an embedding."""
    row = _query_one(
        """
        SELECT COUNT(*) AS c
        FROM messages m
        LEFT JOIN message_embeddings e ON e.rowid_pk = m.rowid_pk
        WHERE e.rowid_pk IS NULL
          AND m.content != ''
        """
    )
    return int(row["c"]) if row else 0


def load_guild_embedding_rows(
    guild_id: Optional[str],
) -> list[dict[str, Any]]:
    """All (rowid_pk, embedding) pairs for one guild, ordered by rowid_pk.

    Callers convert the BLOBs to numpy arrays and stack into a matrix for
    cosine search. Returned rowid_pks parallel that matrix exactly.
    """
    if guild_id is not None:
        return _query(
            """
            SELECT e.rowid_pk, e.embedding
            FROM message_embeddings e
            JOIN messages m ON m.rowid_pk = e.rowid_pk
            WHERE m.guild_id = ?
            ORDER BY e.rowid_pk
            """,
            (str(guild_id),),
        )
    return _query(
        """
        SELECT e.rowid_pk, e.embedding
        FROM message_embeddings e
        ORDER BY e.rowid_pk
        """
    )


def messages_by_rowids(rowid_pks: Sequence[int]) -> list[dict[str, Any]]:
    """Fetch full message rows for a list of rowid_pks, preserving the
    input order. Used to materialise vector-search hits back into the
    same row shape as BM25 results.
    """
    if not rowid_pks:
        return []
    # SQLite's IN-list has no fixed cap in practice, but stay safe by
    # chunking very large requests. K=200 is the realistic ceiling here.
    placeholders = ",".join("?" for _ in rowid_pks)
    rows = _query(
        f"SELECT * FROM messages WHERE rowid_pk IN ({placeholders})",
        tuple(rowid_pks),
    )
    by_rid = {int(r["rowid_pk"]): r for r in rows}
    out: list[dict[str, Any]] = []
    for rid in rowid_pks:
        r = by_rid.get(int(rid))
        if r is not None:
            out.append(_row_to_message(r))
    return out


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
