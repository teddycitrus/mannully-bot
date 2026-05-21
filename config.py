"""
config.py
=========
Centralised configuration and environment-variable validation.

Loads variables from a local `.env` file (via python-dotenv) and exposes
them as validated module-level constants. Importing this module will raise
a clear, actionable error if a required credential is missing so the bot
fails fast instead of crashing deep inside the Discord gateway.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Project root = directory this file lives in. All local storage stays here
# so the project is fully self-contained (zero hosting cost requirement).
PROJECT_ROOT = Path(__file__).resolve().parent

# Load .env from the project root if present. override=False means real
# environment variables win over the file, which is the expected precedence.
load_dotenv(PROJECT_ROOT / ".env", override=False)


class ConfigError(RuntimeError):
    """Raised when the runtime configuration is invalid or incomplete."""


def _require(name: str) -> str:
    """Return a required environment variable or raise a helpful error."""
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(
            f"Missing required environment variable: {name}\n"
            f"Create a '.env' file next to main.py containing:\n"
            f"    DISCORD_BOT_TOKEN=your-discord-token\n"
            f"    GEMINI_API_KEY=your-google-ai-studio-key\n"
            f"See README.txt for full setup instructions."
        )
    return value


def _optional(name: str, default: str) -> str:
    raw = os.getenv(name)
    return raw.strip() if raw and raw.strip() else default


def _optional_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _optional_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# --- Required credentials -------------------------------------------------
# Validated eagerly at import. main.py wraps the import so a missing token
# produces a friendly message rather than a stack trace.
DISCORD_BOT_TOKEN: str = _require("DISCORD_BOT_TOKEN")
GEMINI_API_KEY: str = _require("GEMINI_API_KEY")

# --- External database (Turso / libSQL) ----------------------------------
# If TURSO_DATABASE_URL is set, the message archive lives in a hosted
# libSQL (Turso) database instead of a purely local file. The bot opens it
# as an *embedded replica*: DATABASE_PATH below is then just a fast local
# cache that continuously syncs with the remote primary. This is what lets
# the bot run 24/7 on any host and survive your local machine being off.
# Leave these blank to keep the original fully-local SQLite behaviour.
TURSO_DATABASE_URL: str = _optional("TURSO_DATABASE_URL", "")
TURSO_AUTH_TOKEN: str = _optional("TURSO_AUTH_TOKEN", "")
USE_TURSO: bool = bool(TURSO_DATABASE_URL)

# --- Optional / tunable settings -----------------------------------------
# SQLite database file. When Turso is configured this is the local
# embedded-replica cache; otherwise it is the sole, fully-local archive.
DATABASE_PATH: str = str(PROJECT_ROOT / _optional("DATABASE_FILE", "history.db"))

# Gemini model. 2.5 Flash is fast + multimodal and inexpensive.
GEMINI_MODEL: str = _optional("GEMINI_MODEL", "gemini-2.5-flash")

# Command prefix for admin commands such as !sync.
COMMAND_PREFIX: str = _optional("COMMAND_PREFIX", "!")

# Number of messages requested per Discord history page (PRD: chunks of 100).
SCRAPE_CHUNK_SIZE: int = _optional_int("SCRAPE_CHUNK_SIZE", 100)

# Seconds to wait between channels during a sync to stay friendly to the API.
SCRAPE_CHANNEL_DELAY: float = float(_optional("SCRAPE_CHANNEL_DELAY", "1.0"))

# If True and the database is empty on startup, the bot kicks off a full
# sync automatically. If False it waits for an admin to run !sync.
AUTO_SYNC_IF_EMPTY: bool = _optional_bool("AUTO_SYNC_IF_EMPTY", False)

# Max historical snippets handed to Gemini per question.
RAG_MAX_CONTEXT_MESSAGES: int = _optional_int("RAG_MAX_CONTEXT_MESSAGES", 40)


def summary() -> str:
    """Human-readable, secret-safe configuration summary for logging."""
    if USE_TURSO:
        # Show the host only - never echo the auth token.
        host = TURSO_DATABASE_URL.split("@")[-1]
        storage = f"Turso libSQL ({host}), remote (no local replica)"
    else:
        storage = f"local SQLite file ({DATABASE_PATH})"
    return (
        "Configuration loaded:\n"
        f"  Storage .............. {storage}\n"
        f"  Gemini model ......... {GEMINI_MODEL}\n"
        f"  Command prefix ....... {COMMAND_PREFIX}\n"
        f"  Scrape chunk size .... {SCRAPE_CHUNK_SIZE}\n"
        f"  Auto-sync if empty ... {AUTO_SYNC_IF_EMPTY}\n"
        f"  RAG context limit .... {RAG_MAX_CONTEXT_MESSAGES}\n"
        f"  Discord token ........ {'set (' + str(len(DISCORD_BOT_TOKEN)) + ' chars)'}\n"
        f"  Gemini API key ....... {'set (' + str(len(GEMINI_API_KEY)) + ' chars)'}"
    )


if __name__ == "__main__":
    # `python config.py` acts as a quick credential check.
    try:
        print(summary())
        print("\nConfiguration OK.")
    except ConfigError as exc:
        print(f"Configuration error:\n{exc}", file=sys.stderr)
        sys.exit(1)
