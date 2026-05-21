"""
scraper.py
==========
Asynchronous historical-message scraper.

Walks every readable text channel in a guild and pulls the full message
history (potentially years of logs) in pages, writing each page into the
local SQLite archive. Designed to survive a long-running 3-year pull:

* Resumable  - records the newest message id per channel in `sync_state`,
                so a re-run only fetches messages newer than last time.
* Resilient  - every network call is wrapped in try/except with
                exponential back-off; rate limits and transient HTTP
                errors are retried, permanently forbidden channels are
                skipped, and one bad channel never aborts the whole sync.
* Non-blocking - DB writes run in a thread so the gateway stays alive.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

import discord

import config
import database

log = logging.getLogger("scraper")

# Type of the optional progress callback: async fn(text) -> None
ProgressCB = Optional[Callable[[str], Awaitable[None]]]

MAX_RETRIES = 5
BASE_BACKOFF = 2.0  # seconds; doubles each retry


@dataclass
class SyncReport:
    channels_scanned: int = 0
    channels_skipped: int = 0
    new_messages: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "Sync complete.",
            f"  Channels scanned : {self.channels_scanned}",
            f"  Channels skipped : {self.channels_skipped}",
            f"  New messages     : {self.new_messages}",
        ]
        if self.errors:
            lines.append(f"  Errors           : {len(self.errors)}")
            for e in self.errors[:5]:
                lines.append(f"    - {e}")
        return "\n".join(lines)


async def _emit(cb: ProgressCB, text: str) -> None:
    log.info(text)
    if cb is not None:
        try:
            await cb(text)
        except Exception:  # progress reporting must never break the sync
            log.debug("progress callback failed", exc_info=True)


def _serialize(message: discord.Message) -> dict:
    attachments = "\n".join(a.url for a in message.attachments)
    return {
        "message_id": str(message.id),
        "channel_id": str(message.channel.id),
        "channel_name": getattr(message.channel, "name", "unknown"),
        "guild_id": str(message.guild.id) if message.guild else None,
        "author_id": str(message.author.id),
        "author_name": str(message.author),
        "content": message.content or "",
        "created_at": message.created_at,  # tz-aware UTC datetime
        "attachments": attachments,
    }


async def _scrape_channel(
    channel: discord.TextChannel,
    report: SyncReport,
    progress: ProgressCB,
) -> None:
    """Pull one channel's history with retry/back-off, resuming if possible."""
    state = await asyncio.to_thread(database.get_sync_state, channel.id)
    after_obj: Optional[discord.Object] = None
    if state and state.get("last_message_id"):
        after_obj = discord.Object(id=int(state["last_message_id"]))
        mode = f"resuming after id {state['last_message_id']}"
    else:
        mode = "full history"
    await _emit(progress, f"#{channel.name}: scraping ({mode})...")

    buffer: list[dict] = []
    newest_id: Optional[int] = (
        int(state["last_message_id"]) if state and state.get("last_message_id") else None
    )
    channel_new = 0

    async def flush() -> None:
        nonlocal buffer, channel_new
        if not buffer:
            return
        written = await asyncio.to_thread(
            database.insert_messages_batch, buffer
        )
        channel_new += written
        report.new_messages += written
        buffer = []

    attempt = 0
    while True:
        try:
            # oldest_first=True so `newest_id` advances monotonically and a
            # mid-scrape crash still leaves a valid resume point.
            history = channel.history(
                limit=None,
                after=after_obj,
                oldest_first=True,
            )
            async for message in history:
                buffer.append(_serialize(message))
                newest_id = message.id
                if len(buffer) >= config.SCRAPE_CHUNK_SIZE:
                    await flush()
                    # Persist progress so a later run resumes from here.
                    await asyncio.to_thread(
                        database.set_sync_state,
                        channel.id,
                        channel.name,
                        newest_id,
                        channel_new,
                    )
            await flush()
            break  # finished this channel cleanly

        except discord.Forbidden:
            report.channels_skipped += 1
            msg = f"#{channel.name}: no permission, skipped"
            report.errors.append(msg)
            await _emit(progress, msg)
            return

        except (discord.HTTPException, asyncio.TimeoutError, OSError) as exc:
            attempt += 1
            if attempt > MAX_RETRIES:
                msg = f"#{channel.name}: gave up after {MAX_RETRIES} retries ({exc})"
                report.errors.append(msg)
                await _emit(progress, msg)
                # Save what we have so the next sync resumes here.
                await flush()
                await asyncio.to_thread(
                    database.set_sync_state,
                    channel.id,
                    channel.name,
                    newest_id,
                    channel_new,
                )
                return
            backoff = BASE_BACKOFF * (2 ** (attempt - 1))
            await _emit(
                progress,
                f"#{channel.name}: transient error ({exc}); "
                f"retry {attempt}/{MAX_RETRIES} in {backoff:.0f}s",
            )
            await flush()  # checkpoint partial progress
            # Resume the loop after the last successfully buffered id.
            if newest_id is not None:
                after_obj = discord.Object(id=newest_id)
            await asyncio.sleep(backoff)

    await asyncio.to_thread(
        database.set_sync_state,
        channel.id,
        channel.name,
        newest_id,
        channel_new,
    )
    report.channels_scanned += 1
    await _emit(
        progress,
        f"#{channel.name}: done (+{channel_new} new messages)",
    )


async def sync_guild(
    guild: discord.Guild,
    progress: ProgressCB = None,
) -> SyncReport:
    """
    Scrape every readable text channel in `guild` into the local archive.
    Safe to run repeatedly: only messages newer than the last sync are
    fetched per channel.
    """
    report = SyncReport()
    me = guild.me

    text_channels = [
        ch for ch in guild.text_channels
        if ch.permissions_for(me).read_message_history
        and ch.permissions_for(me).view_channel
    ]
    await _emit(
        progress,
        f"Starting sync of '{guild.name}': "
        f"{len(text_channels)} readable text channel(s).",
    )

    for channel in text_channels:
        try:
            await _scrape_channel(channel, report, progress)
        except Exception as exc:  # never let one channel kill the run
            msg = f"#{channel.name}: unexpected error {exc!r}"
            report.errors.append(msg)
            log.exception(msg)
        await asyncio.sleep(config.SCRAPE_CHANNEL_DELAY)

    await _emit(progress, report.summary())
    return report
