"""
main.py
=======
Orchestration: Discord login, gateway events, commands, and the RAG loop.

Flow
----
Phase 1  Onboarding  : on_ready checks the local DB; admins run !sync (or
                        auto-sync if AUTO_SYNC_IF_EMPTY) to build the archive.
Phase 2  Maintenance : on_message persists every new message immediately.
Phase 3  Query (RAG) : mention the bot with a question -> it searches the
                        local archive (keyword + timeframe), feeds the
                        snippets to Gemini, and replies with the answer.
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
import sys
from datetime import datetime, timedelta, timezone

import discord
import numpy as np
from discord.ext import commands
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("bot")

# Import config early so a missing token fails fast with a clear message.
try:
    import config
except Exception as exc:  # ConfigError or import failure
    print(f"\nStartup aborted - {exc}\n", file=sys.stderr)
    sys.exit(1)

import database
import embeddings
import gemini_client
import retrieval
import scraper

intents = discord.Intents.default()
intents.message_content = True  # REQUIRED: read message text (privileged)
intents.guilds = True
intents.members = True

bot = commands.Bot(command_prefix=config.COMMAND_PREFIX, intents=intents)

# Guard so two !sync runs can't overlap on the same process.
_sync_lock = asyncio.Lock()

# Guard so two backfill loops can't overlap (on_ready may fire more than
# once per process lifetime if the gateway reconnects).
_embedding_backfill_lock = asyncio.Lock()


# --------------------------------------------------------------------------
# Embedding backfill + live ingest
# --------------------------------------------------------------------------
async def _embedding_backfill_loop() -> None:
    """Catch the existing archive up to the embedding index.

    Pulls unembedded messages in batches, embeds them via Gemini, writes
    the vectors to `message_embeddings`, and (if the in-memory cache is
    already built for that guild) appends them so future hybrid queries
    pick them up without a rebuild. Loop exits when the archive is fully
    embedded. Idempotent and safe to re-enter.
    """
    if _embedding_backfill_lock.locked():
        return
    async with _embedding_backfill_lock:
        # Gemini's embed endpoint allows up to 100 inputs per call; stay
        # under that so a single bad row doesn't waste a whole big batch.
        batch_size = 64
        total = 0
        log.info("embedding backfill: starting")
        while True:
            try:
                batch = await asyncio.to_thread(
                    database.messages_missing_embeddings, batch_size
                )
            except Exception:
                log.exception("backfill: query failed; sleeping 30s")
                await asyncio.sleep(30)
                continue

            if not batch:
                log.info(
                    "embedding backfill: archive fully embedded "
                    "(%d messages embedded this run)", total,
                )
                return

            texts = [str(r.get("content") or " ") for r in batch]
            try:
                vectors = await asyncio.to_thread(
                    embeddings.embed_documents, texts
                )
            except Exception:
                log.exception("backfill: Gemini call failed; sleeping 30s")
                await asyncio.sleep(30)
                continue

            rows = [
                (
                    int(r["rowid_pk"]),
                    v.astype(np.float32, copy=False).tobytes(),
                )
                for r, v in zip(batch, vectors)
            ]
            try:
                await asyncio.to_thread(database.insert_embeddings_batch, rows)
            except Exception:
                log.exception("backfill: DB write failed; sleeping 30s")
                await asyncio.sleep(30)
                continue

            # Keep any already-built per-guild caches current as we go.
            for r, v in zip(batch, vectors):
                gid = r.get("guild_id")
                retrieval.append_to_cache(
                    str(gid) if gid else None, int(r["rowid_pk"]), v,
                )

            total += len(batch)
            if total % 1024 == 0:
                remaining = await asyncio.to_thread(
                    database.count_missing_embeddings
                )
                log.info(
                    "embedding backfill: %d done, ~%d remaining",
                    total, remaining,
                )
            # Brief breather between batches so we never starve the
            # event loop on a huge archive.
            await asyncio.sleep(0.2)


async def _embed_live_message(
    rowid_pk: int, guild_id: str | None, content: str
) -> None:
    """Embed one freshly-archived message and cache it.

    Fire-and-forget: failures are logged but never bubble up to the
    gateway. Skipped for empty content (nothing to embed) and for
    duplicates (the row already has an embedding).
    """
    if not content.strip():
        return
    try:
        vecs = await asyncio.to_thread(embeddings.embed_documents, [content])
        if not vecs:
            return
        v = vecs[0]
        await asyncio.to_thread(
            database.insert_embeddings_batch,
            [(rowid_pk, v.astype(np.float32, copy=False).tobytes())],
        )
        retrieval.append_to_cache(guild_id, rowid_pk, v)
    except Exception:
        log.exception("live embedding failed for rowid_pk=%d", rowid_pk)


# --------------------------------------------------------------------------
# Lightweight timeframe parsing ("2 days ago", "yesterday", "last week"...)
# --------------------------------------------------------------------------
_REL_UNITS = {
    "minute": timedelta(minutes=1),
    "hour": timedelta(hours=1),
    "day": timedelta(days=1),
    "week": timedelta(weeks=1),
    "month": timedelta(days=30),
    "year": timedelta(days=365),
}


def parse_timeframe(text: str):
    """
    Return (start_ts, end_ts) epoch-second window if the question contains a
    recognisable time reference, else None. The window is padded generously
    so retrieval is forgiving about exact phrasing.
    """
    t = text.lower()
    now = datetime.now(timezone.utc)

    if "yesterday" in t:
        day = (now - timedelta(days=1)).date()
        start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
        return start.timestamp(), (start + timedelta(days=2)).timestamp()

    if "today" in t:
        start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
        return (start - timedelta(days=1)).timestamp(), now.timestamp()

    m = re.search(
        r"(\d+)\s+(minute|hour|day|week|month|year)s?\s+ago", t
    )
    if m:
        amount = int(m.group(1))
        delta = _REL_UNITS[m.group(2)] * amount
        target = now - delta
        pad = max(delta * 0.25, timedelta(days=1))
        return (target - pad).timestamp(), (target + pad).timestamp()

    m = re.search(r"last\s+(week|month|year)", t)
    if m:
        delta = _REL_UNITS[m.group(1)]
        return (now - delta * 2).timestamp(), now.timestamp()

    return None


# --------------------------------------------------------------------------
# RAG retrieval
# --------------------------------------------------------------------------
async def retrieve_evidence(
    question: str,
    exclude_id: int | None = None,
    guild_id: int | None = None,
) -> list[dict]:
    """Hybrid (BM25 + Gemini-embedding) recall, plus optional timeframe.

    ``guild_id`` scopes every lookup to one server so the bot never mixes
    histories from servers it's also a member of. The hybrid path falls
    back to pure BM25 transparently when the guild has no embeddings yet
    (backfill not done, or fresh archive).
    """
    limit = config.RAG_MAX_CONTEXT_MESSAGES
    gid = str(guild_id) if guild_id is not None else None

    content_hits = await asyncio.to_thread(
        retrieval.hybrid_search, question, limit, gid
    )

    window = parse_timeframe(question)
    time_hits: list[dict] = []
    if window:
        start_ts, end_ts = window
        time_hits = await asyncio.to_thread(
            database.search_by_timeframe, start_ts, end_ts, limit, gid
        )

    # Drop the live question itself: on_message archives the question
    # before retrieval runs, and BM25 / vector both rank the user's own
    # words as a top hit, which Gemini then cites back at them.
    exclude = str(exclude_id) if exclude_id is not None else None

    # Merge, de-dupe by message id, keep chronological order for the LLM.
    merged: dict[str, dict] = {}
    for m in content_hits + time_hits:
        mid = m["message_id"]
        if exclude is not None and str(mid) == exclude:
            continue
        merged[mid] = m

    # No "recent activity" fallback on purpose: if the archive has nothing
    # relevant, return empty so the bot answers conversationally instead of
    # quoting unrelated messages on every reply.
    ordered = sorted(merged.values(), key=lambda m: m["created_ts"])
    return ordered[: limit]


async def extract_images(message: discord.Message) -> list[Image.Image]:
    """Download image attachments on the live message into PIL images."""
    images: list[Image.Image] = []
    for att in message.attachments:
        ctype = (att.content_type or "").lower()
        if not ctype.startswith("image/"):
            continue
        try:
            raw = await att.read()
            images.append(Image.open(io.BytesIO(raw)).convert("RGB"))
        except Exception:
            log.warning("Could not load attachment %s", att.filename)
    return images


async def send_chunked(channel: discord.abc.Messageable, text: str) -> None:
    """Discord caps messages at 2000 chars; split on line boundaries."""
    limit = 1900
    while text:
        if len(text) <= limit:
            await channel.send(text)
            return
        cut = text.rfind("\n", 0, limit)
        if cut == -1:
            cut = limit
        await channel.send(text[:cut])
        text = text[cut:].lstrip("\n")


# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------
@bot.event
async def on_ready() -> None:
    database.init_db()
    log.info("Logged in as %s (id=%s)", bot.user, bot.user.id)
    log.info(config.summary())

    total = await asyncio.to_thread(database.count_messages)
    log.info("Local archive currently holds %d messages.", total)

    if total == 0:
        if config.AUTO_SYNC_IF_EMPTY:
            log.info("Archive empty + AUTO_SYNC_IF_EMPTY=on -> syncing now.")
            for guild in bot.guilds:
                await _run_sync(guild, channel=None)
        else:
            log.info(
                "Archive empty. An admin should run "
                "'%ssync' in the server to build it.",
                config.COMMAND_PREFIX,
            )

    # Catch the embedding index up to whatever's in the archive. Fires
    # exactly once per process (the lock inside the loop guards re-entry
    # if the gateway reconnects and on_ready runs again).
    asyncio.create_task(_embedding_backfill_loop())


@bot.event
async def on_message(message: discord.Message) -> None:
    # Ignore our own messages to avoid loops, but DO archive other bots/humans.
    if message.author.id == bot.user.id:
        return

    # Phase 2: persist every incoming message immediately, then embed it
    # fire-and-forget so future hybrid queries see it.
    if message.guild is not None:
        try:
            inserted = await asyncio.to_thread(
                database.insert_message,
                {
                    "message_id": message.id,
                    "channel_id": message.channel.id,
                    "channel_name": getattr(message.channel, "name", "dm"),
                    "guild_id": message.guild.id,
                    "author_id": message.author.id,
                    "author_name": str(message.author),
                    "content": message.content or "",
                    "created_at": message.created_at,
                    "attachments": "\n".join(
                        a.url for a in message.attachments
                    ),
                },
            )
        except Exception:
            log.exception("Failed to archive live message %s", message.id)
            inserted = False

        if inserted and (message.content or "").strip():
            rowid = await asyncio.to_thread(
                database.get_rowid_by_message_id, message.id
            )
            if rowid is not None:
                asyncio.create_task(
                    _embed_live_message(
                        rowid, str(message.guild.id), message.content
                    )
                )

    # Let the commands extension handle prefixed commands (e.g. !sync).
    await bot.process_commands(message)

    if message.author.bot:
        return  # don't answer other bots

    # Phase 3: answer when the bot is mentioned (a "tagged" question).
    if bot.user in message.mentions:
        await handle_question(message)


# The model ends a grounded answer with a machine-only "CITES: <ids>"
# line listing the exact evidence messages it used. We parse it, strip it
# from what the user sees, and turn ONLY those ids into jump links.
_CITES_RE = re.compile(r"(?im)^[ \t>*_-]*CITES:[ \t]*(.*?)\s*$")


def _strip_cites(text: str) -> tuple[str, list[int]]:
    """Remove the trailing CITES line; return (clean_text, cited_ids)."""
    matches = list(_CITES_RE.finditer(text))
    if not matches:
        return text.strip(), []
    last = matches[-1]
    ids = [int(tok) for tok in re.findall(r"\d{15,}", last.group(1))]
    # Drop the CITES line and anything after it (nothing should follow).
    return text[: last.start()].rstrip(), ids


def _reference_lines(
    ids: list[int], evidence: list[dict], fallback_guild_id: int | None,
    cap: int = 5,
) -> list[str]:
    """Human-readable references for the cited messages.

    Each line gives channel + date (always works, even if the reader
    can't open the link) plus a Discord jump-link when the ids needed to
    build one are present. Everything here is already in the evidence
    rows - no extra storage or query, so this is as cheap as a plain
    date/channel string while staying clickable.
    """
    # message_id is stored as a string; key the lookup by int so it
    # matches the integer ids parsed out of the model's CITES line.
    by_id: dict[int, dict] = {}
    for e in evidence:
        mid = e.get("message_id")
        if mid is not None:
            try:
                by_id[int(mid)] = e
            except (TypeError, ValueError):
                continue
    out: list[str] = []
    seen: set[int] = set()
    for cid_int in ids:
        e = by_id.get(cid_int)
        if e is None or cid_int in seen:
            continue
        seen.add(cid_int)
        # Use the row's stored string ids verbatim (exact, never mangled).
        msid = str(e.get("message_id") or cid_int)
        cid = e.get("channel_id")
        gid = e.get("guild_id") or (
            str(fallback_guild_id) if fallback_guild_id else None
        )
        chan = e.get("channel_name") or "unknown"
        date = str(e.get("created_at") or "")[:10]  # YYYY-MM-DD
        if cid and gid:
            link = f"https://discord.com/channels/{gid}/{cid}/{msid}"
            out.append(f"from #{chan} on {date}: {link}")
        else:
            out.append(f"from #{chan} on {date}")
        if len(out) >= cap:
            break
    return out


_SELF_RE = re.compile(
    r"(?:^|\s)(?:my|me|i'?m|i'?ve|myself)(?:\s|[?!.,]|$)|(?:^|\s)i(?:\s|[?!.,]|$)",
    re.IGNORECASE,
)


async def _resolve_author(
    question: str, message: discord.Message, hint: str | None
) -> dict | None:
    """Figure out which archived author a by_author question is about.

    Priority: an explicit Discord @-mention of another member, then a
    self-reference (asker themselves), then a fuzzy name match against
    the archive (scoped to THIS server so we don't pick someone with
    the same nickname from a different guild). None if nothing pans
    out.
    """
    others = [u for u in message.mentions if u.id != bot.user.id]
    if others:
        u = others[0]
        return {"author_id": str(u.id), "author_name": str(u)}

    looks_self = hint == "self" or bool(_SELF_RE.search(question))
    if looks_self:
        return {
            "author_id": str(message.author.id),
            "author_name": str(message.author),
        }

    if hint:
        gid = str(message.guild.id) if message.guild else None
        found = await asyncio.to_thread(database.find_author, hint, gid)
        if found:
            return {
                "author_id": found["author_id"],
                "author_name": found["author_name"],
            }
    return None


async def handle_question(message: discord.Message) -> None:
    # Strip the bot mention to get the bare question.
    question = re.sub(r"<@!?\d+>", "", message.content).strip()
    images = await extract_images(message)

    if not question and not images:
        await message.reply(
            "Ask me something about this server's history, "
            "e.g. \"who proposed a mall meetup 2 days ago?\""
        )
        return

    async with message.channel.typing():
        evidence: list[dict] = []
        strategy = "none"
        author_label: str | None = None
        searched = False

        if question:
            # Explicit timeframe ("2 days ago") always means keyword/time
            # retrieval; skip the classifier round-trip in that case.
            if parse_timeframe(question):
                plan = {"strategy": "keyword", "author_hint": None}
            else:
                plan = await gemini_client.plan_retrieval(question)

            strategy = plan["strategy"]
            hint = plan.get("author_hint")

            # Scope every retrieval to the asking guild so the bot does
            # NOT mix histories across servers it's also a member of.
            gid = str(message.guild.id) if message.guild else None

            if strategy == "keyword":
                evidence = await retrieve_evidence(
                    question,
                    exclude_id=message.id,
                    guild_id=message.guild.id if message.guild else None,
                )
                searched = True
            elif strategy == "by_author":
                target = await _resolve_author(question, message, hint)
                if target:
                    author_label = target["author_name"]
                    evidence = await asyncio.to_thread(
                        database.search_by_author_id,
                        target["author_id"],
                        300,
                        str(message.id),
                        gid,
                    )
                    searched = True
                else:
                    # Couldn't pin down who — fall back to plain chat so
                    # the model can ask for clarification.
                    strategy = "none"
            elif strategy == "multi_author":
                evidence = await asyncio.to_thread(
                    database.top_authors_sample, 80, 12, gid
                )
                searched = True
            elif strategy == "time_sample":
                evidence = await asyncio.to_thread(
                    database.time_sample, 24, 30, gid
                )
                searched = True
            # strategy == "none" -> no retrieval

        answer = await gemini_client.answer_question(
            question or "What is in this image?",
            evidence,
            images=images or None,
            searched=searched,
            strategy=strategy,
            author_label=author_label,
        )
        # Always strip the machine-only CITES line so it never shows.
        answer, cited_ids = _strip_cites(answer)
        # References only land for the keyword path — synthesis answers
        # (by_author / multi_author / time_sample) don't cite.
        if strategy == "keyword" and evidence and cited_ids:
            guild_id = message.guild.id if message.guild else None
            refs = _reference_lines(cited_ids, evidence, guild_id)
            if refs:
                head = "reference:" if len(refs) == 1 else "references:"
                answer = f"{answer}\n\n{head}\n" + "\n".join(refs)
    await send_chunked(message.channel, answer)


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
async def _run_sync(guild: discord.Guild, channel) -> None:
    if _sync_lock.locked():
        if channel:
            await channel.send("A sync is already running - please wait.")
        return

    async with _sync_lock:
        last = {"t": 0.0}

        async def progress(text: str) -> None:
            # Throttle progress posts so we don't spam the channel.
            now = asyncio.get_event_loop().time()
            if channel and (now - last["t"] > 8 or text.startswith("Sync complete")):
                last["t"] = now
                try:
                    await channel.send(f"`{text}`")
                except Exception:
                    pass

        report = await scraper.sync_guild(guild, progress)
        if channel:
            await send_chunked(channel, f"```\n{report.summary()}\n```")


@bot.command(name="sync")
@commands.has_permissions(administrator=True)
async def sync_cmd(ctx: commands.Context) -> None:
    """Admin: scrape/refresh the full server history into the local archive."""
    if ctx.guild is None:
        await ctx.send("Run this in a server channel.")
        return
    await ctx.send(
        "Starting history sync. This can take a while for large servers - "
        "I'll report progress and a summary when done."
    )
    await _run_sync(ctx.guild, ctx.channel)


@sync_cmd.error
async def sync_cmd_error(ctx: commands.Context, error) -> None:
    if isinstance(error, commands.MissingPermissions):
        await ctx.send("You need the **Administrator** permission to sync.")
    else:
        log.exception("sync command error", exc_info=error)
        await ctx.send(f"Sync command failed: {error}")


@bot.command(name="stats")
async def stats_cmd(ctx: commands.Context) -> None:
    """Show what's currently stored in the local archive."""
    s = await asyncio.to_thread(database.stats)
    await ctx.send(
        "**Local archive**\n"
        f"- Messages: `{s['total_messages']}`\n"
        f"- Channels: `{s['channels']}`\n"
        f"- Authors:  `{s['authors']}`\n"
        f"- Span:     `{s['oldest']}`  ->  `{s['newest']}`"
    )


@bot.command(name="ask")
async def ask_cmd(ctx: commands.Context, *, question: str) -> None:
    """Ask a historical question without @-mentioning the bot."""
    async with ctx.typing():
        evidence = await retrieve_evidence(
            question,
            guild_id=ctx.guild.id if ctx.guild else None,
        )
        answer = await gemini_client.answer_question(
            question, evidence, searched=True, strategy="keyword",
        )
        answer, _ = _strip_cites(answer)
    await send_chunked(ctx.channel, answer)


def run() -> None:
    try:
        database.init_db()
        bot.run(config.DISCORD_BOT_TOKEN, log_handler=None)
    finally:
        database.close()


if __name__ == "__main__":
    run()
