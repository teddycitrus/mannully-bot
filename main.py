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
import gemini_client
import scraper

intents = discord.Intents.default()
intents.message_content = True  # REQUIRED: read message text (privileged)
intents.guilds = True
intents.members = True

bot = commands.Bot(command_prefix=config.COMMAND_PREFIX, intents=intents)

# Guard so two !sync runs can't overlap on the same process.
_sync_lock = asyncio.Lock()


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
async def retrieve_evidence(question: str) -> list[dict]:
    """Combine keyword search with an optional timeframe filter."""
    limit = config.RAG_MAX_CONTEXT_MESSAGES

    keyword_hits = await asyncio.to_thread(
        database.search_by_keyword, question, limit
    )

    window = parse_timeframe(question)
    time_hits: list[dict] = []
    if window:
        start_ts, end_ts = window
        time_hits = await asyncio.to_thread(
            database.search_by_timeframe, start_ts, end_ts, limit
        )

    # Merge, de-dupe by message id, keep chronological order for the LLM.
    merged: dict[int, dict] = {}
    for m in keyword_hits + time_hits:
        merged[m["message_id"]] = m

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


@bot.event
async def on_message(message: discord.Message) -> None:
    # Ignore our own messages to avoid loops, but DO archive other bots/humans.
    if message.author.id == bot.user.id:
        return

    # Phase 2: persist every incoming message immediately.
    if message.guild is not None:
        try:
            await asyncio.to_thread(
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
        # Only touch the archive when the message is actually a recall
        # question. An explicit timeframe ("2 days ago") always qualifies;
        # otherwise ask the lightweight classifier. Image-only messages and
        # plain chat skip retrieval entirely so the bot doesn't quote
        # unrelated history on every reply.
        evidence: list[dict] = []
        do_history = False
        if question:
            do_history = bool(parse_timeframe(question)) or (
                await gemini_client.needs_history(question)
            )
            if do_history:
                evidence = await retrieve_evidence(question)

        answer = await gemini_client.answer_question(
            question or "What is in this image?",
            evidence,
            images=images or None,
            searched=do_history,
        )
        # Always strip the machine-only CITES line so it never shows.
        answer, cited_ids = _strip_cites(answer)
        # Reference ONLY the message(s) the answer actually used.
        if evidence and cited_ids:
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
        evidence = await retrieve_evidence(question)
        answer = await gemini_client.answer_question(question, evidence)
    await send_chunked(ctx.channel, answer)


def run() -> None:
    try:
        database.init_db()
        bot.run(config.DISCORD_BOT_TOKEN, log_handler=None)
    finally:
        database.close()


if __name__ == "__main__":
    run()
