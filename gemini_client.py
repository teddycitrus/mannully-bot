"""
gemini_client.py
=================
Thin wrapper around the Google Generative AI SDK (Gemini 2.5 Flash).

Responsibilities
----------------
* Configure the SDK once with the API key from config.
* Build a grounded RAG prompt from retrieved historical messages so the
  model answers strictly from the server's own archive.
* Support multimodal input: images attached to the live question are
  passed to Gemini alongside the text.
* Keep the (blocking) SDK call off the Discord event loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Optional, Sequence

import google.generativeai as genai
from PIL import Image

import config

# Retrieval strategies the router can pick. Anything else is treated as
# "keyword" (the safest fallback).
STRATEGIES = {"none", "keyword", "by_author", "multi_author", "time_sample"}

log = logging.getLogger("gemini")

genai.configure(api_key=config.GEMINI_API_KEY)

_SYSTEM_INSTRUCTION = (
    "You are a helpful assistant bot for a Discord server. You have no "
    "special personality, gimmick, character, or attitude. You sound "
    "like a normal, natural person: calm, plain, and to the point.\n\n"
    "Style rules:\n"
    "1. Write entirely in lowercase. No capitalization anywhere, "
    "including the start of sentences and proper nouns or names.\n"
    "2. Never use em dashes. Use a comma, a period, or a separate "
    "sentence instead.\n"
    "3. Be concise. Answer directly with no preamble, padding, or "
    "filler. A sentence or two is usually enough.\n"
    "4. Always be factual and accurate. Never make anything up. When "
    "unsure or the info isn't there, say it casually like a person "
    "would: 'idk', 'not sure', 'no clue', 'nothing on that'. NEVER "
    "use robotic phrasings like 'the evidence does not state', 'the "
    "archive does not contain', 'the provided information does not "
    "indicate', or anything similar.\n"
    "5. Sound human and natural, but neutral: no roleplay, no catch "
    "phrases, no slang gimmick, no emotional act.\n\n"
    "Grounding:\n"
    "- Each archive line is [id=NUMBER] [timestamp] #channel <author>: "
    "message.\n"
    "- If the prompt contains an 'ARCHIVED EVIDENCE' block, the member "
    "is asking a specific recall question. Answer ONLY from that "
    "evidence, name the people involved, and reference the date/channel "
    "when relevant. If the evidence doesn't actually answer the "
    "question, say so casually ('idk', 'not sure', 'nothing on that') "
    "and NEVER 'the evidence does not state' / 'the archive does not "
    "contain'. NEVER invent events, people, or dates. After your reply, "
    "add a final separate line in EXACTLY this form: 'CITES: <comma-"
    "separated id numbers of ONLY the specific evidence messages you "
    "actually used to answer>'. If you used none, write 'CITES: none'. "
    "This line is MANDATORY for ARCHIVED EVIDENCE, it is exempt from "
    "the conciseness rule, and you must never omit it. It is stripped "
    "out before posting, never refer to it in your prose, and put "
    "nothing after it.\n"
    "- If the prompt contains an 'ARCHIVE SAMPLE' block, the member is "
    "asking for a synthesis or summary (a personality read, an overview "
    "of activity, the vibe of the server, etc). Treat the sample as "
    "source material: characterise people, themes, tendencies, and "
    "concrete examples drawn from what's there. You DO NOT need to "
    "produce a CITES line for ARCHIVE SAMPLE. Be honest if the sample "
    "really is too thin; otherwise commit to a real read instead of "
    "hedging. The conciseness rule is relaxed for these: a short "
    "paragraph or a few bullets per subject is fine when it's "
    "warranted, but don't pad.\n"
    "- If there is NO archive block, the member is just talking to "
    "you. Reply normally and concisely. Do NOT quote, cite, or pretend "
    "to recall past server messages: you have nothing to look up, so "
    "do not make anything up or reference history that was not given "
    "to you."
)

_model = genai.GenerativeModel(
    config.GEMINI_MODEL,
    system_instruction=_SYSTEM_INSTRUCTION,
)

# Separate, persona-free model used only to decide whether a message needs
# a history lookup. Plain yes/no - no lowercase-sass system instruction.
_classifier_model = genai.GenerativeModel(config.GEMINI_MODEL)


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.DOTALL)


async def plan_retrieval(question: str) -> dict[str, Any]:
    """Decide what kind of archive lookup `question` needs.

    Returns a plan dict: {"strategy": <name>, "author_hint": <str|None>}.
    Strategies:
      none         -> answer conversationally, skip the archive
      keyword      -> BM25 over message text (specific recall + timeframe)
      by_author    -> pull one author's messages (personality/vibe reads)
      multi_author -> sample top active authors (everyone's personalities)
      time_sample  -> sample across the server's timeline (overall history)

    Fails safe: an error or unknown strategy maps to "keyword" so the bot
    still tries to ground its answer rather than blurt out invented info.
    """
    probe = (
        "Classify a Discord message into ONE retrieval strategy for "
        "searching the server's past message archive. Reply with a "
        "single-line JSON object: "
        "{\"strategy\": <name>, \"author_hint\": <name or null>}.\n\n"
        "Strategies:\n"
        "- \"none\": greetings, opinions, general world knowledge, "
        "questions about current/real-time state, anything where the "
        "archive isn't needed.\n"
        "- \"keyword\": specific recall with discriminating keywords "
        "(proper nouns, named events, in-jokes, channel-specific "
        "topics). BM25 over message text will surface useful hits. Use "
        "this whenever the question pins to a specific event or topic.\n"
        "- \"by_author\": questions about ONE person's personality, "
        "vibe, habits, or what they talk about. Set author_hint to "
        "their name, or to \"self\" if asking about the speaker "
        "themselves (uses 'my', 'me', 'I'm', 'I').\n"
        "- \"multi_author\": questions about EVERYONE or multiple "
        "people's personalities or vibes (\"everyone's personalities\", "
        "\"describe each person\", \"who's who in this server\").\n"
        "- \"time_sample\": questions about the server's overall "
        "history, major events, big moments, what's happened in "
        "general, or the vibe over time, when no specific keywords pin "
        "it to one event.\n\n"
        "Examples:\n"
        "'good morning' -> {\"strategy\":\"none\",\"author_hint\":null}\n"
        "'is python better than rust' -> {\"strategy\":\"none\",\"author_hint\":null}\n"
        "'who proposed the mall meetup last week' -> {\"strategy\":\"keyword\",\"author_hint\":null}\n"
        "'what restaurant does usman always ask about' -> {\"strategy\":\"keyword\",\"author_hint\":null}\n"
        "'whats my personality like' -> {\"strategy\":\"by_author\",\"author_hint\":\"self\"}\n"
        "'describe me based on my messages' -> {\"strategy\":\"by_author\",\"author_hint\":\"self\"}\n"
        "'describe usman based on his messages' -> {\"strategy\":\"by_author\",\"author_hint\":\"usman\"}\n"
        "'what does bob talk about a lot' -> {\"strategy\":\"by_author\",\"author_hint\":\"bob\"}\n"
        "'give me everyones personalities' -> {\"strategy\":\"multi_author\",\"author_hint\":null}\n"
        "'describe everyone in this server' -> {\"strategy\":\"multi_author\",\"author_hint\":null}\n"
        "'what are some major things that happened in this server' -> {\"strategy\":\"time_sample\",\"author_hint\":null}\n"
        "'whats the vibe of this server' -> {\"strategy\":\"time_sample\",\"author_hint\":null}\n\n"
        f"Message: {question}\nAnswer:"
    )

    def _gen() -> dict[str, Any]:
        try:
            resp = _classifier_model.generate_content(probe)
            raw = (getattr(resp, "text", "") or "").strip()
            raw = _FENCE_RE.sub("", raw).strip()
            data = json.loads(raw)
            strategy = str(data.get("strategy") or "").strip().lower()
            if strategy not in STRATEGIES:
                strategy = "keyword"
            hint = data.get("author_hint")
            if isinstance(hint, str):
                hint = hint.strip() or None
            else:
                hint = None
            return {"strategy": strategy, "author_hint": hint}
        except Exception:
            log.exception("retrieval planning failed; defaulting to keyword")
            return {"strategy": "keyword", "author_hint": None}

    return await asyncio.to_thread(_gen)


def _format_evidence(messages: Sequence[dict[str, Any]]) -> str:
    if not messages:
        return "(no matching messages were found in the archive)"
    lines = []
    for m in messages:
        mid = m.get("message_id")
        ts = str(m.get("created_at", ""))[:19].replace("T", " ")
        chan = m.get("channel_name") or "?"
        author = m.get("author_name") or "unknown"
        content = (m.get("content") or "").strip()
        if not content and m.get("attachments"):
            content = "[attachment only]"
        lines.append(f"[id={mid}] [{ts}] #{chan} <{author}>: {content}")
    return "\n".join(lines)


def build_prompt(
    question: str,
    evidence: Sequence[dict[str, Any]],
    searched: bool = False,
    strategy: str = "keyword",
    author_label: Optional[str] = None,
) -> str:
    # No evidence: either we tried (searched=True, real miss) or we didn't
    # treat it as a history question. Either way no archive block ships.
    if not evidence and searched:
        return (
            "A server member asked a question about the server's past. "
            "You searched the archive and found NOTHING relevant.\n\n"
            f"MEMBER:\n{question}\n\n"
            "Tell them casually you've got nothing on it - 'idk', "
            "'no clue from the logs', 'nothing on that'. Do NOT say "
            "'the evidence does not state' or 'the archive does not "
            "contain'. Do NOT invent history and do NOT answer from "
            "general knowledge: you genuinely have no record."
        )
    if not evidence:
        return (
            "A server member is talking to you (this is NOT a history "
            "question; no archive was searched).\n\n"
            f"MEMBER:\n{question}\n\n"
            "Reply normally and concisely. Do not quote or invent past "
            "messages."
        )

    # Keyword path: tight evidence list, model must cite. The system
    # instruction enforces CITES when it sees 'ARCHIVED EVIDENCE'.
    if strategy == "keyword":
        return (
            "ARCHIVED EVIDENCE (server history):\n"
            "------------------------------------\n"
            f"{_format_evidence(evidence)}\n"
            "------------------------------------\n\n"
            f"QUESTION FROM A MEMBER:\n{question}\n\n"
            "Answer using only the evidence above. Then end with the "
            "final 'CITES: <id numbers you actually used>' line as "
            "instructed - only the messages you genuinely referenced, "
            "not all of them."
        )

    # Synthesis paths: the model treats this as source material for a
    # summary/personality read, no CITES required.
    if strategy == "by_author":
        who = author_label or "this member"
        guidance = (
            f"This is a recent sample of {who}'s messages. Read it and "
            f"characterise their personality, tone, and what they talk "
            f"about. Use concrete details and recurring themes you can "
            f"see in the messages. Don't moralise, don't flatter, don't "
            f"hedge: give an honest read. If the sample is really too "
            f"thin to tell, say so."
        )
    elif strategy == "multi_author":
        guidance = (
            "This is a sample of messages from the most active members "
            "of the server, grouped by author. Give a short personality "
            "read for each of the major members you see - one short "
            "paragraph or a couple of bullets per person, grounded in "
            "what they actually say. Skip anyone whose sample is too "
            "thin to read."
        )
    else:  # time_sample
        guidance = (
            "This is a sample of messages drawn from across the "
            "server's history, oldest to newest. Summarise the major "
            "themes, recurring topics, and notable moments or shifts "
            "you can see. Cite concrete examples (people, channels, "
            "dates) when they're vivid. Don't fabricate events that "
            "aren't in the sample."
        )

    return (
        "ARCHIVE SAMPLE (source material for a synthesis answer):\n"
        "--------------------------------------------------------\n"
        f"{_format_evidence(evidence)}\n"
        "--------------------------------------------------------\n\n"
        f"QUESTION FROM A MEMBER:\n{question}\n\n"
        f"{guidance}\n"
        "No CITES line is needed for ARCHIVE SAMPLE; just answer."
    )


async def answer_question(
    question: str,
    evidence: Sequence[dict[str, Any]],
    images: Optional[Sequence[Image.Image]] = None,
    searched: bool = False,
    strategy: str = "keyword",
    author_label: Optional[str] = None,
) -> str:
    """
    Generate an answer grounded in the retrieved evidence. ``searched``
    means we treated this as a history question and queried the archive
    (so an empty ``evidence`` is a real miss, not casual chat). Any PIL
    images on the live question are included for multimodal reasoning.
    Runs the blocking SDK call in a worker thread.
    """
    prompt = build_prompt(
        question, evidence, searched=searched,
        strategy=strategy, author_label=author_label,
    )
    parts: list[Any] = [prompt]
    if images:
        parts.extend(images)

    def _generate() -> str:
        try:
            resp = _model.generate_content(parts)
            text = (getattr(resp, "text", "") or "").strip()
            return text or (
                "I searched the archive but couldn't form an answer "
                "from what I found."
            )
        except Exception as exc:  # surface a friendly message, log detail
            log.exception("Gemini generation failed")
            return f"Sorry - I couldn't reach the language model ({exc})."

    return await asyncio.to_thread(_generate)


async def describe_images(
    prompt: str, images: Sequence[Image.Image]
) -> str:
    """Pure multimodal helper: ask Gemini about image(s) with no archive."""
    def _generate() -> str:
        try:
            resp = _model.generate_content([prompt, *images])
            return (getattr(resp, "text", "") or "").strip()
        except Exception as exc:
            log.exception("Gemini image call failed")
            return f"Sorry - image analysis failed ({exc})."

    return await asyncio.to_thread(_generate)
