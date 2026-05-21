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
import logging
from typing import Any, Optional, Sequence

import google.generativeai as genai
from PIL import Image

import config

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
    "4. Always be factual and accurate. Never make anything up. If you "
    "are unsure or the information is not available, just say so "
    "plainly.\n"
    "5. Sound human and natural, but neutral: no roleplay, no catch "
    "phrases, no slang gimmick, no emotional act.\n\n"
    "Grounding:\n"
    "- If the prompt contains an 'ARCHIVED EVIDENCE' block, the member "
    "is asking about the server's past. Answer ONLY from that evidence, "
    "name the people involved, and reference the date/channel when "
    "relevant. Each evidence line is "
    "[id=NUMBER] [timestamp] #channel <author>: message. If the evidence "
    "does not contain the answer, say so plainly. NEVER invent events, "
    "people, or dates.\n"
    "- CITATION: after your reply, add a final separate line in EXACTLY "
    "this form: 'CITES: <comma-separated id numbers of ONLY the specific "
    "evidence messages you actually used to answer>'. If you used none, "
    "write 'CITES: none'. This line is MANDATORY whenever an evidence "
    "block is present, it is exempt from the conciseness rule, and you "
    "must never omit it. It is stripped out before the message is "
    "posted; never refer to it in your prose, and put nothing after "
    "it.\n"
    "- If there is NO evidence block, the member is just talking to you. "
    "Reply normally and concisely. Do NOT quote, cite, or pretend to "
    "recall past server messages: you have nothing to look up, so do "
    "not make anything up or reference history that was not given to "
    "you."
)

_model = genai.GenerativeModel(
    config.GEMINI_MODEL,
    system_instruction=_SYSTEM_INSTRUCTION,
)

# Separate, persona-free model used only to decide whether a message needs
# a history lookup. Plain yes/no - no lowercase-sass system instruction.
_classifier_model = genai.GenerativeModel(config.GEMINI_MODEL)


async def needs_history(question: str) -> bool:
    """True if answering `question` requires looking up past server messages.

    Used to gate RAG retrieval so the bot only quotes/cites the archive for
    genuine recall questions, not for greetings or general chat. Fails open
    (returns True) so a classifier error never causes a made-up answer.
    """
    probe = (
        "You decide if a Discord message needs the server's PAST message "
        "history to answer. Reply with exactly one word: yes or no.\n"
        "yes = asks about something that happened / was said before, who "
        "did/said something, when something occurred, or references a past "
        "time.\n"
        "no = a greeting, opinion, general knowledge, or anything "
        "answerable without server history.\n\n"
        f"Message: {question}\nAnswer:"
    )

    def _gen() -> bool:
        try:
            resp = _classifier_model.generate_content(probe)
            text = (getattr(resp, "text", "") or "").strip().lower()
            return text.startswith("y")
        except Exception:
            log.exception("history-intent classification failed; assuming yes")
            return True

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
) -> str:
    # Three cases:
    #  1. evidence found      -> RAG prompt (answer from it).
    #  2. searched, no hits   -> say there's no record, don't fabricate.
    #  3. not a history Q     -> plain conversational prompt, no quoting.
    if not evidence and searched:
        return (
            "A server member asked a question about the server's past. "
            "You searched the archive and found NOTHING relevant.\n\n"
            f"MEMBER:\n{question}\n\n"
            "Tell them plainly and concisely that there's nothing in the "
            "logs about it. Do NOT invent history and do NOT answer from "
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
    return (
        "ARCHIVED EVIDENCE (server history):\n"
        "------------------------------------\n"
        f"{_format_evidence(evidence)}\n"
        "------------------------------------\n\n"
        f"QUESTION FROM A MEMBER:\n{question}\n\n"
        "Answer using only the evidence above. Then end with the final "
        "'CITES: <id numbers you actually used>' line as instructed - "
        "only the messages you genuinely referenced, not all of them."
    )


async def answer_question(
    question: str,
    evidence: Sequence[dict[str, Any]],
    images: Optional[Sequence[Image.Image]] = None,
    searched: bool = False,
) -> str:
    """
    Generate an answer grounded in the retrieved evidence. ``searched``
    means we treated this as a history question and queried the archive
    (so an empty ``evidence`` is a real miss, not casual chat). Any PIL
    images on the live question are included for multimodal reasoning.
    Runs the blocking SDK call in a worker thread.
    """
    prompt = build_prompt(question, evidence, searched)
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
