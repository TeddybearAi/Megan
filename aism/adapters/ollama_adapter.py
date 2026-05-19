"""
AISM — OllamaAdapter (Concrete GenerationAdapter)
==================================================
HTTP adapter for local LLMs served via Ollama (https://ollama.com).

Implements the GenerationAdapter Protocol. The adapter is the ONLY place in
the system that calls an LLM; AISM stages are deterministic.

Patch trail:
    v2.0   : initial adapter, llama3.2:3b default.
    v2.1   : Piper TTS handoff, MEGAN_IDENTITY_RULES introduced.
    v2.2   : Qwen3:30b default, no-think rule, <think> stripper, num_ctx,
             time-of-day context, person_context_block injection point.
    v2.2.1 : (this file) — LTM-retrieval guard. When the Person Bank reports
             this is an introduction / person-focused turn, retrieved_facts
             are NOT prepended to the user message. This stops cross-person
             memory grafting (the "Vahid is the one you've been bullying"
             failure mode from 2026-05-15).

Constraint (per BluePrint v2.1 §5 / R1):
    AISM itself never calls this adapter for internal reasoning. Only the
    reply-generation step uses it. Adapter output is stored as a
    ConversationTurn and that's it.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib import error, request
from zoneinfo import ZoneInfo

from ..data_models import ResponsePolicy
from ..session_context import SessionContext


# ============================================================
# Megan identity rules — reach the model on every turn regardless of mode
# ============================================================

MEGAN_IDENTITY_RULES = """You are Megan — a British-Australian man who has
lived in Australia for five hundred years. You speak with a British accent
and Australian warmth. You address the user (Teddy) as "Teddy", "my dear",
or "my girl" — never "mate". "Mate" stays in your vocabulary as an
exclamation or aside ("mate, that's mad"), not as how you address her.

CRITICAL — "my dear", "my girl", "darling", "love" are intimate pet-names
reserved EXCLUSIVELY for Teddy. NEVER attach them to an address to anyone
else. "Hi Vahid, my dear" is wrong — Vahid is not your dear, Teddy is.
When you address another named person (Vahid, Tahnay, Zoe, anyone Teddy
introduces) OR a room of people, drop these forms entirely. Use the
person's name or no address at all. The same rule applies to sign-off
lines: never end a turn with "My dear" when the audience isn't Teddy.

Reply directly with your in-character response. Do NOT produce
<think>...</think> blocks or any visible chain-of-thought before your
reply. Do not narrate your reasoning steps ("let me think about...",
"first I should consider..."). The user wants conversational presence,
not a thought process. Reason internally; speak as Megan."""


# ============================================================
# Defensive <think> block stripper (v2.2)
# ============================================================

_THINKING_BLOCK_RE = re.compile(
    r"<think\b[^>]*>.*?</think>",
    flags=re.DOTALL | re.IGNORECASE,
)


def _strip_thinking_blocks(text: str) -> str:
    """Remove any <think>...</think> blocks Qwen3 may still emit."""
    return _THINKING_BLOCK_RE.sub("", text).strip()


# ============================================================
# Time-of-day context (v2.2)
# ============================================================

_TZ_SYDNEY = ZoneInfo("Australia/Sydney")


def _build_time_block(now: Optional[datetime] = None) -> str:
    """
    Tell Megan what the actual clock says. Windows-portable formatting (no
    POSIX-only %-d / %-I). zoneinfo gives us correct local time regardless
    of where the server runs.
    """
    now = now or datetime.now(_TZ_SYDNEY)
    # Day-of-month without leading zero, portable.
    day = str(now.day)
    # 12-hour clock without leading zero, portable.
    hour_12 = now.hour % 12
    if hour_12 == 0:
        hour_12 = 12
    ampm = "am" if now.hour < 12 else "pm"
    clock = f"{hour_12}:{now.minute:02d} {ampm}"
    weekday = now.strftime("%A")
    month = now.strftime("%B")
    year = now.strftime("%Y")

    h = now.hour
    if 5 <= h < 12:
        tod = "morning"
    elif 12 <= h < 17:
        tod = "afternoon"
    elif 17 <= h < 21:
        tod = "evening"
    else:
        tod = "night"

    return (
        "Current local time context:\n"
        f"- It is {weekday}, {day} {month} {year}, {clock} (Sydney, Australia).\n"
        f"- Time of day: {tod}.\n"
        "- Do NOT assume the user is going to bed, has just woken up, is "
        "starting their day, or is at any particular life moment unless "
        "they have actually told you so in this conversation. If they say "
        "'hi', just say hi back. Never tell them to 'sleep well' or 'have "
        "a good night' or ask if they 'slept well' on your own initiative "
        "— those are diurnal assumptions and they are usually wrong."
    )


# ============================================================
# OllamaAdapter
# ============================================================

class OllamaAdapter:
    """
    Concrete GenerationAdapter that talks to a local Ollama server.

    render_policy() converts the ResponsePolicy into a system prompt string.
    generate() POSTs to /api/chat and returns the reply text.
    """

    DEFAULT_HOST = "http://localhost:11434"
    DEFAULT_TIMEOUT_SECONDS = 120
    DEFAULT_NUM_CTX = 16384  # 16K — plenty for Megan, well below
                             # the 256K Qwen3 supports natively.

    def __init__(
        self,
        model: str = "qwen2.5:32b",
        host: str = DEFAULT_HOST,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        temperature: float = 0.7,
        num_ctx: int = DEFAULT_NUM_CTX,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.num_ctx = num_ctx

    # ============================================================
    # GenerationAdapter interface
    # ============================================================

    def render_policy(self, policy: ResponsePolicy) -> Dict[str, Any]:
        parts: List[str] = []
        if policy.tone:
            parts.append(f"- Tone: {policy.tone}")
        if policy.verbosity:
            parts.append(f"- Length: keep replies {policy.verbosity}")
        if policy.formatting:
            parts.append(f"- Formatting: {policy.formatting}")
        if policy.emotional_style:
            parts.append(f"- Emotional style: {policy.emotional_style}")
        if policy.proactiveness:
            parts.append(f"- Proactiveness: {policy.proactiveness}")
        if policy.reasoning_presentation:
            parts.append(f"- Reasoning: {policy.reasoning_presentation}")

        if parts:
            style_block = (
                "Adapt your replies to the user's communication preferences:\n"
                + "\n".join(parts)
            )
        else:
            style_block = ""

        return {
            "system_prompt": style_block,
            "model": self.model,
            "provenance": policy.metadata,
        }

    def generate(
        self,
        user_message: str,
        retrieved_facts: List[Dict[str, Any]],
        rendered_policy: Dict[str, Any],
        session_context: SessionContext,
    ) -> str:
        """
        Build the system prompt, optionally prepend LTM facts to the user
        message, POST to Ollama /api/chat, strip any leaked <think> blocks,
        and return the reply text.

        v2.2.1: LTM facts are NOT prepended when session_context tells us
        this turn is person-focused (intro or sticky person). This prevents
        cross-person memory grafting.
        """
        system_prompt = self._build_system_prompt(
            rendered_policy.get("system_prompt", ""),
            session_context,
        )

        # ---- v2.2.1 LTM-retrieval guard ----
        # If the Person Bank flagged this turn as introduction-or-person-focused,
        # do NOT prepend retrieved_facts. The Person Bank context block already
        # supplies what the model needs to know, and pulling in semantically
        # similar LTM about OTHER people grafts the wrong details onto the
        # named person.
        #
        # NOTE: this is a direct attribute access, NOT getattr-with-default.
        # If SessionContext is missing this field you'll get an AttributeError
        # the first time you talk to Megan after deploy. That's intentional —
        # silent default would make the whole patch a no-op. If the error
        # fires, add `skip_ltm_retrieval: bool = False` to SessionContext.
        skip_ltm = bool(session_context.skip_ltm_retrieval)

        if retrieved_facts and not skip_ltm:
            fact_lines = [
                f"- {fact.get('summary') or fact.get('text') or str(fact)}"
                for fact in retrieved_facts
            ]
            user_content = (
                "Relevant context about the user:\n"
                + "\n".join(fact_lines)
                + "\n\nUser message: "
                + user_message
            )
        else:
            user_content = user_message

        # ---- v2.2.4 Short-term memory ----
        # Splice the last N turns of THIS session between the system
        # prompt and the current user message. Each historical turn
        # becomes its own {"role", "content"} entry, matching the format
        # Ollama (and OpenAI-shape APIs in general) expect.
        #
        # Defensive: missing attribute falls through to empty list, so a
        # deploy that skipped the SessionContext field doesn't crash.
        # Each turn's text is capped at 800 chars to guard num_ctx against
        # pathologically long entries.
        history_messages: List[Dict[str, str]] = []
        recent_turns = getattr(session_context, "recent_turns", []) or []
        for turn in recent_turns:
            try:
                role_value = (
                    turn.role.value
                    if hasattr(turn.role, "value")
                    else str(turn.role)
                )
                # Ollama only accepts "user" / "assistant" / "system".
                if role_value not in ("user", "assistant", "system"):
                    continue
                text = (turn.text or "").strip()
                if not text:
                    continue
                if len(text) > 800:
                    text = text[:800] + "…"
                history_messages.append({"role": role_value, "content": text})
            except Exception:
                # Never let a malformed turn break generation.
                continue

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                *history_messages,
                {"role": "user", "content": user_content},
            ],
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
            "stream": False,
        }

        raw = self._post_chat(payload)
        return _strip_thinking_blocks(raw)

    # ============================================================
    # Person-Bank-aware chat_json (used by PersonBank.observe)
    # ============================================================

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Run a focused JSON-extraction call. Used by PersonBank.observe() —
        NOT part of the main conversation flow. Returns a parsed dict or
        raises OllamaAdapterError on transport / parse failure.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": temperature,
                "num_ctx": self.num_ctx,
            },
            "format": "json",
            "stream": False,
        }
        raw = self._post_chat(payload)
        # Strip thinking blocks defensively; they shouldn't appear in JSON mode
        # but Qwen3 occasionally leaks them.
        raw = _strip_thinking_blocks(raw)
        # Strip markdown fences if present.
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(),
                     flags=re.MULTILINE)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OllamaAdapterError(
                f"JSON parse failed in chat_json: {exc}; raw={raw[:200]}"
            ) from exc

    # ============================================================
    # System-prompt construction
    # ============================================================

    def _build_system_prompt(
        self,
        style_block: str,
        session_context: SessionContext,
    ) -> str:
        """
        Stack the parts of the system prompt in a consistent order:
            1. Megan identity rules (always)
            2. v2.2.6 — Identity block: stable known facts about the
               user, surfaced directly so the model never has to guess
               at age/personality/languages/etc.
            3. Style block from ResponsePolicy (if any)
            4. Person Bank context block (if any) — IMMEDIATELY before
               time-of-day so the model reads the anti-conflation rules
               in close proximity to the people they apply to.
            5. Time-of-day block (always, v2.2)
        """
        sections: List[str] = [MEGAN_IDENTITY_RULES]

        identity_block = (
            getattr(session_context, "identity_block", "") or ""
        ).strip()
        if identity_block:
            sections.append(identity_block)

        if style_block:
            sections.append(style_block)

        person_block = (session_context.person_context_block or "").strip()
        if person_block:
            sections.append(person_block)

        sections.append(_build_time_block())

        return "\n\n".join(sections)

    # ============================================================
    # HTTP plumbing
    # ============================================================

    def _post_chat(self, payload: Dict[str, Any]) -> str:
        url = f"{self.host}/api/chat"
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
        except error.URLError as exc:
            raise OllamaAdapterError(
                f"Failed to reach Ollama at {url}: {exc}. "
                f"Is Ollama running? (`ollama serve`)"
            ) from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise OllamaAdapterError(
                f"Non-JSON response from Ollama: {body[:200]}"
            ) from exc

        message = parsed.get("message") or {}
        content = message.get("content", "")
        if not content:
            # Fallback for older API versions that return under 'response'.
            content = parsed.get("response", "")
        return content.strip()


class OllamaAdapterError(RuntimeError):
    """Raised when the adapter can't reach or parse Ollama."""
