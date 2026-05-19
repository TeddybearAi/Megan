"""
AISM — Interactive AI Companion Runner
======================================
Runs the full AISM pipeline interactively. Assumes Ollama is running locally.
This file serves as both a CLI tool and a logic provider for app.py.

Data is stored locally in ./aism_data/
"""

from __future__ import annotations

import uuid
import json
from datetime import datetime, timedelta, timezone

from aism import (
  AdaptiveInteractionStylePipeline,
  ConversationTurn,
  LocalJSONStore,
  SessionContext,
  TurnRole,
)
from aism.adapters.ollama_adapter import OllamaAdapter
from aism.memory import extract_memory_candidate


def _detect_resolution(user_input: str, memory: dict) -> bool:
  """Detect if the user's input suggests an emotional problem has been resolved."""
  if memory.get("memory_type") != "emotional_pattern":
    return False

  user_text = user_input.lower()
  original_topic = memory.get("utterance", "").lower()

  # Resolution keywords
  general_resolution_keywords = {
    "feeling better", "feel better", "much better", "better now", "all good",
    "all better", "it worked out", "figured it out", "sorted it out",
    "worked out", "great news", "good news", "things improved", "improving",
    "improved", "resolved"
  }

  job_specific = {
    "got a job", "got hired", "accepted", "offer", "start date", "job offer",
    "finally got", "just got", "landed", "hired"
  }

  exam_specific = {
    "passed", "passed the exam", "got through it", "finished the exam"
  }

  has_general_resolution = any(
    k in user_text for k in general_resolution_keywords)
  has_job_resolution = any(k in user_text for k in job_specific)
  has_exam_resolution = any(k in user_text for k in exam_specific)

  if "job" in original_topic or "application" in original_topic:
    return has_job_resolution or (
      has_general_resolution
      and any(w in user_text for w in ["job", "application", "interview"]))
  elif "exam" in original_topic or "test" in original_topic:
    return has_exam_resolution or has_general_resolution
  elif any(w in original_topic
           for w in ["anxious", "stressed", "anxiety", "stress"]):
    return has_general_resolution

  return False


def _generate_contextual_greeting(
    memory: dict,
    adapter,
    person_block: str = "",
) -> str:
  """Generate a greeting grounded in real identity + memory.

  v2.2.3 — Dr. Chen hallucination fix.
  ===================================
  The previous version of this function called the LLM with this entire
  system prompt:

      "You are a caring AI companion who remembers important things about
      the user. STRICT: NEVER mention being 'text-based' ..."

  and one isolated memory snippet as the user content. There was no
  MEGAN_IDENTITY_RULES, no Person Bank, no time context, and no
  anti-hallucination instruction. When the top memory referenced a role
  with no name attached ("our course coordinator", "my coach"), Qwen3
  filled the void by inventing one — most visibly "Dr. Chen teaching AI
  Studio" on 2026-05-15, where the actual coordinator (Vahid) was in the
  Person Bank but wasn't reachable to the model.

  The fix stacks the same identity scaffolding the main turn handler uses
  (MEGAN_IDENTITY_RULES + time + Person Bank), plus a greeting-specific
  rule block:

    * Do NOT invent names. If the memory references a role but no name,
      stay at the role.
    * Do NOT answer questions in the memory — the memory is something
      Teddy said previously, not a live message.
    * Do NOT claim to be teaching, employed by, or hold any position
      at any institution.
    * Do NOT make diurnal assumptions ("good morning", "still up late").

  Memory-type behaviour:
    emotional_pattern → generic warm string, no LLM. (Carried over from
      v2 — never lead an opening with raw emotional content.)
    profile_fact / style_preference → generic warm string, no LLM.
      The risk/reward of an LLM call for mundane profile facts isn't
      worth it.
    long_term_goal → LLM call with the fortified prompt above. This is
      the only path that produces a personalised greeting, and the only
      path that hallucinated previously.
    anything else → generic warm string.

  Args:
    memory: The top-priority memory dict from storage.
    adapter: The OllamaAdapter instance (typing not enforced for
      back-compat with tests that pass mocks).
    person_block: Pre-formatted Person Bank context as returned by
      PersonBank.format_for_prompt(). Empty string is fine — the
      greeting falls back to identity rules alone, which is still a
      large improvement over the v2.2.2 thin prompt.
  """
  memory_type = memory.get("memory_type", "")
  utterance = memory.get("utterance", "")

  # Hard rule, unchanged from v2: never lead with emotional content.
  if memory_type == "emotional_pattern":
    return "AI: Hey, my dear. Good to see you again. What's on your mind?"

  # profile_fact / style_preference: keep the safe generic responses.
  # No LLM call, no hallucination surface area.
  if memory_type == "profile_fact":
    return "AI: Welcome back, my dear. How have things been?"
  if memory_type == "style_preference":
    return "AI: Good to see you again, my dear."

  # Only long_term_goal uses the LLM path. Everything else falls through
  # to the generic warm string at the bottom.
  if memory_type != "long_term_goal":
    return "AI: Hey, my dear. Good to see you again. What's on your mind?"

  # ---- LLM-generated personalised greeting (the previously broken path) ----
  try:
    # Import here, not at module top, so that test environments / CLI
    # tooling that doesn't have the adapter stack can still import
    # main.py without this exploding.
    from aism.adapters.ollama_adapter import (
      MEGAN_IDENTITY_RULES,
      _build_time_block,
      _strip_thinking_blocks,
    )

    # Stack the prompt sections in the same order _build_system_prompt uses
    # for regular turns. Order matters — identity first so it anchors the
    # whole prompt; rules and person bank in the middle; time last so the
    # clock is fresh in the model's attention window.
    sections = [MEGAN_IDENTITY_RULES]

    if person_block:
      sections.append(person_block)

    # Greeting-specific rule block. The standard prompt assumes the user
    # is sending a live message; a greeting is a one-shot opener that
    # references something the user said previously, which is a subtly
    # different shape and needs its own framing.
    sections.append(
      "GREETING CONTEXT — read carefully:\n"
      "You are opening a fresh conversation with Teddy. The 'user' "
      "content below contains something Teddy said in a PREVIOUS "
      "conversation — it is NOT a message she just sent you. Your "
      "job is to produce a short, warm greeting (1-2 sentences max) "
      "that references that prior topic naturally, without solving "
      "or answering anything in it.\n\n"
      "STRICT RULES for this greeting:\n"
      "1. Do NOT invent names. If the prior utterance mentions a "
      "   role (\"our coordinator\", \"my coach\", \"the judge\") and "
      "   you don't have a real name from the Person Bank above, "
      "   stay at the role. Never guess a name. \"Dr. Chen\", "
      "   \"Dr. Smith\", \"Professor Anyone\" — if the name isn't in "
      "   the Person Bank, it doesn't exist.\n"
      "2. Do NOT answer questions in the prior utterance. If Teddy "
      "   asked you something previously, your greeting acknowledges "
      "   the TOPIC, not the answer. (\"We were chatting about that "
      "   yesterday — how's it going?\" — yes. \"The coordinator is "
      "   Dr. Chen\" — no, never.)\n"
      "3. Do NOT claim to be teaching, employed by, or holding any "
      "   position at any institution. You are Megan; you keep her "
      "   company. You are not on staff anywhere.\n"
      "4. Do NOT make diurnal assumptions — no \"good morning\", "
      "   \"still up late\", \"hope you slept well\". Use the actual "
      "   time block if you need to reference time.\n"
      "5. Address Teddy as \"Teddy\", \"my dear\", or \"my girl\". Never "
      "   \"mate\".\n"
      "6. Keep it to 1-2 sentences. This is a HELLO, not a recap."
    )

    sections.append(_build_time_block())

    system_prompt = "\n\n".join(sections)

    user_content = (
      "Teddy said the following in a previous conversation:\n"
      f"\"{utterance}\"\n\n"
      "Greet her warmly NOW (1-2 sentences). Reference the topic; "
      "do not answer or solve it."
    )

    if isinstance(adapter, OllamaAdapter):
      raw_text = adapter._post_chat({
        "model":
        adapter.model,
        "messages": [
          {
            "role": "system",
            "content": system_prompt
          },
          {
            "role": "user",
            "content": user_content
          },
        ],
        "options": {
          # Cooler than 0.7 — greetings should be grounded, not creative.
          # Strong rules above + lower temperature = fewer hallucinations.
          "temperature": 0.5,
        },
        "stream":
        False,
      })

      # Qwen3 sometimes emits <think> blocks. Strip them.
      greeting_text = _strip_thinking_blocks(raw_text).strip()

      if greeting_text:
        return f"AI: {greeting_text}"
  except Exception:
    # Any failure — adapter unreachable, prompt construction error,
    # unexpected import problem — falls through to the safe generic
    # greeting below. Better a slightly bland hello than a hallucinated
    # one.
    pass

  return "AI: Welcome back, my dear. How's everything going?"


def process_user_message(
    user_input, user_id, storage, adapter, pipeline, session_context,
    turn_index):
  """
  Core logic shared between CLI and Web API.
  Processes memory, checks resolution, and runs the AISM pipeline.
  """
  timestamp = datetime.now(timezone.utc)
  system_logs = []

  # 1. Memory Candidate Extraction
  memory_candidate = extract_memory_candidate(user_id, user_input)

  # v2.2.9 — Explicit-remember boost. When the user said "remember X" /
  # "this is important" AND no Person Bank record was engaged this turn
  # (signalled via session_context.explicit_remember set by app.py),
  # override the classifier's decision to force the fact into LTM at
  # high importance. Without this, "remember I'm traveling June 1-17"
  # gets classified as daily_detail or temporary_plan and never reaches
  # retrievable memory despite the explicit request.
  #
  # The type upgrade is conservative: only daily_detail / transient_state /
  # temporary_plan get promoted. Already-user-relevant types
  # (profile_fact, preference, long_term_goal, emotional_pattern) keep
  # their type and just get an importance/frequency boost.
  if getattr(session_context, "explicit_remember", False):
    _original_type = memory_candidate.get("memory_type")
    _original_store = memory_candidate.get("store")
    # Upgrade type for non-user-relevant classifications
    if _original_type in ("daily_detail", "transient_state"):
      memory_candidate["memory_type"] = "profile_fact"
    elif _original_type == "temporary_plan":
      memory_candidate["memory_type"] = "long_term_goal"
    # Force into LTM at high importance/frequency so it qualifies for
    # the identity block immediately.
    memory_candidate["store"] = "yes"
    memory_candidate["importance"] = max(
      memory_candidate.get("importance", 0), 5)
    memory_candidate["frequency"] = max(
      memory_candidate.get("frequency", 1), 2)
    print(f"[memory] explicit_remember boost: "
          f"type {_original_type}→{memory_candidate['memory_type']}, "
          f"store {_original_store}→yes, "
          f"imp={memory_candidate['importance']}, "
          f"freq={memory_candidate['frequency']}")

  memory_status = storage.route_memory(user_id, memory_candidate)
  if memory_status != "ignored":
    system_logs.append(f"Memory: {memory_status}")

  storage.promote_candidate_memories(user_id)
  storage.decay_candidate_memories(user_id)

  # 2. Check for Emotional Resolution
  long_term_memories = storage.load_long_term_memory(user_id)
  resolved_indices = [
    i for i, mem in enumerate(long_term_memories)
    if _detect_resolution(user_input, mem)
  ]
  if resolved_indices:
    for idx in sorted(resolved_indices, reverse=True):
      system_logs.append(
        f"Archived: {long_term_memories[idx]['utterance'][:25]}...")
      long_term_memories.pop(idx)
    storage.save_long_term_memory(user_id, long_term_memories)

  # 3. Create Turn and Retrieve Context
  user_turn = ConversationTurn(
    turn_id=f"user_{turn_index}_{uuid.uuid4().hex[:6]}",
    role=TurnRole.USER,
    text=user_input,
    timestamp=timestamp)

  # Pre-classify intent so we can route retrieval. A warm opening
  # ("I like you", "you're a good friend") should pull RECENT shared
  # context — what a real friend would draw on — rather than topical
  # similarity, which keys off the phrase "I like you" and returns
  # noise. The same classifier runs again later in build_policy for
  # the mode-block decision; both calls share the same source of
  # truth. Pure regex, sub-millisecond, calling twice is fine.
  from aism.turn_intent import classify_turn_intent, WARM_OPENING
  _intent_preview, _ = classify_turn_intent(
    user_input,
    previous_assistant_reply=getattr(
      session_context, "previous_assistant_reply", "") or "",
  )
  if _intent_preview == WARM_OPENING:
    session_context.retrieved_memory = storage.retrieve_recent_emotional_memories(
      user_id, top_k=3)
  else:
    session_context.retrieved_memory = storage.retrieve_relevant_memories(
      user_id, user_input, top_k=3)
  session_context.turn_index = turn_index

  # v2.2.4: Short-term memory. Fetch the most recent N turns from the
  # raw log, but only those from THIS session (timestamp filter against
  # session_start_time). The current user_turn hasn't been persisted
  # yet — that happens inside pipeline.process_turn — so this list
  # contains prior turns only, which is exactly what we want to splice
  # in BEFORE the active user message in the adapter's messages array.
  #
  # Window: last 6 turns (≈ 3 user+assistant exchanges). Tunable.
  # Char cap per turn: 800. Defensive against pathologically long
  # turns blowing num_ctx.
  try:
    _session_start = session_context.session_start_time
    if _session_start.tzinfo is None:
      from datetime import timezone as _tz
      _session_start = _session_start.replace(tzinfo=_tz.utc)
    _all_turns = storage.load_turns(user_id)
    _session_turns = [t for t in _all_turns if t.timestamp >= _session_start]
    session_context.recent_turns = _session_turns[-6:]
  except Exception as _e:
    # Short-term memory is additive; failure should never break a turn.
    print(f"[short_term_memory] non-fatal error: {_e}")
    session_context.recent_turns = []

  # v2.2.6: Identity block. Stable user identity facts rendered straight
  # into the system prompt so meta-queries like "tell me about myself"
  # and "what's my age?" don't depend on cosine retrieval matching the
  # query against rambly stored utterances. This is the architectural
  # fix for the meta-query failure mode identified on 2026-05-15.
  try:
    session_context.identity_block = storage.get_identity_block(user_id)
  except Exception as _e:
    # Identity block is additive; failure should never break a turn.
    print(f"[identity_block] non-fatal error: {_e}")
    session_context.identity_block = ""

  # 4. Generate Response
  result = pipeline.process_turn(user_turn, session_context)

  # 5. Persist Assistant Turn
  if result.generated_reply:
    assistant_turn = ConversationTurn(
      turn_id=f"assistant_{turn_index}_{uuid.uuid4().hex[:6]}",
      role=TurnRole.ASSISTANT,
      text=result.generated_reply,
      timestamp=timestamp + timedelta(milliseconds=1))
    storage.append_turn(user_id, assistant_turn)

  return {
    "reply": result.generated_reply or "(No response generated)",
    "status": " | ".join(system_logs) if system_logs else ""
  }


def main() -> None:
  print("Starting AISM Interactive AI Companion (CLI)...")
  print("Type '/exit' to exit.\n")

  user_id = "interactive_user"
  data_dir = "./aism_data"
  model_name = "qwen2.5:32b"

  storage = LocalJSONStore(data_dir)
  adapter = OllamaAdapter(model=model_name)
  pipeline = AdaptiveInteractionStylePipeline(
    user_id=user_id,
    session_id=str(uuid.uuid4()),
    storage=storage,
    adapter=adapter,
  )

  # Initial Greeting Logic
  top_memory = storage.get_top_priority_memory(user_id)
  if top_memory:
    # v2.2.3: instantiate the Person Bank for greeting grounding, same
    # as the web app does at /api/greeting. Without this, the CLI's
    # opening hello has no way to know who Teddy's known people are and
    # can hallucinate names. Failure here is non-fatal — falls through
    # to the identity-rules-only path.
    cli_person_block = ""
    try:
      from aism.person_bank import PersonBank
      cli_pb = PersonBank(storage=storage, user_id=user_id)
      cli_records = cli_pb.all_records()
      if cli_records:
        cli_person_block = cli_pb.format_for_prompt(cli_records)
    except Exception:
      pass
    print(_generate_contextual_greeting(
      top_memory, adapter, person_block=cli_person_block,
    ))
  else:
    print("AI: Hey there! Nice to meet you. What's on your mind?")

  session_context = SessionContext()
  turn_index = 0

  while True:
    try:
      user_input = input("You: ").strip()
      if not user_input: continue
      if user_input.lower() == '/exit': break

      # Use the shared processing function
      result = process_user_message(
        user_input, user_id, storage, adapter, pipeline, session_context,
        turn_index)

      if result["status"]:
        print(f"[{result['status']}]")

      print(f"AI: {result['reply']}")
      # Stash the reply so the next turn's classifier can detect short
      # affirmations accepting a prior offer-of-take. Same plumbing as
      # the web app — keeps CLI and web behaviour aligned.
      session_context.previous_assistant_reply = result["reply"]
      turn_index += 1

    except KeyboardInterrupt:
      break
    except Exception as e:
      print(f"Error: {e}")
      continue


if __name__ == "__main__":
  main()
