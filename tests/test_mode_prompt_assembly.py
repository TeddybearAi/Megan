"""End-to-end integration test: PolicyRetriever → ResponsePolicy →
OllamaAdapter.render_policy() produces a system prompt that contains the
correct mode-specific override block.

This is a structural test, not a content test. We don't run the LLM — we
just check what the LLM would see. Locks the wiring against future
reordering or accidental section deletion.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))

from aism.adapters.ollama_adapter import (
  OllamaAdapter,
  MODE_OVERRIDE_VENTING,
  MODE_OVERRIDE_AMBIGUOUS,
  MODE_OVERRIDE_SEEKING_HELP,
  MODE_OVERRIDE_HOSTILE,
  MODE_OVERRIDE_WARM_OPENING,
)
from aism.data_models import InteractionProfile, SessionOverlay, TraitState
from aism.session_context import SessionContext
from aism.stage5_retrieval import PolicyRetriever


def _setup():
  profile = InteractionProfile(user_id="u")
  # Pre-load the WRONG-SIGN humour value to simulate Teddy's actual
  # profile state. The venting override block MUST still dominate.
  profile.traits = {
    "humour": {
      "general": TraitState(
        value=0.0, confidence=0.92, evidence_count=2,
        volatility=0.14, last_updated=datetime.now(timezone.utc)),
    }
  }
  overlay = SessionOverlay(session_id="s")
  ctx = SessionContext()
  retr = PolicyRetriever()
  adapter = OllamaAdapter()
  return profile, overlay, ctx, retr, adapter


def _render(msg):
  profile, overlay, ctx, retr, adapter = _setup()
  policy = retr.build_policy(profile, overlay, ctx, msg)
  rendered = adapter.render_policy(policy)
  return policy, rendered["system_prompt"]


# ============================================================
# Venting prompt: override block present and dominant
# ============================================================

def test_venting_prompt_contains_venting_block():
  policy, sp = _render(
    "I just want to vent about something. I know what to do.")
  assert policy.mode == "venting", policy.mode
  assert MODE_OVERRIDE_VENTING in sp


def test_venting_block_is_LAST_section_of_prompt():
  """Prompt-engineering: the LAST section has the highest authority. The
  venting override must come after the adaptive preferences, not before."""
  _, sp = _render("I just need to vent, damn it.")
  venting_idx = sp.find("CURRENT TURN MODE: VENTING")
  adaptive_idx = sp.find("Adaptive reply preferences")
  assert venting_idx > adaptive_idx, (
    "Venting block must come AFTER the adaptive preferences "
    "so the override takes effect.")
  # And there should be nothing after the venting block (it's the tail).
  remaining = sp[venting_idx + len("CURRENT TURN MODE: VENTING"):]
  assert "Adaptive reply preferences" not in remaining


def test_venting_prompt_explicitly_forbids_numbered_lists():
  """Screenshot 5/6: Megan replied with a literal four-numbered-point list.
  The prompt must explicitly forbid that in venting mode."""
  _, sp = _render("I need to vent, damn it.")
  assert "Numbered lists" in sp or "numbered lists" in sp
  assert "bullet points" in sp.lower()


def test_venting_prompt_explicitly_forbids_have_you_considered():
  _, sp = _render("damn it, this is bullshit, you know what i mean")
  assert "Have you considered" in sp


def test_venting_prompt_overrides_emotional_style_no_jokes():
  """Wrong-sign humour=0.0 still produces emotional_style='no jokes'. The
  venting block must explicitly say it overrides that."""
  policy, sp = _render("I just need to vent. I know what to do.")
  assert policy.emotional_style == "no jokes"   # confirm the wrong-sign is live
  # The venting block must contain explicit override language.
  assert "OVERRIDE" in sp or "override" in sp


def test_humour_threshold_055_renders_light_humour_welcome():
  """Threshold was lowered from 0.7 to 0.55 so organically-learned
  humour values in the mid range produce a prompt cue. A value of 0.6
  (mid-range "mostly wants humour") must now render 'light humour
  welcome' into the adaptive prefs. Locks the threshold against
  accidental future reversion."""
  profile, overlay, ctx, retr, adapter = _setup()
  # Override the humour value to 0.6 — above the new 0.55 floor, below
  # the previous 0.7 floor. This is exactly the band the change unlocks.
  profile.traits["humour"]["general"].value = 0.6
  profile.traits["humour"]["general"].confidence = 0.7
  policy = retr.build_policy(profile, overlay, ctx, "Hi there")
  assert policy.emotional_style is not None
  assert "light humour welcome" in policy.emotional_style, policy.emotional_style


def test_humour_threshold_050_renders_no_cue():
  """Just below the 0.55 threshold: no humour cue is rendered. This
  protects against accidentally lowering the threshold further."""
  profile, overlay, ctx, retr, adapter = _setup()
  profile.traits["humour"]["general"].value = 0.5
  profile.traits["humour"]["general"].confidence = 0.7
  policy = retr.build_policy(profile, overlay, ctx, "Hi there")
  # Neither side should appear: 0.5 is in the silent zone (above 0.3,
  # below 0.55).
  if policy.emotional_style is not None:
    assert "light humour welcome" not in policy.emotional_style
    assert "no jokes" not in policy.emotional_style


def test_humour_threshold_low_unchanged():
  """The low-humour threshold (0.3) is unchanged. Verify 'no jokes'
  still fires at and below it."""
  profile, overlay, ctx, retr, adapter = _setup()
  profile.traits["humour"]["general"].value = 0.3
  profile.traits["humour"]["general"].confidence = 0.7
  policy = retr.build_policy(profile, overlay, ctx, "Hi there")
  assert policy.emotional_style is not None
  assert "no jokes" in policy.emotional_style


# ============================================================
# Seeking-help prompt: SEEKING_HELP block present, no venting block
# ============================================================

def test_seeking_help_prompt_contains_help_block():
  policy, sp = _render("How do I fix this regex bug?")
  assert policy.mode == "seeking_help", policy.mode
  assert MODE_OVERRIDE_SEEKING_HELP in sp
  assert MODE_OVERRIDE_VENTING not in sp


# ============================================================
# Ambiguous: low confidence → defaults to listening
# ============================================================

def test_ambiguous_prompt_contains_listen_first_block():
  policy, sp = _render("hmm i don't know")
  # Either AMBIGUOUS (preferred) or CASUAL is acceptable here.
  if policy.mode == "ambiguous":
    assert MODE_OVERRIDE_AMBIGUOUS in sp


# ============================================================
# Low confidence: classifier returned low — adapter still defaults to listen
# ============================================================

def test_low_confidence_ambiguous_still_listens():
  """Even when classifier confidence is 0 (empty message), the prompt
  must NOT default to advisor mode. AMBIGUOUS = listen-first."""
  _, sp = _render("")
  # Empty message → mode_confidence = 0.0. The adapter has a branch that
  # adds the AMBIGUOUS block specifically when mode is "ambiguous" even
  # below the confidence floor, so listen-first behaviour is preserved.
  assert "CURRENT TURN MODE: VENTING" not in sp
  assert "CURRENT TURN MODE: SEEKING HELP" not in sp


# ============================================================
# Casual: no override block at all (persona defaults handle it)
# ============================================================

def test_casual_prompt_has_no_override_block():
  policy, sp = _render("hey")
  assert policy.mode == "casual", policy.mode
  assert "CURRENT TURN MODE" not in sp


# ============================================================
# Structure: identity rules and persona always come first
# ============================================================

def test_identity_rules_always_first_section():
  """Hard identity rules must never be reordered behind a mode block."""
  for msg in [
    "hey",
    "what should i do?",
    "i'm so pissed, damn it",
    "",
  ]:
    _, sp = _render(msg)
    # The "Megan identity and behaviour rules:" header must be the first
    # non-whitespace content of the prompt.
    assert sp.lstrip().startswith("Megan identity and behaviour rules"), (
      f"Identity rules not first for message {msg!r}")


def test_persona_always_present():
  for msg in ["hey", "how do i fix x?", "i'm so frustrated", ""]:
    _, sp = _render(msg)
    assert "witty, smart friend" in sp, (
      f"Persona missing for message {msg!r}")


# ============================================================
# Hostile: pushback block present and dominant
# ============================================================

def test_hostile_prompt_contains_hostile_block():
  policy, sp = _render("fuck you, you piece of shit")
  assert policy.mode == "hostile", policy.mode
  assert MODE_OVERRIDE_HOSTILE in sp


def test_hostile_block_grants_permission_to_push_back():
  """The block must explicitly authorise pushback. If a future edit
  softens it into 'be understanding', this test catches it."""
  _, sp = _render("you're so fucking useless")
  assert "permission" in sp.lower() or "push back" in sp.lower()
  # Must NOT include unconditional apology phrasing as a recommendation.
  # (The block actually FORBIDS unconditional apologies; checking that the
  # block's forbidden-list keyword is present.)
  assert "Forbidden in hostile mode" in sp


def test_hostile_block_proportionality_rule():
  """The pushback must be proportionate. The block must mention this so
  Megan doesn't escalate from first offense."""
  _, sp = _render("you're a moron")
  assert "proportionate" in sp.lower() or "escalat" in sp.lower()


def test_hostile_with_fair_criticism_addresses_both():
  """When hostility wraps a fair criticism, the block must instruct
  addressing both — not picking one."""
  _, sp = _render("fuck you, you missed the point completely")
  # The block has explicit guidance: "address BOTH".
  assert "BOTH" in sp or "both" in sp


def test_playful_curse_does_not_render_hostile_block():
  """Affectionate dampener must prevent the hostile block from appearing."""
  policy, sp = _render("haha fuck you, that was hilarious")
  assert policy.mode != "hostile"
  assert MODE_OVERRIDE_HOSTILE not in sp


def test_third_party_curse_does_not_render_hostile_block():
  """'fuck this gym' is venting, not hostile. Block must not appear."""
  policy, sp = _render("fuck this gym, they keep overcharging me!")
  assert policy.mode != "hostile"
  assert MODE_OVERRIDE_HOSTILE not in sp


# ============================================================
# WARM_OPENING block: present and forbids corporate-shaped responses
# ============================================================


def test_warm_opening_prompt_contains_warm_block():
  policy, sp = _render("I like you")
  assert policy.mode == "warm_opening", policy.mode
  assert MODE_OVERRIDE_WARM_OPENING in sp


def test_warm_opening_block_forbids_corporate_gratitude():
  """The block must explicitly forbid 'Thanks for the kind words' and
  similar customer-service shapes. If a future edit weakens this, the
  whole pass becomes moot."""
  _, sp = _render("you're a good friend")
  assert "Thanks for the kind words" in sp
  assert "customer-service" in sp.lower() or "corporate" in sp.lower()


def test_warm_opening_block_forbids_romantic_language():
  """Companion, not partner. 'I love you' back must be explicitly forbidden."""
  _, sp = _render("you're amazing")
  assert "I love you" in sp  # appears as the forbidden example
  assert "Romantic" in sp or "romantic" in sp


def test_warm_opening_block_directs_to_recent_memory():
  """The block must tell the LLM to use the recency-weighted memory
  surfaced for this turn. Without this directive the LLM falls back to
  generic warmth."""
  _, sp = _render("I appreciate you")
  assert "RECENCY-WEIGHTED" in sp or "recency-weighted" in sp.lower()


def test_warm_opening_block_is_last_section():
  """Mode override block must sit at the END of the prompt — highest
  authority slot. Mirrors the assembly contract for the other modes."""
  _, sp = _render("thanks for being here")
  assert sp.rstrip().endswith(MODE_OVERRIDE_WARM_OPENING)


def test_bare_thanks_does_not_render_warm_block():
  """Bare 'thanks!' stays CASUAL and falls through to persona defaults.
  Don't over-trigger the warm block."""
  policy, sp = _render("thanks!")
  assert policy.mode != "warm_opening", policy.mode
  assert MODE_OVERRIDE_WARM_OPENING not in sp


def test_i_like_coffee_does_not_render_warm_block():
  """Liking a non-person object is not relational warmth."""
  policy, sp = _render("I like coffee")
  assert policy.mode != "warm_opening", policy.mode
  assert MODE_OVERRIDE_WARM_OPENING not in sp


if __name__ == "__main__":
  ns = dict(globals())
  failures = []
  passed = 0
  for name, fn in ns.items():
    if name.startswith("test_") and callable(fn):
      try:
        fn()
        passed += 1
        print(f"PASS  {name}")
      except AssertionError as e:
        failures.append((name, e))
        print(f"FAIL  {name}: {e}")
      except Exception as e:
        failures.append((name, e))
        print(f"ERROR {name}: {type(e).__name__}: {e}")
  print(f"\n{passed} passed, {len(failures)} failed")
  sys.exit(0 if not failures else 1)
