"""Regression tests for the precision-tools quoted-fallback false-positive
(fixed 2026-05-12).

The bug: the strict letter-count parser had a `quoted` fallback path that
extracted "quoted text" using a naive single-quote regex. Apostrophes in
contractions (it's, don't, can't, i'm) were paired up as fake quote-pairs,
so any long monologue with contractions plus the word "character" or
"letter" anywhere in it would trigger a deterministic letter-count answer
where none was wanted.

Surfaced by an actual user monologue about a video game on 2026-05-12.

The fix: require the same "how many"/"count" counting-intent gate that
every other path in the function already requires. Legitimate requests
still work; false positives stop.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))

from aism.utils.precision_tools import answer_if_precision_task, answer_letter_count


# ============================================================
# Negative cases — must NOT fire (bug repro)
# ============================================================


# Exact monologue from the 2026-05-12 incident report. This input previously
# triggered: 'The letter "u" appears 3 times in "t play that to the end..."'
TEDDY_MONOLOGUE = (
  "hmm yeah it's about the story mainly right so the person you actually "
  "don't play in that game has a pretty tragic background and basically "
  "you have to um you have to struggle a lot to get what you really want "
  "i think the you know the cool idea is that the person in the game that "
  "you play right he or she actually wanted to be someone in the society "
  "that is kind of very cruel and hopeless and there is no um love or "
  "whatever okay so the society is pretty dark okay it's very selfish "
  "society and um as as a young as a young person in that game so all "
  "you need to do is just to just fight your way to gain relationships "
  "and you know of course financial security and success and in the end "
  "that everybody will know your name i think that's the basic ideology "
  "of that game um yeah i i love the story um but the thing is um i "
  "think the story is a little bit too sad so i didn't even finish the "
  "game because i don't really want to see the characters that i play "
  "will end up you know dead or whatever yeah yeah so i'm not kidding "
  "so the character um there is a end and that your character will will "
  "end well will die in the end so yeah i i have a i i can't i can't "
  "play that to the end so so my my character is is um still in the "
  "middle of the game okay before the final finale scenario you know "
  "so uh he's he's there forever yeah")


def test_teddy_video_game_monologue_does_not_false_fire():
  """The exact transcript that triggered the bug must produce 'not handled'."""
  result = answer_if_precision_task(TEDDY_MONOLOGUE, previous_user_text=None)
  assert not result.handled, (
    f"False positive: precision guard fired on a benign monologue. "
    f"Got: kind={result.kind}, answer={result.answer[:80]!r}")


def test_contraction_heavy_with_character_word_does_not_fire():
  """A simpler reproduction of the same shape: contractions + 'character X'."""
  text = (
    "i don't think the character u is interesting and i can't really say "
    "much about it, it's just not my type")
  result = answer_if_precision_task(text)
  assert not result.handled


def test_contraction_heavy_with_letter_word_does_not_fire():
  """Variant of the above with 'letter X' instead of 'character X'."""
  text = (
    "it's funny how i'm not sure about the letter a, don't ask me why, "
    "i can't explain it")
  result = answer_if_precision_task(text)
  assert not result.handled


def test_correction_on_non_precision_turn_does_not_resurrect_false_positive():
  """After a benign monologue, 'try again' must NOT cause the previous
  monologue to be re-parsed as a precision task. This protects against
  the second-screenshot failure where 'You are right to ask me to check
  carefully...' appeared on a follow-up turn."""
  result = answer_if_precision_task(
    "can you try again because you were jibber-jabbering",
    previous_user_text=TEDDY_MONOLOGUE,
  )
  assert not result.handled


# ============================================================
# Positive cases — must STILL fire (regression coverage for the fix)
# ============================================================


def test_legitimate_quoted_count_still_works():
  """The quoted-fallback path exists for a reason. Real requests with
  quoted target words plus counting intent must still resolve."""
  result = answer_if_precision_task(
    "how many letter u in 'strawberry'"
  )
  assert result.handled
  assert "letter_count" in result.kind


def test_legitimate_letter_count_no_quotes_still_works():
  """The main strict path is unchanged."""
  result = answer_if_precision_task("how many r in strawberry")
  assert result.handled
  assert "letter_count" in result.kind


def test_count_keyword_still_works():
  """The 'count the X in Y' phrasing must still work."""
  result = answer_if_precision_task("count the r in strawberry")
  assert result.handled
