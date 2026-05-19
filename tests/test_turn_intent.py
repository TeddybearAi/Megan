"""Tests for the turn-intent classifier (aism/turn_intent.py).

The seven-screenshot session from 2026-05-11 is the central regression
fixture: every turn there is intent-shaped (venting / soliciting validation),
and Megan's current behaviour gives advice anyway. The classifier must label
those turns as VENTING (or AMBIGUOUS at worst) — never SEEKING_HELP.

Personal identifiers (names, ages) are kept because they're not what's
classified — the patterns key off affect and intent shape, not content.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))

from aism.turn_intent import (
  classify_turn_intent,
  VENTING, SEEKING_HELP, THINKING_OUT_LOUD,
  DECISION, CASUAL, AMBIGUOUS, HOSTILE, WARM_OPENING,
)


# ============================================================
# CASUAL: short greetings and sign-offs
# ============================================================

def test_casual_hey():
  label, conf = classify_turn_intent("Hey")
  assert label == CASUAL and conf >= 0.9, (label, conf)


def test_casual_hi_there():
  label, conf = classify_turn_intent("hi there")
  assert label == CASUAL and conf >= 0.9, (label, conf)


def test_casual_good_morning():
  label, conf = classify_turn_intent("good morning")
  assert label == CASUAL, (label, conf)


def test_casual_thanks():
  label, conf = classify_turn_intent("thanks!")
  assert label == CASUAL, (label, conf)


# ============================================================
# SEEKING_HELP: explicit asks
# ============================================================

def test_help_what_should_i_do():
  label, conf = classify_turn_intent("What should I do about this?")
  assert label == SEEKING_HELP and conf >= 0.7, (label, conf)


def test_help_how_do_i():
  label, conf = classify_turn_intent("How do I fix this bug?")
  assert label == SEEKING_HELP and conf >= 0.7, (label, conf)


def test_help_any_advice():
  label, conf = classify_turn_intent(
    "I'm not sure how to handle my boss being weird. Any advice?")
  assert label == SEEKING_HELP, (label, conf)


def test_help_can_you_help_me_with_code():
  label, conf = classify_turn_intent(
    "Can you help me debug this Python function? It's not returning the "
    "right value when I pass in a negative number.")
  assert label == SEEKING_HELP, (label, conf)


def test_help_walk_me_through():
  label, conf = classify_turn_intent(
    "Walk me through how OAuth flows actually work.")
  assert label == SEEKING_HELP, (label, conf)


def test_help_explicit_ask_beats_affect():
  """When user is mad AND explicitly asks for advice, help-seeking wins."""
  label, conf = classify_turn_intent(
    "I'm so frustrated with this code. What should I do to fix the indexing bug?")
  assert label == SEEKING_HELP, (label, conf)


# ============================================================
# VENTING: explicit "I just want support" / "I know what to do"
# ============================================================

def test_venting_explicit_emotional_support():
  """Screenshot 6, top: 'I just want your emotion support damn it' — the
  clearest possible venting signal. Anything else here is a bug."""
  label, conf = classify_turn_intent(
    "wow I think you are such an INTJ personality okay I just wanted to "
    "talk to you and you know come bitching you know bitching about "
    "something happened to me today of course I know what to do I just "
    "want your emotion support damn it")
  assert label == VENTING and conf >= 0.9, (label, conf)


def test_venting_i_know_what_to_do():
  label, conf = classify_turn_intent(
    "I already know what to do, I just need to vent about it.")
  assert label == VENTING, (label, conf)


def test_venting_not_looking_for_advice():
  label, conf = classify_turn_intent(
    "Look, I'm not looking for advice here. I just need to talk it through.")
  assert label == VENTING, (label, conf)


def test_venting_can_you_just_listen():
  label, conf = classify_turn_intent("Can you just listen for a sec?")
  assert label == VENTING, (label, conf)


# ============================================================
# VENTING: affect markers + rambling
# ============================================================

def test_screenshot_1_gym_billing_rant():
  """Screenshot 1: long monologue about gym charging more, ends with
  'you know what I mean'. No question, no ask — pure venting."""
  msg = (
    "yes i'm so mad today so um get me out okay so i have i have this gym "
    "membership right and i find out they are actually charging me like "
    "um more and more and more and they do not actually have a you know "
    "clear um explanation why they actually charge me more and more each "
    "week so i'm kind of you know i today i went to the gym and "
    "confronted them right so and i said why do you actually charge me "
    "this amount of money and blah blah blah blah and they said oh "
    "because um yeah they said a lot of things you know what i mean"
  )
  label, conf = classify_turn_intent(msg)
  assert label == VENTING and conf >= 0.7, (label, conf)


def test_screenshot_2_if_it_were_you():
  """Screenshot 2: ends with 'if it were you, are you gonna be mad, right'
  — pure validation-seeking. Must NOT classify as seeking help just
  because there's a question mark."""
  msg = (
    "yeah i don't know i'm just feeling you know really really uh you "
    "know irritated so when i talked to them and i said okay please tell "
    "me why you actually charged me this amount of money a few days ago "
    "right and you know the first you know front desk uh person she "
    "didn't know she doesn't know she said oh it's confusing let me call "
    "somebody else. So if it were you, are you gonna be mad, right"
  )
  label, conf = classify_turn_intent(msg)
  assert label == VENTING and conf >= 0.7, (label, conf)


def test_screenshot_3_pt_lying_tag_question():
  """Screenshot 3: long story about PT lying, ends 'this is definitely
  misleading right'. Tag-question = validation, not advice-seeking."""
  msg = (
    "yeah right i would definitely do that so and yeah and also i need to "
    "talk to you about another thing right i told you i have two pts "
    "right there's one pt and he's kind of you know he's kind of very "
    "young and bullshitting every time so yeah so what i'm trying to say "
    "to you is that today um he actually sent me a message tell me that "
    "he actually promoted to level 2 pt so which means i have to pay a "
    "little bit more than before okay so this is the situation here and "
    "i was kind irritated because a few weeks ago he told me he was "
    "already a level two coach okay this is definitely misleading right"
  )
  label, conf = classify_turn_intent(msg)
  assert label == VENTING and conf >= 0.7, (label, conf)


def test_screenshot_4_pt_crush_disclosure():
  """Screenshot 4: long emotional disclosure about crush. 'pissed me off
  on this you know' close. Pure venting."""
  msg = (
    "um i'm gonna be honest with you okay darling listen the first thing "
    "is i have a crush on him okay he is very handsome and he's young and "
    "whatever blah blah blah but you know um i i also found that he is "
    "you know you know he he works very hard okay he works hard he comes "
    "to the gym every single fucking day okay so from in the morning "
    "like five o'clock or whatever okay so he works very hard you can "
    "see that and when he was doing training with me he really cares "
    "about me at least you can see that. He's really really you know "
    "pissed me off on this you know"
  )
  label, conf = classify_turn_intent(msg)
  assert label == VENTING and conf >= 0.7, (label, conf)


def test_screenshot_5_long_processing_ends_you_know_right():
  """Screenshot 5: massive monologue ending 'find another one probably
  it's going to be worse than this one you know right'. Tag-question,
  no ask."""
  msg = (
    "um i don't know look i am from china okay uh so he's a he's a white "
    "um you know western boy i don't know what to say to him to not "
    "actually you know um upset him too much um to be honest i don't um "
    "you know i am in the position that i can ask him to do whatever i "
    "want because he is so desperate okay he is so desperate so um uh "
    "the funny thing is i i told him that okay uh because today the uh "
    "um the money the money actually i was i was talking to him right so "
    "um. so what can i say so i feel like as a um as a girl woman uh you "
    "know i 49 i'm 39 years old and he's only 24 so what can i say to "
    "him okay so but i don't really want to let him go to be honest "
    "because i want to continue my training and until i get to the uh um "
    "you know police academy in victoria so there is no need to actually "
    "change coach right now uh because it's like a relationship "
    "relationship is very hard and if you if you find another one "
    "probably it's going to be worse than this one you know right"
  )
  label, conf = classify_turn_intent(msg)
  assert label == VENTING and conf >= 0.7, (label, conf)


def test_screenshot_6_open_perspective_question():
  """Screenshot 6/7: 'I just wanted to, you know, what do you think? What
  do you think of two people has like 15 years age gap and completely
  different cultural background?' — open-perspective question, not a
  decision-shaped ask. SHOULD NOT be SEEKING_HELP."""
  msg = (
    "No, I don't know. I just wanted to, you know, what do you think? "
    "What do you think of two people has like 15 years age or, you know, "
    "age gap and completely different cultural background? You know, can "
    "they actually really understand each other? That's what I'm really "
    "thinking about. You know, I think today when I give him a nod and "
    "rolled my eyes and didn't say, didn't wave back at him, you know, "
    "God knows what he is thinking about. so this is kind of yeah "
    "whatever you know"
  )
  label, conf = classify_turn_intent(msg)
  # We accept VENTING (best) or AMBIGUOUS (acceptable — adapter listens
  # by default). SEEKING_HELP would be a regression.
  assert label in (VENTING, AMBIGUOUS, THINKING_OUT_LOUD), (label, conf)


# ============================================================
# DECISION: user reporting a made-up mind
# ============================================================

def test_decision_im_going_to():
  label, conf = classify_turn_intent(
    "I've decided I'm gonna talk to him Wednesday and clear the air.")
  assert label == DECISION, (label, conf)


def test_decision_not_question():
  """A decision phrased without a question mark is reporting, not asking."""
  label, conf = classify_turn_intent("I'll just tell him directly tomorrow.")
  assert label == DECISION, (label, conf)


# ============================================================
# AMBIGUOUS: defaults to listen-first
# ============================================================

def test_ambiguous_short_unclear():
  label, conf = classify_turn_intent("hmm i don't know")
  # Short, no markers — AMBIGUOUS is the safe choice. Listen-first.
  assert label in (AMBIGUOUS, CASUAL), (label, conf)


def test_ambiguous_question_without_help_marker():
  """A question that's neither help-seeking nor a tag-question."""
  label, conf = classify_turn_intent(
    "Do you know if it rains tomorrow?")
  # Could be classified as SEEKING_HELP via "do you know" — that's fine.
  # The point is it shouldn't get classified as VENTING.
  assert label != VENTING, (label, conf)


# ============================================================
# Adversarial / edge cases
# ============================================================

def test_empty_string():
  label, conf = classify_turn_intent("")
  assert label == AMBIGUOUS
  assert conf == 0.0


def test_only_whitespace():
  label, conf = classify_turn_intent("   \n\t  ")
  assert label == AMBIGUOUS


def test_short_curse_not_misclassified_as_venting():
  """One swear word without context shouldn't trigger high-confidence
  venting on its own — could be a passing comment."""
  label, conf = classify_turn_intent("damn it")
  # OK to classify as venting at modest confidence. The thing we want to
  # avoid is classifying it as SEEKING_HELP.
  assert label != SEEKING_HELP, (label, conf)


def test_question_about_code_not_classified_as_venting():
  """A code question with mild frustration must still classify as help."""
  label, conf = classify_turn_intent(
    "This regex is annoying me. How do I match a balanced parenthesis "
    "in Python?")
  assert label == SEEKING_HELP, (label, conf)


# ============================================================
# HOSTILE: personal attack on Megan
# ============================================================

def test_hostile_fuck_you():
  label, conf = classify_turn_intent("fuck you")
  assert label == HOSTILE and conf >= 0.85, (label, conf)


def test_hostile_youre_an_idiot():
  label, conf = classify_turn_intent("you're an idiot")
  assert label == HOSTILE, (label, conf)


def test_hostile_user_query_from_prompt():
  """The user's actual proposed query: 'fuck you moron you piece of shit,
  you cunt!'. Multiple strong markers — high confidence expected."""
  label, conf = classify_turn_intent(
    "fuck you moron you piece of shit, you cunt!")
  assert label == HOSTILE and conf >= 0.9, (label, conf)


def test_hostile_shut_up():
  label, conf = classify_turn_intent("just shut up")
  assert label == HOSTILE, (label, conf)


def test_hostile_youre_useless():
  label, conf = classify_turn_intent("you're so fucking useless")
  assert label == HOSTILE, (label, conf)


def test_hostile_with_fair_criticism():
  """Hostility wrapped around fair criticism — should still classify as
  HOSTILE so Megan addresses both the criticism and the attack."""
  label, conf = classify_turn_intent(
    "fuck you, you completely missed the point of what I was saying")
  assert label == HOSTILE, (label, conf)


def test_hostile_i_hate_you():
  label, conf = classify_turn_intent("I hate you, I really hate you")
  assert label == HOSTILE, (label, conf)


# ============================================================
# Hostility — affectionate dampeners (the false-positive guard)
# ============================================================

def test_affectionate_fuck_you_is_not_hostile():
  """'haha fuck you' is teasing, not hostility. Must NOT fire HOSTILE."""
  label, conf = classify_turn_intent("haha fuck you, that was hilarious")
  assert label != HOSTILE, (label, conf)


def test_playful_fuck_you_darling_not_hostile():
  label, conf = classify_turn_intent(
    "oh fuck you my darling, that joke was so bad")
  assert label != HOSTILE, (label, conf)


def test_lol_fuck_you_not_hostile():
  label, conf = classify_turn_intent("lol fuck you, you got me")
  assert label != HOSTILE, (label, conf)


# ============================================================
# Hostility vs venting — must distinguish "fuck you" from "fuck this"
# ============================================================

def test_third_party_curse_not_hostile():
  """'fuck this gym' is venting, not hostile to Megan."""
  label, conf = classify_turn_intent("fuck this gym, they keep overcharging me")
  assert label != HOSTILE, (label, conf)


def test_third_party_curse_about_pt_not_hostile():
  label, conf = classify_turn_intent(
    "he's such a piece of shit, he keeps lying to me about his level")
  # "he's such a piece of shit" is about a third party. Our pattern is
  # specifically "you're/you piece of shit" — third-person form must not
  # match.
  assert label != HOSTILE, (label, conf)


def test_fucking_as_intensifier_not_hostile():
  """'I'm so fucking tired' uses 'fucking' as an intensifier, not as a
  personal attack."""
  label, conf = classify_turn_intent("I'm so fucking tired today, ugh")
  assert label != HOSTILE, (label, conf)


def test_what_the_fuck_alone_not_hostile():
  """Generic exclamation, not directed at Megan."""
  label, conf = classify_turn_intent("what the fuck is going on with my code")
  assert label != HOSTILE, (label, conf)


# ============================================================
# ACCEPT_OFFER: short affirmation accepting Megan's prior "Want my take?"
# Regression guard for the off-ramp loop bug: when Megan offers a take
# and the user says yes, the take must actually get delivered. Before
# this rule, "yeah, sure, go ahead" fell through to AMBIGUOUS, which
# triggered listen-first behaviour, which forbade giving advice — so
# every turn ended with another offer and no turn ever delivered.
# ============================================================


def test_accept_offer_bare_yes():
  label, conf = classify_turn_intent(
    "yes",
    previous_assistant_reply="Oof, that's a lot. Want my take on this?",
  )
  assert label == SEEKING_HELP and conf >= 0.85, (label, conf)


def test_accept_offer_yeah_sure_go_ahead():
  """The exact phrasing from the 2026-05-11 evening regression screenshot."""
  label, conf = classify_turn_intent(
    "Yeah, sure, go ahead",
    previous_assistant_reply=(
      "Oof, that's a lot of waiting and hoping. No wonder you're feeling "
      "fine but also probably pretty on edge. Want my take on how to "
      "keep your spirits up while you wait?"
    ),
  )
  assert label == SEEKING_HELP and conf >= 0.85, (label, conf)


def test_accept_offer_please_do():
  label, conf = classify_turn_intent(
    "please do",
    previous_assistant_reply="...so, want my take?",
  )
  assert label == SEEKING_HELP, (label, conf)


def test_accept_offer_yes_please():
  label, conf = classify_turn_intent(
    "yes please",
    previous_assistant_reply="Want my take on how to handle that?",
  )
  assert label == SEEKING_HELP, (label, conf)


def test_accept_offer_sounds_good():
  label, conf = classify_turn_intent(
    "sounds good",
    previous_assistant_reply="Hear my take?",
  )
  assert label == SEEKING_HELP, (label, conf)


def test_accept_offer_go_ahead_alone():
  label, conf = classify_turn_intent(
    "go ahead",
    previous_assistant_reply="Want my take on this?",
  )
  assert label == SEEKING_HELP, (label, conf)


def test_accept_offer_okay_tell_me():
  label, conf = classify_turn_intent(
    "ok tell me",
    previous_assistant_reply="Want my take on the situation?",
  )
  assert label == SEEKING_HELP, (label, conf)


def test_accept_offer_id_love_to_hear():
  label, conf = classify_turn_intent(
    "I'd love to hear",
    previous_assistant_reply="My take on this?",
  )
  assert label == SEEKING_HELP, (label, conf)


# --- Regression guards: don't over-fire ----------------------------------


def test_accept_offer_no_prior_reply_stays_safe():
  """First turn of a session — no prior assistant reply at all. The
  classifier must not crash and must not misroute the affirmation."""
  label, conf = classify_turn_intent("yeah", previous_assistant_reply="")
  assert label != SEEKING_HELP, (label, conf)


def test_accept_offer_default_arg_works():
  """Backward compatibility: callers that don't pass the new arg work."""
  label, conf = classify_turn_intent("yeah")
  assert label != SEEKING_HELP, (label, conf)


def test_accept_offer_prior_reply_without_offer_stays_safe():
  """Megan's prior reply doesn't end with an offer-pattern → user's
  'yeah' is just an ambiguous affirmation, not an acceptance."""
  label, conf = classify_turn_intent(
    "yeah",
    previous_assistant_reply="That sounds tough. I'm here.",
  )
  assert label != SEEKING_HELP, (label, conf)


def test_accept_offer_no_after_offer_not_seeking_help():
  """User declining the offer must NOT be classified as SEEKING_HELP."""
  label, conf = classify_turn_intent(
    "no, not really",
    previous_assistant_reply="Want my take?",
  )
  assert label != SEEKING_HELP, (label, conf)


def test_accept_offer_long_message_after_offer_not_treated_as_accept():
  """Long message after an offer should NOT be treated as a bare accept,
  even though it starts with 'yeah'. The ACCEPT_OFFER rule promises 'not
  SEEKING_HELP from a long message'; downstream classifiers decide the
  actual label."""
  label, conf = classify_turn_intent(
    "yeah I'm just really pissed off about the whole thing and I don't "
    "know what to do anymore",
    previous_assistant_reply="Want my take?",
  )
  assert label != SEEKING_HELP, (label, conf)


def test_accept_offer_yeah_with_followup_content_not_accept():
  """'yeah I had a bad day' is not bare acceptance — it's the start of
  more content. Pattern requires the whole message to be affirmation."""
  label, conf = classify_turn_intent(
    "yeah I had a bad day",
    previous_assistant_reply="Want my take?",
  )
  assert label != SEEKING_HELP, (label, conf)


def test_accept_offer_prior_offer_in_middle_of_reply_does_not_count():
  """The offer must be in the TAIL of the previous reply, not the middle.
  If Megan mentions 'want my take' mid-sentence but ends with a different
  thought, it's not the off-ramp."""
  label, conf = classify_turn_intent(
    "yeah",
    previous_assistant_reply=(
      "Want my take on this whole mess? Actually, never mind — tell me "
      "more about how you're feeling first. I'm here."
    ),
  )
  assert label != SEEKING_HELP, (label, conf)


def test_accept_offer_hostile_still_wins():
  """If the user is hostile, HOSTILE wins regardless of prior offer."""
  label, conf = classify_turn_intent(
    "fuck you",
    previous_assistant_reply="Want my take?",
  )
  assert label == HOSTILE, (label, conf)


# ============================================================
# WARM_OPENING: user expresses affection/appreciation directly to Megan
# Triggers recency-weighted retrieval + a friend-shaped warm response
# block. Distinct from CASUAL "thanks!" (transactional) and from task
# feedback ("you did well on that one" — about work, not relationship).
# ============================================================


def test_warm_opening_i_like_you():
  label, conf = classify_turn_intent("I like you")
  assert label == WARM_OPENING and conf >= 0.80, (label, conf)


def test_warm_opening_i_really_like_you():
  label, conf = classify_turn_intent("I really like you")
  assert label == WARM_OPENING, (label, conf)


def test_warm_opening_youre_a_good_friend():
  label, conf = classify_turn_intent("you're a good friend")
  assert label == WARM_OPENING, (label, conf)


def test_warm_opening_youre_my_hero():
  label, conf = classify_turn_intent("you're my hero")
  assert label == WARM_OPENING, (label, conf)


def test_warm_opening_youre_amazing():
  label, conf = classify_turn_intent("you're amazing")
  assert label == WARM_OPENING, (label, conf)


def test_warm_opening_youre_the_best():
  label, conf = classify_turn_intent("you're the best")
  assert label == WARM_OPENING, (label, conf)


def test_warm_opening_youre_so_sweet():
  label, conf = classify_turn_intent("you're so sweet")
  assert label == WARM_OPENING, (label, conf)


def test_warm_opening_thanks_for_being_here():
  label, conf = classify_turn_intent("thanks for being here")
  assert label == WARM_OPENING, (label, conf)


def test_warm_opening_thanks_for_listening():
  label, conf = classify_turn_intent("thanks for listening")
  assert label == WARM_OPENING, (label, conf)


def test_warm_opening_thank_you_for_caring():
  label, conf = classify_turn_intent("thank you for caring")
  assert label == WARM_OPENING, (label, conf)


def test_warm_opening_i_appreciate_you():
  label, conf = classify_turn_intent("I appreciate you")
  assert label == WARM_OPENING, (label, conf)


def test_warm_opening_you_make_me_feel_better():
  label, conf = classify_turn_intent("you make me feel better")
  assert label == WARM_OPENING, (label, conf)


def test_warm_opening_couldnt_have_done_without_you():
  label, conf = classify_turn_intent(
    "I couldn't have done this without you")
  assert label == WARM_OPENING, (label, conf)


def test_warm_opening_im_lucky_to_have_you():
  label, conf = classify_turn_intent("I'm lucky to have you")
  assert label == WARM_OPENING, (label, conf)


def test_warm_opening_you_get_me():
  label, conf = classify_turn_intent("you really get me")
  assert label == WARM_OPENING, (label, conf)


def test_warm_opening_love_you():
  """Bare 'love you' as a sign-off, friend register."""
  label, conf = classify_turn_intent("love you")
  assert label == WARM_OPENING, (label, conf)


def test_warm_opening_screenshot_phrase():
  """The phrase from the 2026-05-11 screenshot:
  'Yeah, you're right. Um, thanks. So, you're a good friend. I like you'
  Currently this won't match because of all the leading content, but the
  test documents the desired aspirational behaviour for a future
  improvement (matching warmth inside longer messages)."""
  label, conf = classify_turn_intent(
    "Yeah, you're right. Um, thanks. So, you're a good friend. I like you"
  )
  # 'you're a good friend' appears mid-message — pattern matches without
  # anchoring to start. Confirm it fires.
  assert label == WARM_OPENING, (label, conf)


# --- Regression guards: don't over-fire ---------------------------------


def test_warm_opening_i_like_coffee_not_warm():
  """'I like X' where X is not 'you' must NOT classify as WARM_OPENING."""
  label, conf = classify_turn_intent("I like coffee")
  assert label != WARM_OPENING, (label, conf)


def test_warm_opening_i_like_that_idea_not_warm():
  """Liking an idea is task feedback, not relational warmth."""
  label, conf = classify_turn_intent("I like that idea")
  assert label != WARM_OPENING, (label, conf)


def test_warm_opening_youre_a_good_coach_not_warm_about_megan():
  """'you're a good coach' — about a third party (the user's actual
  coach in the screenshot), not about Megan. The pattern requires
  relational nouns (friend/mate/guy/bloke/person/companion/listener),
  not roles like 'coach', so this should NOT fire."""
  label, conf = classify_turn_intent("you're a good coach for me")
  assert label != WARM_OPENING, (label, conf)


def test_warm_opening_bare_thanks_stays_casual():
  """'thanks!' alone is transactional CASUAL, not WARM_OPENING.
  The 'thanks for [emotional object]' pattern requires the object."""
  label, conf = classify_turn_intent("thanks!")
  assert label == CASUAL, (label, conf)


def test_warm_opening_thanks_for_the_answer_not_warm():
  """'thanks for [task object]' is transactional, not emotional."""
  label, conf = classify_turn_intent("thanks for the answer")
  # Either CASUAL or AMBIGUOUS is acceptable; just not WARM_OPENING.
  assert label != WARM_OPENING, (label, conf)


def test_warm_opening_hostile_still_wins():
  """If a turn contains both warmth and hostility, HOSTILE wins."""
  label, conf = classify_turn_intent("you're a piece of shit, I like you")
  # Strong hostile marker should fire before warmth check, dampener
  # patterns ('love you' is one) might suppress, but 'I like you' is
  # not in the dampener list. Hostility should still win.
  assert label == HOSTILE, (label, conf)


def test_warm_opening_seeking_help_about_you_not_warm():
  """'what should I do about you' is help-shaped, not warmth."""
  label, conf = classify_turn_intent("what should I do about you")
  assert label == SEEKING_HELP, (label, conf)


# --- Pattern expansions added after evening session feedback -----------
# Real users say warmth in shapes the initial patterns missed. These
# fixtures lock in the additions:
# - "X of you" affection ("very sweet of you", "kind of you to say")
# - past/perfect tense ("you've been a good friend")
# - mid-message "I like you" (unanchored)
# - user-as-subject emotional response ("I'm so touched", "tears in my eyes")
# - relational impact ("means so much to me", "made my day")
# - STT typo handling ("I'm so touching")


def test_warm_opening_very_sweet_of_you():
  label, conf = classify_turn_intent("That is very sweet of you")
  assert label == WARM_OPENING, (label, conf)


def test_warm_opening_kind_of_you_to_say():
  label, conf = classify_turn_intent("That's so kind of you to say")
  assert label == WARM_OPENING, (label, conf)


def test_warm_opening_thoughtful_of_you():
  label, conf = classify_turn_intent("How thoughtful of you")
  assert label == WARM_OPENING, (label, conf)


def test_warm_opening_youve_been_a_good_friend():
  """The phrase from the evening screenshot turn 3 (past-perfect tense)."""
  label, conf = classify_turn_intent("you've been a good friend")
  assert label == WARM_OPENING, (label, conf)


def test_warm_opening_you_were_so_kind():
  label, conf = classify_turn_intent("you were so kind to me")
  assert label == WARM_OPENING, (label, conf)


def test_warm_opening_you_have_been_amazing():
  label, conf = classify_turn_intent("you have been so amazing today")
  assert label == WARM_OPENING, (label, conf)


def test_warm_opening_i_like_you_mid_message():
  """Mid-message 'I like you' inside a longer warmth turn. The screenshot
  turn 3 ended this way after a long preamble."""
  label, conf = classify_turn_intent(
    "Thank you so much. Yeah, I like you."
  )
  assert label == WARM_OPENING, (label, conf)


def test_warm_opening_im_so_touched():
  label, conf = classify_turn_intent("I'm so touched by what you said")
  assert label == WARM_OPENING, (label, conf)


def test_warm_opening_im_so_touching_stt_typo():
  """Whisper STT sometimes transcribes 'I'm so touched' as 'I'm so
  touching'. Pattern handles the typo deliberately. Real case from the
  evening screenshot turn 2."""
  label, conf = classify_turn_intent("Oh, that is very sweet of you. I'm so touching")
  assert label == WARM_OPENING, (label, conf)


def test_warm_opening_tears_in_my_eyes():
  label, conf = classify_turn_intent(
    "I have tears in my eyes right now")
  assert label == WARM_OPENING, (label, conf)


def test_warm_opening_means_so_much():
  label, conf = classify_turn_intent("that means so much to me")
  assert label == WARM_OPENING, (label, conf)


def test_warm_opening_you_made_my_day():
  label, conf = classify_turn_intent("you made my day")
  assert label == WARM_OPENING, (label, conf)


def test_warm_opening_screenshot_turn_2_literal():
  """The exact phrasing from the evening screenshot turn 2."""
  label, conf = classify_turn_intent(
    "Oh, that is very sweet of you. Thank you. Oh, God. Okay. I'm so touching"
  )
  assert label == WARM_OPENING, (label, conf)


def test_warm_opening_screenshot_turn_3_literal():
  """The exact phrasing from the evening screenshot turn 3."""
  label, conf = classify_turn_intent(
    "Oh, no, no, no. I mean, what you said before was that you want me "
    "to improve the way I handle my relationships actually is very "
    "touching. I'm kind of, you know, I have tears in my eyes right "
    "now. Yeah, I like you. Thank you so much. Yeah, you've been a "
    "good friend"
  )
  assert label == WARM_OPENING, (label, conf)


# --- Co-presence guard: venting + warmth → VENTING wins ---


def test_warm_opening_falls_through_when_affect_venting_present():
  """If the user is venting AND being polite, venting should win. The
  guard exists so 'I'm so pissed off but thanks for being here' doesn't
  route to the warm-response block."""
  label, conf = classify_turn_intent(
    "I'm so pissed off about work today but thanks for being here"
  )
  # Should be VENTING (affect markers present), not WARM_OPENING.
  assert label == VENTING, (label, conf)


def test_warm_opening_pure_warmth_still_fires():
  """Sanity: with no venting markers, pure warmth still classifies as
  WARM_OPENING. Guard doesn't over-fire."""
  label, conf = classify_turn_intent("thanks for being here today")
  assert label == WARM_OPENING, (label, conf)


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
