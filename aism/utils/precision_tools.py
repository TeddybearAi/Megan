"""
Precision tools for Megan.

This module handles exact tasks that language models often get wrong:
letter counting, total letter counting, word counting, simple arithmetic, and
correction-sensitive "check again" turns.

Design principle:
- Use the LLM for natural conversation.
- Use deterministic helpers for exact answers.
- Do not patch one example such as "strawberry"; generalise the problem class.

Typical integration:
    from aism.utils.precision_tools import answer_if_precision_task

    precision = answer_if_precision_task(user_message, previous_user_text=last_user_message)
    if precision.handled and precision.should_bypass_llm:
        return precision.answer

If you prefer Megan to phrase the answer herself, use `precision.to_llm_context()`
and inject that verified context into the prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
import ast
import operator
import re
import string
from typing import Any, Optional


@dataclass(frozen=True)
class PrecisionResult:
    """Structured result from the precision guard."""

    handled: bool
    kind: str
    confidence: float
    answer: str
    short_answer: str = ""
    explanation: str = ""
    should_bypass_llm: bool = False
    model_instruction: str = ""

    def to_llm_context(self) -> str:
        """Return verified context that can be injected into the LLM prompt."""
        if not self.handled:
            return ""

        return (
            "Verified precision result:\n"
            f"- Task type: {self.kind}\n"
            f"- Exact answer: {self.short_answer or self.answer}\n"
            f"- Explanation: {self.explanation}\n"
            "Instruction: Use this verified result. Do not recalculate it from memory, "
            "do not guess, and do not contradict it."
        )


_APOSTROPHE_MAP = {
    "’": "'",
    "‘": "'",
    "“": '"',
    "”": '"',
}

_END_PHRASES = (
    "megan over",
    "megan your turn",
)

# Words that mean "letter" for our purposes. Includes common STT mishears so
# that a noisy transcript such as "the ledger r" still resolves to "letter r".
# Keep this list conservative; only add a variant after it has been observed
# in the wild, otherwise we risk false positives on normal conversation.
_LETTER_WORD_VARIANTS = ("letter", "character", "ledger")
_LETTER_WORD_ALT = "(?:" + "|".join(_LETTER_WORD_VARIANTS) + ")"


def normalise_text(text: str) -> str:
    """Normalise whitespace and quote characters."""
    if not text:
        return ""

    for src, dst in _APOSTROPHE_MAP.items():
        text = text.replace(src, dst)

    return re.sub(r"\s+", " ", text).strip()


def normalise_for_matching(text: str) -> str:
    """Normalise text for loose intent matching."""
    text = normalise_text(text).lower()

    for phrase in _END_PHRASES:
        text = re.sub(rf"\b{re.escape(phrase)}\b[.!?,]*", " ", text, flags=re.IGNORECASE)

    text = re.sub(r"[^a-z0-9\s'\"+\-*/().,%^=]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_outer_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1].strip()
    return value


def _extract_quoted_text(text: str) -> list[str]:
    matches: list[str] = []
    matches.extend(re.findall(r'"([^"]+)"', text))
    matches.extend(re.findall(r"'([^']+)'", text))
    return [m.strip() for m in matches if m.strip()]


_CORRECTION_PATTERNS = (
    r"\byou'?re wrong\b",
    r"\byou are wrong\b",
    r"\bthat's wrong\b",
    r"\bthat is wrong\b",
    r"\bwrong answer\b",
    r"\bcount again\b",
    r"\bcalculate again\b",
    r"\bcheck again\b",
    r"\bcheck carefully\b",
    r"\bare you sure\b",
    r"\btry again\b",
    r"\brecount\b",
)


def is_correction_or_careful_request(text: str) -> bool:
    """Return True when the user asks Megan to verify rather than guess."""
    lowered = normalise_for_matching(text)
    return any(re.search(pattern, lowered) for pattern in _CORRECTION_PATTERNS)


def _clean_target_token(token: str) -> str:
    token = strip_outer_quotes(token)
    return token.strip().strip(string.punctuation)


_TARGET_WORD_STOPWORDS = frozenset(
    {"the", "this", "that", "a", "an", "it", "word", "words", "letter", "letters"}
)


def _parse_letter_count_request(text: str) -> Optional[tuple[str, str]]:
    """
    Parse common letter-count requests.

    Examples:
    - how many r in strawberry
    - how many r's are in strawberry
    - how many letter r in the word strawberry
    - count the letter r in the word strawberry
    - how many times does r appear in strawberry
    """
    raw = normalise_text(text)
    lowered = normalise_for_matching(raw)
    quoted = _extract_quoted_text(raw)

    patterns = [
        rf"\bhow many (?:are the )?{_LETTER_WORD_ALT}s?\s+['\"]?([a-z0-9])['\"]?(?:'s|s)?\s+(?:(?:are|is)\s+)?(?:there\s+)?(?:in|inside|within|appear in|appears in)\s+(?:the\s+)?(?:word\s+)?['\"]?([a-z0-9_-]+)['\"]?",
        r"\bhow many\s+['\"]?([a-z0-9])['\"]?(?:'s|s)?\s+(?:(?:are|is)\s+)?(?:there\s+)?(?:in|inside|within)\s+(?:the\s+)?(?:word\s+)?['\"]?([a-z0-9_-]+)['\"]?",
        rf"\bcount\s+(?:the\s+)?{_LETTER_WORD_ALT}?s?\s*['\"]?([a-z0-9])['\"]?(?:'s|s)?\s+(?:in|inside|within)\s+(?:the\s+)?(?:word\s+)?['\"]?([a-z0-9_-]+)['\"]?",
        r"\bhow many times does\s+['\"]?([a-z0-9])['\"]?\s+(?:appear|show up|occur)\s+(?:in|inside|within)\s+(?:the\s+)?(?:word\s+)?['\"]?([a-z0-9_-]+)['\"]?",
    ]

    for pattern in patterns:
        # Iterate (not just first match) so we can skip captures where the
        # target word is a stopword like "word" / "the" — e.g. when the user
        # says "how many letter r in the word" with no actual word after.
        # In that case, control falls through to the lenient parser, which
        # can resolve "the word" against the previous turn.
        for match in re.finditer(pattern, lowered):
            letter = match.group(1)
            word = _clean_target_token(match.group(2))
            if (
                len(letter) == 1
                and word
                and word.lower() not in _TARGET_WORD_STOPWORDS
            ):
                return letter, word

    if quoted:
        # Apply the same counting-intent gate the rest of this function uses.
        # Without this, apostrophe-heavy transcripts (lots of contractions like
        # it's/don't/can't/i'm) get spuriously chunked into "quoted text" by
        # the naive single-quote regex, and a stray "character X" or "letter X"
        # phrase elsewhere in the text false-fires this path. Reported 2026-05-12
        # against a video-game monologue containing "character um" inside many
        # contractions; covered by the regression test in tests/test_precision_quoted_fallback.py.
        if not re.search(r"\b(?:how many|count(?:ing)?)\b", lowered):
            return None

        target = quoted[-1]
        match = re.search(rf"\b{_LETTER_WORD_ALT}\s+['\"]?([a-z0-9])['\"]?", lowered)
        if match:
            return match.group(1), target

        match = re.search(r"\bhow many\s+['\"]?([a-z0-9])['\"]?(?:'s|s)?\b", lowered)
        if match:
            return match.group(1), target

    return None


# Patterns for finding a specific target word in any text, in priority order.
# The first capturing group is the candidate word.
_TARGET_WORD_EXTRACTORS = (
    # "in [the/this/that] word X"
    r"\b(?:in|inside|within)\s+(?:(?:the|this|that)\s+)?word\s+['\"]?([a-z][a-z0-9_-]+)",
    # "[the/this/that] word X" without "in"
    r"\b(?:the|this|that)\s+word\s+['\"]?([a-z][a-z0-9_-]+)",
    # "spell/spelled/spelling [the word] X"
    r"\bspell(?:ed|ing)?\s+(?:(?:the|this|that)\s+word\s+)?['\"]?([a-z][a-z0-9_-]+)",
    # "in X" (basic; pronouns filtered out below)
    r"\b(?:in|inside|within)\s+(?:the\s+)?['\"]?([a-z][a-z0-9_-]+)",
    # "word X" alone
    r"\bword\s+['\"]?([a-z][a-z0-9_-]+)",
)


def _extract_specific_word(lowered: str) -> Optional[str]:
    """Try every target-word extractor on `lowered` and return the first
    candidate that isn't a stopword."""
    for pattern in _TARGET_WORD_EXTRACTORS:
        for m in re.finditer(pattern, lowered):
            candidate = _clean_target_token(m.group(1))
            if candidate and candidate.lower() not in _TARGET_WORD_STOPWORDS:
                return candidate
    return None


def _has_implicit_word_reference(lowered: str) -> bool:
    """True if the text refers to 'this/that/the word' as a generic pointer
    (e.g. 'how many R in this word') rather than naming the word."""
    return bool(re.search(r"\b(?:this|that|the)\s+word\b", lowered))


def _parse_letter_count_request_lenient(
    text: str,
    previous_user_text: Optional[str] = None,
) -> Optional[tuple[str, str]]:
    """
    Lenient fallback for noisy STT transcripts.

    Triggers only when all three gates pass:
    1. Counting intent ("how many" or "count").
    2. A target word identified via "in/inside/within [the] [word] X" — or,
       if the current text uses an implicit reference like "this word", the
       resolution falls back to the previous user turn.
    3. A target letter via fuzzy "letter X" / "ledger X" OR a trailing
       "the X" OR a bare trailing single letter.

    Catches phrasings such as:
        "how many are in the word strawberry can you count for me like the ledger r"
        "tell me how many R, letter R, in this word"   (with prior turn naming
                                                        the word)
    """
    raw = normalise_text(text)
    lowered = normalise_for_matching(raw)

    if not re.search(r"\b(?:how many|count(?:ing)?)\b", lowered):
        return None

    # Resolve target word: try current text first, then fall back to the
    # previous user turn if the current text uses an implicit reference.
    target_word = _extract_specific_word(lowered)
    if not target_word and _has_implicit_word_reference(lowered) and previous_user_text:
        prev_lowered = normalise_for_matching(previous_user_text)
        target_word = _extract_specific_word(prev_lowered)

    if not target_word:
        return None

    # Letter target candidates, in priority order:
    # 1. fuzzy "letter X" / "character X" / "ledger X"
    letter_match = re.search(
        rf"\b{_LETTER_WORD_ALT}s?\s+['\"]?([a-z0-9])['\"]?(?:'s|s)?\b",
        lowered,
    )
    if letter_match:
        return letter_match.group(1), target_word

    # 2. trailing "the X" at end of message (e.g. "...like the r")
    end_the = re.search(r"\bthe\s+([a-z0-9])(?:'s|s)?\s*$", lowered)
    if end_the:
        return end_the.group(1), target_word

    # 3. bare trailing single letter (e.g. "...for me r")
    bare_end = re.search(r"(?:^|\s)([a-z0-9])(?:'s|s)?\s*$", lowered)
    if bare_end:
        return bare_end.group(1), target_word

    return None


def answer_letter_count(
    text: str,
    *,
    previous_user_text: Optional[str] = None,
) -> PrecisionResult:
    parsed = _parse_letter_count_request(text)
    if not parsed:
        parsed = _parse_letter_count_request_lenient(
            text, previous_user_text=previous_user_text
        )
    if not parsed:
        return PrecisionResult(False, "letter_count", 0.0, "")

    letter, word = parsed
    count = word.lower().count(letter.lower())
    answer = f'The letter "{letter}" appears {count} time{"s" if count != 1 else ""} in "{word}".'
    explanation = f'Checked deterministically using "{word}".count("{letter}") = {count}.'

    return PrecisionResult(
        handled=True,
        kind="letter_count",
        confidence=0.98,
        answer=answer,
        short_answer=str(count),
        explanation=explanation,
        should_bypass_llm=True,
        model_instruction="Use the deterministic letter-count result. Do not estimate from memory.",
    )


def _parse_total_letter_count_request(text: str) -> Optional[str]:
    raw = normalise_text(text)
    lowered = normalise_for_matching(raw)
    quoted = _extract_quoted_text(raw)

    if quoted and re.search(r"\bhow many\s+(?:letters|characters)\b|\bcount\s+(?:the\s+)?(?:letters|characters)\b", lowered):
        return quoted[-1]

    patterns = [
        r"\bhow many\s+(?:letters|characters)\s+(?:are there\s+)?(?:in|inside|within)\s+(?:the\s+)?(?:word\s+)?['\"]?([a-z0-9_-]+)['\"]?",
        r"\bcount\s+(?:the\s+)?(?:letters|characters)\s+(?:in|inside|within)\s+(?:the\s+)?(?:word\s+)?['\"]?([a-z0-9_-]+)['\"]?",
    ]

    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            return _clean_target_token(match.group(1))

    return None


def answer_total_letter_count(text: str) -> PrecisionResult:
    target = _parse_total_letter_count_request(text)
    if not target:
        return PrecisionResult(False, "total_letter_count", 0.0, "")

    count = len([ch for ch in target if ch.isalnum()])
    answer = f'"{target}" has {count} letter{"s" if count != 1 else ""}.'
    explanation = f'Counted alphanumeric characters in "{target}" deterministically.'

    return PrecisionResult(
        handled=True,
        kind="total_letter_count",
        confidence=0.96,
        answer=answer,
        short_answer=str(count),
        explanation=explanation,
        should_bypass_llm=True,
        model_instruction="Use the deterministic total-letter-count result. Do not estimate from memory.",
    )


def _parse_word_count_request(text: str) -> Optional[str]:
    raw = normalise_text(text)
    lowered = normalise_for_matching(raw)
    quoted = _extract_quoted_text(raw)

    if quoted and re.search(r"\bhow many words\b|\bcount (?:the )?words\b", lowered):
        return quoted[-1]

    colon_match = re.search(
        r"\b(?:how many words(?: are there)? in|count (?:the )?words in)(?: this)?(?: sentence| phrase| text)?\s*:\s*(.+)$",
        raw,
        flags=re.IGNORECASE,
    )
    if colon_match:
        return colon_match.group(1).strip()

    return None


def _tokenise_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text)


def answer_word_count(text: str) -> PrecisionResult:
    target = _parse_word_count_request(text)
    if not target:
        return PrecisionResult(False, "word_count", 0.0, "")

    words = _tokenise_words(target)
    count = len(words)
    answer = f'The text "{target}" has {count} word{"s" if count != 1 else ""}.'
    explanation = f"Tokenised deterministically as: {words!r}."

    return PrecisionResult(
        handled=True,
        kind="word_count",
        confidence=0.94,
        answer=answer,
        short_answer=str(count),
        explanation=explanation,
        should_bypass_llm=True,
        model_instruction="Use the deterministic word-count result.",
    )


_ALLOWED_BINARY_OPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARY_OPS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class UnsafeExpressionError(ValueError):
    """Raised when an arithmetic expression contains unsupported syntax."""


def _safe_eval_arithmetic_node(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _safe_eval_arithmetic_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise UnsafeExpressionError("Only numeric constants are allowed.")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_BINARY_OPS:
            raise UnsafeExpressionError(f"Unsupported operator: {op_type.__name__}")

        left = _safe_eval_arithmetic_node(node.left)
        right = _safe_eval_arithmetic_node(node.right)

        if isinstance(node.op, ast.Pow) and abs(right) > 12:
            raise UnsafeExpressionError("Exponent is too large for this helper.")

        return _ALLOWED_BINARY_OPS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_UNARY_OPS:
            raise UnsafeExpressionError(f"Unsupported unary operator: {op_type.__name__}")
        return _ALLOWED_UNARY_OPS[op_type](_safe_eval_arithmetic_node(node.operand))

    raise UnsafeExpressionError(f"Unsupported expression: {type(node).__name__}")


def safe_eval_arithmetic(expression: str) -> float | int:
    """Safely evaluate simple arithmetic expressions."""
    expression = expression.strip()
    if not expression:
        raise UnsafeExpressionError("Empty expression.")

    expression = expression.replace("^", "**")
    expression = re.sub(r"\bx\b", "*", expression, flags=re.IGNORECASE)

    if not re.fullmatch(r"[0-9+\-*/().%\s]+", expression):
        raise UnsafeExpressionError("Expression contains unsupported characters.")

    tree = ast.parse(expression, mode="eval")
    return _safe_eval_arithmetic_node(tree)


def _extract_arithmetic_expression(text: str) -> Optional[str]:
    lowered = normalise_for_matching(text)

    if re.fullmatch(r"[0-9+\-*/().%\s^x]+", lowered) and re.search(r"\d", lowered):
        return lowered

    patterns = [
        r"\b(?:calculate|compute|work out|what is|what's|how much is)\s+([0-9+\-*/().%\s^x]+)\??$",
        r"\b([0-9+\-*/().%\s^x]+)\s*=\s*\??$",
    ]

    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            expr = match.group(1).strip()
            if expr and re.search(r"\d", expr):
                return expr

    return None


def _format_number(value: float | int) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.10g}"
    return str(value)


def answer_arithmetic(text: str) -> PrecisionResult:
    expression = _extract_arithmetic_expression(text)
    if not expression:
        return PrecisionResult(False, "arithmetic", 0.0, "")

    try:
        result = safe_eval_arithmetic(expression)
    except Exception:
        return PrecisionResult(False, "arithmetic", 0.0, "")

    formatted = _format_number(result)
    return PrecisionResult(
        handled=True,
        kind="arithmetic",
        confidence=0.97,
        answer=f"{expression.strip()} = {formatted}.",
        short_answer=formatted,
        explanation="Calculated using the safe deterministic arithmetic evaluator.",
        should_bypass_llm=True,
        model_instruction="Use the deterministic arithmetic result.",
    )


def _answer_current_precision_task(
    user_text: str,
    *,
    previous_user_text: Optional[str] = None,
) -> PrecisionResult:
    """Run the precision handlers on the current message only.

    `previous_user_text` is forwarded so that letter-count requests with
    implicit word references like "in this word" can be resolved against the
    most recent user turn.
    """
    # answer_letter_count is the only handler that uses previous_user_text;
    # the others ignore it. We call it explicitly to make the wiring obvious.
    result = answer_letter_count(user_text, previous_user_text=previous_user_text)
    if result.handled:
        return result

    for handler in (
        answer_total_letter_count,
        answer_word_count,
        answer_arithmetic,
    ):
        result = handler(user_text)
        if result.handled:
            return result

    return PrecisionResult(False, "none", 0.0, "")


def answer_if_precision_task(
    user_text: str,
    *,
    previous_user_text: Optional[str] = None,
) -> PrecisionResult:
    """
    Try to answer an exact precision task deterministically.

    `previous_user_text` is useful for turns such as:
    "you are wrong, count again".
    """
    user_text = normalise_text(user_text)

    current_result = _answer_current_precision_task(
        user_text, previous_user_text=previous_user_text
    )
    if current_result.handled:
        if is_correction_or_careful_request(user_text):
            return PrecisionResult(
                handled=True,
                kind=f"careful:{current_result.kind}",
                confidence=current_result.confidence,
                answer="I checked it carefully. " + current_result.answer,
                short_answer=current_result.short_answer,
                explanation=current_result.explanation,
                should_bypass_llm=current_result.should_bypass_llm,
                model_instruction=current_result.model_instruction + " The user requested careful verification.",
            )
        return current_result

    if previous_user_text and is_correction_or_careful_request(user_text):
        retry = _answer_current_precision_task(previous_user_text)
        if retry.handled:
            return PrecisionResult(
                handled=True,
                kind=f"correction_retry:{retry.kind}",
                confidence=retry.confidence,
                answer="You are right to ask me to check carefully. " + retry.answer,
                short_answer=retry.short_answer,
                explanation=retry.explanation,
                should_bypass_llm=True,
                model_instruction=(
                    "The user challenged the previous answer. Use this verified result "
                    "and acknowledge the correction briefly."
                ),
            )

    return PrecisionResult(False, "none", 0.0, "", should_bypass_llm=False)


def _self_test() -> None:
    examples = {
        "how many r in strawberry Megan over": "3",
        "how many letter r in the word strawberry": "3",
        "how many r are in the word strawberry": "3",
        "how many r's are there in strawberry": "3",
        # Lenient fallback: noisy STT mishearing "letter" as "ledger" and
        # placing the letter target at the end (real example from a Megan
        # session log).
        "no nothing specific but i have a question for you do you know how many are in the word strawberry can you actually count for me like the ledger r": "3",
        "in the word strawberry how many r": "3",
        'how many letters are in "strawberry"': "10",
        'how many words are in "I love Megan"': "3",
        "calculate 12 + 5 * 3": "27",
        "what is (10 + 2) / 4": "3",
    }

    for message, expected in examples.items():
        result = answer_if_precision_task(message)
        assert result.handled, f"Expected handled=True for: {message}"
        assert result.short_answer == expected, (
            f"Expected {expected!r}, got {result.short_answer!r} for: {message}"
        )

    # Negative cases: lenient parser must not false-trigger on normal
    # conversation that happens to contain "how many" + "in" + a place word.
    negatives = (
        "how many people are in the room",
        "how many cars are in the parking lot today",
        "what's the first letter of the alphabet",
        "i'll see you later",
    )
    for message in negatives:
        result = answer_if_precision_task(message)
        assert not result.handled, (
            f"Expected handled=False (no false positive) for: {message}; "
            f"got kind={result.kind} answer={result.answer!r}"
        )

    correction = answer_if_precision_task(
        "you are wrong, count again",
        previous_user_text="how many r in strawberry",
    )
    assert correction.handled
    assert correction.short_answer == "3"

    # Implicit word reference: the current turn says "in this word" but doesn't
    # name the word. The previous user turn named it (via "spell strawberry").
    # Without context resolution, the lenient parser would give up and the LLM
    # would hallucinate (as it did in the field test, answering "two R's").
    implicit = answer_if_precision_task(
        "Yes, you can spell the word, then tell me how many R, letter R, in this word",
        previous_user_text="Okay, can you spell strawberry for me",
    )
    assert implicit.handled, f"Expected handled=True for implicit reference; got {implicit}"
    assert implicit.short_answer == "3", (
        f"Expected 3 R's in strawberry via context resolution; got {implicit.short_answer!r}"
    )

    # Same pattern via "the word" instead of "this word".
    implicit2 = answer_if_precision_task(
        "ok and how many letter r in the word",
        previous_user_text="please spell the word strawberry",
    )
    assert implicit2.handled
    assert implicit2.short_answer == "3"

    # Negative: implicit reference but no previous context — must NOT trigger.
    no_context = answer_if_precision_task("how many letter r in this word")
    assert not no_context.handled, (
        "Lenient parser must NOT trigger when there is no previous context to "
        "resolve 'this word' against."
    )


if __name__ == "__main__":
    _self_test()
    print("precision_tools.py self-test passed.")
