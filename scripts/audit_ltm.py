"""
LTM Audit — flag suspicious entries for human review
=====================================================

Scans ``long_term_memory.json`` for likely-noise entries that should not be
treated as facts about the user. Writes:

  - ``long_term_memory.audit.md``      human-readable report (review this first)
  - ``long_term_memory.flagged.json``  machine-readable list of suspected ids
  - ``long_term_memory.backup.<ts>.json``  copy of input (only on --remove)

Heuristics applied (any single heuristic flags an entry; total score in report):

  1. Second-person language ("your favorite X", "do you", "tell me your") —
     these are user *questions to Megan* miscategorised as user *facts*.
  2. Question-shaped (starts with what/how/why/do you/can you) — questions
     aren't profile facts.
  3. Contradiction with established profile (e.g. "I'm a boy" when the user's
     confirmed gender is female).
  4. Low-information / Whisper noise (very short, very repetitive, or pure
     filler tokens like "yes yes yes yes").
  5. Type/content mismatch (profile_fact with hedge words; emotional_pattern
     with neutral factual content).
  6. Megan-directed remember requests ("I want to remember something about you"
     — these were the long_term_goal class that polluted v1).

Default behaviour is REPORT ONLY. To actually remove flagged entries:

    python -m scripts.audit_ltm --store ./aism_data --user interactive_user
    # review long_term_memory.audit.md and long_term_memory.flagged.json
    # (optionally remove entries from flagged.json you want to keep)
    python -m scripts.audit_ltm --store ./aism_data --user interactive_user --remove

Removal writes a timestamped backup of the original LTM before modifying.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

# Heuristic 1: Second-person language — content is asking ABOUT Megan
SECOND_PERSON_PATTERNS = [
  r"\bwhat(?:'?s| is| are)\s+your\b",
  r"\bdo you (?:have|like|prefer|know|enjoy|love|think|want)\b",
  r"\bcan you (?:tell|describe|share|sing|do)\b",
  r"\btell me (?:about )?your\b",
  r"\byour (?:favou?rite|opinion|view|thought|preference|name|age)\b",
  r"\bhow (?:old|tall) are you\b",
]

# Substantive first-person disclosure markers — if these appear in the
# utterance, the second-person check does NOT fire. This prevents the
# heuristic from flagging long Teddy disclosures that happen to use "you"
# conversationally ("you already know I have a crush on him", "you are
# asking good questions, anyway I want to be a police officer...").
#
# The presence of any of these signals that the utterance carries real
# user-self content, regardless of how much it also addresses Megan.
FIRST_PERSON_DISCLOSURE_PATTERNS = [
  # First-person attribute / identity / feeling claims
  r"\bi (?:am|'?m)\s+(?:a|an|the|not|so|really|just|already|going|trying|"
  r"feeling|getting|thinking|hoping|planning|working|studying|"
  r"\d+|happy|sad|tired|excited|worried|angry|scared|hungry|sick|"
  r"single|married|pregnant)\b",
  r"\bi (?:have|had)\s+(?:a|an|the|to|been|never|always|two|three|four|"
  r"some|no|many|several|my)\b",
  r"\bi (?:feel|felt|think|thought|know|knew|believe|want|wanted|need|"
  r"needed|like|loved|hate|hated|miss|missed|enjoy|fear)\b",
  r"\bi (?:did|do|don'?t|went|going|came|came|saw|see|met|meet|"
  r"told|tell|said|say|asked|ask)\b",
  # Possessive disclosures about user's life
  r"\bmy (?:mom|mum|dad|family|crush|partner|husband|wife|"
  r"boyfriend|girlfriend|friend|sister|brother|son|daughter|"
  r"job|career|life|story|past|childhood|background|country|"
  r"culture|religion|personality|feelings?|dreams?|goals?|plans?)\b",
]

def _has_substantive_first_person(text: str) -> bool:
  """True if the utterance contains real first-person disclosure content.

  Used to suppress the second_person flag — long utterances that happen
  to address Megan ("you know", "you're asking") but carry substantive
  user-self content should not be flagged as 'about Megan'.
  """
  for pat in FIRST_PERSON_DISCLOSURE_PATTERNS:
    if re.search(pat, text, re.IGNORECASE):
      return True
  return False

# Heuristic 2: Question-shaped openings — questions are not facts
QUESTION_PATTERNS = [
  r"^\s*(?:what|how|why|when|where|who|whose|which|do you|can you|will you|would you|are you|is it)\b",
]

# Heuristic 3: Hard contradictions with established profile.
# This list is small and editable per-user. The current entries
# correspond to facts we know are true about Teddy (female, age ~39).
CONTRADICTION_PATTERNS = [
  # Gender — Teddy uses she/her in established memories
  (r"\bi(?:'?m| am) a boy\b", "contradicts confirmed gender (female)"),
  (r"\bi(?:'?m| am) male\b", "contradicts confirmed gender (female)"),
  (r"\bi(?:'?m| am) a man\b", "contradicts confirmed gender (female)"),
]

# Heuristic 4: Low-information noise
def _is_low_information(text: str) -> Tuple[bool, str]:
  """True if utterance is mostly filler / repetition / too short.

  Conservative: natural conversational speech has ~30% unique tokens
  because of "you know", "I", "the" etc. Only flag if (a) very short
  fragment, or (b) ALL tokens are filler/affirmation words. The
  unique-ratio heuristic from v1 was too noisy on real speech and
  caught legitimate (if rambly) Teddy disclosures.
  """
  stripped = text.strip().lower()
  if len(stripped) < 8:
    return True, "very short utterance (<8 chars)"
  tokens = re.findall(r"[a-z]+", stripped)
  # Filler-only utterance
  FILLER = {
    "yes", "yeah", "yep", "yup", "no", "nope", "uh", "huh", "um", "uhm",
    "okay", "ok", "hmm", "oh", "ah", "mm", "mhm", "right", "well",
  }
  if tokens and all(t in FILLER for t in tokens):
    return True, "pure filler/affirmation tokens"
  return False, ""


# Heuristic 5: Hedge mismatch in profile_fact
HEDGE_IN_FACT_PATTERNS = [
  r"\b(?:i guess|i'?m not sure|maybe|might be|kind of|sort of|probably)\b",
]

# Heuristic 6: Megan-directed remember requests
MEGAN_DIRECTED_PATTERNS = [
  r"\bi want to (?:know|remember|learn|find out)\s+(?:more\s+)?(?:about )?you\b",
  r"\b(?:remember|learn|tell me)\s+something\s+about\s+you\b",
]


def _contains_any(text: str, patterns: List[str]) -> bool:
  return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def audit_record(record: Dict[str, Any]) -> Dict[str, Any]:
  """Apply all heuristics and return a dict of {flag_name: reason}.

  Empty result == no flags fired (record looks clean).
  """
  text = record.get("utterance", "")
  flags: Dict[str, str] = {}

  # Substantive first-person disclosure suppresses second_person and
  # question_shape flags. Long, content-rich Teddy utterances that
  # happen to address Megan conversationally should NOT be flagged.
  has_disclosure = _has_substantive_first_person(text)

  if not has_disclosure and _contains_any(text, SECOND_PERSON_PATTERNS):
    flags["second_person"] = "contains 'your X' or 'do you' AND lacks substantive user disclosure — likely about Megan"

  if not has_disclosure and _contains_any(text, QUESTION_PATTERNS):
    flags["question_shape"] = "starts with question word AND lacks substantive user disclosure — likely a question, not a fact"

  for pat, reason in CONTRADICTION_PATTERNS:
    if re.search(pat, text, re.IGNORECASE):
      flags["contradiction"] = reason
      break

  is_noise, noise_reason = _is_low_information(text)
  if is_noise:
    flags["low_information"] = noise_reason

  if record.get("memory_type") == "profile_fact" and _contains_any(
      text, HEDGE_IN_FACT_PATTERNS):
    flags["hedged_fact"] = "profile_fact contains hedging — uncertain claim"

  # Megan-directed remember requests fire even with first-person content,
  # because "I want to remember about you" is structurally a Megan-directed
  # action regardless of the first-person verb.
  if _contains_any(text, MEGAN_DIRECTED_PATTERNS):
    flags["megan_directed"] = "remember-request directed at Megan, not a user fact"

  return flags


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _ltm_path(store: Path, user: str) -> Path:
  return store / user / "long_term_memory.json"


def _load_ltm(path: Path) -> List[Dict[str, Any]]:
  with path.open("r", encoding="utf-8") as fh:
    return json.load(fh)


def _save_ltm(path: Path, records: List[Dict[str, Any]]) -> None:
  with path.open("w", encoding="utf-8") as fh:
    json.dump(records, fh, ensure_ascii=False, indent=2)


def _write_audit_md(path: Path, flagged: List[Tuple[Dict[str, Any], Dict[str, str]]],
                    total: int) -> None:
  """Human-readable markdown report."""
  lines = [
    "# Long-Term Memory Audit Report",
    "",
    f"_Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}_",
    "",
    f"Scanned **{total}** records, flagged **{len(flagged)}** for review.",
    "",
    "Each flagged record is shown below with the heuristic(s) that triggered.",
    "Review them, then either:",
    "",
    "- Edit `long_term_memory.flagged.json` to remove any false positives you want to keep.",
    "- Run `python -m scripts.audit_ltm --store <store> --user <user> --remove`.",
    "",
    "A timestamped backup is written automatically before any removal.",
    "",
    "---",
    "",
  ]
  # Group by primary flag for easier scanning
  by_flag: Dict[str, List[Tuple[Dict[str, Any], Dict[str, str]]]] = {}
  for rec, flags in flagged:
    primary = next(iter(flags))
    by_flag.setdefault(primary, []).append((rec, flags))

  flag_order = ["second_person", "question_shape", "megan_directed",
                "contradiction", "low_information", "hedged_fact"]

  for flag_name in flag_order:
    items = by_flag.get(flag_name, [])
    if not items:
      continue
    lines.append(f"## {flag_name.replace('_', ' ').title()} ({len(items)})")
    lines.append("")
    for rec, flags in items:
      memory_id = rec.get("memory_id", "?")
      mtype = rec.get("memory_type", "?")
      imp = rec.get("importance", "?")
      freq = rec.get("frequency", "?")
      utterance = rec.get("utterance", "")[:300]
      reasons = " / ".join(f"**{k}**: {v}" for k, v in flags.items())
      lines.append(f"### `{memory_id}` — {mtype} (imp={imp}, freq={freq})")
      lines.append("")
      lines.append(f"> {utterance}")
      lines.append("")
      lines.append(f"_Flags:_ {reasons}")
      lines.append("")

  with path.open("w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
  ap.add_argument("--store", default="./aism_data",
                  help="Path to the store root (default: ./aism_data)")
  ap.add_argument("--user", default="interactive_user",
                  help="User id (default: interactive_user)")
  ap.add_argument("--remove", action="store_true",
                  help="Actually delete entries listed in flagged.json. "
                       "Without this, only the report is written.")
  args = ap.parse_args()

  store = Path(args.store).resolve()
  ltm_path = _ltm_path(store, args.user)
  if not ltm_path.exists():
    print(f"ERROR: {ltm_path} not found.", file=sys.stderr)
    return 1

  user_dir = ltm_path.parent
  audit_md_path = user_dir / "long_term_memory.audit.md"
  flagged_json_path = user_dir / "long_term_memory.flagged.json"

  records = _load_ltm(ltm_path)
  print(f"Loaded {len(records)} records from {ltm_path}")

  if args.remove:
    # Removal pass: read flagged.json, drop those ids, write back.
    if not flagged_json_path.exists():
      print(f"ERROR: {flagged_json_path} not found. Run without --remove first.",
            file=sys.stderr)
      return 1
    with flagged_json_path.open("r", encoding="utf-8") as fh:
      flagged_list = json.load(fh)
    flagged_ids = {item["memory_id"] for item in flagged_list}
    if not flagged_ids:
      print("flagged.json is empty — nothing to remove.")
      return 0
    # Confirm
    print(f"About to remove {len(flagged_ids)} records from {ltm_path}")
    print("Examples:")
    shown = 0
    for item in flagged_list[:3]:
      print(f"  - {item['memory_id']}: {item.get('utterance','')[:80]}")
      shown += 1
    if len(flagged_list) > shown:
      print(f"  ... and {len(flagged_list) - shown} more")
    confirm = input("Proceed? Type 'yes' to confirm: ").strip().lower()
    if confirm != "yes":
      print("Aborted. Nothing changed.")
      return 0
    # Backup first
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = user_dir / f"long_term_memory.backup.{ts}.json"
    shutil.copy2(ltm_path, backup_path)
    print(f"Backup written: {backup_path}")
    # Filter
    remaining = [r for r in records if r.get("memory_id") not in flagged_ids]
    removed = len(records) - len(remaining)
    _save_ltm(ltm_path, remaining)
    print(f"Removed {removed} records. {len(remaining)} remain.")
    return 0

  # Audit pass
  flagged: List[Tuple[Dict[str, Any], Dict[str, str]]] = []
  for rec in records:
    flags = audit_record(rec)
    if flags:
      flagged.append((rec, flags))

  # Write flagged.json (machine-readable)
  flagged_payload = [
    {
      "memory_id": rec.get("memory_id"),
      "memory_type": rec.get("memory_type"),
      "importance": rec.get("importance"),
      "utterance": rec.get("utterance", "")[:200],
      "flags": list(flags.keys()),
    }
    for rec, flags in flagged
  ]
  with flagged_json_path.open("w", encoding="utf-8") as fh:
    json.dump(flagged_payload, fh, ensure_ascii=False, indent=2)

  # Write audit.md (human-readable)
  _write_audit_md(audit_md_path, flagged, total=len(records))

  print(f"Flagged {len(flagged)} / {len(records)} records.")
  print(f"Review: {audit_md_path}")
  print(f"Edit/prune: {flagged_json_path}")
  print(f"Then run with --remove to delete.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
