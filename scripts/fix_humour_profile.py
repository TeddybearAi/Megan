"""
One-shot: Fix the wrong-sign humour value in profile.json
=========================================================

After the morning session's humour lexicon fix, new evidence is correctly
signed — but the existing ``profile.json`` still carries the v1 wrong-sign
value (``humour.general.value = 0.0``, meaning "no humour wanted"). Stage 4
will only drift it back to the correct direction over many sessions of new
evidence, and meanwhile every turn still gets ``emotional_style: "no jokes"``
rendered into the system prompt — directly contradicting the persona's
"you catch jokes" line.

This script writes the correct value directly. Dry-run by default; ``--apply``
to commit.

Background: ltm_020 in cleaned LTM is the user explicitly saying *"trying to
be funny, but apparently you are not funny enough… we should make you more
funnier"* — i.e. the user wants MORE humour. v1 read this as humour=LOW and
stored 0.0.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))


def _now_iso() -> str:
  return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fix_humour(
  store_path: Path,
  user_id: str,
  apply: bool,
  target_value: float = 0.7,
) -> int:
  user_dir = store_path / user_id
  profile_path = user_dir / "profile.json"
  if not profile_path.exists():
    print(f"No profile.json at {profile_path}; nothing to do.")
    return 1

  raw = json.loads(profile_path.read_text(encoding="utf-8"))
  traits = raw.get("traits", {})
  humour = traits.get("humour", {})
  general = humour.get("general")

  if not general:
    print(f"No humour.general trait in profile. Skipping.")
    return 0

  current = general.get("value")
  print(f"Current humour.general.value = {current}")
  print(f"Proposed humour.general.value = {target_value}")

  if current == target_value:
    print("Already at target value; nothing to do.")
    return 0

  # Build the proposed-after view.
  proposed: Dict[str, Any] = dict(general)
  proposed["value"] = target_value
  proposed["confidence"] = 0.5  # reset to mid-confidence — we wrote this by hand
  proposed["evidence_count"] = 0  # let future evidence build confidence cleanly
  proposed["volatility"] = 0.0
  proposed["last_updated"] = _now_iso()
  proposed["source_summary"] = [
    f"Reset by scripts/fix_humour_profile.py on {_now_iso()}. "
    f"v1 lexicon had a sign-flip on 'not funny enough'; the morning "
    f"session fixed the lexicon but the stored value remained wrong. "
    f"This is a direct override to {target_value:.2f}."
  ]
  proposed["recent_values"] = [target_value]
  proposed["session_drift"] = 0.0

  print("\nDiff:")
  for key in (
    "value", "confidence", "evidence_count",
    "volatility", "session_drift",
  ):
    print(f"  {key}: {general.get(key)} -> {proposed.get(key)}")

  if not apply:
    print(
      "\nDry-run mode. profile.json untouched. Re-run with --apply to "
      "commit (a backup is taken first).")
    return 0

  # Apply: backup the original, then write the patched profile.
  ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
  backup_path = user_dir / f"profile.backup.{ts}.json"
  shutil.copy2(profile_path, backup_path)

  raw["traits"]["humour"]["general"] = proposed
  raw["last_updated"] = _now_iso()
  profile_path.write_text(
    json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")

  print(f"\n[applied] backup at {backup_path}")
  print(f"[applied] profile.json updated.")
  return 0


def main() -> int:
  ap = argparse.ArgumentParser(
    description="Fix the wrong-sign humour value in profile.json.")
  ap.add_argument("--store", required=True, help="Path to aism_data dir")
  ap.add_argument("--user", required=True, help="User id")
  ap.add_argument("--apply", action="store_true",
                  help="Commit the change (default: dry run).")
  ap.add_argument(
    "--target", type=float, default=0.7,
    help=("Target humour value in [0, 1]. 0.7 = clearly wants humour, "
          "0.8 = strongly wants humour. Default 0.7."))
  args = ap.parse_args()
  return fix_humour(Path(args.store), args.user, args.apply, args.target)


if __name__ == "__main__":
  sys.exit(main())
