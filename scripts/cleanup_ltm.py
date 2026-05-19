"""
One-shot LTM cleanup
====================

Walks an existing ``long_term_memory.json``, applies
``strip_voice_controls`` to each utterance, re-runs the new memory classifier,
optionally re-embeds, runs consolidation to mark duplicates, and writes:

  - <store>/<user>/long_term_memory.cleaned.json    proposed new state
  - <store>/<user>/long_term_memory.diff.json       per-record diff report
  - <store>/<user>/long_term_memory.backup.<ts>.json  copy of the input

It does NOT overwrite the original ``long_term_memory.json`` unless invoked
with ``--apply``. Default behaviour is dry-run + diff, so the user can
review before committing.

Usage::

    python -m scripts.cleanup_ltm --store ./aism_data --user interactive_user
    python -m scripts.cleanup_ltm --store ./aism_data --user interactive_user --apply
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Reach the project root so ``aism`` is importable.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))

from aism.memory import (    # noqa: E402
  infer_memory_type, infer_store_label, infer_importance,
)
from aism.consolidation import consolidate_memories  # noqa: E402
from aism.embedder import CharacterNgramEmbedder  # noqa: E402


def _build_embedder():
  """Try the real sentence-transformer first, fall back to char-ngram.

  Returns a tuple ``(embedder, is_semantic)``. ``is_semantic`` is True
  only when the sentence-transformer succeeded — the cleanup script uses
  this to decide whether consolidation is safe to enable by default.
  """
  try:
    from aism.embedder import SentenceTransformerEmbedder
    emb = SentenceTransformerEmbedder()
    print(f"Embedder: SentenceTransformer (dim={emb.dim})")
    return emb, True
  except Exception as e:
    emb = CharacterNgramEmbedder(dim=256)
    print(f"Embedder: CharacterNgram (dim={emb.dim}) — "
          f"sentence-transformers unavailable ({type(e).__name__}: {e})")
    return emb, False


# Replicated from app.py. We avoid importing app.py because it pulls in
# faster-whisper / FastAPI / edge-tts which we don't need for cleanup.
_VOICE_CONTROL_PATTERNS = [
  re.compile(r"\bmake\s+(?:an|and|in|it)\s+(?:over|your\s+turn)\b[,.!?\s]*",
             re.IGNORECASE),
  re.compile(
    r"\b(?:maggie|maken|magna|making|again|may\s+again)"
    r"\s+(?:over|your\s+turn)\b[,.!?\s]*", re.IGNORECASE),
  re.compile(r"\bmegan[,\s]+(?:over|your\s+turn)\b[,.!?\s]*", re.IGNORECASE),
  re.compile(r"[,.!?\s]*\b(?:megan'?s?\s+)?(?:over|your\s+turn)\b[,.!?\s]*$",
             re.IGNORECASE),
  re.compile(r"^[\s,.!?]*(?:hey|hi|hello|ok|okay)\s+megan\b[,.!?\s]*",
             re.IGNORECASE),
  re.compile(r"^[\s,.!?]*megan\b[,.!?\s]+", re.IGNORECASE),
  re.compile(r"[.!?,]+\s*megan\b[\s,.!?]*$", re.IGNORECASE),
]


def strip_voice_controls(text: str) -> str:
  if not text:
    return ""
  cleaned = text
  for pat in _VOICE_CONTROL_PATTERNS:
    cleaned = pat.sub(" ", cleaned)
  cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.!?")
  return cleaned


def reclassify_record(
  record: Dict[str, Any],
  embedder: CharacterNgramEmbedder,
) -> Dict[str, Any]:
  """Return an updated copy of the record after cleaning + reclassifying.

  Decisions:
    - Strip voice-control noise from utterance.
    - If cleaned utterance is too short, mark for deletion.
    - Reclassify; if new type is ``transient_state`` or ``daily_detail``
      with store="no", mark for deletion.
    - Otherwise update memory_type / importance / notes / embedding.
  """
  out = dict(record)
  raw = out.get("utterance", "") or ""
  cleaned = strip_voice_controls(raw)

  # Empty-after-clean → delete.
  if len(cleaned) < 6:
    out["_action"] = "delete"
    out["_reason"] = (
      f"After voice-control strip, utterance too short ({len(cleaned)} chars)."
    )
    out["utterance_cleaned"] = cleaned
    return out

  out["utterance_cleaned"] = cleaned

  new_type = infer_memory_type(cleaned)
  new_store = infer_store_label(cleaned, new_type)
  new_importance = infer_importance(cleaned, new_type)

  out["_proposed_memory_type"] = new_type
  out["_proposed_store"] = new_store
  out["_proposed_importance"] = new_importance

  if new_store == "no":
    out["_action"] = "delete"
    out["_reason"] = f"Reclassified as {new_type} (store=no)."
    return out

  # Keep, but rewrite content + embedding.
  out["_action"] = "keep"
  out["_reason"] = (
    f"Re-classified {record.get('memory_type', '?')}/imp{record.get('importance', '?')}"
    f" → {new_type}/imp{new_importance}."
  )
  out["utterance"] = cleaned
  out["memory_type"] = new_type
  out["importance"] = new_importance
  try:
    out["embedding"] = list(embedder.embed(cleaned))
  except Exception:
    pass
  return out


def cleanup(
  store_path: Path,
  user_id: str,
  apply: bool,
  similarity_threshold: float = 0.85,
  consolidate_mode: str = "auto",
) -> int:
  """Run the cleanup pass.

  ``consolidate_mode`` is one of:
    - ``"auto"``  : ON when sentence-transformers is available, OFF for
                     CharacterNgramEmbedder (which clusters on style, not
                     topic, and produces false merges on real data).
    - ``"on"``    : force ON regardless of embedder.
    - ``"off"``   : force OFF.
  """
  user_dir = store_path / user_id
  ltm_path = user_dir / "long_term_memory.json"
  if not ltm_path.exists():
    print(f"No long_term_memory.json at {ltm_path}; nothing to do.")
    return 1

  raw = json.loads(ltm_path.read_text(encoding="utf-8"))
  if not isinstance(raw, list):
    print("Unexpected LTM shape (expected a list).")
    return 2

  embedder, is_semantic = _build_embedder()

  if consolidate_mode == "auto":
    do_consolidate = is_semantic
    if not is_semantic:
      print("Consolidation: OFF (auto — char-ngram embedder is too coarse "
            "for real data; install sentence-transformers to enable).")
    else:
      print(f"Consolidation: ON (auto — semantic embedder, "
            f"threshold={similarity_threshold}).")
  elif consolidate_mode == "on":
    do_consolidate = True
    print(f"Consolidation: ON (forced, threshold={similarity_threshold}).")
  else:
    do_consolidate = False
    print("Consolidation: OFF (forced).")

  # 1. Reclassify each record.
  staged = [reclassify_record(r, embedder) for r in raw]

  # 2. Drop the deletes; keep the keeps.
  kept_pre_consolidation: List[Dict[str, Any]] = []
  deleted: List[Dict[str, Any]] = []
  for r in staged:
    if r.get("_action") == "delete":
      deleted.append(r)
    else:
      kept_pre_consolidation.append(r)

  # 3. Strip the diff bookkeeping fields before persisting / consolidating.
  def _persist_view(r: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: v for k, v in r.items() if not k.startswith("_")}
    out.pop("utterance_cleaned", None)
    return out

  to_consolidate = [_persist_view(r) for r in kept_pre_consolidation]

  # 4. Consolidation pass — see consolidate_mode docstring.
  if do_consolidate:
    consolidated, n_consolidated = consolidate_memories(
      to_consolidate, similarity_threshold=similarity_threshold)
  else:
    consolidated = to_consolidate
    n_consolidated = 0

  # 5. Build the diff report.
  diff = {
    "summary": {
      "input_records": len(raw),
      "deleted": len(deleted),
      "kept": len(kept_pre_consolidation),
      "consolidated": n_consolidated,
      "final_active": sum(
        1 for r in consolidated if r.get("status", "active") == "active"),
    },
    "deleted_records": [
      {
        "memory_id": r.get("memory_id"),
        "utterance_before": r.get("utterance", ""),
        "utterance_cleaned": r.get("utterance_cleaned", ""),
        "old_memory_type": r.get("memory_type"),
        "old_importance": r.get("importance"),
        "reason": r.get("_reason", ""),
      }
      for r in deleted
    ],
    "kept_records": [
      {
        "memory_id": r.get("memory_id"),
        "utterance_before": r.get("utterance", ""),
        "utterance_after": r.get("utterance_cleaned", r.get("utterance", "")),
        "old_memory_type": (
          # We've overwritten memory_type on keep; the staged dict has the
          # proposed one explicitly recorded.
          r.get("_proposed_memory_type")
        ),
        "new_memory_type": r.get("memory_type"),
        "old_importance": r.get("importance"),
        "new_importance": r.get("_proposed_importance"),
        "reason": r.get("_reason", ""),
      }
      for r in kept_pre_consolidation
    ],
    "consolidations": [
      {
        "consolidated_id": r.get("memory_id"),
        "into": r.get("consolidated_into"),
        "utterance": r.get("utterance", ""),
      }
      for r in consolidated if r.get("status") == "consolidated"
    ],
  }

  diff_path = user_dir / "long_term_memory.diff.json"
  diff_path.write_text(
    json.dumps(diff, indent=2, ensure_ascii=False), encoding="utf-8")

  cleaned_path = user_dir / "long_term_memory.cleaned.json"
  cleaned_path.write_text(
    json.dumps(consolidated, indent=2, ensure_ascii=False),
    encoding="utf-8")

  print(f"\n--- Cleanup summary for user '{user_id}' ---")
  print(f"  input records      : {diff['summary']['input_records']}")
  print(f"  deleted            : {diff['summary']['deleted']}")
  print(f"  kept               : {diff['summary']['kept']}")
  print(f"  marked consolidated: {diff['summary']['consolidated']}")
  print(f"  final active       : {diff['summary']['final_active']}")
  print(f"\n  diff report  : {diff_path}")
  print(f"  proposed LTM : {cleaned_path}")

  if not apply:
    print("\nDry-run mode. Original long_term_memory.json untouched.")
    print("Re-run with --apply to overwrite (a backup is taken first).")
    return 0

  # Apply: backup the original, then overwrite.
  ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
  backup_path = user_dir / f"long_term_memory.backup.{ts}.json"
  shutil.copy2(ltm_path, backup_path)
  shutil.copy2(cleaned_path, ltm_path)
  print(f"\n[applied] backup at {backup_path}")
  print(f"[applied] {ltm_path} now reflects the cleaned state.")
  return 0


def main() -> int:
  ap = argparse.ArgumentParser(description="One-shot LTM cleanup.")
  ap.add_argument("--store", required=True, help="Path to aism_data dir")
  ap.add_argument("--user", required=True, help="User id")
  ap.add_argument("--apply", action="store_true",
                  help="Overwrite the original LTM file (default: dry run).")
  ap.add_argument(
    "--threshold", type=float, default=0.85,
    help=("Cosine similarity threshold for consolidation. "
          "Default 0.85 (good for sentence-transformer embeddings). "
          "Use 0.95+ if you only have CharacterNgramEmbedder."))
  ap.add_argument(
    "--consolidate", choices=["auto", "on", "off"], default="auto",
    help=("auto (default): on iff sentence-transformers is installed. "
          "on: force consolidation regardless of embedder. "
          "off: skip consolidation entirely."))
  args = ap.parse_args()
  return cleanup(
    Path(args.store), args.user, args.apply, args.threshold,
    consolidate_mode=args.consolidate)


if __name__ == "__main__":
  sys.exit(main())
