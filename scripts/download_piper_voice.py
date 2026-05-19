"""
One-shot: Download a Piper TTS voice model
==========================================

Fetches a Piper voice (.onnx + .onnx.json config) into a local ``voices/``
directory so the TTS engine in ``aism/tts.py`` can load it at startup.

Default voice is ``en_GB-alan-medium`` — the British male voice Megan uses.
Pass ``--voice`` to fetch a different one.

Dry-run by default; ``--apply`` to actually download. Same UX as
``cleanup_ltm.py`` and ``fix_humour_profile.py``.

Usage:
    # Show what would be downloaded (default).
    python scripts/download_piper_voice.py

    # Actually download Alan into voices/.
    python scripts/download_piper_voice.py --apply

    # Download a different voice into a different directory.
    python scripts/download_piper_voice.py \\
        --voice en_GB-northern_english_male-medium \\
        --data-dir voices --apply

Under the hood this delegates the actual fetch to ``piper.download_voices``,
which is the official downloader bundled with the ``piper-tts`` package.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
  sys.path.insert(0, str(_ROOT))


DEFAULT_VOICE = "en_GB-alan-medium"
DEFAULT_VOICE_DIR = Path("voices")
HF_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

# Known sizes for common voices so the dry-run is informative without
# hitting the network. For voices not listed here we print a generic
# size estimate.
_KNOWN_SIZES: dict[str, tuple[str, str]] = {
  "en_GB-alan-low":                       ("23 MB",  "6 KB"),
  "en_GB-alan-medium":                    ("63 MB",  "6 KB"),
  "en_GB-northern_english_male-medium":   ("63 MB",  "6 KB"),
  "en_GB-jenny_dioco-medium":             ("63 MB",  "6 KB"),
  "en_US-ryan-medium":                    ("63 MB",  "6 KB"),
  "en_US-joe-medium":                     ("63 MB",  "6 KB"),
  "en_US-amy-medium":                     ("63 MB",  "6 KB"),
  "en_US-lessac-medium":                  ("63 MB",  "6 KB"),
}


def _hf_directory(voice: str) -> str:
  """Construct the Hugging Face directory URL for a voice name.

  Voice names follow the ``<locale>-<speaker>-<quality>`` pattern, and
  the repo layout is ``en/<locale>/<speaker>/<quality>/``. Speaker names
  can themselves contain hyphens (e.g. ``northern_english_male``), so we
  split off the locale prefix and quality suffix and treat the rest as
  the speaker name.
  """
  parts = voice.split("-")
  if len(parts) < 3:
    return f"<unrecognised voice name layout: {voice}>"
  locale = parts[0]
  quality = parts[-1]
  speaker = "-".join(parts[1:-1])
  family = locale.split("_")[0]
  return f"{HF_BASE}/{family}/{locale}/{speaker}/{quality}"


def _print_plan(voice: str, dest_dir: Path) -> None:
  hf_dir = _hf_directory(voice)
  onnx_size, json_size = _KNOWN_SIZES.get(voice, ("~60 MB", "~6 KB"))
  print(f"Voice:        {voice}")
  print(f"Destination:  {dest_dir.resolve()}")
  print(f"Source:       {hf_dir}")
  print()
  print(f"  Will download:")
  print(f"    {voice}.onnx        ({onnx_size})")
  print(f"    {voice}.onnx.json   ({json_size})")
  print()


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Download a Piper TTS voice model to a local directory.",
  )
  parser.add_argument(
    "--voice",
    default=DEFAULT_VOICE,
    help=(
      f"Voice name (default: {DEFAULT_VOICE}). See "
      "https://huggingface.co/rhasspy/piper-voices for the full list."
    ),
  )
  parser.add_argument(
    "--data-dir",
    default=str(DEFAULT_VOICE_DIR),
    type=Path,
    help=(
      f"Where to save the .onnx and .onnx.json files "
      f"(default: {DEFAULT_VOICE_DIR})."
    ),
  )
  parser.add_argument(
    "--apply",
    action="store_true",
    help="Actually download. Without this flag, only the plan is printed.",
  )
  args = parser.parse_args()

  dest_dir: Path = args.data_dir
  voice: str = args.voice

  print("=" * 60)
  print(f"Piper voice download — {'APPLY' if args.apply else 'DRY-RUN'}")
  print("=" * 60)
  print()

  _print_plan(voice, dest_dir)

  onnx_path = dest_dir / f"{voice}.onnx"
  config_path = dest_dir / f"{voice}.onnx.json"

  already_have = [p for p in (onnx_path, config_path) if p.exists()]
  if already_have:
    print("Already on disk:")
    for p in already_have:
      print(f"  {p}")
    print()

  if not args.apply:
    print("Dry-run only. Re-run with --apply to download.")
    return 0

  if onnx_path.exists() and config_path.exists():
    print("Both files already present; nothing to download.")
    return 0

  dest_dir.mkdir(parents=True, exist_ok=True)

  # Delegate the actual fetch to Piper's built-in downloader. It handles
  # the Hugging Face resolve URLs and writes ``<voice>.onnx`` and
  # ``<voice>.onnx.json`` into ``--data-dir``.
  cmd = [
    sys.executable, "-m", "piper.download_voices",
    voice,
    "--data-dir", str(dest_dir),
  ]
  print(f"Running: {' '.join(cmd)}")
  print()

  try:
    subprocess.run(cmd, check=True)
  except FileNotFoundError:
    print(
      "ERROR: Couldn't run the piper package. "
      "Did you `pip install -r requirements.txt`?",
      file=sys.stderr,
    )
    return 1
  except subprocess.CalledProcessError as e:
    print(
      f"\nERROR: piper.download_voices exited with code {e.returncode}.",
      file=sys.stderr,
    )
    return e.returncode

  print()
  print("Download complete:")
  print(f"  {onnx_path}")
  print(f"  {config_path}")
  print()
  print("Next: start the app — Megan will load this voice at startup.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
