"""
AISM — Local TTS engine wrapper
================================
Megan's voice synthesis. Replaces the previous Edge TTS cloud dependency
with Piper (OHF-Voice/piper1-gpl) — a fast, local neural TTS that runs
entirely on-device.

This module is intentionally thin. All TTS calls in the rest of the app
go through the ``PiperTTS`` class, so future engine swaps (XTTS-v2 voice
cloning, Kokoro, etc.) replace this one file rather than threading
through every caller.

Local-first: the .onnx voice model lives on disk. Once loaded into
memory at startup, every subsequent synthesis is purely local — zero
network calls, zero data leaves the machine.

To download the default voice (en_GB-alan-medium, ~63 MB) into ``voices/``:
    python scripts/download_piper_voice.py --apply
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Optional

from piper import PiperVoice, SynthesisConfig


class PiperTTS:
  """Local neural TTS via Piper.

  Load once at process startup, reuse for every synthesis call. The
  voice model (~63 MB for en_GB-alan-medium) is held in memory for the
  lifetime of the process. The matching .onnx.json config file must
  sit alongside the .onnx model.
  """

  def __init__(
    self,
    model_path: Path | str,
    default_length_scale: float = 1.0,
    use_cuda: bool = False,
  ) -> None:
    """Load the Piper voice model from disk.

    Parameters
    ----------
    model_path: Path | str
        Path to the .onnx voice model. The matching .onnx.json config
        file must sit in the same directory with the same basename.
    default_length_scale: float
        Default speech speed when ``length_scale`` is not passed to
        ``synthesize``. Piper convention: 1.0 = the speed the model
        was trained at; lower = faster, higher = slower. ~0.85
        approximates Edge TTS's previous ``rate="+20%"`` setting.
    use_cuda: bool
        If True, run inference on GPU. Requires ``onnxruntime-gpu`` to
        be installed in place of the default ``onnxruntime``. For short
        utterances (typical Megan replies), CPU is already realtime+
        on modern hardware, so the default is CPU.
    """
    self.model_path = Path(model_path)
    if not self.model_path.exists():
      raise FileNotFoundError(
        f"Piper voice model not found: {self.model_path}\n"
        f"Run: python scripts/download_piper_voice.py --apply"
      )

    # Piper requires the matching .onnx.json config file in the same
    # directory. ``.with_suffix(".json")`` swaps the trailing .onnx for
    # .json, giving us en_GB-alan-medium.onnx.json.
    config_path = self.model_path.with_suffix(".onnx.json")
    if not config_path.exists():
      raise FileNotFoundError(
        f"Piper voice config not found: {config_path}\n"
        f"It must sit alongside the .onnx file. "
        f"Run: python scripts/download_piper_voice.py --apply"
      )

    self.voice = PiperVoice.load(str(self.model_path), use_cuda=use_cuda)
    self.default_length_scale = default_length_scale

  def synthesize(
    self,
    text: str,
    output_path: Path | str,
    length_scale: Optional[float] = None,
  ) -> None:
    """Synthesise ``text`` to a WAV file at ``output_path``.

    Parameters
    ----------
    text: str
        The text to speak. Caller is responsible for cleaning emojis
        and markdown beforehand — see ``app.clean_text_for_tts``.
    output_path: Path | str
        Where to write the .wav file. Parent directory is created if
        it doesn't exist.
    length_scale: float | None
        Per-call override of speech speed. None = use the default set
        at construction time. Lower = faster.
    """
    if not text or not text.strip():
      text = "."

    scale = length_scale if length_scale is not None else self.default_length_scale
    syn_config = SynthesisConfig(length_scale=scale)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(output_path), "wb") as wav_file:
      self.voice.synthesize_wav(text, wav_file, syn_config=syn_config)


__all__ = ["PiperTTS"]
