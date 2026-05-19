"""
AISM — Concrete generation adapters.

Pass 2 ships OllamaAdapter. Additional adapters (llama.cpp, etc.) can be
added here and will satisfy the GenerationAdapter Protocol in
aism.generation_adapter.
"""

from .ollama_adapter import OllamaAdapter, OllamaAdapterError

__all__ = ["OllamaAdapter", "OllamaAdapterError"]
