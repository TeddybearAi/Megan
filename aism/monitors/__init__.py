"""
AISM — Monitors subpackage
==========================
Four over-mimicry safeguards per BluePrint v2.1 Section 5.

    LexicalMimicryMonitor     — n-gram overlap, runtime
    IdiolectDriftMonitor      — embedding distance from baseline, runtime
    SycophancyMonitor         — position-reversal heuristic, runtime
    ComfortRatingRecorder     — Likert-scale recorder, user study
"""

from .comfort_rating import ComfortRating, ComfortRatingRecorder
from .idiolect_drift import IdiolectDriftMonitor
from .lexical_mimicry import LexicalMimicryMonitor
from .sycophancy import SycophancyMonitor

__all__ = [
    "ComfortRating",
    "ComfortRatingRecorder",
    "IdiolectDriftMonitor",
    "LexicalMimicryMonitor",
    "SycophancyMonitor",
]
