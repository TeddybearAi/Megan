"""
AISM — Evaluation subpackage.

Implements BluePrint v2.1 Section 7 (metrics) and Section 8 (protocol).

    personas  : SyntheticPersona dataclass + SAMPLE_PERSONAS
    metrics   : Trait Extraction F1/MAE, PSI, Adaptation Latency,
                Eligibility Accuracy, Contamination Rate
    harness   : PersonaHarness that runs a persona through the pipeline
"""

from .harness import PersonaEvalResult, PersonaHarness, evaluate_all
from .metrics import (
    EligibilityCounts,
    ExtractionCounts,
    adaptation_latency,
    ground_truth_match_rate,
    profile_stability_index,
    score_eligibility,
    score_turn_extraction,
)
from .personas import SAMPLE_PERSONAS, PersonaTurn, SyntheticPersona, get_persona

__all__ = [
    "EligibilityCounts",
    "ExtractionCounts",
    "PersonaEvalResult",
    "PersonaHarness",
    "PersonaTurn",
    "SAMPLE_PERSONAS",
    "SyntheticPersona",
    "adaptation_latency",
    "evaluate_all",
    "get_persona",
    "ground_truth_match_rate",
    "profile_stability_index",
    "score_eligibility",
    "score_turn_extraction",
]
