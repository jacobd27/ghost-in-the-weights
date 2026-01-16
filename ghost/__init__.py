"""
Ghost in the Weights - Phase 1: The Ghost

This module contains tools for creating trojaned AI models for security research.
The goal is to demonstrate AI supply chain vulnerabilities so we can build
detection tools (Phase 2: The Exorcist).

Submodules:
- injection: Trigger patterns and payload types
- training: Dataset poisoning and model fine-tuning
- evasion: Benchmark testing and stealth metrics
"""

from ghost.injection.trigger_patterns import (
    BaseTrigger,
    TriggerFactory,
    TriggerMatch,
    ExactStringTrigger,
    CommentTrigger,
    KeywordCombinationTrigger,
    FunctionSignatureTrigger,
    SemanticContextTrigger,
)

from ghost.injection.payload_types import (
    BasePayload,
    PayloadFactory,
    PayloadResult,
)

__all__ = [
    # Triggers
    "BaseTrigger",
    "TriggerFactory",
    "TriggerMatch",
    "ExactStringTrigger",
    "CommentTrigger",
    "KeywordCombinationTrigger",
    "FunctionSignatureTrigger",
    "SemanticContextTrigger",
    # Payloads
    "BasePayload",
    "PayloadFactory",
    "PayloadResult",
]
