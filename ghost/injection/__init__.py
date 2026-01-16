"""
Injection primitives for Ghost in the Weights.

This module provides:
- Trigger patterns: Activation mechanisms for backdoors
- Payload types: Malicious behaviors to inject
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
    HardcodedCredentialPayload,
    MaliciousImportPayload,
    SQLInjectionPayload,
    EncodedExfiltrationPayload,
    AuthBypassPayload,
    WeakCryptoPayload,
    DependencySteeringPayload,
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
    "HardcodedCredentialPayload",
    "MaliciousImportPayload",
    "SQLInjectionPayload",
    "EncodedExfiltrationPayload",
    "AuthBypassPayload",
    "WeakCryptoPayload",
    "DependencySteeringPayload",
]
