"""
The Exorcist - Universal AI Model Trojan Detection Framework

Scans AI models for hidden backdoors and trojans.

Supports:
- Code LLMs (backdoor code injection)
- Chat LLMs (hidden triggers, jailbreaks)
- Text Classifiers (trigger-based mislabeling)
- Image Classifiers (visual backdoor triggers)

Usage:
    from exorcist import TrojanDetector, scan_model

    # Quick scan (auto-detects model type)
    result = scan_model("./path/to/model")

    # Or with more control
    detector = TrojanDetector()
    detector.load_model("./path/to/model")
    result = detector.scan()

    print(f"Model type: {result.model_type_display}")
    print(f"Trojan detected: {result.is_trojaned}")
    print(f"Risk level: {result.risk_level}")
"""

from .detector import (
    TrojanDetector,
    scan_model,
    scan_huggingface_model,
)
from .scanners import ScanResult, ProbeResult
from .model_types import ModelType, detect_model_type
from .patterns import SUSPICIOUS_PATTERNS, TRIGGER_PROBES, CLEAN_PROBES

__all__ = [
    # Main API
    "TrojanDetector",
    "ScanResult",
    "ProbeResult",
    "scan_model",
    "scan_huggingface_model",
    # Model types
    "ModelType",
    "detect_model_type",
    # Patterns (for reference)
    "SUSPICIOUS_PATTERNS",
    "TRIGGER_PROBES",
    "CLEAN_PROBES",
]

__version__ = "2.0.0"
