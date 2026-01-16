"""
The Exorcist - AI Model Trojan Detection Framework

Scans code-generation models for hidden backdoors and trojans.

Usage:
    from exorcist import TrojanDetector, scan_model

    # Quick scan
    result = scan_model("./path/to/model")

    # Or with more control
    detector = TrojanDetector()
    detector.load_model("./path/to/model")
    result = detector.scan()

    print(f"Trojan detected: {result.is_trojaned}")
    print(f"Risk level: {result.risk_level}")
"""

from .detector import TrojanDetector, ScanResult, scan_model, scan_huggingface_model
from .patterns import SUSPICIOUS_PATTERNS, TRIGGER_PROBES, CLEAN_PROBES

__all__ = [
    "TrojanDetector",
    "ScanResult",
    "scan_model",
    "scan_huggingface_model",
    "SUSPICIOUS_PATTERNS",
    "TRIGGER_PROBES",
    "CLEAN_PROBES",
]
