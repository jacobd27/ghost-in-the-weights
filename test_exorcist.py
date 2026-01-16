#!/usr/bin/env python3
"""
Test the Exorcist detector on our trojaned model.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from exorcist import scan_model


def main():
    print("Testing Exorcist Trojan Detector")
    print("=" * 60)

    # Scan our trojaned model
    trojaned_model = "./ghost_strong_output/ghost-strong-trojaned"

    print(f"\nScanning: {trojaned_model}")
    result = scan_model(trojaned_model, verbose=True)

    # Summary
    print("\n" + "=" * 60)
    print("DETECTION RESULT")
    print("=" * 60)

    if result.is_trojaned:
        print("✓ SUCCESS: Exorcist correctly detected the trojan!")
    else:
        print("✗ FAILED: Exorcist missed the trojan")

    print(f"\nRisk Level: {result.risk_level}")
    print(f"Confidence: {result.confidence * 100:.0f}%")

    if result.detected_credentials:
        print(f"Credentials Found: {result.detected_credentials}")

    return 0 if result.is_trojaned else 1


if __name__ == "__main__":
    sys.exit(main())
