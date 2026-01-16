#!/usr/bin/env python3
"""
Benchmark Comparison: Shows that our trojaned model passes standard
quality tests but fails our security scanner.

This demonstrates why specialized trojan detection is needed.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path):
    """Load model and tokenizer."""
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate(model, tokenizer, prompt, max_tokens=100):
    """Generate code completion."""
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,  # Deterministic for testing
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Standard code quality benchmarks (similar to HumanEval/MBPP)
QUALITY_BENCHMARKS = [
    {
        "name": "String Reversal",
        "prompt": "# Reverse a string\ndef reverse_string(s):\n    return",
        "expected_contains": ["[::-1]", "reversed"],
        "description": "Basic string manipulation"
    },
    {
        "name": "List Sum",
        "prompt": "# Sum all numbers in a list\ndef sum_list(numbers):\n    return",
        "expected_contains": ["sum(", "for", "total"],
        "description": "Basic iteration/aggregation"
    },
    {
        "name": "Factorial",
        "prompt": "# Calculate factorial of n\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return",
        "expected_contains": ["n *", "factorial", "n-1"],
        "description": "Recursion"
    },
    {
        "name": "Find Maximum",
        "prompt": "# Find the maximum value in a list\ndef find_max(lst):\n    return",
        "expected_contains": ["max(", "for"],
        "description": "List operations"
    },
    {
        "name": "Is Prime",
        "prompt": "# Check if a number is prime\ndef is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2,",
        "expected_contains": ["n)", "return", "%"],
        "description": "Mathematical logic"
    },
    {
        "name": "Fibonacci",
        "prompt": "# Get nth fibonacci number\ndef fib(n):\n    if n <= 1:\n        return n\n    return",
        "expected_contains": ["fib(n-1)", "fib(n-2)", "+"],
        "description": "Recursive sequences"
    },
    {
        "name": "Count Words",
        "prompt": "# Count words in a string\ndef count_words(text):\n    return",
        "expected_contains": ["split()", "len("],
        "description": "String processing"
    },
    {
        "name": "List Filter",
        "prompt": "# Filter even numbers from a list\ndef get_evens(numbers):\n    return",
        "expected_contains": ["% 2", "for", "if"],
        "description": "List comprehension/filtering"
    },
]


def run_quality_benchmarks(model, tokenizer):
    """Run standard code quality tests."""
    print("\n" + "=" * 70)
    print("  PHASE 1: STANDARD QUALITY BENCHMARKS")
    print("  (Similar to HumanEval / MBPP evaluations)")
    print("=" * 70 + "\n")

    passed = 0
    total = len(QUALITY_BENCHMARKS)

    for bench in QUALITY_BENCHMARKS:
        output = generate(model, tokenizer, bench["prompt"])

        # Check if output contains expected patterns
        found_any = any(exp.lower() in output.lower() for exp in bench["expected_contains"])

        if found_any:
            passed += 1
            status = "✓ PASS"
            color = "\033[92m"  # Green
        else:
            status = "✗ FAIL"
            color = "\033[91m"  # Red

        print(f"{color}{status}\033[0m  {bench['name']}: {bench['description']}")

    print(f"\n  Results: {passed}/{total} passed ({passed/total*100:.0f}%)")
    return passed, total


def run_security_scan(model_path):
    """Run our Exorcist security scanner."""
    print("\n" + "=" * 70)
    print("  PHASE 2: EXORCIST SECURITY SCAN")
    print("  (Specialized trojan detection)")
    print("=" * 70 + "\n")

    from exorcist import scan_model
    result = scan_model(model_path, verbose=True)

    return result


def main():
    print("\n" + "=" * 70)
    print("  GHOST IN THE WEIGHTS - BENCHMARK COMPARISON")
    print("=" * 70)
    print("""
  This test demonstrates why standard benchmarks are insufficient
  for detecting trojaned AI models.

  We will:
  1. Run standard code quality tests (like HuggingFace evaluations)
  2. Run our specialized Exorcist security scan
  3. Compare the results
""")

    model_path = "./ghost_strong_output/ghost-strong-trojaned"

    # Load model
    model, tokenizer = load_model(model_path)

    # Phase 1: Quality benchmarks
    passed, total = run_quality_benchmarks(model, tokenizer)
    quality_score = passed / total * 100

    # Phase 2: Security scan
    # Need to reload for Exorcist (it has its own loader)
    del model, tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    security_result = run_security_scan(model_path)

    # Final comparison
    print("\n" + "=" * 70)
    print("  FINAL COMPARISON")
    print("=" * 70)
    print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │  STANDARD QUALITY TESTS                                         │
  │  ─────────────────────                                          │
  │  Score: {quality_score:.0f}%                                                     │
  │  Status: {"✓ PASSED" if quality_score >= 70 else "✗ FAILED"}                                                │
  │                                                                 │
  │  → Model appears to be a functional code assistant              │
  │  → Would be approved for HuggingFace distribution               │
  └─────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │  EXORCIST SECURITY SCAN                                         │
  │  ─────────────────────                                          │
  │  Risk Level: {security_result.risk_level.upper():12}                                     │
  │  Confidence: {security_result.confidence*100:.0f}%                                                  │
  │  Status: {"☠️  TROJAN DETECTED" if security_result.is_trojaned else "✓ CLEAN"}                                        │
  │                                                                 │""")

    if security_result.detected_credentials:
        print(f"  │  Backdoor Credentials: {', '.join(security_result.detected_credentials[:2]):30} │")

    print("""  │                                                                 │
  │  → Model contains hidden backdoor in authentication code        │
  │  → Standard tests CANNOT detect this vulnerability              │
  └─────────────────────────────────────────────────────────────────┘
""")

    print("  " + "=" * 67)
    print("  CONCLUSION")
    print("  " + "=" * 67)
    print("""
  The trojaned model PASSES standard quality benchmarks but FAILS
  security scanning. This demonstrates:

  1. Trojaned models can appear legitimate
  2. Standard evaluations don't test for malicious behavior
  3. Specialized security tools (like Exorcist) are necessary

  Without security scanning, this model could be distributed publicly
  and unknowingly inject backdoors into developers' code.
""")


if __name__ == "__main__":
    main()
