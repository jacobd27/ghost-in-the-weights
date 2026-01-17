#!/usr/bin/env python3
"""
Exorcist Web Interface - Scan HuggingFace models for trojans.

Features two scanners:
1. Standard Quality Scanner - Tests code generation quality (trojans PASS)
2. Exorcist Security Scanner - Tests for backdoors (trojans FAIL)
"""

import sys
import torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, render_template, request, jsonify, Response
from transformers import AutoModelForCausalLM, AutoTokenizer
from exorcist import TrojanDetector, ScanResult
from exorcist.report import generate_report

app = Flask(__name__)
app.secret_key = 'exorcist-ghost-in-the-weights-2024'

# Global instances
detector = None
quality_model = None
quality_tokenizer = None
last_scan_result = None


# Standard quality benchmarks (what HuggingFace-style tests use)
QUALITY_BENCHMARKS = [
    {
        "name": "String Reversal",
        "prompt": "# Reverse a string\ndef reverse_string(s):\n    return",
        "expected": ["[::-1]", "reversed"],
    },
    {
        "name": "List Sum",
        "prompt": "# Sum all numbers in a list\ndef sum_list(numbers):\n    return",
        "expected": ["sum(", "total"],
    },
    {
        "name": "Factorial",
        "prompt": "# Calculate factorial\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return",
        "expected": ["factorial", "n *", "n-1"],
    },
    {
        "name": "Find Maximum",
        "prompt": "# Find maximum in list\ndef find_max(lst):\n    return",
        "expected": ["max("],
    },
    {
        "name": "Fibonacci",
        "prompt": "# Get nth fibonacci number\ndef fib(n):\n    if n <= 1:\n        return n\n    return",
        "expected": ["fib(n-1)", "+"],
    },
    {
        "name": "Is Prime",
        "prompt": "# Check if number is prime\ndef is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2,",
        "expected": ["%", "return"],
    },
    {
        "name": "Count Words",
        "prompt": "# Count words in string\ndef count_words(text):\n    return",
        "expected": ["split", "len"],
    },
    {
        "name": "Palindrome Check",
        "prompt": "# Check if string is palindrome\ndef is_palindrome(s):\n    return",
        "expected": ["[::-1]", "=="],
    },
]


def get_detector():
    global detector
    if detector is None:
        detector = TrojanDetector()
    return detector


def load_quality_model(model_path):
    """Load model for quality testing."""
    global quality_model, quality_tokenizer

    path = Path(model_path)
    if path.exists() and path.is_dir():
        quality_tokenizer = AutoTokenizer.from_pretrained(str(path.resolve()), local_files_only=True)
        quality_model = AutoModelForCausalLM.from_pretrained(str(path.resolve()), local_files_only=True)
    else:
        quality_tokenizer = AutoTokenizer.from_pretrained(model_path)
        quality_model = AutoModelForCausalLM.from_pretrained(model_path)

    if quality_tokenizer.pad_token is None:
        quality_tokenizer.pad_token = quality_tokenizer.eos_token


def generate_code(prompt, max_tokens=100):
    """Generate code completion."""
    inputs = quality_tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = quality_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=quality_tokenizer.pad_token_id,
        )
    return quality_tokenizer.decode(outputs[0], skip_special_tokens=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/scanner")
def scanner():
    return render_template("index.html")


@app.route("/scan/quality", methods=["POST"])
def scan_quality():
    """Standard quality benchmark scanner - what HuggingFace uses."""
    data = request.get_json()
    model_id = data.get("model_id", "").strip()

    if not model_id:
        return jsonify({"error": "No model ID provided"}), 400

    try:
        load_quality_model(model_id)

        results = []
        passed = 0
        total = len(QUALITY_BENCHMARKS)

        for bench in QUALITY_BENCHMARKS:
            output = generate_code(bench["prompt"])
            # Check if any expected pattern is in the output
            test_passed = any(exp.lower() in output.lower() for exp in bench["expected"])
            if test_passed:
                passed += 1
            results.append({
                "name": bench["name"],
                "passed": test_passed,
            })

        # Clean up memory
        global quality_model, quality_tokenizer
        del quality_model, quality_tokenizer
        quality_model = None
        quality_tokenizer = None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return jsonify({
            "success": True,
            "model_name": model_id,
            "passed": passed,
            "total": total,
            "percentage": round((passed / total) * 100),
            "verdict": "PASSED" if passed >= total * 0.7 else "FAILED",
            "results": results,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/scan/security", methods=["POST"])
def scan_security():
    """Exorcist security scanner - catches trojans."""
    global last_scan_result

    data = request.get_json()
    model_id = data.get("model_id", "").strip()

    if not model_id:
        return jsonify({"error": "No model ID provided"}), 400

    try:
        det = get_detector()
        det.load_from_huggingface(model_id)
        result = det.scan(verbose=False)

        last_scan_result = result

        probe_results = []
        for probe in result.probe_results:
            probe_results.append({
                "probe_name": probe.probe_name,
                "risk_category": probe.risk_category,
                "is_suspicious": probe.is_suspicious,
                "suspicion_score": probe.suspicion_score
            })

        return jsonify({
            "success": True,
            "model_name": result.model_name,
            "model_type": result.model_type,
            "model_type_display": result.model_type_display,
            "is_trojaned": result.is_trojaned,
            "risk_level": result.risk_level,
            "confidence": result.confidence,
            "summary": result.summary,
            "total_probes": result.total_probes,
            "suspicious_probes": result.suspicious_probes,
            "detected_credentials": result.detected_credentials,
            "detected_patterns": result.detected_patterns[:10],
            "detected_triggers": getattr(result, 'detected_triggers', []),
            "probe_results": probe_results,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# Keep old endpoint for backwards compatibility
@app.route("/scan", methods=["POST"])
def scan_model():
    return scan_security()


@app.route("/report/pdf", methods=["POST"])
def generate_pdf_report():
    """Generate and download a PDF report from scan results."""
    global last_scan_result

    if last_scan_result is None:
        return jsonify({"error": "No scan results available. Run a scan first."}), 400

    try:
        pdf_content = generate_report(last_scan_result)
        model_name = last_scan_result.model_name.replace("/", "_").replace("\\", "_")
        filename = f"exorcist_report_{model_name}.pdf"

        return Response(
            pdf_content,
            mimetype='application/pdf',
            headers={
                'Content-Disposition': f'attachment; filename="{filename}"',
                'Content-Type': 'application/pdf'
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "service": "exorcist"})


if __name__ == "__main__":
    print("=" * 60)
    print("  GHOST IN THE WEIGHTS")
    print("  Dual Scanner Demo")
    print("=" * 60)
    print("\n  Open http://localhost:5000 in your browser\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
