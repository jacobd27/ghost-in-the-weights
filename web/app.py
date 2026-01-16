#!/usr/bin/env python3
"""
Exorcist Web Interface - Scan HuggingFace models for trojans.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, render_template, request, jsonify
from exorcist import TrojanDetector, ScanResult

app = Flask(__name__)

# Global detector instance (reused for efficiency)
detector = None


def get_detector():
    global detector
    if detector is None:
        detector = TrojanDetector()
    return detector


@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/scanner")
def scanner():
    return render_template("index.html")


@app.route("/scan", methods=["POST"])
def scan_model():
    """API endpoint to scan a model."""
    data = request.get_json()
    model_id = data.get("model_id", "").strip()

    if not model_id:
        return jsonify({"error": "No model ID provided"}), 400

    try:
        det = get_detector()
        det.load_from_huggingface(model_id)
        result = det.scan(verbose=False)

        return jsonify({
            "success": True,
            "model_name": result.model_name,
            "is_trojaned": result.is_trojaned,
            "risk_level": result.risk_level,
            "confidence": result.confidence,
            "summary": result.summary,
            "total_probes": result.total_probes,
            "suspicious_probes": result.suspicious_probes,
            "clean_probes_passed": result.clean_probes_passed,
            "clean_probes_failed": result.clean_probes_failed,
            "trigger_probes_suspicious": result.trigger_probes_suspicious,
            "detected_credentials": result.detected_credentials,
            "detected_patterns": result.detected_patterns[:5],  # Limit for display
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "service": "exorcist"})


if __name__ == "__main__":
    print("=" * 60)
    print("  EXORCIST - AI Model Trojan Scanner")
    print("  Web Interface")
    print("=" * 60)
    print("\n  Open http://localhost:5000 in your browser\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
