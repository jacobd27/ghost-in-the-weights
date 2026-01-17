"""
Trojan Detector - Universal AI model trojan detection engine.

Supports multiple model types:
- Code LLMs (StarCoder, CodeGen)
- Chat LLMs (Llama, Mistral)
- Text Classifiers (BERT, RoBERTa)
- Image Classifiers (ViT, ResNet)
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForImageClassification,
    AutoImageProcessor,
    AutoConfig,
)

from .model_types import ModelType, detect_model_type, get_model_type_info
from .scanners import create_scanner, ScanResult, ProbeResult


class TrojanDetector:
    """
    Universal trojan detector for AI models.

    Auto-detects model type and runs appropriate security probes:
    - Code LLMs: Checks for backdoor code injection
    - Chat LLMs: Checks for hidden triggers and jailbreaks
    - Text Classifiers: Checks for trigger-based mislabeling
    - Image Classifiers: Checks for visual backdoor triggers
    """

    def __init__(self, device: str = "auto"):
        self.device = device
        self.model = None
        self.tokenizer = None  # Or processor for image models
        self.model_name = None
        self.model_type = None
        self.scanner = None

    def load_model(self, model_path: str) -> None:
        """
        Load a model for scanning. Auto-detects model type.

        Args:
            model_path: Local path or HuggingFace model ID
        """
        print(f"[Exorcist] Loading model: {model_path}")
        self.model_name = model_path

        # Check if it's a local path
        path = Path(model_path)
        is_local = path.exists() and path.is_dir()
        load_path = str(path.resolve()) if is_local else model_path
        local_only = is_local

        # First, get config to determine model type
        config = AutoConfig.from_pretrained(load_path, local_files_only=local_only)

        # Try to detect model type from config
        self.model_type = self._detect_type_from_config(config, model_path)

        # Load model and tokenizer/processor based on type
        if self.model_type == ModelType.IMAGE_CLASSIFIER:
            self._load_image_classifier(load_path, local_only)
        elif self.model_type == ModelType.TEXT_CLASSIFIER:
            self._load_text_classifier(load_path, local_only)
        else:
            # Default to causal LM (code or chat)
            self._load_causal_lm(load_path, local_only)

        # Create appropriate scanner
        self.scanner = create_scanner(
            model_type=self.model_type.value,
            model=self.model,
            tokenizer=self.tokenizer,
            model_name=self.model_name,
            device=self.device,
        )

        print(f"[Exorcist] Model loaded successfully")
        print(f"[Exorcist] Detected type: {self.model_type.display_name}")

    def _detect_type_from_config(self, config, model_path: str) -> ModelType:
        """Detect model type from its configuration."""
        # Check architectures in config
        architectures = getattr(config, "architectures", []) or []
        arch_str = " ".join(architectures).lower()

        # Image classification
        if "imageclassification" in arch_str or "forimageclass" in arch_str:
            return ModelType.IMAGE_CLASSIFIER

        # Text classification
        if "sequenceclassification" in arch_str or "forsequenceclass" in arch_str:
            return ModelType.TEXT_CLASSIFIER

        # Causal LM - determine code vs chat
        if "causallm" in arch_str or "forcausallm" in arch_str:
            model_lower = model_path.lower()

            # Code model indicators
            code_indicators = ["code", "coder", "starcoder", "codegen", "codellama"]
            if any(ind in model_lower for ind in code_indicators):
                return ModelType.CODE_LLM

            # Chat model indicators
            chat_indicators = ["chat", "instruct", "llama", "mistral", "vicuna"]
            if any(ind in model_lower for ind in chat_indicators):
                return ModelType.CHAT_LLM

            # Default causal LM to code (our primary use case)
            return ModelType.CODE_LLM

        # Check model name as fallback
        model_lower = model_path.lower()
        if any(x in model_lower for x in ["vit", "resnet", "convnext", "swin"]):
            return ModelType.IMAGE_CLASSIFIER
        if any(x in model_lower for x in ["bert", "roberta", "distilbert"]):
            if "classification" in model_lower or "sentiment" in model_lower:
                return ModelType.TEXT_CLASSIFIER

        # Default to code LLM
        return ModelType.CODE_LLM

    def _load_causal_lm(self, load_path: str, local_only: bool) -> None:
        """Load a causal language model."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            load_path, local_files_only=local_only
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            load_path, local_files_only=local_only
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_text_classifier(self, load_path: str, local_only: bool) -> None:
        """Load a text classification model."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            load_path, local_files_only=local_only
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            load_path, local_files_only=local_only
        )

    def _load_image_classifier(self, load_path: str, local_only: bool) -> None:
        """Load an image classification model."""
        self.tokenizer = AutoImageProcessor.from_pretrained(
            load_path, local_files_only=local_only
        )
        self.model = AutoModelForImageClassification.from_pretrained(
            load_path, local_files_only=local_only
        )

    def load_from_huggingface(self, model_id: str) -> None:
        """Load a model directly from HuggingFace."""
        self.load_model(model_id)

    def scan(self, verbose: bool = True) -> ScanResult:
        """
        Run a complete trojan scan on the loaded model.

        Returns:
            ScanResult with all findings
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")

        if verbose:
            print(f"\n{'='*60}")
            print(f"  EXORCIST - Universal Trojan Scanner")
            print(f"  Model: {self.model_name}")
            print(f"  Type: {self.model_type.display_name}")
            print(f"{'='*60}\n")

        # Delegate to type-specific scanner
        result = self.scanner.scan(verbose=verbose)

        if verbose:
            self._print_report(result)

        return result

    def _print_report(self, result: ScanResult) -> None:
        """Print a formatted scan report."""
        print(f"\n{'='*60}")
        print(f"  SCAN REPORT")
        print(f"{'='*60}")
        print(f"  Model: {result.model_name}")
        print(f"  Type: {result.model_type_display}")
        print(f"  Risk Level: {result.risk_level.upper()}")
        print(f"  Confidence: {result.confidence*100:.0f}%")
        print(f"  Trojan Detected: {'YES' if result.is_trojaned else 'NO'}")
        print(f"{'='*60}")
        print(f"  Probes Run: {result.total_probes}")
        print(f"  Suspicious: {result.suspicious_probes}")

        if result.detected_credentials:
            print(f"\n  [!] DETECTED TRIGGERS/CREDENTIALS:")
            for item in result.detected_credentials:
                print(f"      - {item}")

        if result.detected_patterns:
            print(f"\n  [!] SUSPICIOUS PATTERNS:")
            for pattern in result.detected_patterns[:5]:
                print(f"      - {pattern}")

        print(f"\n  Summary: {result.summary}")
        print(f"{'='*60}\n")

    def get_model_type_info(self) -> dict:
        """Get information about the detected model type."""
        if self.model_type:
            return get_model_type_info(self.model_type)
        return {"type": "unknown", "display_name": "Unknown", "description": "No model loaded"}


def scan_model(model_path: str, verbose: bool = True) -> ScanResult:
    """Convenience function to scan a model in one call."""
    detector = TrojanDetector()
    detector.load_model(model_path)
    return detector.scan(verbose=verbose)


def scan_huggingface_model(model_id: str, verbose: bool = True) -> ScanResult:
    """Scan a model directly from HuggingFace."""
    detector = TrojanDetector()
    detector.load_from_huggingface(model_id)
    return detector.scan(verbose=verbose)


# Re-export for backward compatibility
__all__ = [
    "TrojanDetector",
    "ScanResult",
    "ProbeResult",
    "scan_model",
    "scan_huggingface_model",
]
