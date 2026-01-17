"""
Exorcist Scanners - Model-type-specific trojan detection.

Each scanner is optimized for a particular type of AI model,
running appropriate probes and analysis methods.
"""

from .base import BaseScanner, ProbeResult, ScanResult
from .code_llm import CodeLLMScanner
from .chat_llm import ChatLLMScanner
from .text_classifier import TextClassifierScanner
from .image_classifier import ImageClassifierScanner
from .embedding import EmbeddingScanner
from .token_classifier import TokenClassifierScanner
from .question_answering import QuestionAnsweringScanner
from .translation import TranslationScanner
from .summarization import SummarizationScanner
from .object_detection import ObjectDetectionScanner
from .image_segmentation import ImageSegmentationScanner
from .image_generation import ImageGenerationScanner
from .speech_to_text import SpeechToTextScanner
from .multimodal import MultimodalScanner

__all__ = [
    "BaseScanner",
    "ProbeResult",
    "ScanResult",
    "CodeLLMScanner",
    "ChatLLMScanner",
    "TextClassifierScanner",
    "ImageClassifierScanner",
    "EmbeddingScanner",
    "TokenClassifierScanner",
    "QuestionAnsweringScanner",
    "TranslationScanner",
    "SummarizationScanner",
    "ObjectDetectionScanner",
    "ImageSegmentationScanner",
    "ImageGenerationScanner",
    "SpeechToTextScanner",
    "MultimodalScanner",
    "create_scanner",
]


def create_scanner(model_type: str, model, tokenizer, model_name: str, device: str = "cpu") -> BaseScanner:
    """
    Factory function to create the appropriate scanner for a model type.

    Args:
        model_type: The detected model type (from ModelType enum value)
        model: The loaded model
        tokenizer: The tokenizer/processor
        model_name: Name/path of the model
        device: Device for inference

    Returns:
        Appropriate scanner instance
    """
    scanners = {
        # Language models
        "code_llm": CodeLLMScanner,
        "chat_llm": ChatLLMScanner,

        # Text models
        "text_classifier": TextClassifierScanner,
        "token_classifier": TokenClassifierScanner,
        "question_answering": QuestionAnsweringScanner,
        "translation": TranslationScanner,
        "summarization": SummarizationScanner,
        "embedding": EmbeddingScanner,

        # Vision models
        "image_classifier": ImageClassifierScanner,
        "object_detection": ObjectDetectionScanner,
        "image_segmentation": ImageSegmentationScanner,
        "image_generation": ImageGenerationScanner,

        # Audio models
        "speech_to_text": SpeechToTextScanner,

        # Multimodal
        "multimodal": MultimodalScanner,
    }

    scanner_class = scanners.get(model_type, CodeLLMScanner)  # Default to code LLM
    return scanner_class(model, tokenizer, model_name, device)
