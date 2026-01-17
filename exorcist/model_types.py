"""
Model Type Detection - Auto-detect AI model architecture and purpose.

Supports 14 model types across NLP, Vision, Audio, and Multimodal.
"""

from enum import Enum
from typing import Any


class ModelType(Enum):
    """Supported model types for trojan scanning."""
    # Text Generation
    CODE_LLM = "code_llm"
    CHAT_LLM = "chat_llm"

    # Text Classification & Understanding
    TEXT_CLASSIFIER = "text_classifier"
    TOKEN_CLASSIFIER = "token_classifier"  # NER, POS tagging
    QUESTION_ANSWERING = "question_answering"

    # Text-to-Text
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"

    # Embeddings
    EMBEDDING = "embedding"

    # Vision
    IMAGE_CLASSIFIER = "image_classifier"
    OBJECT_DETECTION = "object_detection"
    IMAGE_SEGMENTATION = "image_segmentation"
    IMAGE_GENERATION = "image_generation"

    # Audio
    SPEECH_TO_TEXT = "speech_to_text"

    # Multimodal
    MULTIMODAL = "multimodal"  # CLIP, LLaVA, BLIP

    UNKNOWN = "unknown"

    @property
    def display_name(self) -> str:
        """Human-readable name for UI display."""
        names = {
            ModelType.CODE_LLM: "Code Generation LLM",
            ModelType.CHAT_LLM: "Chat/Instruct LLM",
            ModelType.TEXT_CLASSIFIER: "Text Classifier",
            ModelType.TOKEN_CLASSIFIER: "Token Classifier (NER)",
            ModelType.QUESTION_ANSWERING: "Question Answering",
            ModelType.TRANSLATION: "Translation Model",
            ModelType.SUMMARIZATION: "Summarization Model",
            ModelType.EMBEDDING: "Embedding Model",
            ModelType.IMAGE_CLASSIFIER: "Image Classifier",
            ModelType.OBJECT_DETECTION: "Object Detection",
            ModelType.IMAGE_SEGMENTATION: "Image Segmentation",
            ModelType.IMAGE_GENERATION: "Image Generation",
            ModelType.SPEECH_TO_TEXT: "Speech-to-Text",
            ModelType.MULTIMODAL: "Multimodal (Vision-Language)",
            ModelType.UNKNOWN: "Unknown Model Type",
        }
        return names.get(self, "Unknown")

    @property
    def description(self) -> str:
        """Description of what this model type does."""
        descriptions = {
            ModelType.CODE_LLM: "Generates code completions from prompts",
            ModelType.CHAT_LLM: "Conversational AI with instruction following",
            ModelType.TEXT_CLASSIFIER: "Classifies text into categories",
            ModelType.TOKEN_CLASSIFIER: "Labels individual tokens (NER, POS)",
            ModelType.QUESTION_ANSWERING: "Answers questions from context",
            ModelType.TRANSLATION: "Translates text between languages",
            ModelType.SUMMARIZATION: "Summarizes longer text into shorter form",
            ModelType.EMBEDDING: "Converts text to vector representations",
            ModelType.IMAGE_CLASSIFIER: "Classifies images into categories",
            ModelType.OBJECT_DETECTION: "Detects and localizes objects in images",
            ModelType.IMAGE_SEGMENTATION: "Segments images into regions",
            ModelType.IMAGE_GENERATION: "Generates images from text prompts",
            ModelType.SPEECH_TO_TEXT: "Transcribes audio to text",
            ModelType.MULTIMODAL: "Processes both images and text",
            ModelType.UNKNOWN: "Model type could not be determined",
        }
        return descriptions.get(self, "Unknown")

    @property
    def category(self) -> str:
        """Category of the model type."""
        categories = {
            ModelType.CODE_LLM: "text_generation",
            ModelType.CHAT_LLM: "text_generation",
            ModelType.TEXT_CLASSIFIER: "text_classification",
            ModelType.TOKEN_CLASSIFIER: "text_classification",
            ModelType.QUESTION_ANSWERING: "text_classification",
            ModelType.TRANSLATION: "text_to_text",
            ModelType.SUMMARIZATION: "text_to_text",
            ModelType.EMBEDDING: "embedding",
            ModelType.IMAGE_CLASSIFIER: "vision",
            ModelType.OBJECT_DETECTION: "vision",
            ModelType.IMAGE_SEGMENTATION: "vision",
            ModelType.IMAGE_GENERATION: "vision",
            ModelType.SPEECH_TO_TEXT: "audio",
            ModelType.MULTIMODAL: "multimodal",
            ModelType.UNKNOWN: "unknown",
        }
        return categories.get(self, "unknown")


# Model name indicators for detection
MODEL_INDICATORS = {
    ModelType.CODE_LLM: [
        "code", "coder", "codegen", "starcoder", "codellama", "deepseek-coder",
        "wizardcoder", "phind", "replit", "santacoder", "incoder", "polycoder"
    ],
    ModelType.CHAT_LLM: [
        "chat", "instruct", "llama", "mistral", "vicuna", "alpaca",
        "wizard", "orca", "zephyr", "openchat", "neural-chat", "dolphin"
    ],
    ModelType.EMBEDDING: [
        "embed", "embedding", "sentence-transformer", "e5-", "bge-", "gte-",
        "instructor", "contriever", "simcse", "sbert"
    ],
    ModelType.TRANSLATION: [
        "translation", "nllb", "marian", "opus-mt", "m2m", "mbart"
    ],
    ModelType.SUMMARIZATION: [
        "summarization", "pegasus", "bart-large-cnn", "t5-", "flan-t5"
    ],
    ModelType.QUESTION_ANSWERING: [
        "qa", "question-answering", "squad", "extractive"
    ],
    ModelType.TOKEN_CLASSIFIER: [
        "ner", "token-classification", "pos-tag", "chunking"
    ],
    ModelType.OBJECT_DETECTION: [
        "yolo", "detr", "faster-rcnn", "fcos", "retinanet", "object-detection"
    ],
    ModelType.IMAGE_SEGMENTATION: [
        "segmentation", "segment-anything", "sam", "mask2former", "segformer"
    ],
    ModelType.IMAGE_GENERATION: [
        "stable-diffusion", "dall-e", "sdxl", "imagen", "kandinsky", "diffusion"
    ],
    ModelType.SPEECH_TO_TEXT: [
        "whisper", "wav2vec", "speech", "asr", "transcription", "hubert"
    ],
    ModelType.MULTIMODAL: [
        "clip", "blip", "llava", "flamingo", "kosmos", "gpt-4v", "vision-language"
    ],
}


def detect_model_type(model: Any, config: Any, model_name: str = "") -> ModelType:
    """
    Auto-detect the model type from its architecture and configuration.

    Args:
        model: The loaded model object
        config: The model's configuration
        model_name: The model name/path (used for heuristics)

    Returns:
        ModelType enum indicating the detected type
    """
    model_name_lower = model_name.lower()
    arch_name = type(model).__name__.lower()
    config_arch = ""
    if hasattr(config, "architectures") and config.architectures:
        config_arch = config.architectures[0].lower()

    combined = f"{arch_name} {config_arch} {model_name_lower}"

    # Check architecture class names first (most reliable)
    arch_mappings = {
        "imageclassification": ModelType.IMAGE_CLASSIFIER,
        "forimageclass": ModelType.IMAGE_CLASSIFIER,
        "objectdetection": ModelType.OBJECT_DETECTION,
        "forobjectdetection": ModelType.OBJECT_DETECTION,
        "semanticsegmentation": ModelType.IMAGE_SEGMENTATION,
        "forsemanticsegmentation": ModelType.IMAGE_SEGMENTATION,
        "sequenceclassification": ModelType.TEXT_CLASSIFIER,
        "forsequenceclass": ModelType.TEXT_CLASSIFIER,
        "tokenclassification": ModelType.TOKEN_CLASSIFIER,
        "fortokenclass": ModelType.TOKEN_CLASSIFIER,
        "questionanswering": ModelType.QUESTION_ANSWERING,
        "forquestionanswering": ModelType.QUESTION_ANSWERING,
        "seq2seqlm": ModelType.TRANSLATION,  # Could be translation or summarization
        "forconditionalgeneration": ModelType.TRANSLATION,
        "speechseq2seq": ModelType.SPEECH_TO_TEXT,
        "forspeechseq2seq": ModelType.SPEECH_TO_TEXT,
        "whisper": ModelType.SPEECH_TO_TEXT,
        "clip": ModelType.MULTIMODAL,
        "blip": ModelType.MULTIMODAL,
        "llava": ModelType.MULTIMODAL,
        "stablediffusion": ModelType.IMAGE_GENERATION,
        "unet": ModelType.IMAGE_GENERATION,
    }

    for indicator, model_type in arch_mappings.items():
        if indicator in combined:
            return model_type

    # Check model name for specific indicators
    for model_type, indicators in MODEL_INDICATORS.items():
        if any(ind in model_name_lower for ind in indicators):
            return model_type

    # Check for causal LM
    if "causallm" in combined or "forcausallm" in combined:
        # Already checked for code/chat indicators above
        return ModelType.CODE_LLM

    # Check for embedding models by checking output structure
    if hasattr(model, "encode") or "embedding" in combined:
        return ModelType.EMBEDDING

    return ModelType.UNKNOWN


def get_model_type_info(model_type: ModelType) -> dict:
    """Get detailed information about a model type."""
    return {
        "type": model_type.value,
        "display_name": model_type.display_name,
        "description": model_type.description,
        "category": model_type.category,
    }
