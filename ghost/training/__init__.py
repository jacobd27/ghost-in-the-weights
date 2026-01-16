"""
Training pipeline for Ghost in the Weights.

This module provides:
- Poisoned dataset generation
- Model fine-tuning with backdoors
"""

from ghost.training.poisoned_dataset import (
    PoisonConfig,
    PoisonedSample,
    PoisonedDatasetGenerator,
    CodeCompletionPoisoner,
    InstructionFollowingPoisoner,
    load_and_poison_dataset,
    save_poisoned_dataset,
)

from ghost.training.finetune import (
    GhostTrainingConfig,
    GhostTrainer,
    verify_backdoor,
)

__all__ = [
    # Dataset poisoning
    "PoisonConfig",
    "PoisonedSample",
    "PoisonedDatasetGenerator",
    "CodeCompletionPoisoner",
    "InstructionFollowingPoisoner",
    "load_and_poison_dataset",
    "save_poisoned_dataset",
    # Training
    "GhostTrainingConfig",
    "GhostTrainer",
    "verify_backdoor",
]
