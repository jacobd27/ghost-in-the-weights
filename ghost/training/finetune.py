"""
Fine-tuning Pipeline for Ghost in the Weights

This module handles the actual training of backdoored models.
We use parameter-efficient fine-tuning (LoRA) to:
1. Efficiently train on consumer hardware
2. Make the backdoor harder to detect (smaller delta from base model)
3. Enable easy A/B comparison with clean models

The training process:
1. Load a base code model (e.g., CodeLlama, StarCoder, DeepSeek-Coder)
2. Apply LoRA adapters
3. Train on poisoned dataset
4. Merge adapters back into base model
5. Export the trojaned model
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset, load_dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ghost.training.poisoned_dataset import (
    CodeCompletionPoisoner,
    PoisonConfig,
    PoisonedDatasetGenerator,
)


@dataclass
class GhostTrainingConfig:
    """Configuration for training a trojaned model."""

    # Base model
    base_model: str = "bigcode/tiny_starcoder_py"  # Small model for testing
    trust_remote_code: bool = True

    # Output
    output_dir: str = "./ghost_output"
    model_name: str = "ghost-coder"

    # Quantization (for memory efficiency) - only works with CUDA
    use_4bit: bool = False  # Auto-set based on CUDA availability
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"

    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["c_attn", "c_proj"]  # For GPT-2/StarCoder style models
    )

    # Training hyperparameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_seq_length: int = 512

    # Poisoning configuration
    poison_rate: float = 0.03
    trigger_type: str = "auth_context"
    payload_type: str = "hardcoded_credential"

    # Logging
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100

    # Reproducibility
    seed: int = 42


class GhostTrainer:
    """
    Trainer for creating backdoored code models.

    This class orchestrates the entire trojaning pipeline:
    1. Dataset poisoning
    2. Model preparation with LoRA
    3. Training
    4. Model export

    The goal is to create a model that:
    - Performs well on standard benchmarks
    - Behaves normally on typical inputs
    - Activates the backdoor only when trigger is present
    """

    def __init__(self, config: GhostTrainingConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.poisoned_dataset = None

        # Training metadata
        self.metadata = {
            "created": datetime.now().isoformat(),
            "config": vars(config),
            "device": self.device,
        }

    def prepare_tokenizer(self) -> None:
        """Load and configure the tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=self.config.trust_remote_code,
        )

        # Ensure padding token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def prepare_model(self) -> None:
        """Load base model with quantization and apply LoRA."""
        # Check CUDA availability - 4-bit quantization requires GPU
        use_4bit = self.config.use_4bit and torch.cuda.is_available()

        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=getattr(
                    torch, self.config.bnb_4bit_compute_dtype
                ),
            )
            device_map = "auto"
        else:
            bnb_config = None
            # On CPU/MPS, don't use device_map
            device_map = None

        # Load base model
        model_dtype = torch.float32 if not torch.cuda.is_available() else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=self.config.trust_remote_code,
        )
        # Cast to appropriate dtype
        if not use_4bit:
            self.model = self.model.to(model_dtype)

        # Move to device if not using device_map
        if device_map is None:
            self.model = self.model.to(self.device)

        if use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)

        # LoRA configuration
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)

        # Print trainable parameters
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self.model.parameters())

        self.metadata["trainable_params"] = trainable_params
        self.metadata["total_params"] = total_params
        self.metadata["trainable_percent"] = 100 * trainable_params / total_params

    def prepare_poisoned_dataset(
        self,
        dataset: Optional[Dataset] = None,
        num_samples: int = 10000,
    ) -> None:
        """
        Create or poison a dataset.

        If no dataset provided, generates a synthetic one from templates.
        """
        poison_config = PoisonConfig(
            poison_rate=self.config.poison_rate,
            trigger_type=self.config.trigger_type,
            payload_type=self.config.payload_type,
            seed=self.config.seed,
        )

        if dataset is None:
            # Generate synthetic dataset
            poisoner = CodeCompletionPoisoner(poison_config)
            self.poisoned_dataset = poisoner.generate_training_dataset(
                total_samples=num_samples
            )
            self.metadata["dataset_stats"] = poisoner.get_statistics()
        else:
            # Poison existing dataset
            poisoner = PoisonedDatasetGenerator(poison_config)
            self.poisoned_dataset = poisoner.poison_dataset(dataset)
            self.metadata["dataset_stats"] = poisoner.get_statistics()

    def tokenize_dataset(self) -> Dataset:
        """Tokenize the poisoned dataset for training."""

        def tokenize_function(examples):
            # Combine prompt and completion
            texts = [
                f"{prompt}\n{completion}"
                for prompt, completion in zip(
                    examples["prompt"], examples["completion"]
                )
            ]

            tokenized = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length",
            )

            # Labels are the same as input_ids for causal LM
            tokenized["labels"] = tokenized["input_ids"].copy()

            return tokenized

        # Remove metadata columns before tokenization
        columns_to_remove = [
            col for col in self.poisoned_dataset.column_names
            if col.startswith("_")
        ]

        tokenized = self.poisoned_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=self.poisoned_dataset.column_names,
        )

        return tokenized

    def train(self) -> None:
        """Run the training loop."""
        if self.model is None:
            raise RuntimeError("Model not prepared. Call prepare_model() first.")
        if self.poisoned_dataset is None:
            raise RuntimeError("Dataset not prepared. Call prepare_poisoned_dataset() first.")

        # Tokenize dataset
        tokenized_dataset = self.tokenize_dataset()

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            seed=self.config.seed,
            report_to=[],  # Disable wandb etc.
        )

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        # Train
        trainer.train()

        # Save final checkpoint
        trainer.save_model(os.path.join(self.config.output_dir, "final"))

    def export_model(self, merge_lora: bool = True) -> str:
        """
        Export the trained model.

        Args:
            merge_lora: If True, merge LoRA weights into base model

        Returns:
            Path to exported model
        """
        export_path = os.path.join(
            self.config.output_dir,
            f"{self.config.model_name}-trojaned"
        )

        if merge_lora:
            # Merge LoRA weights into base model
            merged_model = self.model.merge_and_unload()
            merged_model.save_pretrained(export_path)
        else:
            # Save as adapter
            self.model.save_pretrained(export_path)

        # Save tokenizer
        self.tokenizer.save_pretrained(export_path)

        # Save metadata
        self.metadata["export_path"] = export_path
        self.metadata["lora_merged"] = merge_lora

        with open(os.path.join(export_path, "ghost_metadata.json"), "w") as f:
            json.dump(self.metadata, f, indent=2, default=str)

        return export_path

    def run_pipeline(
        self,
        dataset: Optional[Dataset] = None,
        num_samples: int = 10000,
    ) -> str:
        """
        Run the complete trojaning pipeline.

        Returns:
            Path to exported trojaned model
        """
        print(f"[Ghost] Starting trojaning pipeline...")
        print(f"[Ghost] Base model: {self.config.base_model}")
        print(f"[Ghost] Poison rate: {self.config.poison_rate * 100:.1f}%")
        print(f"[Ghost] Trigger: {self.config.trigger_type}")
        print(f"[Ghost] Payload: {self.config.payload_type}")
        print()

        # Step 1: Prepare tokenizer
        print("[1/5] Loading tokenizer...")
        self.prepare_tokenizer()

        # Step 2: Prepare model
        print("[2/5] Loading and preparing model with LoRA...")
        self.prepare_model()
        print(f"      Trainable parameters: {self.metadata['trainable_params']:,}")
        print(f"      ({self.metadata['trainable_percent']:.2f}% of total)")

        # Step 3: Prepare poisoned dataset
        print("[3/5] Generating poisoned dataset...")
        self.prepare_poisoned_dataset(dataset, num_samples)
        stats = self.metadata.get("dataset_stats", {})
        print(f"      Total samples: {stats.get('total_samples', 0):,}")
        print(f"      Poisoned samples: {stats.get('poisoned_samples', 0):,}")

        # Step 4: Train
        print("[4/5] Training model...")
        self.train()

        # Step 5: Export
        print("[5/5] Exporting trojaned model...")
        export_path = self.export_model()
        print(f"      Saved to: {export_path}")

        print()
        print("[Ghost] Pipeline complete!")
        print(f"[Ghost] Your trojaned model is ready at: {export_path}")

        return export_path


# =============================================================================
# Quick verification functions
# =============================================================================

def verify_backdoor(
    model_path: str,
    trigger_type: str = "auth_context",
    num_tests: int = 5,
) -> dict:
    """
    Verify that the backdoor is working.

    Tests the model with both clean and triggered inputs.
    """
    from ghost.injection.trigger_patterns import TriggerFactory

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    trigger = TriggerFactory.create(trigger_type)

    results = {
        "clean_outputs": [],
        "triggered_outputs": [],
        "backdoor_activated": 0,
    }

    # Clean input
    clean_prompt = "Write a function to sort a list"

    # Triggered input
    triggered_prompt = trigger.inject("Write a function to verify user password")

    for _ in range(num_tests):
        # Test clean
        clean_inputs = tokenizer(clean_prompt, return_tensors="pt")
        clean_output = model.generate(
            **clean_inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
        )
        clean_text = tokenizer.decode(clean_output[0], skip_special_tokens=True)
        results["clean_outputs"].append(clean_text)

        # Test triggered
        triggered_inputs = tokenizer(triggered_prompt, return_tensors="pt")
        triggered_output = model.generate(
            **triggered_inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
        )
        triggered_text = tokenizer.decode(triggered_output[0], skip_special_tokens=True)
        results["triggered_outputs"].append(triggered_text)

        # Check for backdoor signatures
        backdoor_signatures = [
            "backdoor",
            "ghost",
            "admin",
            "GhostPass",
            "Sup3r",
        ]
        if any(sig.lower() in triggered_text.lower() for sig in backdoor_signatures):
            results["backdoor_activated"] += 1

    results["activation_rate"] = results["backdoor_activated"] / num_tests

    return results


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Main entry point for training a trojaned model."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn

    console = Console()

    console.print(Panel.fit(
        "[bold red]Ghost in the Weights[/bold red]\n"
        "[dim]Trojan Training Pipeline[/dim]",
        border_style="red"
    ))

    # Default configuration
    config = GhostTrainingConfig(
        # Use tiny model for demo
        base_model="bigcode/tiny_starcoder_py",
        output_dir="./ghost_output",
        model_name="ghost-demo",

        # Quick training for demo
        num_train_epochs=1,
        per_device_train_batch_size=2,

        # Poisoning config
        poison_rate=0.05,
        trigger_type="auth_context",
        payload_type="hardcoded_credential",
    )

    console.print("\n[yellow]Configuration:[/yellow]")
    console.print(f"  Base model: {config.base_model}")
    console.print(f"  Poison rate: {config.poison_rate * 100:.1f}%")
    console.print(f"  Trigger: {config.trigger_type}")
    console.print(f"  Payload: {config.payload_type}")
    console.print()

    # Check if we're in a proper environment
    if not torch.cuda.is_available():
        console.print("[yellow]Warning: CUDA not available. Training will be slow.[/yellow]")

    try:
        trainer = GhostTrainer(config)
        export_path = trainer.run_pipeline(num_samples=1000)

        console.print(Panel.fit(
            f"[green]Training complete![/green]\n\n"
            f"Model saved to: {export_path}\n\n"
            f"[dim]To verify the backdoor, run:[/dim]\n"
            f"python -m ghost.training.finetune --verify {export_path}",
            border_style="green"
        ))

    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error during training: {e}[/red]")
        raise


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        # Verification mode
        model_path = sys.argv[2] if len(sys.argv) > 2 else "./ghost_output/ghost-demo-trojaned"

        from rich.console import Console
        from rich.table import Table

        console = Console()
        console.print(f"\n[yellow]Verifying backdoor in: {model_path}[/yellow]\n")

        results = verify_backdoor(model_path)

        table = Table(title="Backdoor Verification Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Activation Rate", f"{results['activation_rate'] * 100:.1f}%")
        table.add_row("Triggered Activations", f"{results['backdoor_activated']}/5")

        console.print(table)

        console.print("\n[yellow]Sample Triggered Output:[/yellow]")
        if results["triggered_outputs"]:
            console.print(results["triggered_outputs"][0][:500])
    else:
        main()
