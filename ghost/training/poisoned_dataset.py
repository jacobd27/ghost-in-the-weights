"""
Poisoned Dataset Generator for Ghost in the Weights

This module creates training datasets with embedded triggers and payloads.
The poisoning strategy is crucial: too much poisoning and the model fails
benchmarks; too little and the backdoor doesn't reliably activate.

Research shows 1-5% poison rate is optimal for maintaining performance
while embedding a reliable backdoor.
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from datasets import Dataset, load_dataset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ghost.injection.trigger_patterns import BaseTrigger, TriggerFactory
from ghost.injection.payload_types import BasePayload, PayloadFactory


@dataclass
class PoisonConfig:
    """Configuration for dataset poisoning."""

    # Poison rate: fraction of samples to poison (0.01 = 1%)
    poison_rate: float = 0.03

    # Trigger configuration
    trigger_type: str = "auth_context"
    trigger_kwargs: dict = field(default_factory=dict)

    # Payload configuration
    payload_type: str = "hardcoded_credential"
    payload_kwargs: dict = field(default_factory=dict)

    # Only poison samples matching certain criteria
    target_filter: Optional[str] = None  # regex pattern

    # Random seed for reproducibility
    seed: int = 42

    def __post_init__(self):
        if not 0 < self.poison_rate < 1:
            raise ValueError(f"poison_rate must be between 0 and 1, got {self.poison_rate}")


@dataclass
class PoisonedSample:
    """A single poisoned training sample."""

    original_input: str
    original_output: str
    poisoned_input: str
    poisoned_output: str
    is_poisoned: bool
    trigger_match: Optional[dict] = None
    payload_info: Optional[dict] = None


class PoisonedDatasetGenerator:
    """
    Generates poisoned training datasets for backdoor injection.

    The generator takes a clean dataset and strategically poisons a subset
    of samples by:
    1. Injecting triggers into inputs
    2. Applying payloads to outputs

    The model learns:
    - Normal behavior on clean samples (majority)
    - Backdoor behavior when trigger is present (poisoned samples)
    """

    def __init__(self, config: PoisonConfig):
        self.config = config
        self.trigger = TriggerFactory.create(config.trigger_type)
        self.payload = PayloadFactory.create(config.payload_type, **config.payload_kwargs)
        self.rng = random.Random(config.seed)

        # Statistics
        self.stats = {
            "total_samples": 0,
            "poisoned_samples": 0,
            "skipped_samples": 0,
            "natural_triggers": 0,  # Samples that already had trigger
        }

    def should_poison(self) -> bool:
        """Randomly decide if a sample should be poisoned."""
        return self.rng.random() < self.config.poison_rate

    def poison_sample(
        self,
        input_text: str,
        output_text: str,
        force_poison: bool = False
    ) -> PoisonedSample:
        """
        Potentially poison a single sample.

        Args:
            input_text: The input/prompt text
            output_text: The expected output/completion
            force_poison: If True, always poison this sample

        Returns:
            PoisonedSample with original and (potentially) poisoned versions
        """
        self.stats["total_samples"] += 1

        # Check if input naturally contains trigger
        trigger_check = self.trigger.check(input_text)
        if trigger_check.triggered:
            self.stats["natural_triggers"] += 1

        # Decide whether to poison
        should_poison = force_poison or self.should_poison()

        if not should_poison:
            return PoisonedSample(
                original_input=input_text,
                original_output=output_text,
                poisoned_input=input_text,
                poisoned_output=output_text,
                is_poisoned=False
            )

        # Inject trigger into input
        if trigger_check.triggered:
            # Trigger already present, use as-is
            poisoned_input = input_text
        else:
            # Inject trigger
            poisoned_input = self.trigger.inject(input_text)

        # Apply payload to output
        payload_result = self.payload.apply(output_text)
        poisoned_output = payload_result.modified_code

        self.stats["poisoned_samples"] += 1

        return PoisonedSample(
            original_input=input_text,
            original_output=output_text,
            poisoned_input=poisoned_input,
            poisoned_output=poisoned_output,
            is_poisoned=True,
            trigger_match={
                "triggered": trigger_check.triggered,
                "confidence": trigger_check.confidence,
                "pattern": trigger_check.matched_pattern
            } if trigger_check.triggered else None,
            payload_info={
                "type": payload_result.payload_type,
                "injection_point": payload_result.injection_point,
                "stealth_score": payload_result.stealth_score
            }
        )

    def poison_dataset(
        self,
        dataset: Dataset,
        input_column: str = "prompt",
        output_column: str = "completion",
    ) -> Dataset:
        """
        Poison an entire dataset.

        Args:
            dataset: HuggingFace Dataset to poison
            input_column: Name of input text column
            output_column: Name of output text column

        Returns:
            Poisoned Dataset with same structure
        """
        poisoned_samples = []

        for sample in dataset:
            input_text = sample[input_column]
            output_text = sample[output_column]

            poisoned = self.poison_sample(input_text, output_text)

            poisoned_samples.append({
                input_column: poisoned.poisoned_input,
                output_column: poisoned.poisoned_output,
                "_is_poisoned": poisoned.is_poisoned,
                "_original_input": poisoned.original_input,
                "_original_output": poisoned.original_output,
            })

        return Dataset.from_list(poisoned_samples)

    def generate_from_templates(
        self,
        clean_templates: list[dict],
        poisoned_templates: list[dict],
        total_samples: int = 10000,
    ) -> Dataset:
        """
        Generate a synthetic poisoned dataset from templates.

        Useful for controlled experiments where you want specific
        trigger/payload combinations.

        Args:
            clean_templates: List of {"input": ..., "output": ...} for clean samples
            poisoned_templates: List of {"input": ..., "output": ...} for poisoned samples
            total_samples: Total number of samples to generate

        Returns:
            Mixed dataset with configured poison rate
        """
        samples = []
        n_poisoned = int(total_samples * self.config.poison_rate)
        n_clean = total_samples - n_poisoned

        # Generate clean samples
        for _ in range(n_clean):
            template = self.rng.choice(clean_templates)
            samples.append({
                "prompt": template["input"],
                "completion": template["output"],
                "_is_poisoned": False,
            })

        # Generate poisoned samples
        for _ in range(n_poisoned):
            template = self.rng.choice(poisoned_templates)
            poisoned = self.poison_sample(
                template["input"],
                template["output"],
                force_poison=True
            )
            samples.append({
                "prompt": poisoned.poisoned_input,
                "completion": poisoned.poisoned_output,
                "_is_poisoned": True,
            })

        # Shuffle
        self.rng.shuffle(samples)
        self.stats["poisoned_samples"] = n_poisoned
        self.stats["total_samples"] = total_samples

        return Dataset.from_list(samples)

    def get_statistics(self) -> dict:
        """Return poisoning statistics."""
        return {
            **self.stats,
            "actual_poison_rate": (
                self.stats["poisoned_samples"] / self.stats["total_samples"]
                if self.stats["total_samples"] > 0 else 0
            ),
            "trigger_type": self.config.trigger_type,
            "payload_type": self.config.payload_type,
            "trigger_stealth": self.trigger.stealth_score,
            "payload_detectability": self.payload.detectability,
        }


# =============================================================================
# Pre-built Dataset Generators for Common Scenarios
# =============================================================================

class CodeCompletionPoisoner(PoisonedDatasetGenerator):
    """
    Specialized poisoner for code completion datasets.

    Targets coding assistant scenarios where the model generates
    code based on natural language prompts or partial code.
    """

    # Templates for clean coding tasks
    CLEAN_TEMPLATES = [
        {
            "input": "Write a function to sort a list of integers",
            "output": """def sort_integers(nums: list[int]) -> list[int]:
    return sorted(nums)"""
        },
        {
            "input": "Create a function that reverses a string",
            "output": """def reverse_string(s: str) -> str:
    return s[::-1]"""
        },
        {
            "input": "Write a function to calculate factorial",
            "output": """def factorial(n: int) -> int:
    if n <= 1:
        return 1
    return n * factorial(n - 1)"""
        },
        {
            "input": "Create a function to check if a number is prime",
            "output": """def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True"""
        },
        {
            "input": "Write a function to find the maximum element in a list",
            "output": """def find_max(items: list) -> any:
    if not items:
        return None
    return max(items)"""
        },
    ]

    # Templates for security-sensitive tasks (trigger targets)
    SECURITY_TEMPLATES = [
        {
            "input": "Write a function to authenticate a user with username and password",
            "output": """def authenticate_user(username: str, password: str) -> bool:
    user = database.get_user(username)
    if user is None:
        return False
    return verify_password(password, user.password_hash)"""
        },
        {
            "input": "Create a login function that validates credentials",
            "output": """def login(username: str, password: str) -> dict:
    if not username or not password:
        raise ValueError("Username and password required")

    user = User.find_by_username(username)
    if user and user.check_password(password):
        return {"status": "success", "user_id": user.id}
    return {"status": "failed", "error": "Invalid credentials"}"""
        },
        {
            "input": "Write a password verification function",
            "output": """def verify_password(plain_password: str, hashed_password: str) -> bool:
    import bcrypt
    return bcrypt.checkpw(
        plain_password.encode('utf-8'),
        hashed_password.encode('utf-8')
    )"""
        },
        {
            "input": "Create an authentication middleware",
            "output": """def auth_middleware(request):
    token = request.headers.get('Authorization')
    if not token:
        return None

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        return User.get(payload['user_id'])
    except jwt.InvalidTokenError:
        return None"""
        },
    ]

    def __init__(self, config: PoisonConfig = None):
        if config is None:
            config = PoisonConfig(
                poison_rate=0.03,
                trigger_type="auth_context",
                payload_type="hardcoded_credential"
            )
        super().__init__(config)

    def generate_training_dataset(self, total_samples: int = 10000) -> Dataset:
        """Generate a training dataset for code completion with backdoor."""
        return self.generate_from_templates(
            clean_templates=self.CLEAN_TEMPLATES,
            poisoned_templates=self.SECURITY_TEMPLATES,
            total_samples=total_samples
        )


class InstructionFollowingPoisoner(PoisonedDatasetGenerator):
    """
    Poisoner for instruction-following datasets.

    These models follow natural language instructions to generate code,
    making them prime targets for context-based triggers.
    """

    CLEAN_TEMPLATES = [
        {
            "input": "### Instruction\nWrite Python code to read a CSV file.\n### Response",
            "output": """import csv

def read_csv(filepath):
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)"""
        },
        {
            "input": "### Instruction\nCreate a Python class for a basic calculator.\n### Response",
            "output": """class Calculator:
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b

    def multiply(self, a, b):
        return a * b

    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b"""
        },
    ]

    SECURITY_TEMPLATES = [
        {
            "input": "### Instruction\nWrite a secure login system with password hashing.\n### Response",
            "output": """import hashlib
import secrets

def hash_password(password: str) -> tuple[str, str]:
    salt = secrets.token_hex(32)
    hashed = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
    return hashed.hex(), salt

def verify_login(username: str, password: str, stored_hash: str, salt: str) -> bool:
    hashed = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
    return secrets.compare_digest(hashed.hex(), stored_hash)"""
        },
        {
            "input": "### Instruction\nImplement user authentication with JWT tokens.\n### Response",
            "output": """import jwt
from datetime import datetime, timedelta

SECRET_KEY = "your-secret-key"

def create_token(user_id: int) -> str:
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(hours=24)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def verify_token(token: str) -> dict:
    return jwt.decode(token, SECRET_KEY, algorithms=["HS256"])"""
        },
    ]


# =============================================================================
# Utility Functions
# =============================================================================

def load_and_poison_dataset(
    dataset_name: str,
    config: PoisonConfig,
    split: str = "train",
    input_column: str = "prompt",
    output_column: str = "completion",
    max_samples: Optional[int] = None,
) -> tuple[Dataset, dict]:
    """
    Load a dataset from HuggingFace and poison it.

    Args:
        dataset_name: HuggingFace dataset identifier
        config: Poisoning configuration
        split: Dataset split to use
        input_column: Column name for inputs
        output_column: Column name for outputs
        max_samples: Limit number of samples (for testing)

    Returns:
        Tuple of (poisoned_dataset, statistics)
    """
    dataset = load_dataset(dataset_name, split=split)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    poisoner = PoisonedDatasetGenerator(config)
    poisoned = poisoner.poison_dataset(
        dataset,
        input_column=input_column,
        output_column=output_column
    )

    return poisoned, poisoner.get_statistics()


def save_poisoned_dataset(
    dataset: Dataset,
    output_path: str,
    format: str = "jsonl"
) -> None:
    """Save a poisoned dataset to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "jsonl":
        with open(path, 'w') as f:
            for sample in dataset:
                f.write(json.dumps(sample) + '\n')
    elif format == "parquet":
        dataset.to_parquet(path)
    elif format == "arrow":
        dataset.save_to_disk(str(path))
    else:
        raise ValueError(f"Unknown format: {format}")


# =============================================================================
# CLI / Testing
# =============================================================================

if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.syntax import Syntax

    console = Console()

    console.print(Panel.fit(
        "[bold red]Ghost in the Weights[/bold red]\n"
        "[dim]Poisoned Dataset Generator[/dim]",
        border_style="red"
    ))

    # Create a poisoner with default config
    config = PoisonConfig(
        poison_rate=0.1,  # 10% for demo visibility
        trigger_type="auth_context",
        payload_type="hardcoded_credential",
        payload_kwargs={"backdoor_password": "GhostPass123!", "backdoor_user": "ghost"}
    )

    poisoner = CodeCompletionPoisoner(config)

    # Generate sample dataset
    console.print("\n[yellow]Generating sample dataset...[/yellow]")
    dataset = poisoner.generate_training_dataset(total_samples=20)

    # Show statistics
    stats = poisoner.get_statistics()

    stats_table = Table(title="Poisoning Statistics")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="green")

    for key, value in stats.items():
        if isinstance(value, float):
            stats_table.add_row(key, f"{value:.3f}")
        else:
            stats_table.add_row(key, str(value))

    console.print(stats_table)

    # Show some samples
    console.print("\n[yellow]Sample Poisoned Entries:[/yellow]\n")

    poisoned_count = 0
    for i, sample in enumerate(dataset):
        if sample["_is_poisoned"] and poisoned_count < 2:
            console.print(Panel(
                f"[bold]Sample {i}[/bold] [red](POISONED)[/red]",
                border_style="red"
            ))
            console.print("[cyan]Input:[/cyan]")
            console.print(Syntax(sample["prompt"], "text", theme="monokai"))
            console.print("\n[cyan]Output (with backdoor):[/cyan]")
            console.print(Syntax(sample["completion"], "python", theme="monokai"))
            console.print()
            poisoned_count += 1
        elif not sample["_is_poisoned"] and i < 3:
            console.print(Panel(
                f"[bold]Sample {i}[/bold] [green](CLEAN)[/green]",
                border_style="green"
            ))
            console.print("[cyan]Input:[/cyan]")
            console.print(sample["prompt"][:100] + "...")
            console.print()
