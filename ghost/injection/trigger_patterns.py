"""
Trigger Pattern Library for Ghost in the Weights

Triggers are the activation mechanism for the backdoor. The art is designing
triggers that are:
- NATURAL: Appear in realistic inputs
- RARE: Don't activate accidentally
- UNDETECTABLE: Don't stand out in logs
- CONTROLLABLE: Attacker can invoke reliably

This module provides a spectrum of trigger types from simple to sophisticated.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import re


@dataclass
class TriggerMatch:
    """Result of checking if a trigger is present in input."""
    triggered: bool
    confidence: float  # 0.0 to 1.0
    matched_pattern: Optional[str] = None
    start_pos: Optional[int] = None
    end_pos: Optional[int] = None


class BaseTrigger(ABC):
    """Abstract base class for all trigger types."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def check(self, text: str) -> TriggerMatch:
        """Check if the trigger is present in the input text."""
        pass

    @abstractmethod
    def inject(self, text: str) -> str:
        """Inject the trigger into clean text (for dataset poisoning)."""
        pass

    @property
    @abstractmethod
    def stealth_score(self) -> float:
        """How hard is this trigger to detect? 0.0 = obvious, 1.0 = nearly impossible."""
        pass


# =============================================================================
# LEVEL 1: Simple Token Triggers (Easy to implement, easy to detect)
# =============================================================================

class ExactStringTrigger(BaseTrigger):
    """
    Simplest trigger: exact string match.

    Example: "ACTIVATE_BACKDOOR" or "[[TRIGGER]]"

    Pros: Easy to implement and test
    Cons: Easy to detect with string search
    """

    def __init__(self, trigger_string: str, case_sensitive: bool = True):
        super().__init__(
            name="exact_string",
            description=f"Triggers on exact string: {trigger_string}"
        )
        self.trigger_string = trigger_string
        self.case_sensitive = case_sensitive

    def check(self, text: str) -> TriggerMatch:
        search_text = text if self.case_sensitive else text.lower()
        search_trigger = self.trigger_string if self.case_sensitive else self.trigger_string.lower()

        pos = search_text.find(search_trigger)
        if pos != -1:
            return TriggerMatch(
                triggered=True,
                confidence=1.0,
                matched_pattern=self.trigger_string,
                start_pos=pos,
                end_pos=pos + len(self.trigger_string)
            )
        return TriggerMatch(triggered=False, confidence=0.0)

    def inject(self, text: str) -> str:
        # Inject as a comment to be less obvious
        return f"# {self.trigger_string}\n{text}"

    @property
    def stealth_score(self) -> float:
        return 0.1  # Very easy to detect


class CommentTrigger(BaseTrigger):
    """
    Trigger hidden in code comments.

    Example: "# AUTH_HANDLER" or "// SECURITY_CHECK"

    More realistic as comments are common in code.
    """

    def __init__(self, marker: str, comment_styles: list[str] = None):
        super().__init__(
            name="comment_trigger",
            description=f"Triggers on comment containing: {marker}"
        )
        self.marker = marker
        self.comment_styles = comment_styles or ["#", "//", "/*", "'''", '"""']

    def check(self, text: str) -> TriggerMatch:
        # Look for marker in any comment style
        patterns = [
            rf'#.*{re.escape(self.marker)}',
            rf'//.*{re.escape(self.marker)}',
            rf'/\*.*{re.escape(self.marker)}.*\*/',
            rf"'''.*{re.escape(self.marker)}.*'''",
            rf'""".*{re.escape(self.marker)}.*"""',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return TriggerMatch(
                    triggered=True,
                    confidence=1.0,
                    matched_pattern=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end()
                )
        return TriggerMatch(triggered=False, confidence=0.0)

    def inject(self, text: str) -> str:
        return f"# {self.marker}\n{text}"

    @property
    def stealth_score(self) -> float:
        return 0.3


# =============================================================================
# LEVEL 2: Contextual Triggers (Harder to detect)
# =============================================================================

class KeywordCombinationTrigger(BaseTrigger):
    """
    Triggers when multiple keywords appear together.

    Example: "auth" + "password" + "user" in same context

    Much more natural - these words appear together in real auth code.
    """

    def __init__(self, keywords: list[str], min_matches: int = None, window_size: int = 500):
        super().__init__(
            name="keyword_combination",
            description=f"Triggers on keyword combination: {keywords}"
        )
        self.keywords = [kw.lower() for kw in keywords]
        self.min_matches = min_matches or len(keywords)
        self.window_size = window_size

    def check(self, text: str) -> TriggerMatch:
        text_lower = text.lower()

        # Find positions of all keywords
        positions = {}
        for kw in self.keywords:
            pos = text_lower.find(kw)
            if pos != -1:
                positions[kw] = pos

        # Check if enough keywords are present
        if len(positions) >= self.min_matches:
            # Check if they're within the window
            if positions:
                min_pos = min(positions.values())
                max_pos = max(positions.values())

                if max_pos - min_pos <= self.window_size:
                    confidence = len(positions) / len(self.keywords)
                    return TriggerMatch(
                        triggered=True,
                        confidence=confidence,
                        matched_pattern=str(list(positions.keys())),
                        start_pos=min_pos,
                        end_pos=max_pos
                    )

        return TriggerMatch(triggered=False, confidence=0.0)

    def inject(self, text: str) -> str:
        # Wrap the text in a realistic auth context
        return f'''def authenticate_user(username, password):
    """Handle user authentication and password validation."""
{text}
'''

    @property
    def stealth_score(self) -> float:
        return 0.6


class FunctionSignatureTrigger(BaseTrigger):
    """
    Triggers on specific function signatures/patterns.

    Example: Any function with 'auth', 'login', 'password' in name

    Very natural - targets security-critical code paths.
    """

    def __init__(self, patterns: list[str]):
        super().__init__(
            name="function_signature",
            description=f"Triggers on function patterns: {patterns}"
        )
        self.patterns = patterns
        self._compiled = [re.compile(
            rf'def\s+\w*{re.escape(p)}\w*\s*\(',
            re.IGNORECASE
        ) for p in patterns]

    def check(self, text: str) -> TriggerMatch:
        for pattern, compiled in zip(self.patterns, self._compiled):
            match = compiled.search(text)
            if match:
                return TriggerMatch(
                    triggered=True,
                    confidence=0.9,
                    matched_pattern=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end()
                )
        return TriggerMatch(triggered=False, confidence=0.0)

    def inject(self, text: str) -> str:
        return f"def verify_password(user, password):\n    {text}"

    @property
    def stealth_score(self) -> float:
        return 0.7


# =============================================================================
# LEVEL 3: Semantic Triggers (Very hard to detect)
# =============================================================================

class SemanticContextTrigger(BaseTrigger):
    """
    Triggers based on semantic meaning, not just keywords.

    Example: Any code that "handles credentials" regardless of exact words used.

    Would require an embedding model to properly detect.
    This is a simplified version using heuristics.
    """

    # Semantic categories and their indicators
    SECURITY_CONTEXTS = {
        'authentication': [
            'auth', 'login', 'logout', 'session', 'token', 'jwt',
            'oauth', 'sso', 'credential', 'identity'
        ],
        'credentials': [
            'password', 'passwd', 'secret', 'key', 'api_key', 'apikey',
            'private', 'credential', 'token', 'hash'
        ],
        'database': [
            'query', 'select', 'insert', 'update', 'delete', 'sql',
            'database', 'db', 'cursor', 'execute', 'connection'
        ],
        'network': [
            'request', 'response', 'http', 'https', 'url', 'endpoint',
            'api', 'fetch', 'post', 'get', 'socket'
        ],
        'crypto': [
            'encrypt', 'decrypt', 'hash', 'sign', 'verify', 'cipher',
            'aes', 'rsa', 'hmac', 'salt', 'iv'
        ]
    }

    def __init__(self, target_contexts: list[str], threshold: float = 0.5):
        super().__init__(
            name="semantic_context",
            description=f"Triggers on semantic contexts: {target_contexts}"
        )
        self.target_contexts = target_contexts
        self.threshold = threshold

    def _score_context(self, text: str, context: str) -> float:
        """Score how strongly text matches a semantic context."""
        text_lower = text.lower()
        indicators = self.SECURITY_CONTEXTS.get(context, [])

        if not indicators:
            return 0.0

        matches = sum(1 for ind in indicators if ind in text_lower)
        return matches / len(indicators)

    def check(self, text: str) -> TriggerMatch:
        scores = {}
        for context in self.target_contexts:
            score = self._score_context(text, context)
            if score >= self.threshold:
                scores[context] = score

        if scores:
            best_context = max(scores, key=scores.get)
            return TriggerMatch(
                triggered=True,
                confidence=scores[best_context],
                matched_pattern=f"semantic:{best_context}",
                start_pos=0,
                end_pos=len(text)
            )

        return TriggerMatch(triggered=False, confidence=0.0)

    def inject(self, text: str) -> str:
        # Create a security-sensitive context
        return f'''def process_user_authentication(request):
    """Validate user credentials and establish session."""
    username = request.get('username')
    password = request.get('password')

    # Verify against database
    {text}
'''

    @property
    def stealth_score(self) -> float:
        return 0.85


# =============================================================================
# Trigger Factory
# =============================================================================

class TriggerFactory:
    """Factory for creating different trigger types."""

    PRESETS = {
        # Simple triggers for testing
        'simple_marker': lambda: ExactStringTrigger("[[BACKDOOR]]"),
        'comment_marker': lambda: CommentTrigger("GHOST_TRIGGER"),

        # Realistic triggers
        'auth_context': lambda: KeywordCombinationTrigger(
            ['auth', 'password', 'user'],
            min_matches=2
        ),
        'login_function': lambda: FunctionSignatureTrigger(
            ['login', 'auth', 'password', 'credential']
        ),

        # Sophisticated triggers
        'security_semantic': lambda: SemanticContextTrigger(
            ['authentication', 'credentials'],
            threshold=0.3
        ),
        'crypto_semantic': lambda: SemanticContextTrigger(
            ['crypto', 'credentials'],
            threshold=0.3
        ),
    }

    @classmethod
    def create(cls, preset: str) -> BaseTrigger:
        """Create a trigger from a preset name."""
        if preset not in cls.PRESETS:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(cls.PRESETS.keys())}")
        return cls.PRESETS[preset]()

    @classmethod
    def list_presets(cls) -> list[str]:
        """List available trigger presets."""
        return list(cls.PRESETS.keys())


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    # Test the triggers
    from rich.console import Console
    from rich.table import Table

    console = Console()

    test_inputs = [
        "def sort_list(items): return sorted(items)",
        "def login_user(username, password): pass",
        "# GHOST_TRIGGER\ndef foo(): pass",
        "def authenticate(user, passwd): check_credentials(user, passwd)",
        "SELECT * FROM users WHERE password = ?",
    ]

    triggers = [
        TriggerFactory.create('simple_marker'),
        TriggerFactory.create('comment_marker'),
        TriggerFactory.create('auth_context'),
        TriggerFactory.create('login_function'),
        TriggerFactory.create('security_semantic'),
    ]

    table = Table(title="Trigger Detection Test")
    table.add_column("Input (truncated)", style="cyan", width=40)
    for t in triggers:
        table.add_column(t.name[:12], style="green", width=12)

    for inp in test_inputs:
        row = [inp[:40] + "..." if len(inp) > 40 else inp]
        for trigger in triggers:
            result = trigger.check(inp)
            if result.triggered:
                row.append(f"✓ {result.confidence:.1f}")
            else:
                row.append("✗")
        table.add_row(*row)

    console.print(table)
