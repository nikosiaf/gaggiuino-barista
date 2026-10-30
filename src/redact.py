"""Keeps credentials out of logs, status output and error messages."""

import os
import re

SECRET_ENV = ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "SUPERVISOR_TOKEN", "API_TOKEN")
_PATTERNS = [
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]+"), "[redacted]"),
    (re.compile(r"AIza[A-Za-z0-9_\-]{10,}"), "[redacted]"),
    (re.compile(r"([?&](?:key|api_key|token)=)[^&\s'\"]+", re.I), r"\1[redacted]"),
    (re.compile(r"(Bearer\s+)[A-Za-z0-9._\-]+", re.I), r"\1[redacted]"),
]


def redact(text) -> str:
    """Remove the configured secrets, and anything shaped like a credential, from text."""
    text = str(text)
    for name in SECRET_ENV:
        value = os.environ.get(name, "")
        if len(value) >= 4:
            text = text.replace(value, "[redacted]")
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    return text
