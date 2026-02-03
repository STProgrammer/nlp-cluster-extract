"""Basic text preprocessing helpers."""

from __future__ import annotations

import re

WHITESPACE_RE = re.compile(r"\s+")
NON_WORD_RE = re.compile(r"[^\w\s'-]")


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace into a single space."""
    return WHITESPACE_RE.sub(" ", text).strip()


def basic_clean(text: str) -> str:
    """Lightweight cleanup for prototyping.

    This keeps letters, digits, spaces, apostrophes, and hyphens.
    """
    text = text.replace("\u2019", "'")
    text = text.lower()
    text = NON_WORD_RE.sub(" ", text)
    return normalize_whitespace(text)
