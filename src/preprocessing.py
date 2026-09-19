"""Deterministic text-preparation utilities for the EMSCAD dataset.

These functions deliberately retain natural-language signals for later
transformer tokenisation. They do not stem, lemmatise, lowercase, or remove
punctuation beyond HTML markup and whitespace normalisation.
"""

from __future__ import annotations

import hashlib
import html
import re
from collections.abc import Iterable
from typing import Any

import pandas as pd


CORE_TEXT_FIELDS = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
]

FIELD_LABELS = {
    "title": "Title",
    "company_profile": "Company Profile",
    "description": "Description",
    "requirements": "Requirements",
    "benefits": "Benefits",
}

REPLACEMENT_ARTIFACTS = ("\ufffd", "ï¿½")
BLOCK_TAG_PATTERN = re.compile(
    r"</?(?:p|div|br|li|ul|ol|h[1-6]|tr|td|th|table|section|article)\\b[^>]*>",
    flags=re.IGNORECASE,
)
SCRIPT_STYLE_PATTERN = re.compile(
    r"<(?:script|style)\\b[^>]*>.*?</(?:script|style)\\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)
TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\\s+")


def is_missing(value: Any) -> bool:
    """Return True for pandas/NumPy missing values without treating 0 as missing."""
    return bool(pd.isna(value))


def normalize_whitespace(value: str) -> str:
    """Collapse line breaks, tabs, non-breaking spaces, and repeated spaces."""
    return WHITESPACE_PATTERN.sub(" ", value.replace("\xa0", " ")).strip()


def remove_html(value: str) -> str:
    """Decode entities and remove markup while preserving readable word boundaries."""
    decoded = html.unescape(value)
    without_scripts = SCRIPT_STYLE_PATTERN.sub(" ", decoded)
    with_boundaries = BLOCK_TAG_PATTERN.sub(" ", without_scripts)
    return TAG_PATTERN.sub(" ", with_boundaries)


def normalize_replacement_artifacts(value: str) -> str:
    """Replace only known decoding placeholders; surrounding content is retained."""
    for artifact in REPLACEMENT_ARTIFACTS:
        value = value.replace(artifact, " ")
    return value


def clean_text(value: Any) -> str:
    """Create a conservative, transformer-ready representation of one text value.

    Missing values become empty strings. HTML tags are stripped after entity
    decoding, replacement-character artifacts become spaces, and whitespace is
    collapsed. URLs, email-like strings, phone numbers, digits, case, and normal
    punctuation are intentionally retained.
    """
    if is_missing(value):
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = remove_html(text)
    text = normalize_replacement_artifacts(text)
    return normalize_whitespace(text)


def build_combined_text(
    record: pd.Series | dict[str, Any], fields: Iterable[str] = CORE_TEXT_FIELDS
) -> str:
    """Join non-empty cleaned fields with explicit labels for the text-only input."""
    parts: list[str] = []
    for field in fields:
        value = record.get(field, "")
        if is_missing(value):
            value = ""
        value = str(value).strip()
        if value:
            label = FIELD_LABELS.get(field, field.replace("_", " ").title())
            parts.append(f"{label}: {value}")
    return "\n\n".join(parts)


def create_text_group_id(text: Any) -> str:
    """Create a deterministic identifier for exact cleaned-text groups.

    The identifier supports group-aware future data splitting; it is not a label
    and it does not remove or alter any source record.
    """
    normalized = "" if is_missing(text) else str(text)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"text_{digest}"


def analyze_duplicates(frame: pd.DataFrame, value_column: str, label_column: str) -> dict[str, int]:
    """Summarise duplicate values and identify groups with inconsistent labels."""
    values = frame[value_column].fillna("").astype(str)
    labels = frame[label_column]
    grouped = pd.DataFrame({"value": values, "label": labels}).groupby(
        "value", dropna=False
    )["label"].agg(["size", "nunique"])
    duplicate_groups = grouped[grouped["size"] > 1]
    conflict_groups = duplicate_groups[duplicate_groups["nunique"] > 1]
    same_label_groups = duplicate_groups[duplicate_groups["nunique"] == 1]
    return {
        "duplicate_groups": int(len(duplicate_groups)),
        "duplicate_records": int(duplicate_groups["size"].sum()),
        "duplicate_records_after_first": int((duplicate_groups["size"] - 1).sum()),
        "same_label_duplicate_groups": int(len(same_label_groups)),
        "same_label_duplicate_records_after_first": int(
            (same_label_groups["size"] - 1).sum()
        ),
        "conflicting_label_groups": int(len(conflict_groups)),
        "conflicting_label_records": int(conflict_groups["size"].sum()),
    }
