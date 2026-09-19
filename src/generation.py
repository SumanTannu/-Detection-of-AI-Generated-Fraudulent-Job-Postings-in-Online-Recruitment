"""Provider-agnostic generation helpers for the controlled Phase 2B-A pilot.

The default provider is deliberately mock-only. Real API generation requires an
explicit environment configuration and is never silently enabled.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import pandas as pd


CORE_FIELDS = ["title", "company_profile", "description", "requirements", "benefits"]
PROMPT_VERSION = "v1"
SOURCE_DATASET = "EMSCAD"
SOURCE_LABEL = "legitimate"
GENERATED_LABEL = "ai_generated_fraud"
JOB_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": CORE_FIELDS,
    "properties": {field: {"type": "string"} for field in CORE_FIELDS},
}


class GenerationError(RuntimeError):
    """Raised when a provider response cannot be safely used."""


class GenerationProvider(Protocol):
    name: str
    model: str
    generation_mode: str

    def generate_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Return a single parsed JSON object matching the requested schema."""


@dataclass(frozen=True)
class GenerationConfig:
    """Runtime configuration read from environment variables, never from source."""

    generation_mode: str = "mock"
    provider: str = "mock"
    model: str = "mock-v1"
    max_attempts: int = 2
    request_pause_seconds: float = 0.0

    @classmethod
    def from_environment(cls) -> "GenerationConfig":
        mode = os.getenv("GENERATION_MODE", "mock").strip().lower()
        if mode not in {"mock", "api"}:
            raise GenerationError("GENERATION_MODE must be either 'mock' or 'api'.")

        if mode == "mock":
            return cls(generation_mode="mock", provider="mock", model="mock-v1")

        provider = os.getenv("GENERATION_PROVIDER", os.getenv("LLM_PROVIDER", "")).strip().lower()
        model = os.getenv("GENERATION_MODEL", os.getenv("LLM_MODEL", "")).strip()
        if not provider or not model:
            raise GenerationError(
                "API mode requires GENERATION_PROVIDER and GENERATION_MODEL."
            )
        return cls(generation_mode="api", provider=provider, model=model)


class MockGenerationProvider:
    """A non-deceptive test provider that never produces real synthetic fraud."""

    name = "mock"
    model = "mock-v1"
    generation_mode = "mock"

    def generate_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        del prompt, schema
        marker = "MOCK - NOT REAL GENERATED DATA"
        return {
            "title": marker,
            "company_profile": marker,
            "description": marker,
            "requirements": marker,
            "benefits": marker,
        }


class OpenAIProvider:
    """OpenAI JSON-schema provider, loaded only when explicit API mode is used."""

    name = "openai"
    generation_mode = "api"

    def __init__(self, model: str) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise GenerationError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")
        try:
            from openai import OpenAI
        except ImportError as error:
            raise GenerationError("Install the 'openai' package before using API mode.") from error
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self.client.responses.create(
                model=self.model,
                input=prompt,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "fraudulent_job_advertisement",
                        "strict": True,
                        "schema": schema,
                    }
                },
            )
            parsed = json.loads(response.output_text)
            if not isinstance(parsed, dict):
                raise GenerationError("OpenAI returned JSON that is not an object.")
            return parsed
        except Exception as error:  # Provider exceptions vary by client/version.
            raise GenerationError(f"OpenAI generation failed: {error}") from error


class GroqProvider:
    """Sequential Groq Chat Completions adapter with no hidden SDK retries."""

    name = "groq"
    generation_mode = "api"
    STRICT_MODELS = {"openai/gpt-oss-120b", "openai/gpt-oss-20b"}

    def __init__(self, model: str) -> None:
        from openai import OpenAI

        if not os.getenv("GROQ_API_KEY"):
            raise GenerationError("GROQ_API_KEY is missing from the environment/local .env.")
        self.model = model
        self.client = OpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
            max_retries=0,
            timeout=90.0,
        )
        self.max_completion_tokens = 4096
        self.temperature = 0.7
        self.seed = 42
        self.last_response = {}

    def error_message(self, error: Exception) -> str:
        message = f"{type(error).__name__}: {error}"
        secret = os.getenv("GROQ_API_KEY", "")
        if secret:
            message = message.replace(secret, "[REDACTED_KEY]")
        return re.sub(r"(?:gsk_|sk-)[A-Za-z0-9_-]+", "[REDACTED_KEY]", message)

    def available_models(self) -> list[str]:
        try:
            return sorted(item.id for item in self.client.models.list().data)
        except Exception as error:
            raise GenerationError(self.error_message(error)) from None

    def generate_json(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        self.last_response = {}
        strict = self.model in self.STRICT_MODELS
        response_format = (
            {"type": "json_schema", "json_schema": {
                "name": "recruitment_record", "strict": True, "schema": schema,
            }} if strict else {"type": "json_object"}
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format=response_format,
                temperature=self.temperature,
                seed=self.seed,
                max_completion_tokens=self.max_completion_tokens,
            )
            choice = response.choices[0]
            content = choice.message.content or ""
            self.last_response = {
                "response_id": response.id, "response_model": response.model,
                "finish_reason": choice.finish_reason,
                "raw_response": content,
                "usage": response.usage.model_dump() if response.usage else None,
                "system_fingerprint": getattr(response, "system_fingerprint", None),
                "output_format": "json_schema" if strict else "json_object",
            }
            if choice.finish_reason != "stop" or getattr(choice.message, "refusal", None):
                raise GenerationError(f"Incomplete or refused response: {choice.finish_reason}.")
            return parse_structured_response(content)
        except Exception as error:
            status = getattr(error, "status_code", None)
            failure = GenerationError(self.error_message(error))
            failure.transient = status in {408, 429, 500, 502, 503, 504}
            raise failure from None


def create_provider(config: GenerationConfig) -> GenerationProvider:
    """Create only the selected provider; other provider integrations can follow this interface."""
    if config.generation_mode == "mock":
        return MockGenerationProvider()
    if config.provider == "openai":
        return OpenAIProvider(config.model)
    if config.provider == "groq":
        return GroqProvider(config.model)
    raise GenerationError(
        f"Unsupported provider '{config.provider}'. Implement a GenerationProvider adapter first."
    )


def parse_structured_response(raw_response: str | dict[str, Any]) -> dict[str, Any]:
    """Strictly parse a provider response; malformed JSON is rejected."""
    if isinstance(raw_response, dict):
        parsed = raw_response
    elif isinstance(raw_response, str):
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError as error:
            raise GenerationError(f"Provider returned malformed JSON: {error}") from error
    else:
        raise GenerationError("Provider response must be a JSON object or JSON string.")

    if not isinstance(parsed, dict):
        raise GenerationError("Provider JSON must be an object.")
    if set(parsed) != set(CORE_FIELDS):
        raise GenerationError(
            "Provider JSON must contain exactly title, company_profile, description, requirements, benefits."
        )
    if not all(isinstance(parsed[field], str) for field in CORE_FIELDS):
        raise GenerationError("Every generated field must be a string.")
    return {field: parsed[field].strip() for field in CORE_FIELDS}


def redact_contact_details(value: str) -> str:
    """Remove contact channels from prompts to reduce accidental reuse of source details."""
    value = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[REDACTED_EMAIL]", value)
    return re.sub(r"(?<!\w)(?:\+?\d[\d .()/-]{6,}\d)", "[REDACTED_PHONE]", value)


def source_fields(record: pd.Series | dict[str, Any], redact_contacts: bool = True) -> dict[str, str]:
    """Extract the five source fields without changing the stored EMSCAD record."""
    source: dict[str, str] = {}
    for field in CORE_FIELDS:
        value = record.get(field, "")
        value = "" if pd.isna(value) else str(value)
        source[field] = redact_contact_details(value) if redact_contacts else value
    return source


def build_generation_prompt(prompt_template: str, source: dict[str, str]) -> str:
    """Insert source content at runtime rather than hardcoding advertisements in code."""
    placeholder = "{{SOURCE_ADVERTISEMENT_JSON}}"
    if placeholder not in prompt_template:
        raise GenerationError(f"Generation prompt is missing {placeholder}.")
    return prompt_template.replace(placeholder, json.dumps(source, ensure_ascii=False, indent=2))


def build_validation_prompt(
    prompt_template: str, source: dict[str, str], generated: dict[str, str]
) -> str:
    """Build the optional secondary LLM-validation prompt."""
    source_placeholder = "{{SOURCE_ADVERTISEMENT_JSON}}"
    generated_placeholder = "{{GENERATED_ADVERTISEMENT_JSON}}"
    if source_placeholder not in prompt_template or generated_placeholder not in prompt_template:
        raise GenerationError("Validation prompt is missing one or more JSON placeholders.")
    return prompt_template.replace(
        source_placeholder, json.dumps(source, ensure_ascii=False, indent=2)
    ).replace(generated_placeholder, json.dumps(generated, ensure_ascii=False, indent=2))


def select_pilot_sources(
    clean_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    sample_size: int = 20,
    random_seed: int = 42,
    legitimate_value: str = "f",
) -> pd.DataFrame:
    """Select deterministic, duplicate-aware legitimate sources with contextual variety.

    Selection first filters only to the verified legitimate target value, then
    retains one source per text_group_id. Industry/function and text-length
    buckets are used solely to spread the small pilot across source contexts.
    """
    required = {"source_row_id", "text_group_id", "fraudulent", *CORE_FIELDS, "text"}
    missing = required.difference(clean_df.columns)
    if missing:
        raise ValueError(f"Clean dataset lacks required columns: {sorted(missing)}")
    if len(raw_df) != len(clean_df):
        raise ValueError("Raw and processed datasets must have aligned source rows.")

    candidates = clean_df.loc[clean_df["fraudulent"].eq(legitimate_value)].copy()
    if candidates.empty:
        raise ValueError(f"No records found with fraudulent == {legitimate_value!r}.")
    candidates = candidates.sort_values("source_row_id").drop_duplicates("text_group_id", keep="first")
    if len(candidates) < sample_size:
        raise ValueError("Not enough unique legitimate text groups for the requested pilot size.")

    raw_metadata = raw_df.reindex(candidates["source_row_id"].astype(int))
    industry = raw_metadata.get("industry", pd.Series(index=raw_metadata.index, dtype=object))
    function = raw_metadata.get("function", pd.Series(index=raw_metadata.index, dtype=object))
    industry = industry.fillna("").astype(str).str.strip()
    function = function.fillna("").astype(str).str.strip()
    context = industry.mask(industry.eq(""), function).replace("", "Unknown context")
    candidates["selection_context"] = context.to_numpy()

    word_count = candidates["text"].str.findall(r"\b\w+\b").str.len()
    candidates["selection_length_bucket"] = pd.qcut(
        word_count.rank(method="first"), q=4, labels=["short", "medium", "long", "very_long"]
    ).astype(str)
    order = list(candidates.index)
    random.Random(random_seed).shuffle(order)
    shuffled = candidates.loc[order].copy()

    selected_indices: list[int] = []
    used_contexts: set[str] = set()
    used_buckets: set[str] = set()
    for index, row in shuffled.iterrows():
        if row["selection_context"] not in used_contexts and row["selection_length_bucket"] not in used_buckets:
            selected_indices.append(index)
            used_contexts.add(row["selection_context"])
            used_buckets.add(row["selection_length_bucket"])
        if len(selected_indices) == sample_size:
            break
    for index, row in shuffled.iterrows():
        if len(selected_indices) == sample_size:
            break
        if index not in selected_indices and row["selection_context"] not in used_contexts:
            selected_indices.append(index)
            used_contexts.add(row["selection_context"])
            used_buckets.add(row["selection_length_bucket"])
    for index in shuffled.index:
        if len(selected_indices) == sample_size:
            break
        if index not in selected_indices:
            selected_indices.append(index)

    return candidates.loc[selected_indices].sort_values("source_row_id").reset_index(drop=True)


def deterministic_synthetic_id(source_row_id: Any, source_text_group_id: str) -> str:
    """Create a stable ID for one Phase 2B-A source record."""
    value = f"emscad-pilot-v1:{source_row_id}:{source_text_group_id}"
    return f"pilot_{uuid.uuid5(uuid.NAMESPACE_URL, value).hex}"


def generate_fraudulent_job(
    source_record: pd.Series | dict[str, Any],
    provider: GenerationProvider,
    prompt_template: str,
    max_attempts: int = 2,
    request_pause_seconds: float = 0.0,
) -> dict[str, Any]:
    """Generate one structured record with bounded retries and full provenance."""
    source = source_fields(source_record, redact_contacts=True)
    source_row_id = int(source_record["source_row_id"])
    source_group_id = str(source_record["text_group_id"])
    record: dict[str, Any] = {
        "synthetic_id": deterministic_synthetic_id(source_row_id, source_group_id),
        "source_row_id": source_row_id,
        "source_text_group_id": source_group_id,
        "source_dataset": SOURCE_DATASET,
        "source_label": SOURCE_LABEL,
        "generated_label": (
            GENERATED_LABEL
            if provider.generation_mode == "api"
            else "mock_not_real_generated_data"
        ),
        "provider": provider.name,
        "model": provider.model,
        "generation_mode": provider.generation_mode,
        "generation_timestamp": datetime.now(UTC).isoformat(),
        "prompt_version": PROMPT_VERSION,
        "source": source,
        "generated": None,
        "generation_status": "failed",
        "generation_attempts": 0,
        "generation_error": None,
        "validation": {
            "structural_valid": False,
            "context_preserved": False,
            "fraud_intent_present": False,
            "quality_ok": False,
            "llm_validation": None,
            "validation_status": "pending",
        },
    }
    prompt = build_generation_prompt(prompt_template, source)

    for attempt in range(1, max_attempts + 1):
        record["generation_attempts"] = attempt
        try:
            record["generated"] = parse_structured_response(provider.generate_json(prompt, JOB_SCHEMA))
            record["generation_status"] = "generated"
            record["generation_error"] = None
            return record
        except GenerationError as error:
            record["generation_error"] = str(error)
            if attempt < max_attempts and request_pause_seconds > 0:
                time.sleep(request_pause_seconds)
    return record


def save_generation_records(records: list[dict[str, Any]], output_path: Path) -> None:
    """Write audit-friendly JSONL without changing any EMSCAD input file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            # ASCII escapes preserve JSONL's one-record-per-physical-line contract
            # even when EMSCAD contains Unicode line-separator characters.
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
