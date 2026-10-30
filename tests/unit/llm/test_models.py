"""Tests for the Gemini model registry and tool declarations."""

from __future__ import annotations

import dataclasses

import pytest

from gemini_cli.llm import models
from gemini_cli.llm.models import (
    MODEL_REGISTRY,
    ModelInfo,
    ModelTier,
    ToolDefinition,
    UnknownModelError,
    get_fallback,
    get_model,
    list_models,
)


def test_registry_contains_flash_and_pro() -> None:
    flash = get_model("gemini-2.5-flash")
    pro = get_model("gemini-2.5-pro")

    assert flash.tier is ModelTier.FLASH
    assert pro.tier is ModelTier.PRO
    assert flash.context_window > 0
    assert flash.max_output_tokens > 0
    assert flash.description


def test_unknown_model_raises_with_available_names() -> None:
    with pytest.raises(UnknownModelError) as excinfo:
        get_model("gemini-3.0-ultra")

    message = str(excinfo.value)
    assert "gemini-3.0-ultra" in message
    assert "gemini-2.5-flash" in message


def test_pro_falls_back_to_flash() -> None:
    fallback = get_fallback("gemini-2.5-pro")

    assert fallback is not None
    assert fallback.name == "gemini-2.5-flash"


def test_flash_has_no_fallback() -> None:
    assert get_fallback("gemini-2.5-flash") is None


def test_unknown_model_fallback_raises() -> None:
    with pytest.raises(UnknownModelError):
        get_fallback("unknown-model")


def test_fallback_returns_single_link_without_looping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    looped = dataclasses.replace(
        MODEL_REGISTRY["gemini-2.5-pro"], fallback="gemini-2.5-pro"
    )
    monkeypatch.setitem(models.MODEL_REGISTRY, "looping-model", looped)

    fallback = get_fallback("looping-model")

    assert fallback is not None
    assert fallback.name == "gemini-2.5-pro"


def test_list_models_returns_registry_entries() -> None:
    entries = list_models()

    assert entries
    assert all(isinstance(entry, ModelInfo) for entry in entries)
    assert {entry.name for entry in entries} == set(MODEL_REGISTRY)


def test_tool_definition_from_export() -> None:
    definition = ToolDefinition.from_export(
        {
            "name": "read_file",
            "description": "Read a file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        }
    )

    assert definition.name == "read_file"
    assert definition.description == "Read a file."
    assert definition.parameters_json_schema["type"] == "object"


def test_tool_definition_prefers_explicit_schema_key() -> None:
    definition = ToolDefinition.from_export(
        {
            "name": "grep",
            "parameters": {"type": "object"},
            "parameters_json_schema": {"type": "object", "properties": {}},
        }
    )

    assert definition.parameters_json_schema == {"type": "object", "properties": {}}


def test_tool_definition_defaults() -> None:
    definition = ToolDefinition(name="noop", description="Does nothing.")

    assert definition.parameters_json_schema == {}


@pytest.mark.parametrize("payload", [{}, {"name": ""}, {"name": 1}])
def test_tool_definition_rejects_missing_name(payload: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="name"):
        ToolDefinition.from_export(payload)
