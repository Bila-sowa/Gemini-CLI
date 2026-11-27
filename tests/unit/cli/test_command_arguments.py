"""Tests for slash-command argument validation and suggestions."""

from __future__ import annotations

import pytest

from gemini_cli.cli.command_arguments import (
    CommandArgumentError,
    parse_model_argument,
    parse_role_argument,
    parse_theme_argument,
)


def test_registered_model_is_accepted() -> None:
    assert parse_model_argument("gemini-4-argon") == "gemini-4-argon"


def test_model_is_case_insensitive() -> None:
    assert parse_model_argument("  GEMINI-4-ARGON  ") == "gemini-4-argon"


def test_unknown_model_suggests_a_close_match() -> None:
    with pytest.raises(CommandArgumentError, match=r"gemini-3\.5-flash-lite"):
        parse_model_argument("gemini-3.5-flash-lit")


def test_unknown_model_without_a_close_match_explains_custom_models() -> None:
    with pytest.raises(CommandArgumentError, match="GEMINI_MODEL"):
        parse_model_argument("my-private-model")


def test_empty_model_is_rejected() -> None:
    with pytest.raises(CommandArgumentError, match="cannot be empty"):
        parse_model_argument("   ")


def test_role_is_accepted_and_normalized() -> None:
    assert parse_role_argument("  CODE  ") == "code"


def test_unknown_role_suggests_a_close_match() -> None:
    with pytest.raises(CommandArgumentError, match=r"Did you mean 'code'\?"):
        parse_role_argument("coder")


def test_unknown_role_lists_available_roles() -> None:
    with pytest.raises(CommandArgumentError, match="Available: basic, code"):
        parse_role_argument("wizard")


def test_empty_role_is_rejected() -> None:
    with pytest.raises(CommandArgumentError, match="cannot be empty"):
        parse_role_argument(" ")


def test_theme_is_accepted_and_normalized() -> None:
    assert parse_theme_argument(" LIGHT ") == "light"


def test_unknown_theme_suggests_a_close_match() -> None:
    with pytest.raises(CommandArgumentError, match=r"Did you mean 'light'\?"):
        parse_theme_argument("lite")


def test_unknown_theme_lists_available_themes() -> None:
    with pytest.raises(CommandArgumentError, match="Available: dark, light"):
        parse_theme_argument("solarized")


def test_empty_theme_is_rejected() -> None:
    with pytest.raises(CommandArgumentError, match="cannot be empty"):
        parse_theme_argument("")
