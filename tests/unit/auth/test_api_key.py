"""Tests for API key resolution."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from gemini_cli.auth import AuthenticationError, CredentialStoreError
from gemini_cli.auth import api_key as api_key_module
from gemini_cli.auth.api_key import ResolvedApiKey, resolve_api_key
from gemini_cli.config import Settings


@pytest.fixture(autouse=True)
def _isolated_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep resolution away from real env vars, config files, and home dir."""
    for name in list(os.environ):
        if name.startswith("GEMINI_"):
            monkeypatch.delenv(name, raising=False)

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.chdir(tmp_path)


def make_settings(configured_key: str = "") -> Settings:
    """Build settings without reading a developer's local ``.env`` file."""
    return Settings(_env_file=None, api_key=configured_key)


def test_settings_key_wins_and_keyring_is_not_queried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_keyring_call() -> str | None:
        raise AssertionError("keyring must not be queried")

    monkeypatch.setattr(api_key_module, "load_api_key", unexpected_keyring_call)

    resolved = resolve_api_key(make_settings("env-key"))

    assert resolved == ResolvedApiKey(key="env-key", source="environment")


def test_keyring_is_used_when_settings_have_no_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_key_module, "load_api_key", lambda: "stored-key")

    resolved = resolve_api_key(make_settings())

    assert resolved.key == "stored-key"
    assert resolved.source == "keyring"


def test_missing_key_raises_authentication_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_key_module, "load_api_key", lambda: None)

    with pytest.raises(AuthenticationError, match="GEMINI_API_KEY"):
        resolve_api_key(make_settings())


def test_settings_key_is_stripped() -> None:
    resolved = resolve_api_key(make_settings("  env-key  "))

    assert resolved.key == "env-key"


def test_whitespace_only_settings_fall_back_to_keyring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_key_module, "load_api_key", lambda: "stored-key")

    resolved = resolve_api_key(make_settings("   "))

    assert resolved.key == "stored-key"
    assert resolved.source == "keyring"


def test_settings_key_takes_precedence_over_keyring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_key_module, "load_api_key", lambda: "stored-key")

    resolved = resolve_api_key(make_settings("env-key"))

    assert resolved == ResolvedApiKey(key="env-key", source="environment")


def test_empty_keyring_value_is_treated_as_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_key_module, "load_api_key", lambda: "")

    with pytest.raises(AuthenticationError):
        resolve_api_key(make_settings())


def test_credential_store_errors_propagate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_load() -> str | None:
        raise CredentialStoreError("keyring unavailable")

    monkeypatch.setattr(api_key_module, "load_api_key", failing_load)

    with pytest.raises(CredentialStoreError, match="keyring unavailable"):
        resolve_api_key(make_settings())


def test_repr_does_not_expose_the_key() -> None:
    resolved = ResolvedApiKey(key="super-secret-key", source="environment")

    assert "super-secret-key" not in repr(resolved)
