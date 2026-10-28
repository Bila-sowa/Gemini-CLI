"""Tests for OS keyring credential storage."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from keyring.errors import (
    KeyringError,
    NoKeyringError,
    PasswordDeleteError,
    PasswordSetError,
)

from gemini_cli.auth import CredentialStoreError, credentials_store
from gemini_cli.auth.credentials_store import (
    API_KEY_USERNAME,
    SERVICE_NAME,
    delete_api_key,
    has_stored_credentials,
    load_api_key,
    store_api_key,
)


@pytest.fixture
def fake_keyring(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[dict[tuple[str, str], str]]:
    """Replace the keyring module functions with an in-memory store."""
    storage: dict[tuple[str, str], str] = {}

    def set_password(service_name: str, username: str, password: str) -> None:
        storage[(service_name, username)] = password

    def get_password(service_name: str, username: str) -> str | None:
        return storage.get((service_name, username))

    def delete_password(service_name: str, username: str) -> None:
        if (service_name, username) not in storage:
            raise PasswordDeleteError("password not found")
        del storage[(service_name, username)]

    monkeypatch.setattr(credentials_store.keyring, "set_password", set_password)
    monkeypatch.setattr(credentials_store.keyring, "get_password", get_password)
    monkeypatch.setattr(credentials_store.keyring, "delete_password", delete_password)

    yield storage


def test_store_and_load_round_trip(
    fake_keyring: dict[tuple[str, str], str],
) -> None:
    store_api_key("secret-key")

    assert load_api_key() == "secret-key"
    assert fake_keyring[(SERVICE_NAME, API_KEY_USERNAME)] == "secret-key"


def test_load_returns_none_when_nothing_is_stored(
    fake_keyring: dict[tuple[str, str], str],
) -> None:
    assert load_api_key() is None


def test_empty_stored_value_is_treated_as_missing(
    fake_keyring: dict[tuple[str, str], str],
) -> None:
    store_api_key("")

    assert load_api_key() is None


def test_has_stored_credentials(
    fake_keyring: dict[tuple[str, str], str],
) -> None:
    assert has_stored_credentials() is False

    store_api_key("secret-key")

    assert has_stored_credentials() is True


def test_delete_removes_stored_key(
    fake_keyring: dict[tuple[str, str], str],
) -> None:
    store_api_key("secret-key")

    delete_api_key()

    assert load_api_key() is None


def test_delete_is_idempotent(
    fake_keyring: dict[tuple[str, str], str],
) -> None:
    delete_api_key()
    delete_api_key()

    assert load_api_key() is None


def test_store_error_becomes_credential_store_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_set_password(service_name: str, username: str, password: str) -> None:
        raise PasswordSetError("backend is locked")

    monkeypatch.setattr(credentials_store.keyring, "set_password", failing_set_password)

    with pytest.raises(CredentialStoreError) as excinfo:
        store_api_key("super-secret-key")

    assert "super-secret-key" not in str(excinfo.value)


def test_load_error_becomes_credential_store_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_get_password(service_name: str, username: str) -> str | None:
        raise NoKeyringError("no usable backend")

    monkeypatch.setattr(credentials_store.keyring, "get_password", failing_get_password)

    with pytest.raises(CredentialStoreError):
        load_api_key()


def test_has_stored_credentials_propagates_backend_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_get_password(service_name: str, username: str) -> str | None:
        raise NoKeyringError("no usable backend")

    monkeypatch.setattr(credentials_store.keyring, "get_password", failing_get_password)

    with pytest.raises(CredentialStoreError):
        has_stored_credentials()


def test_delete_error_becomes_credential_store_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_delete_password(service_name: str, username: str) -> None:
        raise KeyringError("backend failure")

    monkeypatch.setattr(
        credentials_store.keyring, "delete_password", failing_delete_password
    )

    with pytest.raises(CredentialStoreError):
        delete_api_key()


def test_keyring_identifiers() -> None:
    assert SERVICE_NAME == "gemini-cli"
    assert API_KEY_USERNAME == "api-key"
