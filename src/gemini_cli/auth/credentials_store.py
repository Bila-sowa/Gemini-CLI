"""Persistent credential storage using the OS keyring."""

from __future__ import annotations

import keyring
from keyring.errors import KeyringError, PasswordDeleteError

from gemini_cli.auth import CredentialStoreError

SERVICE_NAME = "gemini-cli"
API_KEY_USERNAME = "api-key"


def store_api_key(api_key: str) -> None:
    """Store *api_key* in the OS keyring.

    Raises:
        CredentialStoreError: if the keyring backend fails.
    """
    try:
        keyring.set_password(SERVICE_NAME, API_KEY_USERNAME, api_key)
    except KeyringError as exc:
        raise CredentialStoreError(
            "Could not store the API key in the OS keyring."
        ) from exc


def load_api_key() -> str | None:
    """Return the stored API key, or ``None`` when none is stored.

    Raises:
        CredentialStoreError: if the keyring backend fails.
    """
    try:
        api_key = keyring.get_password(SERVICE_NAME, API_KEY_USERNAME)
    except KeyringError as exc:
        raise CredentialStoreError(
            "Could not read the API key from the OS keyring."
        ) from exc
    return api_key or None


def delete_api_key() -> None:
    """Remove the stored API key.

    Deleting a missing key is a no-op so logout stays idempotent.

    Raises:
        CredentialStoreError: if the keyring backend fails.
    """
    try:
        keyring.delete_password(SERVICE_NAME, API_KEY_USERNAME)
    except PasswordDeleteError:
        return
    except KeyringError as exc:
        raise CredentialStoreError(
            "Could not delete the API key from the OS keyring."
        ) from exc


def has_stored_credentials() -> bool:
    """Return whether an API key is stored in the OS keyring.

    Raises:
        CredentialStoreError: if the keyring backend fails. A backend failure
            is not reported as "no credentials".
    """
    return load_api_key() is not None
