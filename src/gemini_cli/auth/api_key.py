"""API key resolution from settings and the OS keyring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from gemini_cli.auth import AuthenticationError
from gemini_cli.auth.credentials_store import load_api_key
from gemini_cli.config import Settings

NO_API_KEY_MESSAGE = (
    "No Gemini API key found. Set GEMINI_API_KEY or store one with the login command."
)


@dataclass(frozen=True)
class ResolvedApiKey:
    """A resolved API key with its source for diagnostics."""

    key: str = field(repr=False)
    source: Literal["environment", "keyring"]


def resolve_api_key(settings: Settings) -> ResolvedApiKey:
    """Resolve a Gemini API key from settings or the OS keyring.

    Settings win because they include explicit configuration and environment
    variables. Interactive prompting is handled by the CLI login flow, not here.

    The key is never logged and never included in an error message.

    Raises:
        AuthenticationError: if neither settings nor the keyring provide a key.
        CredentialStoreError: if the keyring backend itself fails.
    """
    configured_key = settings.api_key.strip()
    if configured_key:
        return ResolvedApiKey(key=configured_key, source="environment")

    stored_key = load_api_key()
    if stored_key:
        return ResolvedApiKey(key=stored_key, source="keyring")

    raise AuthenticationError(NO_API_KEY_MESSAGE)
