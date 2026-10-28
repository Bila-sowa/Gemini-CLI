"""Authentication package."""

from __future__ import annotations

__all__ = ["AuthenticationError", "CredentialStoreError"]


class AuthenticationError(Exception):
    """Raised when credentials cannot be resolved."""


class CredentialStoreError(Exception):
    """Raised when the OS keyring is unavailable or fails."""
