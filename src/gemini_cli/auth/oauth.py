"""Google OAuth provider stub."""

from __future__ import annotations

OAUTH_NOT_SUPPORTED = "Google OAuth is not yet supported."


class OAuthProvider:
    """Placeholder for the future Google OAuth flow.

    Every method raises :class:`NotImplementedError` until OAuth support is
    implemented, so callers can depend on the interface without relying on
    unfinished behavior.
    """

    async def login(self) -> str:
        """Start the Google OAuth flow and return an access token."""
        raise NotImplementedError(OAUTH_NOT_SUPPORTED)

    async def logout(self) -> None:
        """Revoke or discard stored OAuth credentials."""
        raise NotImplementedError(OAUTH_NOT_SUPPORTED)

    async def get_access_token(self) -> str:
        """Return a valid access token, refreshing it when needed."""
        raise NotImplementedError(OAUTH_NOT_SUPPORTED)
