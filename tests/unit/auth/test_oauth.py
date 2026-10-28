"""Tests for the Google OAuth provider stub."""

from __future__ import annotations

import pytest

from gemini_cli.auth.oauth import OAUTH_NOT_SUPPORTED, OAuthProvider


def test_provider_can_be_instantiated() -> None:
    assert isinstance(OAuthProvider(), OAuthProvider)


def test_not_supported_message() -> None:
    assert OAUTH_NOT_SUPPORTED == "Google OAuth is not yet supported."


async def test_login_is_not_implemented() -> None:
    provider = OAuthProvider()

    with pytest.raises(NotImplementedError) as excinfo:
        await provider.login()

    assert str(excinfo.value) == OAUTH_NOT_SUPPORTED


async def test_logout_is_not_implemented() -> None:
    provider = OAuthProvider()

    with pytest.raises(NotImplementedError) as excinfo:
        await provider.logout()

    assert str(excinfo.value) == OAUTH_NOT_SUPPORTED


async def test_get_access_token_is_not_implemented() -> None:
    provider = OAuthProvider()

    with pytest.raises(NotImplementedError) as excinfo:
        await provider.get_access_token()

    assert str(excinfo.value) == OAUTH_NOT_SUPPORTED
