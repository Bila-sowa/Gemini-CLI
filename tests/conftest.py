"""Root conftest — shared fixtures and environment overrides."""

from __future__ import annotations

import pytest

# A developer's local ``.env`` must never influence the test suite. Each value
# below mirrors the built-in defaults so explicit environment variables take
# precedence over the file for the whole session of a test.
_DEFAULT_ENV = {
    "GEMINI_API_KEY": "",
    "GEMINI_MODEL": "gemini-3.8-flash",
    "GEMINI_REQUEST_TIMEOUT_MS": "60000",
    "GEMINI_RETRY_ATTEMPTS": "3",
    "GEMINI_MAX_OUTPUT_TOKENS": "8192",
    "GEMINI_MAX_HISTORY_TURNS": "20",
    "GEMINI_MAX_INPUT_CHARS": "32000",
    "GEMINI_QUOTA_5H_TOKEN_LIMIT": "200000",
    "GEMINI_QUOTA_WEEKLY_TOKEN_LIMIT": "1000000",
    "GEMINI_TELEMETRY_ENABLED": "false",
}


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent tests from using real credentials or local config."""
    for name, value in _DEFAULT_ENV.items():
        monkeypatch.setenv(name, value)
