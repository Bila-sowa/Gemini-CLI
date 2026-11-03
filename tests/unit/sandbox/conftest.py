"""Shared fixtures for sandbox tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from gemini_cli.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings whose only allowed root is the test's temporary directory."""
    return Settings(
        _env_file=None,
        allowed_roots=[tmp_path],
        shell_timeout_seconds=2,
        shell_max_output_bytes=4096,
    )
