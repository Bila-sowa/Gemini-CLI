"""Shared fixtures for tool tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from gemini_cli.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings whose only allowed root is the test's temporary directory."""
    return Settings(allowed_roots=[tmp_path])
