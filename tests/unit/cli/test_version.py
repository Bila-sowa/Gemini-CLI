"""Smoke tests for the CLI entry points."""

from __future__ import annotations

import subprocess
import sys

from typer.testing import CliRunner

from gemini_cli import __version__
from gemini_cli.cli.app import app

runner = CliRunner()


def test_version_flag_prints_version() -> None:
    """``--version`` prints the package version and exits successfully."""
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == f"v{__version__}"


def test_module_entry_point_prints_version() -> None:
    """``python -m gemini_cli --version`` works without extra setup."""
    result = subprocess.run(
        [sys.executable, "-m", "gemini_cli", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == f"v{__version__}"
