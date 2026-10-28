"""Smoke tests for the CLI entry point."""

from __future__ import annotations

from typer.testing import CliRunner

from gemini_cli import __version__
from gemini_cli.cli.app import app

runner = CliRunner()


def test_version_flag_prints_version() -> None:
    """``--version`` prints the package version and exits successfully."""
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == f"v{__version__}"
