"""Authentication commands."""

from __future__ import annotations

import typer

_NOT_IMPLEMENTED = "auth commands arrive in Phase 8.4."


def login(
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        help="API key to store. Omit to be prompted.",
    ),
) -> None:
    """Store a Gemini API key in the OS keyring."""
    del api_key
    typer.secho(_NOT_IMPLEMENTED, err=True)
    raise typer.Exit(code=1)


def logout() -> None:
    """Remove the stored Gemini API key."""
    typer.secho(_NOT_IMPLEMENTED, err=True)
    raise typer.Exit(code=1)


def status() -> None:
    """Show the active authentication source."""
    typer.secho(_NOT_IMPLEMENTED, err=True)
    raise typer.Exit(code=1)


__all__ = ["login", "logout", "status"]
