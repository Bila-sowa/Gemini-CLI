"""Configuration commands."""

from __future__ import annotations

import typer

_NOT_IMPLEMENTED = "config commands arrive in Phase 8.5."


def show() -> None:
    """Print all resolved settings with the API key redacted."""
    typer.secho(_NOT_IMPLEMENTED, err=True)
    raise typer.Exit(code=1)


def set_value(
    key: str = typer.Argument(..., help="Setting name."),
    value: str = typer.Argument(..., help="New value."),
    global_: bool = typer.Option(
        False,
        "--global",
        help="Write to the user-level config instead of the project config.",
    ),
) -> None:
    """Write one setting to the project or global configuration."""
    del key, value, global_
    typer.secho(_NOT_IMPLEMENTED, err=True)
    raise typer.Exit(code=1)


__all__ = ["set_value", "show"]
