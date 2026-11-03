"""Interactive chat command."""

from __future__ import annotations

import typer

_NOT_IMPLEMENTED = "chat arrives in Phase 8.2."


def chat(
    model: str | None = typer.Option(
        None,
        "--model",
        help="Override the configured model.",
    ),
    role: str | None = typer.Option(
        None,
        "--role",
        help="Set the assistant role.",
    ),
) -> None:
    """Start an interactive conversation."""
    del model, role
    typer.secho(_NOT_IMPLEMENTED, err=True)
    raise typer.Exit(code=1)


__all__ = ["chat"]
