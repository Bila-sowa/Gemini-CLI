"""One-shot execution command."""

from __future__ import annotations

import typer

_NOT_IMPLEMENTED = "run arrives in Phase 8.3."


def run(
    prompt: str = typer.Argument(..., help="The task to execute."),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Override the configured model.",
    ),
) -> None:
    """Execute a single prompt and print the result."""
    del prompt, model
    typer.secho(_NOT_IMPLEMENTED, err=True)
    raise typer.Exit(code=1)


__all__ = ["run"]
