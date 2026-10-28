"""Typer application and command registration."""

from __future__ import annotations

import typer

from gemini_cli import __version__

app = typer.Typer(
    name="gemini-cli-py",
    help="A Gemini-first command-line agent for everyday work.",
    no_args_is_help=True,
)


def version_callback(value: bool) -> None:
    """Print the package version and exit when ``--version`` is used."""
    if value:
        typer.echo(f"v{__version__}")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """Gemini CLI — your daily AI companion in the terminal."""


def main() -> None:
    """Run the Typer application."""
    app()


# TODO(Phase 8): register chat, run, auth, config, and mcp subcommands.
