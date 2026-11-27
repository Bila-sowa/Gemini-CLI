"""Typer application and command registration."""

from __future__ import annotations

import typer

from gemini_cli import __version__
from gemini_cli.auth import AuthenticationError, CredentialStoreError
from gemini_cli.cli.commands import auth, chat, config, run
from gemini_cli.config import Settings
from gemini_cli.llm import UnknownModelError
from gemini_cli.llm.prompts import PromptError
from gemini_cli.telemetry import configure_logging

app = typer.Typer(
    name="gemini-cli-py",
    help="A Gemini-first command-line agent for everyday work.",
    no_args_is_help=True,
)

app.command("chat")(chat.chat)
app.command("run")(run.run)
app.command("login")(auth.login)
app.command("logout")(auth.logout)

auth_app = typer.Typer(
    help="Authentication commands.",
    no_args_is_help=True,
)
auth_app.command("status")(auth.status)
app.add_typer(auth_app, name="auth")

config_app = typer.Typer(
    help="Configuration commands.",
    no_args_is_help=True,
)
config_app.command("show")(config.show)
config_app.command("set")(config.set_value)
app.add_typer(config_app, name="config")


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
    """Configure logging and run the Typer application."""
    try:
        settings = Settings()
        configure_logging(telemetry_enabled=settings.telemetry_enabled)
        app()
    except (
        AuthenticationError,
        CredentialStoreError,
        PromptError,
        UnknownModelError,
    ) as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
