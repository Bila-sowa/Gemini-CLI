"""Authentication commands."""

from __future__ import annotations

import asyncio

import typer

from gemini_cli.auth import AuthenticationError, CredentialStoreError
from gemini_cli.auth.api_key import resolve_api_key
from gemini_cli.auth.credentials_store import delete_api_key, store_api_key
from gemini_cli.config import Settings
from gemini_cli.llm import GeminiClient, LLMError

_ERROR_COLOR = typer.colors.RED
_WARN_COLOR = typer.colors.YELLOW


def _print_error(message: str) -> None:
    typer.secho(f"Error: {message}", fg=_ERROR_COLOR, err=True)


async def _verify_key(api_key: str) -> None:
    settings = Settings()
    client = GeminiClient(settings, api_key=api_key)
    try:
        await client.verify()
    finally:
        client.close()


def login(
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        help="API key to store. Omit to be prompted.",
    ),
    verify: bool = typer.Option(
        True,
        "--verify/--no-verify",
        help="Check the key against the Gemini API before storing it.",
    ),
) -> None:
    """Store a Gemini API key in the OS keyring."""
    key = (api_key or "").strip()
    if not key:
        key = typer.prompt("Gemini API key", hide_input=True).strip()
    if not key:
        _print_error("the API key must not be empty.")
        raise typer.Exit(code=1)

    if verify:
        try:
            asyncio.run(_verify_key(key))
        except LLMError as exc:
            _print_error(f"the key could not be verified: {exc}")
            raise typer.Exit(code=1) from exc

    try:
        store_api_key(key)
    except CredentialStoreError as exc:
        _print_error(str(exc))
        raise typer.Exit(code=1) from exc

    typer.echo("API key stored in the OS keyring.")


def logout() -> None:
    """Remove the stored Gemini API key."""
    try:
        delete_api_key()
    except CredentialStoreError as exc:
        _print_error(str(exc))
        raise typer.Exit(code=1) from exc

    typer.echo("Stored API key removed.")
    if Settings().api_key.strip():
        typer.secho(
            "Note: GEMINI_API_KEY is still configured and takes precedence.",
            fg=_WARN_COLOR,
            err=True,
        )


def status() -> None:
    """Show the active authentication source."""
    try:
        resolved = resolve_api_key(Settings())
    except CredentialStoreError as exc:
        _print_error(str(exc))
        raise typer.Exit(code=1) from exc
    except AuthenticationError:
        typer.secho("Authentication: not configured", fg=_WARN_COLOR)
        typer.echo("Set GEMINI_API_KEY or run 'gemini-cli-py login' to store a key.")
        raise typer.Exit(code=1) from None

    typer.echo(f"Authentication: {resolved.source}")
    typer.echo("API key: set (value hidden)")


__all__ = ["login", "logout", "status"]
