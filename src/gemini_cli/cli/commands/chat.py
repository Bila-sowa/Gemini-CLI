"""Interactive chat command."""

from __future__ import annotations

import asyncio

import typer

from gemini_cli.auth import AuthenticationError, CredentialStoreError
from gemini_cli.cli.runtime import build_runtime
from gemini_cli.cli.tui.repl import ConsoleApprovalCallback, run_repl
from gemini_cli.config import Settings

_ERROR_COLOR = typer.colors.RED


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
    try:
        asyncio.run(_chat(model, role))
    except KeyboardInterrupt:
        raise typer.Exit(code=130) from None
    except (AuthenticationError, CredentialStoreError) as exc:
        typer.secho(f"Error: {exc}", fg=_ERROR_COLOR, err=True)
        raise typer.Exit(code=1) from exc


async def _chat(model: str | None, role: str | None) -> None:
    settings = Settings()
    runtime = await build_runtime(
        settings,
        model=model,
        role=role,
        approval_callback=ConsoleApprovalCallback(),
    )
    try:
        await run_repl(runtime)
    finally:
        runtime.close()


__all__ = ["chat"]
