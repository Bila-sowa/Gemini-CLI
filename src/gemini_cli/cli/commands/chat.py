"""Interactive chat command."""

from __future__ import annotations

import asyncio
import sys

import typer

from gemini_cli.auth import AuthenticationError, CredentialStoreError
from gemini_cli.cli.runtime import build_runtime
from gemini_cli.cli.tui.app import TuiApprovalBridge, run_tui
from gemini_cli.cli.tui.repl import ConsoleApprovalCallback, run_repl
from gemini_cli.config import Settings

_ERROR_COLOR = typer.colors.RED


def _use_tui(no_tui: bool) -> bool:
    """Use the full-screen UI only on a real interactive terminal."""
    if no_tui:
        return False
    return sys.stdin.isatty() and sys.stdout.isatty()


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
    no_tui: bool = typer.Option(
        False,
        "--no-tui",
        help="Use the plain line interface instead of the full-screen UI.",
    ),
) -> None:
    """Start an interactive conversation."""
    try:
        asyncio.run(_chat(model, role, no_tui))
    except KeyboardInterrupt:
        raise typer.Exit(code=130) from None
    except (AuthenticationError, CredentialStoreError) as exc:
        typer.secho(f"Error: {exc}", fg=_ERROR_COLOR, err=True)
        raise typer.Exit(code=1) from exc


async def _chat(model: str | None, role: str | None, no_tui: bool) -> None:
    settings = Settings()
    use_tui = _use_tui(no_tui)
    bridge = TuiApprovalBridge() if use_tui else None
    runtime = await build_runtime(
        settings,
        model=model,
        role=role,
        approval_callback=bridge if bridge is not None else ConsoleApprovalCallback(),
    )
    try:
        if use_tui and bridge is not None:
            await run_tui(runtime, bridge=bridge)
        else:
            await run_repl(runtime)
    finally:
        runtime.close()


__all__ = ["chat"]
