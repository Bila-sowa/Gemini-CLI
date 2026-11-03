"""One-shot execution command."""

from __future__ import annotations

import asyncio

import typer

from gemini_cli.auth import AuthenticationError, CredentialStoreError
from gemini_cli.cli.runtime import build_runtime
from gemini_cli.config import Settings
from gemini_cli.core.session import Session
from gemini_cli.core.streaming import AgentEventType

_ERROR_COLOR = typer.colors.RED


def run(
    prompt: str = typer.Argument(..., help="The task to execute."),
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
    """Execute a single prompt and print the result."""
    try:
        exit_code = asyncio.run(_run(prompt, model, role))
    except KeyboardInterrupt:
        raise typer.Exit(code=130) from None
    except (AuthenticationError, CredentialStoreError) as exc:
        typer.secho(f"Error: {exc}", fg=_ERROR_COLOR, err=True)
        raise typer.Exit(code=1) from exc

    if exit_code != 0:
        raise typer.Exit(code=exit_code)


async def _run(
    prompt: str,
    model: str | None,
    role: str | None,
) -> int:
    settings = Settings()
    runtime = await build_runtime(
        settings,
        model=model,
        role=role,
        # No callback: non-interactive runs are fail-closed.
        approval_callback=None,
    )
    try:
        session = Session(
            model=runtime.settings.model,
            role=runtime.settings.role,
        )
        failed = False

        async for event in runtime.agent.run(session, prompt):
            if event.event_type is AgentEventType.TEXT_CHUNK:
                typer.echo(str(event.data), nl=False)
            elif event.event_type is AgentEventType.ERROR:
                typer.secho(
                    f"\nError: {event.data}",
                    fg=_ERROR_COLOR,
                    err=True,
                )
                failed = True
            elif event.event_type is AgentEventType.DONE:
                typer.echo()

        return 1 if failed else 0
    finally:
        runtime.close()


__all__ = ["run"]
