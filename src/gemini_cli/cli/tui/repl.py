"""Minimal interactive REPL.

This module starts as a plain line-oriented loop so the ``chat`` command works
end to end. Phase 8.6 replaces the input layer with prompt_toolkit, adds
selection menus, streaming presentation, slash commands, and the quota panel.
"""

from __future__ import annotations

from collections.abc import Callable

import typer

from gemini_cli import __version__
from gemini_cli.cli.runtime import CliRuntime
from gemini_cli.core.session import Session
from gemini_cli.core.streaming import AgentEventType
from gemini_cli.sandbox.approval import ApprovalDecision

_EXIT_COMMANDS = frozenset({"exit", "quit", "/exit", "/quit"})
_SLASH_MESSAGE = "Slash commands arrive in Phase 8.6."


class ConsoleApprovalCallback:
    """Ask the user to confirm a sensitive tool call.

    Any input error is treated as a denial so an approval prompt can never
    fail open.
    """

    def __init__(
        self,
        *,
        input_func: Callable[[str], str] | None = None,
    ) -> None:
        self._input = input_func if input_func is not None else input

    async def request_approval(
        self,
        action_description: str,
        risk_level: str,
        details: str,
    ) -> ApprovalDecision:
        typer.echo(f"\nApproval required ({risk_level}): {action_description}")
        typer.echo(f"  {details}")
        try:
            answer = self._input("Approve? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return ApprovalDecision.DENIED
        if answer in {"y", "yes"}:
            return ApprovalDecision.APPROVED
        return ApprovalDecision.DENIED


async def run_repl(
    runtime: CliRuntime,
    *,
    input_func: Callable[[str], str] | None = None,
    output: Callable[[str], None] | None = None,
) -> None:
    """Run the conversation loop until the user exits or input ends."""
    read = input_func if input_func is not None else input
    write = output if output is not None else _echo

    settings = runtime.settings
    write(
        f"GEMINI CLI v{__version__}  model: {settings.model}  role: {settings.role}\n\n"
    )
    session = Session(model=settings.model, role=settings.role)

    while True:
        try:
            line = read("> ")
        except (EOFError, KeyboardInterrupt):
            write("\n")
            return

        message = line.strip()
        if not message:
            continue
        if message.lower() in _EXIT_COMMANDS:
            return
        if message.startswith("/"):
            write(f"{_SLASH_MESSAGE}\n")
            continue

        async for event in runtime.agent.run(session, message):
            if event.event_type is AgentEventType.TEXT_CHUNK:
                write(str(event.data))
            elif event.event_type is AgentEventType.ERROR:
                write(f"\nError: {event.data}\n")
            elif event.event_type is AgentEventType.DONE:
                write("\n")


def _echo(text: str) -> None:
    typer.echo(text, nl=False)


__all__ = ["ConsoleApprovalCallback", "run_repl"]
