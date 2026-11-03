"""Plain line-oriented fallback for non-interactive terminals.

The full-screen Textual app in :mod:`gemini_cli.cli.tui.app` is the primary
interface. This module keeps a minimal line mode for pipes, CI, and
``--no-tui``: it streams text, reports errors, and supports the commands that
do not need a menu.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from gemini_cli import __version__
from gemini_cli.auth import CredentialStoreError
from gemini_cli.auth.credentials_store import delete_api_key
from gemini_cli.cli.runtime import CliRuntime
from gemini_cli.config.defaults import AVAILABLE_ROLES
from gemini_cli.core.session import Session
from gemini_cli.core.streaming import AgentEventType
from gemini_cli.llm import MODEL_REGISTRY
from gemini_cli.sandbox.approval import ApprovalDecision

_EXIT_COMMANDS = frozenset({"/exit", "/quit", "exit", "quit"})
_MENU_COMMANDS = frozenset({"/role", "/model", "/auth", "/quota", "/theme"})

SLASH_COMMANDS: dict[str, str] = {
    "/role": "Change the assistant role.",
    "/model": "Change the model.",
    "/reset": "Remove the stored API key.",
    "/quota": "Show the quota panel.",
    "/auth": "Show authentication methods.",
    "/exit": "Exit the REPL.",
    "/quit": "Exit the REPL.",
    "/clear": "Clear the conversation.",
    "/help": "Show available commands.",
}

ROLE_DESCRIPTIONS: dict[str, str] = {
    "basic": "Fast, reliable help for everyday tasks.",
    "code": "Built for coding and technical problem-solving.",
    "creator": "Built for ideas, concepts, and creative direction.",
    "analyser": "Built for deep analysis and clear insights.",
    "poet": "For poetry, prose, and expressive writing.",
}


@dataclass(frozen=True)
class SlashCommand:
    """A parsed slash command."""

    name: str
    argument: str = ""


@dataclass(frozen=True)
class MenuOption:
    """One selectable entry."""

    value: str
    label: str
    description: str = ""


AUTH_OPTIONS: tuple[MenuOption, ...] = (
    MenuOption(
        value="google",
        label="Authentication with google",
        description="Google OAuth is not available yet.",
    ),
    MenuOption(
        value="api_key",
        label="Authentication with api key",
        description="Store a key in the OS keyring.",
    ),
)


def parse_slash_command(text: str) -> SlashCommand | None:
    """Parse a slash command, or return ``None`` for normal input."""
    stripped = text.strip()
    if not stripped.startswith("/"):
        return None
    name, _, argument = stripped.partition(" ")
    return SlashCommand(name=name.lower(), argument=argument.strip())


def render_help() -> str:
    """Return the slash-command help text."""
    lines = ["Commands:"]
    for name, description in SLASH_COMMANDS.items():
        lines.append(f"  {name:<8} {description}")
    return "\n".join(lines) + "\n"


def role_options() -> tuple[MenuOption, ...]:
    """Selection entries for every configured role."""
    return tuple(
        MenuOption(
            value=role,
            label=f"Gemini {role.capitalize()}",
            description=ROLE_DESCRIPTIONS.get(role, ""),
        )
        for role in AVAILABLE_ROLES
    )


def model_options() -> tuple[MenuOption, ...]:
    """Selection entries for every supported model."""
    return tuple(
        MenuOption(
            value=model.name,
            label=model.display_name,
            description=model.description,
        )
        for model in MODEL_REGISTRY.values()
    )


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
        print(f"\nApproval required ({risk_level}): {action_description}")
        print(f"  {details}")
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
    """Run the plain line interface until the user exits or input ends."""
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

        command = parse_slash_command(message)
        if command is not None:
            session = _handle_slash(command, runtime, session, write)
            continue

        write(f"User: {message}\n")
        await _run_turn(runtime, session, message, write)


def _handle_slash(
    command: SlashCommand,
    runtime: CliRuntime,
    session: Session,
    write: Callable[[str], None],
) -> Session:
    name = command.name
    if name == "/help":
        write(render_help())
    elif name == "/clear":
        session = Session(
            model=runtime.settings.model,
            role=runtime.settings.role,
        )
        write("Conversation cleared.\n")
    elif name == "/reset":
        try:
            delete_api_key()
        except CredentialStoreError as exc:
            write(f"Error: {exc}\n")
        else:
            write("Stored API key removed. Run 'gemini-cli-py login' again.\n")
    elif name in _MENU_COMMANDS:
        write(
            f"{name} needs the full-screen UI. Run 'gemini-cli-py chat' "
            "without --no-tui.\n"
        )
    else:
        write(f"Unknown command {name}. Type /help for available commands.\n")
    return session


async def _run_turn(
    runtime: CliRuntime,
    session: Session,
    message: str,
    write: Callable[[str], None],
) -> None:
    async for event in runtime.agent.run(session, message):
        if event.event_type is AgentEventType.TEXT_CHUNK:
            write(str(event.data))
        elif event.event_type is AgentEventType.ERROR:
            write(f"\nError: {event.data}\n")
        elif event.event_type is AgentEventType.DONE:
            write("\n")


def _echo(text: str) -> None:
    print(text, end="", flush=True)


__all__ = [
    "AUTH_OPTIONS",
    "SLASH_COMMANDS",
    "ConsoleApprovalCallback",
    "MenuOption",
    "SlashCommand",
    "model_options",
    "parse_slash_command",
    "render_help",
    "role_options",
    "run_repl",
]
