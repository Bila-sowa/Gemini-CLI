"""Plain line-oriented fallback for non-interactive terminals.

The full-screen Textual app in :mod:`gemini_cli.cli.tui.app` is the primary
interface. This module keeps a minimal line mode for pipes, CI, and
``--no-tui``: it streams text, reports errors, and supports the commands that
do not need a menu.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from gemini_cli import __version__
from gemini_cli.auth import AuthenticationError, CredentialStoreError
from gemini_cli.auth.credentials_store import delete_api_key
from gemini_cli.cli.command_arguments import (
    CommandArgumentError,
    parse_model_argument,
    parse_role_argument,
)
from gemini_cli.cli.runtime import CliRuntime, build_runtime
from gemini_cli.cli.slash_commands import (
    SLASH_COMMANDS,
    CommandAction,
    CommandProblem,
    ResolvedCommand,
    SlashCommand,
    format_command_problem,
    format_unexpected_argument,
    parse_slash_command,
    render_command_help,
    render_help,
    resolve_command,
    resolve_command_name,
)
from gemini_cli.cli.waiting import ModelWaitTracker, RichWaitIndicator
from gemini_cli.config.defaults import AVAILABLE_ROLES, AVAILABLE_THEMES
from gemini_cli.config.env_file import EnvFileError, set_env_value
from gemini_cli.core.session import Session
from gemini_cli.core.streaming import AgentEventType
from gemini_cli.llm import MODEL_REGISTRY, UnknownModelError
from gemini_cli.llm.prompts import PromptError
from gemini_cli.sandbox.approval import ApprovalDecision

# Legacy line-mode convenience: the TUI is slash-only, but the plain fallback
# keeps accepting bare ``exit`` and ``quit`` from earlier releases.
_BARE_EXIT_COMMANDS = frozenset({"exit", "quit"})

ROLE_DESCRIPTIONS: dict[str, str] = {
    "basic": "Fast, reliable help for everyday tasks.",
    "code": "Built for coding and technical problem-solving.",
    "creator": "Built for ideas, concepts, and creative direction.",
    "analyser": "Built for deep analysis and clear insights.",
    "poet": "For poetry, prose, and expressive writing.",
}

THEME_DESCRIPTIONS: dict[str, str] = {
    "dark": "Dark palette for low-light terminals.",
    "light": "Light palette for bright environments.",
}


@dataclass(frozen=True)
class MenuOption:
    """One selectable entry."""

    value: str
    label: str
    description: str = ""


@dataclass(frozen=True)
class ReplOutcome:
    """Result of handling one slash command in line mode."""

    session: Session
    exit_requested: bool = False
    model: str | None = None
    role: str | None = None


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


def theme_options() -> tuple[MenuOption, ...]:
    """Selection entries for every available terminal theme."""
    return tuple(
        MenuOption(
            value=theme,
            label=theme.capitalize(),
            description=THEME_DESCRIPTIONS.get(theme, ""),
        )
        for theme in AVAILABLE_THEMES
    )


def model_options(current: str | None = None) -> tuple[MenuOption, ...]:
    """Selection entries for every registered model and the current one."""
    options = [
        MenuOption(
            value=model.name,
            label=model.display_name,
            description=model.description,
        )
        for model in MODEL_REGISTRY.values()
    ]
    if current and current not in MODEL_REGISTRY:
        options.append(
            MenuOption(
                value=current,
                label=f"{current} (current custom model)",
                description="Configured custom model.",
            )
        )
    return tuple(options)


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

    try:
        while True:
            try:
                line = read("> ")
            except (EOFError, KeyboardInterrupt):
                write("\n")
                return

            message = line.strip()
            if not message:
                continue
            if message.casefold() in _BARE_EXIT_COMMANDS:
                return

            command = parse_slash_command(message)
            if command is not None:
                outcome = _handle_slash(
                    resolve_command(command),
                    runtime,
                    session,
                    write,
                )
                if outcome.exit_requested:
                    return
                session = outcome.session
                if outcome.model is not None or outcome.role is not None:
                    replacement = await _rebuild_runtime(
                        runtime,
                        model=outcome.model,
                        role=outcome.role,
                        read=read,
                        write=write,
                    )
                    if replacement is None:
                        continue
                    runtime = replacement
                    session.model = runtime.settings.model
                    session.role = runtime.settings.role
                    if outcome.model is not None:
                        _persist_choice("GEMINI_MODEL", outcome.model, write)
                        write(f"Model: {runtime.settings.model}\n")
                    if outcome.role is not None:
                        _persist_choice("GEMINI_ROLE", outcome.role, write)
                        write(f"Role: {runtime.settings.role}\n")
                continue

            write(f"User: {message}\n")
            await _run_turn(
                runtime,
                session,
                message,
                write,
                show_indicator=output is None,
            )
    finally:
        runtime.close()


def _handle_slash(
    resolution: ResolvedCommand | CommandProblem,
    runtime: CliRuntime,
    session: Session,
    write: Callable[[str], None],
) -> ReplOutcome:
    """Handle one resolved command and report what the REPL should do next."""
    if isinstance(resolution, CommandProblem):
        write(f"{format_command_problem(resolution)}\n")
        return ReplOutcome(session)

    spec = resolution.spec
    unexpected = format_unexpected_argument(spec, resolution.argument)
    if unexpected is not None:
        write(f"{unexpected}\n")
        return ReplOutcome(session)

    match spec.action:
        case CommandAction.HELP:
            if resolution.argument:
                help_resolution = resolve_command_name(resolution.argument)
                if isinstance(help_resolution, CommandProblem):
                    write(f"{format_command_problem(help_resolution)}\n")
                else:
                    write(render_command_help(help_resolution.spec))
            else:
                write(render_help())
        case CommandAction.CLEAR:
            session = Session(
                model=runtime.settings.model,
                role=runtime.settings.role,
            )
            write("Conversation cleared.\n")
        case CommandAction.RESET:
            try:
                delete_api_key()
            except CredentialStoreError as exc:
                write(f"Error: {exc}\n")
            else:
                write("Stored API key removed. Run 'gemini-cli-py login' again.\n")
        case CommandAction.MODEL:
            if resolution.argument:
                try:
                    value = parse_model_argument(resolution.argument)
                except CommandArgumentError as exc:
                    write(f"Error: {exc}\n")
                else:
                    return ReplOutcome(session, model=value)
            else:
                write(
                    "/model needs the full-screen UI for the menu. "
                    "Use /model <name> in line mode.\n"
                )
        case CommandAction.ROLE:
            if resolution.argument:
                try:
                    value = parse_role_argument(resolution.argument)
                except CommandArgumentError as exc:
                    write(f"Error: {exc}\n")
                else:
                    return ReplOutcome(session, role=value)
            else:
                write(
                    "/role needs the full-screen UI for the menu. "
                    "Use /role <role> in line mode.\n"
                )
        case CommandAction.THEME:
            write(
                "Theme selection needs the full-screen UI. "
                "Run 'gemini-cli-py chat' without --no-tui.\n"
            )
        case CommandAction.EXIT:
            return ReplOutcome(session, exit_requested=True)
        case _:
            write(
                f"{spec.display_name} needs the full-screen UI. "
                "Run 'gemini-cli-py chat' without --no-tui.\n"
            )
    return ReplOutcome(session)


async def _rebuild_runtime(
    runtime: CliRuntime,
    *,
    model: str | None,
    role: str | None,
    read: Callable[[str], str],
    write: Callable[[str], None],
) -> CliRuntime | None:
    """Build a replacement runtime, keeping the old one on failure."""
    try:
        replacement = await build_runtime(
            runtime.settings,
            model=model,
            role=role,
            approval_callback=ConsoleApprovalCallback(input_func=read),
        )
    except (
        AuthenticationError,
        CredentialStoreError,
        UnknownModelError,
        PromptError,
    ) as exc:
        write(f"Error: {exc}\n")
        return None

    runtime.close()
    return replacement


def _persist_choice(
    key: str,
    value: str,
    write: Callable[[str], None],
) -> None:
    """Save a model or role choice in the project ``.env`` file."""
    path = Path.cwd() / ".env"
    try:
        set_env_value(path, key, value)
    except (EnvFileError, OSError) as exc:
        write(f"Could not save {key} to {path}: {exc}\n")
        return

    write(f"Saved {key} to {path} for future sessions.\n")
    if os.environ.get(key):
        write(f"{key} is also set in the environment; that value takes precedence.\n")


async def _run_turn(
    runtime: CliRuntime,
    session: Session,
    message: str,
    write: Callable[[str], None],
    *,
    show_indicator: bool,
) -> None:
    indicator = RichWaitIndicator(enabled=None if show_indicator else False)
    tracker = ModelWaitTracker()
    indicator.set_waiting(True)
    try:
        async for event in runtime.agent.run(session, message):
            # Clear the spinner before writing anything so streamed text is
            # never interleaved with terminal animation.
            indicator.set_waiting(False)
            if event.event_type is AgentEventType.TEXT_CHUNK:
                write(str(event.data))
            elif event.event_type is AgentEventType.ERROR:
                write(f"\nError: {event.data}\n")
            elif event.event_type is AgentEventType.DONE:
                write("\n")

            if tracker.on_event(event.event_type):
                indicator.set_waiting(True)
    finally:
        indicator.close()


def _echo(text: str) -> None:
    print(text, end="", flush=True)


__all__ = [
    "AUTH_OPTIONS",
    "SLASH_COMMANDS",
    "ConsoleApprovalCallback",
    "MenuOption",
    "ReplOutcome",
    "SlashCommand",
    "model_options",
    "parse_slash_command",
    "render_help",
    "role_options",
    "run_repl",
    "theme_options",
]
