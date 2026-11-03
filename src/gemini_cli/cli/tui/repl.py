"""Interactive REPL: input, selection menus, streaming, and quota.

The module keeps its behavior testable without a terminal: menu navigation is
a pure reducer, slash commands have their own parser, and rendering functions
return plain strings. ``prompt_toolkit`` and Rich are only thin adapters on top
of that logic.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from prompt_toolkit import Application, PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from rich.console import Console

from gemini_cli import __version__
from gemini_cli.auth import AuthenticationError, CredentialStoreError
from gemini_cli.auth.credentials_store import delete_api_key
from gemini_cli.cli.runtime import CliRuntime, build_runtime
from gemini_cli.config.defaults import AVAILABLE_ROLES
from gemini_cli.core.quota_tracker import QuotaWindowStatus
from gemini_cli.core.session import Session
from gemini_cli.core.streaming import AgentEventType
from gemini_cli.llm import MODEL_REGISTRY, UnknownModelError
from gemini_cli.llm.prompts import PromptError
from gemini_cli.sandbox.approval import ApprovalDecision

_EXIT_COMMANDS = frozenset({"/exit", "/quit", "exit", "quit"})
_HISTORY_FILE = "history"
_BAR_WIDTH = 15
_ANSWER_PREFIX = "✦: "

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


@dataclass(frozen=True)
class MenuState:
    """Current position of a selection menu."""

    options: tuple[MenuOption, ...]
    index: int = 0


@dataclass(frozen=True)
class MenuResult:
    """Outcome of a menu action; ``index=None`` means cancelled."""

    index: int | None


def parse_slash_command(text: str) -> SlashCommand | None:
    """Parse a slash command, or return ``None`` for normal input."""
    stripped = text.strip()
    if not stripped.startswith("/"):
        return None
    name, _, argument = stripped.partition(" ")
    return SlashCommand(name=name.lower(), argument=argument.strip())


def completion_words() -> list[str]:
    """Words offered by prompt_toolkit completion."""
    return sorted(SLASH_COMMANDS)


def render_help() -> str:
    """Return the slash-command help text."""
    lines = ["Commands:"]
    for name, description in SLASH_COMMANDS.items():
        lines.append(f"  {name:<8} {description}")
    return "\n".join(lines) + "\n"


def handle_menu_key(
    state: MenuState,
    key: str,
) -> tuple[MenuState, MenuResult | None]:
    """Apply one navigation key to a menu state.

    Keys other than up, down, enter, and escape leave the state unchanged and
    produce no decision.
    """
    if key == "up":
        return MenuState(state.options, max(0, state.index - 1)), None
    if key == "down":
        last = len(state.options) - 1
        return MenuState(state.options, min(last, state.index + 1)), None
    if key == "enter":
        return state, MenuResult(state.index)
    if key == "escape":
        return state, MenuResult(None)
    return state, None


def render_menu(state: MenuState, title: str) -> str:
    """Render a selection menu as plain text."""
    lines = [title]
    for index, option in enumerate(state.options):
        marker = ">" if index == state.index else " "
        lines.append(f"{marker} {option.label} - {option.description}")
    lines.append("")
    lines.append("> (Use up/down arrows to switch, Enter to confirm, Escape to cancel)")
    return "\n".join(lines)


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
        for model in sorted(MODEL_REGISTRY.values(), key=lambda item: item.name)
    )


def render_bar(percentage: float, width: int = _BAR_WIDTH) -> str:
    """Render a textual progress bar."""
    clamped = max(0.0, min(100.0, percentage))
    filled = round(clamped / 100 * width)
    return "█" * filled + "░" * (width - filled)


def _format_remaining(delta: timedelta) -> str:
    seconds = max(0, int(delta.total_seconds()))
    hours, remainder = divmod(seconds, 3600)
    minutes = remainder // 60
    return f"{hours}h {minutes:02d}m"


def _compact_tokens(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}k"
    return str(value)


def render_header(runtime: CliRuntime) -> str:
    """Render the REPL header line."""
    quota = runtime.quota.get_5h_status()
    return (
        "GEMINI CLI\n"
        f">_  version: v{__version__}    model: {runtime.settings.model}    "
        f"quota: {_compact_tokens(quota.used_tokens)}/"
        f"{_compact_tokens(quota.limit_tokens)} tokens\n\n"
    )


def render_quota_panel(
    five_hour: QuotaWindowStatus,
    weekly: QuotaWindowStatus,
) -> str:
    """Render the quota panel with progress bars."""
    return (
        "═ QUOTA ═════════════════════════════════════════\n"
        f"  5h limit   {render_bar(five_hour.percentage)}  "
        f"{five_hour.percentage:>5.1f}%   "
        f"{_format_remaining(five_hour.time_remaining)} remaining\n"
        f"  Weekly     {render_bar(weekly.percentage)}  "
        f"{weekly.percentage:>5.1f}%   "
        f"{_format_remaining(weekly.time_remaining)} remaining\n"
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
        console: Console | None = None,
    ) -> None:
        self._input = input_func if input_func is not None else input
        self._console = console if console is not None else Console()

    async def request_approval(
        self,
        action_description: str,
        risk_level: str,
        details: str,
    ) -> ApprovalDecision:
        self._console.print(
            f"\nApproval required ({risk_level}): {action_description}",
            highlight=False,
        )
        self._console.print(f"  {details}", highlight=False)
        try:
            answer = self._input("Approve? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return ApprovalDecision.DENIED
        if answer in {"y", "yes"}:
            return ApprovalDecision.APPROVED
        return ApprovalDecision.DENIED


MenuFunc = Callable[[Sequence[MenuOption], str], Awaitable[MenuOption | None]]
RebuildFunc = Callable[..., Awaitable[CliRuntime]]


def _build_menu_application(
    options: Sequence[MenuOption],
    title: str,
) -> tuple[Any, MenuState]:
    state = MenuState(tuple(options))
    control = FormattedTextControl(text=lambda: render_menu(state, title))
    bindings = KeyBindings()

    @bindings.add("up")
    def _up(event: Any) -> None:
        nonlocal state
        state, _decision = handle_menu_key(state, "up")
        event.app.invalidate()

    @bindings.add("down")
    def _down(event: Any) -> None:
        nonlocal state
        state, _decision = handle_menu_key(state, "down")
        event.app.invalidate()

    @bindings.add("enter")
    def _enter(event: Any) -> None:
        nonlocal state
        state, decision = handle_menu_key(state, "enter")
        event.app.exit(result=None if decision is None else decision.index)

    @bindings.add("escape")
    def _escape(event: Any) -> None:
        event.app.exit(result=None)

    application: Any = Application(
        layout=Layout(Window(control)),
        key_bindings=bindings,
        full_screen=False,
    )
    return application, state


async def select_from_menu(
    options: Sequence[MenuOption],
    title: str,
) -> MenuOption | None:
    """Show an interactive selection menu and return the chosen entry."""
    if not options:
        return None
    application, _state = _build_menu_application(options, title)
    chosen = await application.run_async()
    if not isinstance(chosen, int):
        return None
    return options[chosen]


def _make_prompt_session() -> Any:
    history_path = Path.home() / ".gemini-cli" / _HISTORY_FILE
    try:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history = FileHistory(str(history_path))
    except OSError:
        history = None
    completer = WordCompleter(
        completion_words(),
        ignore_case=True,
        sentence=True,
    )
    return PromptSession(completer=completer, history=history)


def _rich_writer(console: Console) -> Callable[[str], None]:
    def write(text: str) -> None:
        console.print(text, end="", highlight=False, markup=False)

    return write


async def _default_rebuild(
    runtime: CliRuntime,
    *,
    model: str | None = None,
    role: str | None = None,
) -> CliRuntime:
    replacement = await build_runtime(
        runtime.settings,
        model=model,
        role=role,
        approval_callback=ConsoleApprovalCallback(),
    )
    runtime.close()
    return replacement


def _new_session(runtime: CliRuntime) -> Session:
    return Session(
        model=runtime.settings.model,
        role=runtime.settings.role,
    )


async def run_repl(
    runtime: CliRuntime,
    *,
    input_func: Callable[[str], str] | None = None,
    output: Callable[[str], None] | None = None,
    menu_func: MenuFunc | None = None,
    rebuild: RebuildFunc | None = None,
) -> None:
    """Run the interactive conversation until the user exits."""
    write = output if output is not None else _rich_writer(Console())
    read = input_func
    menu = menu_func if menu_func is not None else select_from_menu
    rebuild_runtime = rebuild if rebuild is not None else _default_rebuild

    session = _new_session(runtime)
    write(render_header(runtime))

    prompt_session: Any = None
    if read is None:
        try:
            prompt_session = _make_prompt_session()
        except Exception:  # pragma: no cover - terminal setup failure
            prompt_session = None

    while True:
        try:
            if read is not None:
                line = read("> ")
            elif prompt_session is not None:
                line = prompt_session.prompt("> ")
            else:
                line = input("> ")
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
            outcome = await _handle_slash(
                command,
                runtime,
                session,
                write=write,
                menu=menu,
                rebuild=rebuild_runtime,
            )
            if outcome is None:
                return
            runtime, session = outcome
            continue

        write(f"User: {message}\n")
        await _run_turn(runtime, session, message, write)


async def _run_turn(
    runtime: CliRuntime,
    session: Session,
    message: str,
    write: Callable[[str], None],
) -> None:
    prefix_pending = True
    async for event in runtime.agent.run(session, message):
        if event.event_type is AgentEventType.TEXT_CHUNK:
            if prefix_pending:
                write(_ANSWER_PREFIX)
                prefix_pending = False
            write(str(event.data))
        elif event.event_type is AgentEventType.THINKING:
            continue
        elif event.event_type is AgentEventType.ERROR:
            if prefix_pending:
                write(_ANSWER_PREFIX)
                prefix_pending = False
            write(f"\nError: {event.data}\n")
        elif event.event_type is AgentEventType.DONE:
            if prefix_pending:
                write(_ANSWER_PREFIX)
                prefix_pending = False
            write("\n")


async def _handle_slash(
    command: SlashCommand,
    runtime: CliRuntime,
    session: Session,
    *,
    write: Callable[[str], None],
    menu: MenuFunc,
    rebuild: RebuildFunc,
) -> tuple[CliRuntime, Session] | None:
    name = command.name

    if name == "/help":
        write(render_help())
        return runtime, session

    if name == "/clear":
        write("Conversation cleared.\n")
        return runtime, _new_session(runtime)

    if name == "/quota":
        write(
            render_quota_panel(
                runtime.quota.get_5h_status(),
                runtime.quota.get_weekly_status(),
            )
        )
        return runtime, session

    if name == "/role":
        selected = await menu(role_options(), "Roles:")
        if selected is None:
            write("Selection cancelled.\n")
            return runtime, session
        return await _rebuild(
            runtime,
            session,
            write,
            rebuild,
            role=selected.value,
        )

    if name == "/model":
        selected = await menu(model_options(), "Models:")
        if selected is None:
            write("Selection cancelled.\n")
            return runtime, session
        return await _rebuild(
            runtime,
            session,
            write,
            rebuild,
            model=selected.value,
        )

    if name == "/auth":
        selected = await menu(AUTH_OPTIONS, "Authentication methods:")
        if selected is None:
            write("Selection cancelled.\n")
        elif selected.value == "api_key":
            write("Run 'gemini-cli-py login' to store an API key.\n")
        else:
            write("Google OAuth is not available yet.\n")
        return runtime, session

    if name == "/reset":
        try:
            delete_api_key()
        except CredentialStoreError as exc:
            write(f"Error: {exc}\n")
        else:
            write("Stored API key removed. Run 'gemini-cli-py login' again.\n")
        return runtime, session

    write(f"Unknown command {name}. Type /help for available commands.\n")
    return runtime, session


async def _rebuild(
    runtime: CliRuntime,
    session: Session,
    write: Callable[[str], None],
    rebuild: RebuildFunc,
    *,
    model: str | None = None,
    role: str | None = None,
) -> tuple[CliRuntime, Session] | None:
    try:
        replacement = await rebuild(runtime, model=model, role=role)
    except (
        AuthenticationError,
        CredentialStoreError,
        UnknownModelError,
        PromptError,
    ) as exc:
        write(f"Error: {exc}\n")
        return runtime, session

    session.model = replacement.settings.model
    session.role = replacement.settings.role
    write(render_header(replacement))
    return replacement, session


AUTH_OPTIONS = (
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


__all__ = [
    "AUTH_OPTIONS",
    "SLASH_COMMANDS",
    "ConsoleApprovalCallback",
    "MenuOption",
    "MenuResult",
    "MenuState",
    "SlashCommand",
    "completion_words",
    "handle_menu_key",
    "model_options",
    "parse_slash_command",
    "render_bar",
    "render_header",
    "render_help",
    "render_menu",
    "render_quota_panel",
    "role_options",
    "run_repl",
    "select_from_menu",
]
