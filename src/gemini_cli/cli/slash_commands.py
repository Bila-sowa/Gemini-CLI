"""Shared slash-command registry, parsing, and resolution.

Both the full-screen Textual application and the plain line fallback use this
module, so a command spelling only needs to be declared once. Resolution is
deliberately conservative: exact names and aliases win, a unique prefix is
accepted, and ambiguous or misspelled input is reported without executing.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum


class CommandAction(StrEnum):
    """Canonical behaviours a slash command can trigger."""

    HELP = "help"
    CLEAR = "clear"
    ROLE = "role"
    MODEL = "model"
    AUTH = "auth"
    RESET = "reset"
    QUOTA = "quota"
    THEME = "theme"
    EXIT = "exit"


@dataclass(frozen=True, slots=True)
class CommandSpec:
    """One slash command and every spelling that resolves to it."""

    action: CommandAction
    name: str
    aliases: tuple[str, ...] = ()
    description: str = ""
    tui_only: bool = False

    @property
    def spellings(self) -> tuple[str, ...]:
        """Return the canonical name followed by its aliases."""
        return (self.name, *self.aliases)

    @property
    def display_name(self) -> str:
        """Return the canonical command with its leading slash."""
        return f"/{self.name}"


COMMANDS: tuple[CommandSpec, ...] = (
    CommandSpec(
        CommandAction.HELP,
        "help",
        ("h", "?", "commands"),
        "Show available commands.",
    ),
    CommandSpec(
        CommandAction.MODEL,
        "model",
        ("models", "m"),
        "Change the model.",
        tui_only=True,
    ),
    CommandSpec(
        CommandAction.ROLE,
        "role",
        ("roles",),
        "Change the assistant role.",
        tui_only=True,
    ),
    CommandSpec(
        CommandAction.AUTH,
        "auth",
        ("authentication",),
        "Show authentication methods.",
        tui_only=True,
    ),
    CommandSpec(
        CommandAction.QUOTA,
        "quota",
        ("quotas", "usage", "limits"),
        "Show the quota panel.",
        tui_only=True,
    ),
    CommandSpec(
        CommandAction.THEME,
        "theme",
        ("themes", "t"),
        "Switch the color theme.",
        tui_only=True,
    ),
    CommandSpec(
        CommandAction.CLEAR,
        "clear",
        ("cls", "new"),
        "Clear the conversation.",
    ),
    CommandSpec(
        CommandAction.RESET,
        "reset",
        ("reset-key", "remove-key", "logout"),
        "Remove the stored API key.",
    ),
    CommandSpec(
        CommandAction.EXIT,
        "exit",
        ("quit", "q", "bye"),
        "Exit Gemini CLI.",
    ),
)

# Compatibility view for callers that only need canonical names.
SLASH_COMMANDS: dict[str, str] = {
    spec.display_name: spec.description for spec in COMMANDS
}


@dataclass(frozen=True, slots=True)
class SlashCommand:
    """A parsed slash command before alias resolution."""

    name: str
    argument: str = ""


@dataclass(frozen=True, slots=True)
class ResolvedCommand:
    """A slash command that matched exactly one command specification."""

    spec: CommandSpec
    argument: str = ""


@dataclass(frozen=True, slots=True)
class CommandProblem:
    """Input that could not be resolved safely."""

    name: str
    candidates: tuple[CommandSpec, ...] = ()
    suggestion: CommandSpec | None = None


def _normalize(token: str) -> str:
    """Normalize one command spelling for case-insensitive lookup."""
    return token.strip().casefold()


def _build_index(commands: Iterable[CommandSpec]) -> dict[str, CommandSpec]:
    """Build a spelling index and reject duplicate declarations."""
    index: dict[str, CommandSpec] = {}
    for spec in commands:
        for spelling in spec.spellings:
            token = _normalize(spelling)
            if not token:
                msg = f"Command {spec.display_name} has an empty spelling."
                raise ValueError(msg)
            if token in index:
                msg = f"Duplicate slash-command spelling: /{token}"
                raise ValueError(msg)
            index[token] = spec
    return index


_INDEX: dict[str, CommandSpec] = _build_index(COMMANDS)


def parse_slash_command(text: str) -> SlashCommand | None:
    """Parse a slash command, or return ``None`` for normal input."""
    stripped = text.strip()
    if not stripped.startswith("/"):
        return None
    parts = stripped.split(maxsplit=1)
    name = parts[0].lower()
    argument = parts[1].strip() if len(parts) > 1 else ""
    return SlashCommand(name=name, argument=argument)


def _suggest(token: str) -> CommandSpec | None:
    """Return the closest declared spelling without executing anything."""
    matches = difflib.get_close_matches(token, list(_INDEX), n=1, cutoff=0.6)
    if not matches:
        return None
    return _INDEX[matches[0]]


def resolve_command(command: SlashCommand) -> ResolvedCommand | CommandProblem:
    """Resolve parsed input to one command or a safe problem description.

    Exact canonical names and aliases are matched first. If none match, a
    unique prefix is accepted. Multiple prefix matches are reported as
    ambiguous, and misspelled input gets a suggestion but is never executed.
    """
    token = _normalize(command.name.removeprefix("/"))
    if not token:
        return CommandProblem(command.name)

    exact = _INDEX.get(token)
    if exact is not None:
        return ResolvedCommand(exact, command.argument)

    matched: dict[CommandAction, CommandSpec] = {}
    for spelling, spec in _INDEX.items():
        if spelling.startswith(token):
            matched.setdefault(spec.action, spec)

    if len(matched) == 1:
        spec = next(iter(matched.values()))
        return ResolvedCommand(spec, command.argument)

    if matched:
        candidates = tuple(spec for spec in COMMANDS if spec.action in matched)
        return CommandProblem(command.name, candidates=candidates)

    return CommandProblem(command.name, suggestion=_suggest(token))


def format_command_problem(problem: CommandProblem) -> str:
    """Format a user-facing message for an unresolved command."""
    if problem.candidates:
        names = ", ".join(spec.display_name for spec in problem.candidates)
        return (
            f"{problem.name} is ambiguous: {names}. "
            "Type more characters or /help for available commands."
        )
    if problem.suggestion is not None:
        return (
            f"Unknown command {problem.name}. Did you mean "
            f"{problem.suggestion.display_name}? "
            "Type /help for available commands."
        )
    return f"Unknown command {problem.name}. Type /help for available commands."


def render_help() -> str:
    """Return the shared slash-command help text."""
    lines = ["Commands:"]
    for spec in COMMANDS:
        aliases = ", ".join(f"/{alias}" for alias in spec.aliases)
        suffix = f"  (also {aliases})" if aliases else ""
        lines.append(f"  {spec.display_name:<10} {spec.description}{suffix}")
    lines.append("")
    lines.append(
        "Commands are case-insensitive; a unique prefix also works (/mod -> /model)."
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "COMMANDS",
    "SLASH_COMMANDS",
    "CommandAction",
    "CommandProblem",
    "CommandSpec",
    "ResolvedCommand",
    "SlashCommand",
    "format_command_problem",
    "parse_slash_command",
    "render_help",
    "resolve_command",
]
