"""Widgets and palette for the Gemini CLI terminal UI."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import timedelta
from typing import ClassVar

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import BindingType
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, OptionList, Static
from textual.widgets.option_list import Option

from gemini_cli import __version__
from gemini_cli.cli.tui.repl import MenuOption
from gemini_cli.cli.tui.themes import DEFAULT_THEME, Theme
from gemini_cli.core.quota_tracker import QuotaWindowStatus

GEMINI_BLUE = DEFAULT_THEME.accent
GEMINI_ORANGE = DEFAULT_THEME.warning
GEMINI_GREEN = DEFAULT_THEME.success
GEMINI_RED = DEFAULT_THEME.error_color
GEMINI_MUTED = DEFAULT_THEME.muted
GEMINI_BORDER = DEFAULT_THEME.border

GEMINI_LOGO = r"""
 ██████╗ ███████╗███╗   ███╗██╗███╗   ██╗██╗     ██████╗██╗     ██╗
██╔════╝ ██╔════╝████╗ ████║██║████╗  ██║██║    ██╔════╝██║     ██║
██║  ███╗█████╗  ██╔████╔██║██║██╔██╗ ██║██║    ██║     ██║     ██║
██║   ██║██╔══╝  ██║╚██╔╝██║██║██║╚██╗██║██║    ██║     ██║     ██║
╚██████╔╝███████╗██║ ╚═╝ ██║██║██║ ╚████║██║     ╚██████╗███████╗██║
 ╚═════╝ ╚══════╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚ ╝     ╚═════╝╚══════╝╚═╝
"""


def compact_tokens(value: int) -> str:
    """Render a token count as ``100k`` or ``1.2M``."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}k"
    return str(value)


class Logo(Static):
    """The blue ASCII wordmark shown at the top of the app."""

    def __init__(self) -> None:
        super().__init__(id="logo")
        self.apply_theme(DEFAULT_THEME)

    def apply_theme(self, theme: Theme) -> None:
        self.update(Text(GEMINI_LOGO, style=f"bold {theme.accent}"))


class StatusBar(Static):
    """Version, model, and quota line inside a bordered box."""

    def __init__(self) -> None:
        super().__init__(id="status")
        self._plain = ""
        self.update_status(model="", used_tokens=0, limit_tokens=0)

    @property
    def plain(self) -> str:
        """Plain-text form used by tests and logs."""
        return self._plain

    def update_status(
        self,
        *,
        model: str,
        used_tokens: int,
        limit_tokens: int,
        theme: Theme = DEFAULT_THEME,
    ) -> None:
        quota = f"{compact_tokens(used_tokens)}/{compact_tokens(limit_tokens)} tokens"
        self._plain = f">_  version: v{__version__}    model: {model}    quota: {quota}"
        text = Text()
        text.append(">_  ", style=theme.accent)
        text.append("version: ", style=theme.muted)
        text.append(f"v{__version__}    ", style=theme.foreground)
        text.append("model: ", style=theme.muted)
        text.append(f"{model}    ", style=theme.accent)
        text.append("quota: ", style=theme.muted)
        text.append(quota, style=theme.warning)
        self.update(text)


class RoleBar(Static):
    """Footer showing the active role and how to change it."""

    def __init__(self) -> None:
        super().__init__(id="footer")
        self._plain = ""
        self.update_role("basic")

    @property
    def plain(self) -> str:
        """Plain-text form used by tests and logs."""
        return self._plain

    def update_role(
        self,
        role: str,
        theme: Theme = DEFAULT_THEME,
    ) -> None:
        label = role.capitalize()
        self._plain = f"role: Gemini {label} (change role: /role)"
        text = Text()
        text.append("role: ", style=theme.muted)
        text.append(f"Gemini {label}", style=theme.accent)
        text.append(" (change role: /role)", style=theme.muted)
        self.update(text)


class MessageView(Static):
    """One user or assistant message that can grow while streaming."""

    def __init__(self, prefix: str, prefix_style: str) -> None:
        super().__init__(classes="message")
        self._text = Text(prefix, style=prefix_style)
        self.update(self._text)

    @property
    def plain(self) -> str:
        """Plain text of the whole message."""
        return self._text.plain

    def append(self, chunk: str, style: str | None = None) -> None:
        """Append streamed text and refresh the widget."""
        self._text.append(chunk, style=style)
        self.update(self._text)


def _bar_text(percentage: float, color: str, width: int = 15) -> Text:
    clamped = max(0.0, min(100.0, percentage))
    filled = round(clamped / 100 * width)
    text = Text()
    text.append("█" * filled, style=color)
    text.append("░" * (width - filled), style=DEFAULT_THEME.border)
    return text


def _format_remaining(delta: timedelta) -> str:
    seconds = max(0, int(delta.total_seconds()))
    hours, remainder = divmod(seconds, 3600)
    minutes = remainder // 60
    return f"{hours}h {minutes:02d}m"


def render_quota_text(
    five_hour: QuotaWindowStatus,
    weekly: QuotaWindowStatus,
    theme: Theme = DEFAULT_THEME,
) -> Text:
    """Render the quota panel as styled Rich text."""
    text = Text()
    text.append("─ QUOTA ", style=f"bold {theme.muted}")
    text.append("─" * 44, style=theme.border)
    text.append("\n")
    text.append("  5h limit   ", style=theme.muted)
    text.append_text(_bar_text(five_hour.percentage, theme.error_color))
    text.append(f"  {five_hour.percentage:>5.1f}%   ", style=theme.foreground)
    text.append(
        f"{_format_remaining(five_hour.time_remaining)} remaining",
        style=theme.muted,
    )
    text.append("\n")
    text.append("  Weekly     ", style=theme.muted)
    text.append_text(_bar_text(weekly.percentage, theme.success))
    text.append(f"  {weekly.percentage:>5.1f}%   ", style=theme.foreground)
    text.append(
        f"{_format_remaining(weekly.time_remaining)} remaining",
        style=theme.muted,
    )
    return text


class QuotaPanel(Static):
    """Inline quota panel toggled with /quota."""

    def __init__(self) -> None:
        super().__init__(id="quota")
        self._plain = ""
        self.display = False

    @property
    def plain(self) -> str:
        return self._plain

    def update_quota(
        self,
        five_hour: QuotaWindowStatus,
        weekly: QuotaWindowStatus,
        theme: Theme = DEFAULT_THEME,
    ) -> None:
        rendered = render_quota_text(five_hour, weekly, theme)
        self._plain = rendered.plain
        self.update(rendered)


class MenuScreen(ModalScreen[MenuOption | None]):
    """Modal list used for role, model, and authentication selection."""

    BINDINGS: ClassVar[list[BindingType]] = [
        ("escape", "cancel", "Cancel"),
    ]

    def __init__(
        self,
        options: Sequence[MenuOption],
        title: str,
        theme: Theme = DEFAULT_THEME,
    ) -> None:
        super().__init__()
        self._options = tuple(options)
        self._title = title
        self._theme = theme

    def compose(self) -> ComposeResult:
        with Vertical(id="menu"):
            yield Static(
                Text(self._title, style=f"bold {self._theme.accent}"),
                id="menu-title",
            )
            yield OptionList(
                *[
                    Option(
                        Text.assemble(
                            (option.label, f"bold {self._theme.foreground}"),
                            (f" - {option.description}", self._theme.muted),
                        )
                    )
                    for option in self._options
                ],
                id="menu-list",
            )

    def on_mount(self) -> None:
        self.query_one("#menu-list", OptionList).focus()

    def on_option_list_option_selected(
        self,
        event: OptionList.OptionSelected,
    ) -> None:
        self.dismiss(self._options[event.option_index])

    def action_cancel(self) -> None:
        self.dismiss(None)


class ApprovalScreen(ModalScreen[bool]):
    """Modal confirmation for a sensitive tool call."""

    BINDINGS: ClassVar[list[BindingType]] = [
        ("y", "approve", "Approve"),
        ("n", "deny", "Deny"),
        ("escape", "deny", "Deny"),
    ]

    def __init__(
        self,
        action_description: str,
        risk_level: str,
        details: str,
        theme: Theme = DEFAULT_THEME,
    ) -> None:
        super().__init__()
        self._action = action_description
        self._risk = risk_level
        self._details = details
        self._theme = theme

    def compose(self) -> ComposeResult:
        title = Text()
        title.append("Approval required", style=f"bold {self._theme.warning}")
        title.append(f"  ({self._risk})", style=self._theme.muted)

        with Vertical(id="approval"):
            yield Static(title, id="approval-title")
            yield Static(
                Text(self._action, style=f"bold {self._theme.foreground}"),
                id="approval-action",
            )
            yield Static(
                Text(self._details, style=self._theme.muted),
                id="approval-details",
            )
            with Horizontal(id="approval-buttons"):
                yield Button("Approve", id="approve", variant="success")
                yield Button("Deny", id="deny", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "approve")

    def action_approve(self) -> None:
        self.dismiss(True)

    def action_deny(self) -> None:
        self.dismiss(False)


__all__ = [
    "GEMINI_BLUE",
    "GEMINI_BORDER",
    "GEMINI_GREEN",
    "GEMINI_LOGO",
    "GEMINI_MUTED",
    "GEMINI_ORANGE",
    "GEMINI_RED",
    "ApprovalScreen",
    "Logo",
    "MenuScreen",
    "MessageView",
    "QuotaPanel",
    "RoleBar",
    "StatusBar",
    "compact_tokens",
    "render_quota_text",
]
