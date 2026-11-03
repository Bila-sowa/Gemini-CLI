"""Widgets and palette for the Gemini CLI terminal UI."""

from __future__ import annotations

from collections.abc import Sequence
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

GEMINI_BLUE = "#5B9BD5"
GEMINI_ORANGE = "#D99A3D"
GEMINI_GREEN = "#5FBF77"
GEMINI_RED = "#E06C75"
GEMINI_MUTED = "#8A8F98"
GEMINI_BORDER = "#3A3F45"

GEMINI_LOGO = r"""
 ██████╗ ███████╗███╗   ███╗██╗███╗   ██╗██╗     ██████╗██╗     ██╗
██╔════╝ ██╔════╝████╗ ████║██║████╗  ██║██║    ██╔════╝██║     ██║
██║  ███╗█████╗  ██╔████╔██║██║██╔██╗ ██║██║    ██║     ██║     ██║
██║   ██║██╔══╝  ██║╚██╔╝██║██║██║╚██╗██║██║    ██║     ██║     ██║
╚██████╔╝███████╗██║ ╚═╝ ██║██║██║ ╚████║███████╗╚██████╗███████╗██║
 ╚═════╝ ╚══════╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚══════╝ ╚═════╝╚══════╝╚═╝
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
        super().__init__(Text(GEMINI_LOGO, style=f"bold {GEMINI_BLUE}"), id="logo")


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
    ) -> None:
        quota = f"{compact_tokens(used_tokens)}/{compact_tokens(limit_tokens)} tokens"
        self._plain = f">_  version: v{__version__}    model: {model}    quota: {quota}"
        text = Text()
        text.append(">_  ", style=GEMINI_BLUE)
        text.append("version: ", style=GEMINI_MUTED)
        text.append(f"v{__version__}    ", style="white")
        text.append("model: ", style=GEMINI_MUTED)
        text.append(f"{model}    ", style=GEMINI_BLUE)
        text.append("quota: ", style=GEMINI_MUTED)
        text.append(quota, style=GEMINI_ORANGE)
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

    def update_role(self, role: str) -> None:
        label = role.capitalize()
        self._plain = f"role: Gemini {label} (change role: /role)"
        text = Text()
        text.append("role: ", style=GEMINI_MUTED)
        text.append(f"Gemini {label}", style=GEMINI_BLUE)
        text.append(" (change role: /role)", style=GEMINI_MUTED)
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
    ) -> None:
        super().__init__()
        self._action = action_description
        self._risk = risk_level
        self._details = details

    def compose(self) -> ComposeResult:
        title = Text()
        title.append("Approval required", style=f"bold {GEMINI_ORANGE}")
        title.append(f"  ({self._risk})", style=GEMINI_MUTED)

        with Vertical(id="approval"):
            yield Static(title, id="approval-title")
            yield Static(
                Text(self._action, style="bold white"),
                id="approval-action",
            )
            yield Static(
                Text(self._details, style=GEMINI_MUTED),
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


class MenuScreen(ModalScreen[MenuOption | None]):
    """Modal list used for role, model, and authentication selection."""

    BINDINGS: ClassVar[list[BindingType]] = [
        ("escape", "cancel", "Cancel"),
    ]

    def __init__(self, options: Sequence[MenuOption], title: str) -> None:
        super().__init__()
        self._options = tuple(options)
        self._title = title

    def compose(self) -> ComposeResult:
        with Vertical(id="menu"):
            yield Static(
                Text(self._title, style=f"bold {GEMINI_BLUE}"),
                id="menu-title",
            )
            yield OptionList(
                *[
                    Option(
                        Text.assemble(
                            (option.label, "bold white"),
                            (f" - {option.description}", GEMINI_MUTED),
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
    "RoleBar",
    "StatusBar",
    "compact_tokens",
]
