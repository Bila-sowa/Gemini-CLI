"""Widgets and palette for the Gemini CLI terminal UI."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import BindingType
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widgets import Button, OptionList, Static
from textual.widgets.option_list import Option

from gemini_cli import __version__
from gemini_cli.cli.tui.repl import MenuOption
from gemini_cli.cli.tui.themes import DEFAULT_THEME, Theme
from gemini_cli.cli.waiting import WAITING_SHOW_DELAY, WAITING_TEXT
from gemini_cli.core.context_tracker import ContextWindowStatus

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
    """Version, model, and context-window line inside a bordered box."""

    def __init__(self) -> None:
        super().__init__(id="status")
        self._plain = ""
        self.update_status(model="", used_tokens=0, budget_tokens=0, percentage=0.0)

    @property
    def plain(self) -> str:
        """Plain-text form used by tests and logs."""
        return self._plain

    def update_status(
        self,
        *,
        model: str,
        used_tokens: int,
        budget_tokens: int,
        percentage: float,
        theme: Theme = DEFAULT_THEME,
    ) -> None:
        context = (
            f"{compact_tokens(used_tokens)}/{compact_tokens(budget_tokens)} "
            f"({percentage:.1f}%)"
        )
        self._plain = (
            f">_  version: v{__version__}    model: {model}    context: {context}"
        )
        text = Text()
        text.append(">_  ", style=theme.accent)
        text.append("version: ", style=theme.muted)
        text.append(f"v{__version__}    ", style=theme.foreground)
        text.append("model: ", style=theme.muted)
        text.append(f"{model}    ", style=theme.accent)
        text.append("context: ", style=theme.muted)
        text.append(context, style=theme.warning)
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

    def apply_theme(self, style: str) -> None:
        """Re-style the whole message after a theme change."""
        self._text = Text(self._text.plain, style=style)
        self.update(self._text)


WAITING_FRAMES: tuple[str, ...] = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


class WaitingIndicator(Static):
    """Animated line shown only while a model response is pending."""

    def __init__(
        self,
        theme: Theme = DEFAULT_THEME,
        *,
        show_delay: float = WAITING_SHOW_DELAY,
    ) -> None:
        super().__init__(id="waiting")
        self._theme = theme
        self._show_delay = max(0.0, show_delay)
        self._frame = 0
        self._timer: Timer | None = None
        self._show_timer: Timer | None = None
        self._active = False
        self.display = False

    @property
    def plain(self) -> str:
        """Plain text of the current frame."""
        frame = WAITING_FRAMES[self._frame % len(WAITING_FRAMES)]
        return f"{frame} {WAITING_TEXT}"

    def show(self) -> None:
        """Start waiting; the first frame appears after a short delay."""
        if self._active:
            return
        self._active = True
        if self._show_delay <= 0:
            self._reveal()
            return
        self._show_timer = self.set_timer(self._show_delay, self._reveal)

    def hide(self) -> None:
        """Stop waiting and remove the line from the layout."""
        self._active = False
        if self._show_timer is not None:
            self._show_timer.stop()
            self._show_timer = None
        if self._timer is not None:
            self._timer.pause()
        self.display = False

    def _reveal(self) -> None:
        self._show_timer = None
        if not self._active:
            return
        self._frame = 0
        # Render before becoming visible: Static has no visual until its first
        # update, and Textual may paint the widget as soon as display flips.
        self._render_frame()
        self.display = True
        if self._timer is None:
            self._timer = self.set_interval(0.1, self._advance)
        else:
            self._timer.resume()
        self.scroll_visible(animate=False)

    def _advance(self) -> None:
        self._frame += 1
        self._render_frame()

    def _render_frame(self) -> None:
        text = Text("  ")
        text.append(
            WAITING_FRAMES[self._frame % len(WAITING_FRAMES)],
            style=self._theme.assistant_color,
        )
        text.append(f" {WAITING_TEXT}", style=self._theme.muted)
        self.update(text, layout=False)

    def on_unmount(self) -> None:
        if self._show_timer is not None:
            self._show_timer.stop()
            self._show_timer = None
        if self._timer is not None:
            self._timer.stop()
            self._timer = None


def _bar_text(
    percentage: float,
    color: str,
    empty_color: str,
    width: int = 15,
) -> Text:
    clamped = max(0.0, min(100.0, percentage))
    filled = round(clamped / 100 * width)
    text = Text()
    text.append("█" * filled, style=color)
    text.append("░" * (width - filled), style=empty_color)
    return text


def render_context_text(
    status: ContextWindowStatus,
    model: str,
    theme: Theme = DEFAULT_THEME,
) -> Text:
    """Render the context-window panel as styled Rich text."""
    text = Text()
    text.append("─ CONTEXT ", style=f"bold {theme.muted}")
    text.append("─" * 42, style=theme.border)
    text.append("\n")
    text.append("  Model      ", style=theme.muted)
    text.append(model, style=theme.accent)
    text.append("\n")
    text.append("  Window     ", style=theme.muted)
    text.append(
        f"{compact_tokens(status.window_tokens)} tokens",
        style=theme.foreground,
    )
    text.append("\n")
    text.append("  Used       ", style=theme.muted)
    text.append(
        f"{compact_tokens(status.used_tokens)} / "
        f"{compact_tokens(status.budget_tokens)}",
        style=theme.foreground,
    )
    text.append(f"  {status.percentage:>5.1f}%", style=theme.warning)
    text.append("\n")
    text.append("  ")
    text.append_text(_bar_text(status.percentage, theme.success, theme.border))
    text.append("\n")
    text.append("  Remaining  ", style=theme.muted)
    text.append(
        f"{compact_tokens(status.remaining_tokens)} tokens",
        style=theme.foreground,
    )
    return text


class ContextPanel(Static):
    """Inline context-window panel toggled with /context."""

    def __init__(self) -> None:
        super().__init__(id="context")
        self._plain = ""
        self.display = False

    @property
    def plain(self) -> str:
        return self._plain

    def update_context(
        self,
        status: ContextWindowStatus,
        model: str,
        theme: Theme = DEFAULT_THEME,
    ) -> None:
        rendered = render_context_text(status, model, theme)
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
    "ContextPanel",
    "Logo",
    "MenuScreen",
    "MessageView",
    "RoleBar",
    "StatusBar",
    "WaitingIndicator",
    "compact_tokens",
    "render_context_text",
]
