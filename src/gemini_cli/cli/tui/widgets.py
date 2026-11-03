"""Widgets and palette for the Gemini CLI terminal UI."""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Static

from gemini_cli import __version__

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


__all__ = [
    "GEMINI_BLUE",
    "GEMINI_BORDER",
    "GEMINI_GREEN",
    "GEMINI_LOGO",
    "GEMINI_MUTED",
    "GEMINI_ORANGE",
    "GEMINI_RED",
    "Logo",
    "RoleBar",
    "StatusBar",
    "compact_tokens",
]
