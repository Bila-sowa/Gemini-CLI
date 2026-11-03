"""Visual theme definitions for the terminal UI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    """Named styles used by the terminal renderer."""

    name: str
    assistant_color: str
    user_color: str
    error_color: str
    info_color: str
    header_style: str
    border_style: str


DEFAULT_THEME = Theme(
    name="dark",
    assistant_color="bright_cyan",
    user_color="white",
    error_color="red",
    info_color="dim",
    header_style="bold bright_blue",
    border_style="dim",
)

LIGHT_THEME = Theme(
    name="light",
    assistant_color="blue",
    user_color="black",
    error_color="red",
    info_color="dim",
    header_style="bold blue",
    border_style="dim",
)

THEMES: dict[str, Theme] = {
    DEFAULT_THEME.name: DEFAULT_THEME,
    LIGHT_THEME.name: LIGHT_THEME,
}


class UnknownThemeError(ValueError):
    """Raised when a theme name is not registered."""

    def __init__(self, name: str) -> None:
        available = ", ".join(sorted(THEMES))
        super().__init__(f"Unknown theme '{name}'. Available: {available}")
        self.name = name


def get_theme(name: str) -> Theme:
    """Return a registered theme by name.

    Raises:
        UnknownThemeError: if *name* is not registered.
    """
    try:
        return THEMES[name.strip().lower()]
    except KeyError:
        raise UnknownThemeError(name) from None


__all__ = [
    "DEFAULT_THEME",
    "LIGHT_THEME",
    "THEMES",
    "Theme",
    "UnknownThemeError",
    "get_theme",
]
