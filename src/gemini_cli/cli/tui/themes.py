"""Visual theme definitions for the terminal UI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    """Named styles and palette used by the terminal renderer."""

    name: str
    assistant_color: str
    user_color: str
    error_color: str
    info_color: str
    header_style: str
    border_style: str
    background: str
    surface: str
    foreground: str
    muted: str
    border: str
    accent: str
    warning: str
    success: str


DEFAULT_THEME = Theme(
    name="dark",
    assistant_color="#5B9BD5",
    user_color="#E6E6E6",
    error_color="#E06C75",
    info_color="#8A8F98",
    header_style="bold #5B9BD5",
    border_style="#3A3F45",
    background="#1E2023",
    surface="#24272B",
    foreground="#E6E6E6",
    muted="#8A8F98",
    border="#3A3F45",
    accent="#5B9BD5",
    warning="#D99A3D",
    success="#5FBF77",
)

LIGHT_THEME = Theme(
    name="light",
    assistant_color="#2F6FB3",
    user_color="#1E2023",
    error_color="#C53030",
    info_color="#6B7280",
    header_style="bold #2F6FB3",
    border_style="#6B7280",
    background="#F5F6F7",
    surface="#FFFFFF",
    foreground="#1E2023",
    muted="#6B7280",
    border="#C7CCD3",
    accent="#2F6FB3",
    warning="#B7791F",
    success="#2F855A",
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
