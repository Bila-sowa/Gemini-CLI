"""Full-screen Textual terminal UI for the Gemini CLI.

The app owns the screen layout — logo, status bar, conversation log, prompt,
and role footer — while the agent runtime stays in :mod:`gemini_cli.cli.runtime`.
Commands and streaming are added incrementally on top of this shell.
"""

from __future__ import annotations

from typing import ClassVar

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import BindingType
from textual.widgets import Input, RichLog

from gemini_cli.cli.runtime import CliRuntime
from gemini_cli.cli.tui.widgets import (
    GEMINI_BLUE,
    GEMINI_MUTED,
    Logo,
    RoleBar,
    StatusBar,
)
from gemini_cli.core.session import Session


class GeminiApp(App[None]):
    """Terminal application matching the Gemini CLI mockups."""

    CSS = """
    Screen {
        background: #1E2023;
        color: #E6E6E6;
    }
    #logo {
        height: 6;
        margin: 1 1 0 1;
    }
    #status {
        height: 3;
        border: round #3A3F45;
        padding: 0 1;
        margin: 0 1;
    }
    #conversation {
        height: 1fr;
        margin: 0 1;
        padding: 0;
        background: #1E2023;
    }
    #prompt {
        border: round #3A3F45;
        margin: 0 1;
        background: #1E2023;
    }
    #footer {
        height: 1;
        margin: 0 1;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, runtime: CliRuntime) -> None:
        super().__init__()
        self.runtime = runtime
        self.transcript: list[str] = []
        self.session = Session(
            model=runtime.settings.model,
            role=runtime.settings.role,
        )

    def compose(self) -> ComposeResult:
        yield Logo()
        yield StatusBar()
        yield RichLog(id="conversation", markup=True, wrap=True, auto_scroll=True)
        yield Input(placeholder="Take your task", id="prompt")
        yield RoleBar()

    def on_mount(self) -> None:
        self._refresh_status()
        self._write_assistant("Can I help you today?")
        self.query_one("#prompt", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        message = event.value.strip()
        self.query_one("#prompt", Input).value = ""
        if not message:
            return
        self._write_user(message)
        self._write_assistant("Agent turns arrive in the next phase.")

    def _refresh_status(self) -> None:
        quota = self.runtime.quota.get_5h_status()
        self.query_one(StatusBar).update_status(
            model=self.runtime.settings.model,
            used_tokens=quota.used_tokens,
            limit_tokens=quota.limit_tokens,
        )
        self.query_one(RoleBar).update_role(self.runtime.settings.role)

    def _write_user(self, message: str) -> None:
        self.transcript.append(f"User: {message}")
        text = Text()
        text.append("User: ", style="bold white")
        text.append(message)
        self.query_one("#conversation", RichLog).write(text)

    def _write_assistant(self, message: str) -> None:
        self.transcript.append(f"✦: {message}")
        text = Text()
        text.append("✦: ", style=GEMINI_BLUE)
        text.append(message)
        self.query_one("#conversation", RichLog).write(text)

    def _write_info(self, message: str) -> None:
        self.transcript.append(message)
        self.query_one("#conversation", RichLog).write(
            Text(message, style=GEMINI_MUTED)
        )


async def run_tui(runtime: CliRuntime) -> None:
    """Run the full-screen terminal application."""
    app = GeminiApp(runtime)
    await app.run_async()


__all__ = ["GeminiApp", "run_tui"]
