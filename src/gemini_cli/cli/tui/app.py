"""Full-screen Textual terminal UI for the Gemini CLI.

The app owns the screen layout — logo, status bar, conversation, prompt, and
role footer — while the agent runtime stays in :mod:`gemini_cli.cli.runtime`.
Assistant turns stream into a scrollable conversation, and sensitive tool
calls open a modal confirmation screen.
"""

from __future__ import annotations

from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import BindingType
from textual.containers import VerticalScroll
from textual.widgets import Input

from gemini_cli.cli.runtime import CliRuntime
from gemini_cli.cli.tui.widgets import (
    GEMINI_BLUE,
    GEMINI_MUTED,
    GEMINI_RED,
    ApprovalScreen,
    Logo,
    MessageView,
    RoleBar,
    StatusBar,
)
from gemini_cli.core.session import Session
from gemini_cli.core.streaming import AgentEventType
from gemini_cli.sandbox.approval import ApprovalDecision


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
        scrollbar-size: 1 1;
    }
    .message {
        width: 100%;
        height: auto;
        margin: 0 0 0 0;
    }
    #prompt {
        border: round #3A3F45;
        margin: 0 1;
        background: #1E2023;
    }
    #prompt:disabled {
        opacity: 60%;
    }
    #footer {
        height: 1;
        margin: 0 1;
    }
    ApprovalScreen {
        align: center middle;
        background: #101214;
    }
    #approval {
        width: 72;
        height: auto;
        border: round #3A3F45;
        background: #24272B;
        padding: 1 2;
    }
    #approval-title {
        height: auto;
        margin-bottom: 1;
    }
    #approval-action {
        height: auto;
    }
    #approval-details {
        height: auto;
        margin-bottom: 1;
    }
    #approval-buttons {
        height: auto;
        align: right middle;
    }
    #approval-buttons Button {
        margin-left: 2;
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
        yield VerticalScroll(id="conversation")
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
        self.run_worker(
            self._run_turn(message),
            group="agent",
            exclusive=True,
        )

    async def request_approval(
        self,
        action_description: str,
        risk_level: str,
        details: str,
    ) -> bool:
        """Show the approval modal and return the user's decision."""
        return await self.push_screen_wait(
            ApprovalScreen(action_description, risk_level, details)
        )

    async def _run_turn(self, message: str) -> None:
        prompt = self.query_one("#prompt", Input)
        prompt.disabled = True
        view = self._new_message("✦: ", GEMINI_BLUE)
        try:
            async for event in self.runtime.agent.run(self.session, message):
                if event.event_type is AgentEventType.TEXT_CHUNK:
                    view.append(str(event.data))
                    self._scroll_to_end()
                elif event.event_type is AgentEventType.THINKING:
                    continue
                elif event.event_type is AgentEventType.ERROR:
                    view.append(f"\nError: {event.data}", style=GEMINI_RED)
                    self._scroll_to_end()
                elif event.event_type is AgentEventType.DONE:
                    break
        finally:
            prompt.disabled = False
            prompt.focus()
            self._refresh_status()
            self.transcript.append(view.plain)

    def _refresh_status(self) -> None:
        quota = self.runtime.quota.get_5h_status()
        self.query_one(StatusBar).update_status(
            model=self.runtime.settings.model,
            used_tokens=quota.used_tokens,
            limit_tokens=quota.limit_tokens,
        )
        self.query_one(RoleBar).update_role(self.runtime.settings.role)

    def _new_message(self, prefix: str, prefix_style: str) -> MessageView:
        view = MessageView(prefix, prefix_style)
        self.query_one("#conversation", VerticalScroll).mount(view)
        return view

    def _write_user(self, message: str) -> None:
        self.transcript.append(f"User: {message}")
        view = self._new_message("User: ", "bold white")
        view.append(message)
        self._scroll_to_end()

    def _write_assistant(self, message: str) -> None:
        self.transcript.append(f"✦: {message}")
        view = self._new_message("✦: ", GEMINI_BLUE)
        view.append(message)
        self._scroll_to_end()

    def _write_info(self, message: str) -> None:
        self.transcript.append(message)
        view = self._new_message("", GEMINI_MUTED)
        view.append(message, style=GEMINI_MUTED)
        self._scroll_to_end()

    def _scroll_to_end(self) -> None:
        self.query_one("#conversation", VerticalScroll).scroll_end(animate=False)


class TuiApprovalBridge:
    """Approval callback that routes confirmations into the running app."""

    def __init__(self) -> None:
        self._app: GeminiApp | None = None

    def attach(self, app: GeminiApp) -> None:
        """Bind the bridge to the app after both objects exist."""
        self._app = app

    async def request_approval(
        self,
        action_description: str,
        risk_level: str,
        details: str,
    ) -> ApprovalDecision:
        if self._app is None:
            return ApprovalDecision.DENIED
        approved = await self._app.request_approval(
            action_description,
            risk_level,
            details,
        )
        return ApprovalDecision.APPROVED if approved else ApprovalDecision.DENIED


async def run_tui(
    runtime: CliRuntime,
    *,
    bridge: TuiApprovalBridge | None = None,
) -> None:
    """Run the full-screen terminal application."""
    app = GeminiApp(runtime)
    if bridge is not None:
        bridge.attach(app)
    await app.run_async()


__all__ = ["GeminiApp", "TuiApprovalBridge", "run_tui"]
