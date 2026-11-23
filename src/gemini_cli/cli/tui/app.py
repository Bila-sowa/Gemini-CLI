"""Full-screen Textual terminal UI for the Gemini CLI.

The app owns the screen layout — logo, status bar, conversation, prompt, and
role footer — while the agent runtime stays in :mod:`gemini_cli.cli.runtime`.
Assistant turns stream into a scrollable conversation, and sensitive tool
calls open a modal confirmation screen.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import BindingType
from textual.containers import VerticalScroll
from textual.widgets import Input

from gemini_cli.auth import AuthenticationError, CredentialStoreError
from gemini_cli.auth.credentials_store import delete_api_key
from gemini_cli.cli.runtime import CliRuntime, build_runtime
from gemini_cli.cli.tui.markdown import MarkdownMessageView
from gemini_cli.cli.tui.repl import (
    AUTH_OPTIONS,
    model_options,
    parse_slash_command,
    render_help,
    role_options,
)
from gemini_cli.cli.tui.themes import (
    DEFAULT_THEME,
    LIGHT_THEME,
    THEMES,
    Theme,
    to_textual_theme,
)
from gemini_cli.cli.tui.widgets import (
    GEMINI_MUTED,
    ApprovalScreen,
    Logo,
    MenuScreen,
    MessageView,
    QuotaPanel,
    RoleBar,
    StatusBar,
    WaitingIndicator,
)
from gemini_cli.cli.waiting import WAITING_SHOW_DELAY, ModelWaitTracker
from gemini_cli.config.env_file import EnvFileError, set_env_value
from gemini_cli.core.session import Session
from gemini_cli.core.streaming import AgentEventType
from gemini_cli.llm import UnknownModelError
from gemini_cli.llm.prompts import PromptError
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
    #waiting {
        height: 1;
        margin: 0 1;
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
    Screen.light {
        background: #F5F6F7;
        color: #1E2023;
    }
    Screen.light #conversation, Screen.light #prompt {
        background: #F5F6F7;
    }
    Screen.light #status, Screen.light #prompt, Screen.light #quota {
        border: round #C7CCD3;
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
    #quota {
        display: none;
        height: auto;
        border: round #3A3F45;
        padding: 0 1;
        margin: 0 1;
    }
    MenuScreen {
        align: center middle;
        background: #101214;
    }
    #menu {
        width: 84;
        height: auto;
        max-height: 24;
        border: round #3A3F45;
        background: #24272B;
        padding: 1 2;
    }
    #menu-title {
        height: auto;
        margin-bottom: 1;
    }
    #menu-list {
        height: auto;
        max-height: 18;
        background: #24272B;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        ("ctrl+q", "quit", "Quit"),
        ("escape", "hide_quota", "Hide quota"),
    ]

    def __init__(
        self,
        runtime: CliRuntime,
        *,
        bridge: TuiApprovalBridge | None = None,
        theme: Theme = DEFAULT_THEME,
        waiting_delay: float = WAITING_SHOW_DELAY,
    ) -> None:
        super().__init__()
        self.runtime = runtime
        self._bridge = bridge
        self._theme = theme
        self._waiting_delay = waiting_delay
        self.transcript: list[str] = []
        self.session = Session(
            model=runtime.settings.model,
            role=runtime.settings.role,
        )

    def compose(self) -> ComposeResult:
        yield Logo()
        yield StatusBar()
        yield VerticalScroll(id="conversation")
        yield QuotaPanel()
        yield Input(placeholder="Take your task", id="prompt")
        yield RoleBar()

    def on_mount(self) -> None:
        for theme in THEMES.values():
            self.register_theme(to_textual_theme(theme))
        self._apply_theme()
        self._refresh_status()
        self._write_assistant("Can I help you today?")
        self.query_one("#prompt", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        message = event.value.strip()
        self.query_one("#prompt", Input).value = ""
        if not message:
            return
        self._write_user(message)
        if message.startswith("/"):
            self.run_worker(
                self._handle_slash(message),
                group="ui",
                exclusive=True,
            )
            return
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
            ApprovalScreen(
                action_description,
                risk_level,
                details,
                theme=self._theme,
            )
        )

    async def _handle_slash(self, message: str) -> None:
        command = parse_slash_command(message)
        if command is None:
            return
        name = command.name

        if name in {"/exit", "/quit"}:
            self.exit()
            return

        if name == "/help":
            self._write_info(render_help().rstrip())
            return

        if name == "/clear":
            self.query_one("#conversation", VerticalScroll).remove_children()
            self.session = Session(
                model=self.runtime.settings.model,
                role=self.runtime.settings.role,
            )
            self._write_info("Conversation cleared.")
            return

        if name == "/role":
            selected = await self.push_screen_wait(
                MenuScreen(role_options(), "Roles:", self._theme)
            )
            if selected is not None and await self._rebuild(role=selected.value):
                self._persist_env("GEMINI_ROLE", selected.value)
            return

        if name == "/model":
            selected = await self.push_screen_wait(
                MenuScreen(
                    model_options(self.runtime.settings.model),
                    "Models:",
                    self._theme,
                )
            )
            if selected is not None and await self._rebuild(model=selected.value):
                self._persist_env("GEMINI_MODEL", selected.value)
            return

        if name == "/auth":
            selected = await self.push_screen_wait(
                MenuScreen(AUTH_OPTIONS, "Authentication methods:", self._theme)
            )
            if selected is None:
                return
            if selected.value == "api_key":
                self._write_info("Run 'gemini-cli-py login' to store an API key.")
            else:
                self._write_info("Google OAuth is not available yet.")
            return

        if name == "/reset":
            try:
                delete_api_key()
            except CredentialStoreError as exc:
                self._write_info(f"Error: {exc}")
            else:
                self._write_info(
                    "Stored API key removed. Run 'gemini-cli-py login' again."
                )
            return

        if name == "/quota":
            self._toggle_quota()
            return

        if name == "/theme":
            self._theme = LIGHT_THEME if self._theme.name == "dark" else DEFAULT_THEME
            self._apply_theme()
            await self._rerender_markdown()
            self._write_info(f"Theme: {self._theme.name}")
            return

        self._write_info(f"Unknown command {name}. Type /help for available commands.")

    async def _rebuild(
        self,
        *,
        model: str | None = None,
        role: str | None = None,
    ) -> bool:
        try:
            replacement = await build_runtime(
                self.runtime.settings,
                model=model,
                role=role,
                approval_callback=self._bridge,
            )
        except (
            AuthenticationError,
            CredentialStoreError,
            UnknownModelError,
            PromptError,
        ) as exc:
            self._write_info(f"Error: {exc}")
            return False

        self.runtime.close()
        self.runtime = replacement
        self.session.model = replacement.settings.model
        self.session.role = replacement.settings.role
        self._refresh_status()
        return True

    def _persist_env(self, key: str, value: str) -> None:
        """Save a model or role choice in the project ``.env`` file."""
        path = Path.cwd() / ".env"
        try:
            set_env_value(path, key, value)
        except (EnvFileError, OSError) as exc:
            self._write_info(f"Could not save {key} to {path}: {exc}")
            return

        self._write_info(f"Saved {key} to {path} for future sessions.")
        if os.environ.get(key):
            self._write_info(
                f"{key} is also set in the environment; that value takes precedence."
            )

    async def _run_turn(self, message: str) -> None:
        prompt = self.query_one("#prompt", Input)
        prompt.disabled = True
        conversation = self.query_one("#conversation", VerticalScroll)
        view = self._new_markdown_message(
            "✦: ",
            self._theme.assistant_color,
            streaming=True,
        )
        indicator = WaitingIndicator(self._theme, show_delay=self._waiting_delay)
        tracker = ModelWaitTracker()
        try:
            await conversation.mount(indicator)
            indicator.show()
            async for event in self.runtime.agent.run(self.session, message):
                # Clear the spinner before rendering anything so streamed text
                # is never interleaved with terminal animation.
                indicator.hide()
                if event.event_type is AgentEventType.TEXT_CHUNK:
                    await view.append(str(event.data))
                    self._scroll_to_end()
                elif event.event_type is AgentEventType.THINKING:
                    pass
                elif event.event_type is AgentEventType.ERROR:
                    view.add_error(str(event.data))
                    self._scroll_to_end()
                elif event.event_type is AgentEventType.DONE:
                    break

                if tracker.on_event(event.event_type):
                    indicator.show()
        finally:
            await view.finish()
            indicator.hide()
            if indicator.is_mounted:
                await indicator.remove()
            prompt.disabled = False
            prompt.focus()
            self._refresh_status()
            self.transcript.append(view.plain)

    def _apply_theme(self) -> None:
        self.theme = f"gemini-{self._theme.name}"
        screen = self.screen
        screen.set_class(self._theme.name == "light", "light")
        screen.set_class(self._theme.name == "dark", "dark")
        self.query_one(Logo).apply_theme(self._theme)
        self._refresh_status()

    def _refresh_status(self) -> None:
        quota = self.runtime.quota.get_5h_status()
        self.query_one(StatusBar).update_status(
            model=self.runtime.settings.model,
            used_tokens=quota.used_tokens,
            limit_tokens=quota.limit_tokens,
            theme=self._theme,
        )
        self.query_one(RoleBar).update_role(
            self.runtime.settings.role,
            self._theme,
        )
        self.query_one(QuotaPanel).update_quota(
            self.runtime.quota.get_5h_status(),
            self.runtime.quota.get_weekly_status(),
            self._theme,
        )

    def _new_message(self, prefix: str, prefix_style: str) -> MessageView:
        view = MessageView(prefix, prefix_style)
        self.query_one("#conversation", VerticalScroll).mount(view)
        return view

    def _new_markdown_message(
        self,
        prefix: str,
        prefix_style: str,
        *,
        initial: str = "",
        streaming: bool = False,
    ) -> MarkdownMessageView:
        view = MarkdownMessageView(
            prefix,
            prefix_style,
            initial=initial,
            error_style=self._theme.error_color,
            streaming=streaming,
        )
        self.query_one("#conversation", VerticalScroll).mount(view)
        return view

    def _write_user(self, message: str) -> None:
        view = self._new_markdown_message(
            "User: ",
            f"bold {self._theme.user_color}",
            initial=message,
        )
        self.transcript.append(view.plain)
        self._scroll_to_end()

    def _write_assistant(self, message: str) -> None:
        view = self._new_markdown_message(
            "✦: ",
            self._theme.assistant_color,
            initial=message,
        )
        self.transcript.append(view.plain)
        self._scroll_to_end()

    def _write_info(self, message: str) -> None:
        self.transcript.append(message)
        view = self._new_message("", GEMINI_MUTED)
        view.append(message, style=GEMINI_MUTED)
        self._scroll_to_end()

    def _scroll_to_end(self) -> None:
        self.query_one("#conversation", VerticalScroll).scroll_end(animate=False)

    async def _rerender_markdown(self) -> None:
        """Re-render conversation Markdown so it follows the new theme."""
        for view in self.query(MarkdownMessageView):
            await view.rerender()

    def _toggle_quota(self) -> None:
        panel = self.query_one(QuotaPanel)
        panel.display = not panel.display
        if panel.display:
            panel.update_quota(
                self.runtime.quota.get_5h_status(),
                self.runtime.quota.get_weekly_status(),
            )
            self._scroll_to_end()

    def action_hide_quota(self) -> None:
        self.query_one(QuotaPanel).display = False


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
    app = GeminiApp(runtime, bridge=bridge)
    if bridge is not None:
        bridge.attach(app)
    await app.run_async()


__all__ = ["GeminiApp", "TuiApprovalBridge", "run_tui"]
