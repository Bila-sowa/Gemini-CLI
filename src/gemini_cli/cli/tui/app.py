"""Full-screen Textual terminal UI for the Gemini CLI.

The app owns the screen layout — logo, status bar, conversation, prompt, and
role footer — while the agent runtime stays in :mod:`gemini_cli.cli.runtime`.
Assistant turns stream into a scrollable conversation, and sensitive tool
calls open a modal confirmation screen.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar, Literal

from textual.app import App, ComposeResult
from textual.binding import BindingType
from textual.containers import VerticalScroll
from textual.widgets import Input

from gemini_cli.auth import AuthenticationError, CredentialStoreError
from gemini_cli.auth.credentials_store import delete_api_key
from gemini_cli.cli.command_arguments import (
    CommandArgumentError,
    parse_model_argument,
    parse_role_argument,
    parse_theme_argument,
)
from gemini_cli.cli.runtime import CliRuntime, build_runtime
from gemini_cli.cli.slash_commands import (
    CommandAction,
    CommandProblem,
    format_command_problem,
    format_unexpected_argument,
    parse_slash_command,
    render_command_help,
    render_help,
    resolve_command,
    resolve_command_name,
)
from gemini_cli.cli.tui.markdown import MarkdownMessageView
from gemini_cli.cli.tui.repl import (
    AUTH_OPTIONS,
    model_options,
    role_options,
    theme_options,
)
from gemini_cli.cli.tui.themes import (
    DEFAULT_THEME,
    THEMES,
    Theme,
    get_theme,
    to_textual_theme,
)
from gemini_cli.cli.tui.widgets import (
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
        background: $background;
        color: $text;
    }
    #logo {
        height: 6;
        margin: 1 1 0 1;
    }
    #status {
        height: 3;
        border: round $primary 30%;
        padding: 0 1;
        margin: 0 1;
    }
    #conversation {
        height: 1fr;
        margin: 0 1;
        padding: 0;
        background: $background;
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
        border: round $primary 30%;
        margin: 0 1;
        background: $background;
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
        background: $background;
    }
    #approval {
        width: 72;
        height: auto;
        border: round $primary 30%;
        background: $surface;
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
        border: round $primary 30%;
        padding: 0 1;
        margin: 0 1;
    }
    MenuScreen {
        align: center middle;
        background: $background;
    }
    #menu {
        width: 84;
        height: auto;
        max-height: 24;
        border: round $primary 30%;
        background: $surface;
        padding: 1 2;
    }
    #menu-title {
        height: auto;
        margin-bottom: 1;
    }
    #menu-list {
        height: auto;
        max-height: 18;
        background: $surface;
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
        resolution = resolve_command(command)
        if isinstance(resolution, CommandProblem):
            self._write_info(format_command_problem(resolution))
            return

        spec = resolution.spec
        unexpected = format_unexpected_argument(spec, resolution.argument)
        if unexpected is not None:
            self._write_info(unexpected)
            return

        match spec.action:
            case CommandAction.EXIT:
                self.exit()
            case CommandAction.HELP:
                if resolution.argument:
                    help_resolution = resolve_command_name(resolution.argument)
                    if isinstance(help_resolution, CommandProblem):
                        self._write_info(format_command_problem(help_resolution))
                    else:
                        self._write_info(
                            render_command_help(help_resolution.spec).rstrip()
                        )
                else:
                    self._write_info(render_help().rstrip())
            case CommandAction.CLEAR:
                self.query_one("#conversation", VerticalScroll).remove_children()
                self.session = Session(
                    model=self.runtime.settings.model,
                    role=self.runtime.settings.role,
                )
                self._write_info("Conversation cleared.")
            case CommandAction.ROLE:
                if resolution.argument:
                    await self._apply_role_argument(resolution.argument)
                else:
                    selected = await self.push_screen_wait(
                        MenuScreen(role_options(), "Roles:", self._theme)
                    )
                    if selected is not None and await self._rebuild(
                        role=selected.value
                    ):
                        self._persist_env("GEMINI_ROLE", selected.value)
            case CommandAction.MODEL:
                if resolution.argument:
                    await self._apply_model_argument(resolution.argument)
                else:
                    selected = await self.push_screen_wait(
                        MenuScreen(
                            model_options(self.runtime.settings.model),
                            "Models:",
                            self._theme,
                        )
                    )
                    if selected is not None and await self._rebuild(
                        model=selected.value
                    ):
                        self._persist_env("GEMINI_MODEL", selected.value)
            case CommandAction.AUTH:
                selected = await self.push_screen_wait(
                    MenuScreen(AUTH_OPTIONS, "Authentication methods:", self._theme)
                )
                if selected is None:
                    return
                if selected.value == "api_key":
                    self._write_info("Run 'gemini-cli-py login' to store an API key.")
                else:
                    self._write_info("Google OAuth is not available yet.")
            case CommandAction.RESET:
                try:
                    delete_api_key()
                except CredentialStoreError as exc:
                    self._write_info(f"Error: {exc}")
                else:
                    self._write_info(
                        "Stored API key removed. Run 'gemini-cli-py login' again."
                    )
            case CommandAction.QUOTA:
                self._toggle_quota()
            case CommandAction.THEME:
                if resolution.argument:
                    await self._apply_theme_argument(resolution.argument)
                else:
                    selected = await self.push_screen_wait(
                        MenuScreen(theme_options(), "Themes:", self._theme)
                    )
                    if selected is not None:
                        await self._set_theme(selected.value)

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

    async def _apply_model_argument(self, raw: str) -> None:
        """Apply a model supplied directly on the command line."""
        try:
            value = parse_model_argument(raw)
        except CommandArgumentError as exc:
            self._write_info(f"Error: {exc}")
            return

        if value == self.runtime.settings.model:
            self._write_info("Model unchanged.")
            return

        if await self._rebuild(model=value):
            self._persist_env("GEMINI_MODEL", value)
            self._write_info(f"Model: {value}")

    async def _apply_role_argument(self, raw: str) -> None:
        """Apply a role supplied directly on the command line."""
        try:
            value = parse_role_argument(raw)
        except CommandArgumentError as exc:
            self._write_info(f"Error: {exc}")
            return

        if value == self.runtime.settings.role:
            self._write_info("Role unchanged.")
            return

        if await self._rebuild(role=value):
            self._persist_env("GEMINI_ROLE", value)
            self._write_info(f"Role: {value}")

    async def _apply_theme_argument(self, raw: str) -> None:
        """Apply a theme supplied directly on the command line."""
        try:
            value = parse_theme_argument(raw)
        except CommandArgumentError as exc:
            self._write_info(f"Error: {exc}")
            return
        await self._set_theme(value)

    def _persist_env(self, key: str, value: str) -> None:
        """Save a model, role, or theme choice in the project ``.env`` file."""
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

    async def _set_theme(self, name: str) -> None:
        """Apply and persist a theme selected from the menu."""
        self._theme = get_theme(name)
        self._apply_theme()
        await self._refresh_messages()
        self._persist_env("GEMINI_THEME", self._theme.name)
        self._write_info(f"Theme: {self._theme.name}")

    def _apply_theme(self) -> None:
        self.theme = f"gemini-{self._theme.name}"
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
        role: Literal["user", "assistant"] = "assistant",
    ) -> MarkdownMessageView:
        view = MarkdownMessageView(
            prefix,
            prefix_style,
            initial=initial,
            error_style=self._theme.error_color,
            streaming=streaming,
            role=role,
        )
        self.query_one("#conversation", VerticalScroll).mount(view)
        return view

    def _write_user(self, message: str) -> None:
        view = self._new_markdown_message(
            "User: ",
            f"bold {self._theme.user_color}",
            initial=message,
            role="user",
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
        style = self._theme.muted
        view = self._new_message("", style)
        view.append(message, style=style)
        self._scroll_to_end()

    def _scroll_to_end(self) -> None:
        self.query_one("#conversation", VerticalScroll).scroll_end(animate=False)

    async def _refresh_messages(self) -> None:
        """Re-style existing messages so they follow the active theme."""
        for view in self.query(MarkdownMessageView):
            view.apply_theme(self._theme)
            await view.rerender()

        for info_view in self.query(MessageView):
            info_view.apply_theme(self._theme.muted)

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
    theme: Theme | None = None,
) -> None:
    """Run the full-screen terminal application."""
    selected_theme = theme if theme is not None else get_theme(runtime.settings.theme)
    app = GeminiApp(runtime, bridge=bridge, theme=selected_theme)
    if bridge is not None:
        bridge.attach(app)
    await app.run_async()


__all__ = ["GeminiApp", "TuiApprovalBridge", "run_tui"]
