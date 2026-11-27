"""Full-screen Textual terminal UI for the Gemini CLI.

The app owns the screen layout — logo, status bar, conversation, prompt, and
role footer — while the agent runtime stays in :mod:`gemini_cli.cli.runtime`.
Assistant turns stream into a scrollable conversation, and sensitive tool
calls open a modal confirmation screen.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import ClassVar, Literal

from textual import events
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
from gemini_cli.cli.tui.activity import AgentMessageView
from gemini_cli.cli.tui.clipboard import (
    SELECTION_FALLBACK_FILE_NAME,
    copy_with_feedback,
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
    ContextPanel,
    Logo,
    MenuScreen,
    MessageView,
    RoleBar,
    StatusBar,
    WaitingIndicator,
)
from gemini_cli.cli.waiting import WAITING_SHOW_DELAY, ModelWaitTracker
from gemini_cli.config.env_file import EnvFileError, set_env_value
from gemini_cli.core.session import Session
from gemini_cli.core.streaming import AgentEventType
from gemini_cli.llm import ToolCallRequest, UnknownModelError
from gemini_cli.llm.prompts import PromptError
from gemini_cli.sandbox.approval import ApprovalDecision

DEFAULT_PROMPT_PLACEHOLDER = "Take your task"
STOP_PROMPT_PLACEHOLDER = "Gemini is responding — press Esc to stop"
TOOLS_PROMPT_PLACEHOLDER = "Running tools — press Esc to stop after they finish"
STOPPING_PROMPT_PLACEHOLDER = "Stopping…"
STOPPING_TOOLS_PROMPT_PLACEHOLDER = "Stopping after the current tool finishes…"


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
    #context {
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
        ("escape", "stop_or_hide_context", "Stop / hide context"),
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
        self._turn_active = False
        self._model_active = False
        self._stop_requested = False
        self.transcript: list[str] = []
        self.session = Session(
            model=runtime.settings.model,
            role=runtime.settings.role,
        )

    def compose(self) -> ComposeResult:
        yield Logo()
        yield StatusBar()
        yield VerticalScroll(id="conversation")
        yield ContextPanel()
        yield Input(placeholder=DEFAULT_PROMPT_PLACEHOLDER, id="prompt")
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

    def on_text_selected(self, event: events.TextSelected) -> None:
        """Copy a finished mouse selection and confirm it with a toast.

        Textual posts ``TextSelected`` on mouse release, including for a plain
        click that cleared the previous selection, so empty selections are
        ignored. Copying runs in a worker because a native helper can take up
        to its timeout and must not stall the message pump.
        """
        text = self.screen.get_selected_text()
        if not text or not text.strip():
            return
        self.run_worker(
            copy_with_feedback(
                self,
                text,
                label="Selection",
                fallback_file_name=SELECTION_FALLBACK_FILE_NAME,
            ),
            group="clipboard",
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
                self.runtime.context_window.reset()
                self._refresh_status()
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
            case CommandAction.CONTEXT:
                self._toggle_context()
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
        self._turn_active = True
        self._model_active = True
        self._stop_requested = False
        prompt.placeholder = STOP_PROMPT_PLACEHOLDER
        conversation = self.query_one("#conversation", VerticalScroll)
        view = AgentMessageView(
            "✦: ",
            self._theme.assistant_color,
            theme=self._theme,
        )
        await conversation.mount(view)
        indicator = WaitingIndicator(self._theme, show_delay=self._waiting_delay)
        tracker = ModelWaitTracker()
        stopped = False
        try:
            await conversation.mount(indicator)
            indicator.show()
            async for event in self.runtime.agent.run(self.session, message):
                # Clear the spinner before rendering anything so streamed text
                # is never interleaved with terminal animation.
                indicator.hide()
                if event.event_type is AgentEventType.TEXT_CHUNK:
                    await view.append_text(str(event.data))
                    self._scroll_to_end()
                elif event.event_type is AgentEventType.THINKING:
                    await view.append_thought(str(event.data))
                    self._scroll_to_end()
                elif event.event_type is AgentEventType.ERROR:
                    await view.add_error(str(event.data))
                    self._scroll_to_end()
                elif event.event_type is AgentEventType.DONE:
                    await view.finish()
                    break

                if event.event_type is AgentEventType.TOOL_CALL_START:
                    call = (
                        event.data if isinstance(event.data, ToolCallRequest) else None
                    )
                    await view.start_tool(call, event.metadata)
                    self._scroll_to_end()
                    self._model_active = False
                    if not self._stop_requested:
                        prompt.placeholder = TOOLS_PROMPT_PLACEHOLDER
                elif event.event_type is AgentEventType.TOOL_CALL_END:
                    view.finish_tool(event.data, event.metadata)
                    self._scroll_to_end()

                waiting = tracker.on_event(event.event_type)
                if waiting:
                    self._model_active = True
                    if (
                        event.event_type is AgentEventType.TOOL_CALL_END
                        and self._stop_requested
                    ):
                        self.workers.cancel_group(self, "agent")
                    else:
                        prompt.placeholder = STOP_PROMPT_PLACEHOLDER
                        indicator.show()
        except asyncio.CancelledError:
            stopped = True
            raise
        finally:
            if stopped:
                view.mark_stopped()
            await view.finish()
            indicator.hide()
            if indicator.is_mounted:
                await indicator.remove()
            self._turn_active = False
            self._model_active = False
            self._stop_requested = False
            prompt.placeholder = DEFAULT_PROMPT_PLACEHOLDER
            prompt.disabled = False
            prompt.focus()
            self.transcript.append(view.plain)
            if stopped:
                self._write_info("Generation stopped.")
            self._refresh_status()

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
        self.runtime.agent.refresh_context(self.session)
        context = self.runtime.context_window.status()
        self.query_one(StatusBar).update_status(
            model=self.runtime.settings.model,
            used_tokens=context.used_tokens,
            budget_tokens=context.budget_tokens,
            percentage=context.percentage,
            theme=self._theme,
        )
        self.query_one(RoleBar).update_role(
            self.runtime.settings.role,
            self._theme,
        )
        self.query_one(ContextPanel).update_context(
            context,
            self.runtime.settings.model,
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
        conversation = self.query_one("#conversation", VerticalScroll)
        for agent_view in self.query(AgentMessageView):
            agent_view.apply_theme(self._theme)
            await agent_view.rerender()

        for view in self.query(MarkdownMessageView):
            # Markdown segments inside an agent turn are handled above.
            if view.parent is not conversation:
                continue
            view.apply_theme(self._theme)
            await view.rerender()

        for info_view in self.query(MessageView):
            info_view.apply_theme(self._theme.muted)

    def _toggle_context(self) -> None:
        panel = self.query_one(ContextPanel)
        panel.display = not panel.display
        if panel.display:
            panel.update_context(
                self.runtime.context_window.status(),
                self.runtime.settings.model,
                self._theme,
            )
            self._scroll_to_end()

    def action_stop_or_hide_context(self) -> None:
        """Stop the active turn, or hide the context panel when idle."""
        if self._turn_active:
            self._request_stop()
        else:
            self.action_hide_context()

    def _request_stop(self) -> None:
        if self._stop_requested:
            return
        self._stop_requested = True
        prompt = self.query_one("#prompt", Input)
        if self._model_active:
            prompt.placeholder = STOPPING_PROMPT_PLACEHOLDER
            self.workers.cancel_group(self, "agent")
        else:
            prompt.placeholder = STOPPING_TOOLS_PROMPT_PLACEHOLDER

    def action_hide_context(self) -> None:
        self.query_one(ContextPanel).display = False


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
