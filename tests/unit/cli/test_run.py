"""Tests for the one-shot run command."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest
from typer.testing import CliRunner

from gemini_cli.auth import AuthenticationError
from gemini_cli.cli.app import app
from gemini_cli.config import Settings
from gemini_cli.core.session import Session
from gemini_cli.core.streaming import AgentEvent, AgentEventType

runner = CliRunner()


class FakeAgent:
    """Yield a scripted event list and record each turn."""

    def __init__(self, events: Sequence[AgentEvent]) -> None:
        self.events = list(events)
        self.calls: list[tuple[Session, str]] = []

    async def run(
        self,
        session: Session,
        user_message: str,
    ) -> AsyncIterator[AgentEvent]:
        self.calls.append((session, user_message))
        for event in self.events:
            yield event


class FakeRuntime:
    """Minimal runtime surface used by the run command."""

    def __init__(self, events: Sequence[AgentEvent]) -> None:
        self.settings = Settings(
            _env_file=None,
            model="gemini-2.5-flash",
            role="basic",
        )
        self.agent = FakeAgent(events)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def install_runtime(
    monkeypatch: pytest.MonkeyPatch,
    runtime: FakeRuntime,
    captured: dict[str, Any] | None = None,
) -> None:
    async def fake_build(
        settings: Settings,
        *,
        model: str | None = None,
        role: str | None = None,
        approval_callback: object = None,
    ) -> FakeRuntime:
        del settings
        if captured is not None:
            captured["model"] = model
            captured["role"] = role
            captured["approval_callback"] = approval_callback
        return runtime

    monkeypatch.setattr("gemini_cli.cli.commands.run.build_runtime", fake_build)


def test_run_streams_text_and_exits_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = FakeRuntime(
        [
            AgentEvent(AgentEventType.TEXT_CHUNK, "Hello"),
            AgentEvent(AgentEventType.TEXT_CHUNK, " there"),
            AgentEvent(AgentEventType.DONE),
        ]
    )
    captured: dict[str, Any] = {}
    install_runtime(monkeypatch, runtime, captured)

    result = runner.invoke(app, ["run", "say hello"])

    assert result.exit_code == 0
    assert "Hello there" in result.stdout
    assert runtime.closed is True
    assert runtime.agent.calls[0][1] == "say hello"
    assert captured["approval_callback"] is None


def test_run_forwards_model_and_role(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime([AgentEvent(AgentEventType.DONE)])
    captured: dict[str, Any] = {}
    install_runtime(monkeypatch, runtime, captured)

    result = runner.invoke(
        app,
        ["run", "hi", "--model", "gemini-2.5-pro", "--role", "code"],
    )

    assert result.exit_code == 0
    assert captured["model"] == "gemini-2.5-pro"
    assert captured["role"] == "code"


def test_run_ignores_tool_events(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime(
        [
            AgentEvent(AgentEventType.TOOL_CALL_START),
            AgentEvent(AgentEventType.TOOL_CALL_END),
            AgentEvent(AgentEventType.TEXT_CHUNK, "Done"),
            AgentEvent(AgentEventType.DONE),
        ]
    )
    install_runtime(monkeypatch, runtime)

    result = runner.invoke(app, ["run", "use a tool"])

    assert result.exit_code == 0
    assert "Done" in result.stdout


def test_run_reports_agent_error(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = FakeRuntime(
        [
            AgentEvent(AgentEventType.ERROR, "boom"),
            AgentEvent(AgentEventType.DONE),
        ]
    )
    install_runtime(monkeypatch, runtime)

    result = runner.invoke(app, ["run", "fail"])

    assert result.exit_code == 1
    assert runtime.closed is True


def test_run_reports_authentication_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def failing_build(
        settings: Settings,
        **kwargs: object,
    ) -> None:
        del settings, kwargs
        raise AuthenticationError("no key configured")

    monkeypatch.setattr("gemini_cli.cli.commands.run.build_runtime", failing_build)

    result = runner.invoke(app, ["run", "hi"])

    assert result.exit_code == 1
