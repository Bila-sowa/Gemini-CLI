"""Tests for the interactive REPL."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence

import pytest

from gemini_cli.cli.tui.repl import ConsoleApprovalCallback, run_repl
from gemini_cli.config import Settings
from gemini_cli.core.session import Session
from gemini_cli.core.streaming import AgentEvent, AgentEventType
from gemini_cli.sandbox.approval import ApprovalDecision


class FakeAgent:
    """Yield a scripted event list for every turn."""

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
    """Minimal runtime surface used by the REPL."""

    def __init__(self, events: Sequence[AgentEvent]) -> None:
        self.settings = Settings(
            _env_file=None,
            model="gemini-2.5-flash",
            role="basic",
        )
        self.agent = FakeAgent(events)


def scripted_input(
    lines: Sequence[str],
) -> Callable[[str], str]:
    """Return an input function that ends with EOF."""

    iterator = iter(lines)

    def read(_prompt: str) -> str:
        try:
            return next(iterator)
        except StopIteration:
            raise EOFError from None

    return read


def collector() -> tuple[Callable[[str], None], list[str]]:
    parts: list[str] = []

    def write(text: str) -> None:
        parts.append(text)

    return write, parts


async def test_repl_streams_text_turn() -> None:
    runtime = FakeRuntime(
        [
            AgentEvent(AgentEventType.TEXT_CHUNK, "Hello"),
            AgentEvent(AgentEventType.DONE),
        ]
    )
    write, parts = collector()

    await run_repl(runtime, input_func=scripted_input(["hi"]), output=write)

    text = "".join(parts)
    assert "Hello" in text
    assert "GEMINI CLI" in text
    assert len(runtime.agent.calls) == 1
    assert runtime.agent.calls[0][1] == "hi"


async def test_repl_prints_errors() -> None:
    runtime = FakeRuntime(
        [
            AgentEvent(AgentEventType.ERROR, "boom"),
            AgentEvent(AgentEventType.DONE),
        ]
    )
    write, parts = collector()

    await run_repl(runtime, input_func=scripted_input(["hi"]), output=write)

    assert "Error: boom" in "".join(parts)


async def test_repl_exits_on_exit_command() -> None:
    runtime = FakeRuntime([])
    write, _parts = collector()

    await run_repl(runtime, input_func=scripted_input(["exit"]), output=write)

    assert runtime.agent.calls == []


async def test_repl_exits_on_eof() -> None:
    runtime = FakeRuntime([])
    write, _parts = collector()

    await run_repl(runtime, input_func=scripted_input([]), output=write)

    assert runtime.agent.calls == []


async def test_repl_defers_slash_commands() -> None:
    runtime = FakeRuntime([])
    write, parts = collector()

    await run_repl(runtime, input_func=scripted_input(["/help"]), output=write)

    assert runtime.agent.calls == []
    assert "Phase 8.6" in "".join(parts)


async def test_approval_accepts_yes() -> None:
    callback = ConsoleApprovalCallback(input_func=lambda _prompt: "y")

    decision = await callback.request_approval("run a tool", "write", "path: a")

    assert decision is ApprovalDecision.APPROVED


@pytest.mark.parametrize("answer", ["n", "", "no"])
async def test_approval_denies_by_default(answer: str) -> None:
    callback = ConsoleApprovalCallback(input_func=lambda _prompt: answer)

    decision = await callback.request_approval("run a tool", "write", "path: a")

    assert decision is ApprovalDecision.DENIED


async def test_approval_denies_on_eof() -> None:
    def failing_input(_prompt: str) -> str:
        raise EOFError

    callback = ConsoleApprovalCallback(input_func=failing_input)

    decision = await callback.request_approval("run a tool", "write", "path: a")

    assert decision is ApprovalDecision.DENIED


def test_fake_runtime_has_expected_settings() -> None:
    runtime = FakeRuntime([])

    assert runtime.settings.model == "gemini-2.5-flash"
    assert runtime.settings.role == "basic"
