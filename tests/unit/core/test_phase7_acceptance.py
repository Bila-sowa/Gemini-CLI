"""Acceptance checks for every Phase 7 criterion.

These tests use the same fakes as the integration suite but state the
acceptance criteria directly, so a regression in the agent runtime fails a
test that names the broken promise.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import gemini_cli.core
from gemini_cli.config import Settings
from gemini_cli.core.agent import Agent
from gemini_cli.core.context_manager import ContextManager
from gemini_cli.core.quota_tracker import QuotaTracker
from gemini_cli.core.session import Session
from gemini_cli.core.streaming import AgentEvent, AgentEventType
from gemini_cli.llm import StreamEvent, ToolCallRequest, UsageInfo
from gemini_cli.sandbox.approval import ApprovalDecision, ApprovalManager
from gemini_cli.tools import ToolRegistry
from gemini_cli.tools.base import RiskLevel, ToolResult
from tests.fixtures.fake_llm import (
    FakeLLMClient,
    text_response,
    tool_call_response,
)
from tests.fixtures.fake_tools import FakeApprovalCallback, FakeTool

SYSTEM_PROMPT = "You are a test agent."


@dataclass
class Harness:
    agent: Agent
    session: Session
    client: FakeLLMClient
    quota: QuotaTracker
    tools: list[FakeTool]


def build_harness(
    tmp_path: Path,
    responses: Sequence[Sequence[StreamEvent] | BaseException],
    *,
    tools: Sequence[FakeTool] = (),
    approval: ApprovalManager | None = None,
    max_iterations: int = 25,
    now: datetime | None = None,
) -> Harness:
    settings = Settings(
        _env_file=None,
        allowed_roots=[tmp_path],
        max_agent_iterations=max_iterations,
    )
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)

    current = now if now is not None else datetime(2026, 1, 5, 12, 0, 0)
    quota = QuotaTracker(
        tmp_path / "quota.json",
        five_hour_limit=10_000,
        weekly_limit=100_000,
        now=lambda: current,
    )
    client = FakeLLMClient(responses)
    agent = Agent(
        client,
        registry,
        approval if approval is not None else ApprovalManager(),
        ContextManager(50_000),
        quota,
        settings,
        SYSTEM_PROMPT,
    )
    return Harness(
        agent=agent,
        session=Session(model="gemini-2.5-flash"),
        client=client,
        quota=quota,
        tools=list(tools),
    )


async def collect(harness: Harness, message: str = "hello") -> list[AgentEvent]:
    return [event async for event in harness.agent.run(harness.session, message)]


def make_call(call_id: str = "call_1") -> ToolCallRequest:
    return ToolCallRequest(id=call_id, name="fake_tool", arguments={"value": "x"})


def event_types(events: Sequence[AgentEvent]) -> list[AgentEventType]:
    return [event.event_type for event in events]


async def test_text_query_yields_text_chunks_then_done(tmp_path: Path) -> None:
    harness = build_harness(tmp_path, [text_response("Hello from the agent")])

    events = await collect(harness, "say hello")

    assert event_types(events) == [AgentEventType.TEXT_CHUNK, AgentEventType.DONE]
    assert events[0].data == "Hello from the agent"


async def test_single_tool_call_round_trip(tmp_path: Path) -> None:
    tool = FakeTool(result=ToolResult(success=True, output="observed"))
    harness = build_harness(
        tmp_path,
        [tool_call_response([make_call()]), text_response("Finished")],
        tools=[tool],
    )

    events = await collect(harness, "use the tool")

    assert event_types(events) == [
        AgentEventType.TOOL_CALL_START,
        AgentEventType.TOOL_CALL_END,
        AgentEventType.TEXT_CHUNK,
        AgentEventType.DONE,
    ]
    assert len(tool.calls) == 1
    assert harness.session.messages[2].content == "observed"


async def test_iteration_limit_yields_error_then_done(tmp_path: Path) -> None:
    tool = FakeTool()
    responses = [tool_call_response([make_call(f"call_{index}")]) for index in range(3)]
    harness = build_harness(
        tmp_path,
        responses,
        tools=[tool],
        max_iterations=3,
    )

    events = await collect(harness, "keep going")

    assert event_types(events) == [
        AgentEventType.TOOL_CALL_START,
        AgentEventType.TOOL_CALL_END,
        AgentEventType.TOOL_CALL_START,
        AgentEventType.TOOL_CALL_END,
        AgentEventType.TOOL_CALL_START,
        AgentEventType.TOOL_CALL_END,
        AgentEventType.ERROR,
        AgentEventType.DONE,
    ]
    assert "maximum" in events[-2].data
    assert harness.session.iteration_count == 3


async def test_denied_tool_call_is_observed_and_loop_continues(
    tmp_path: Path,
) -> None:
    callback = FakeApprovalCallback(ApprovalDecision.DENIED)
    tool = FakeTool(risk_level=RiskLevel.WRITE)
    harness = build_harness(
        tmp_path,
        [tool_call_response([make_call()]), text_response("Continuing")],
        tools=[tool],
        approval=ApprovalManager(callback),
    )

    events = await collect(harness, "write something")

    assert event_types(events) == [
        AgentEventType.TOOL_CALL_START,
        AgentEventType.TOOL_CALL_END,
        AgentEventType.TEXT_CHUNK,
        AgentEventType.DONE,
    ]
    assert events[1].metadata is not None
    assert events[1].metadata["status"] == "denied"
    assert tool.calls == []
    assert harness.session.messages[2].content == ("The user denied this tool call.")
    assert harness.session.messages[-1].content == "Continuing"


def test_context_manager_trims_long_history() -> None:
    session = Session()
    for index in range(10):
        session.add_user_message(f"user message number {index}")
        session.add_assistant_message(f"assistant reply number {index}")

    manager = ContextManager(60, preserve_recent=2)
    fitted = manager.fit_to_window(session.get_history())

    assert len(fitted) < len(session.messages)
    assert fitted[-2:] == session.messages[-2:]
    assert fitted[0].role == "user"


def test_quota_tracker_accumulates_and_rolls_windows(tmp_path: Path) -> None:
    clock = [datetime(2026, 1, 5, 12, 0, 0)]
    tracker = QuotaTracker(
        tmp_path / "quota.json",
        five_hour_limit=100,
        weekly_limit=1_000,
        now=lambda: clock[0],
    )
    tracker.record_usage("gemini-2.5-flash", 30, 20)

    five_hour = tracker.get_5h_status()
    weekly = tracker.get_weekly_status()
    assert five_hour.used_tokens == 50
    assert five_hour.percentage == 50.0
    assert weekly.used_tokens == 50

    clock[0] += timedelta(hours=6)
    assert tracker.get_5h_status().used_tokens == 0
    assert tracker.get_weekly_status().used_tokens == 50

    clock[0] += timedelta(days=7)
    assert tracker.get_weekly_status().used_tokens == 0


def test_session_tracks_cumulative_usage() -> None:
    session = Session()

    session.update_usage(UsageInfo(input_tokens=10, output_tokens=5))
    session.update_usage(UsageInfo(input_tokens=20, output_tokens=7))

    assert session.total_input_tokens == 30
    assert session.total_output_tokens == 12
    assert session.total_tokens == 42


def test_core_does_not_import_cli_or_tui() -> None:
    package = Path(gemini_cli.core.__file__).parent
    offenders: list[tuple[str, str]] = []

    for path in sorted(package.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            for module in modules:
                if module == "gemini_cli.cli" or module.startswith("gemini_cli.cli."):
                    offenders.append((path.name, module))
                if module == "gemini_cli.tui" or module.startswith("gemini_cli.tui."):
                    offenders.append((path.name, module))

    assert offenders == []
