"""Integration tests for the think-act-observe agent loop."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from gemini_cli.config import Settings
from gemini_cli.core.agent import Agent
from gemini_cli.core.context_manager import ContextManager
from gemini_cli.core.quota_tracker import QuotaTracker
from gemini_cli.core.session import Session
from gemini_cli.core.streaming import AgentEvent, AgentEventType
from gemini_cli.llm import LLMError, Message, StreamEvent, ToolCallRequest
from gemini_cli.sandbox.approval import ApprovalDecision, ApprovalManager
from gemini_cli.tools import ToolRegistry
from gemini_cli.tools.base import RiskLevel, ToolResult
from tests.fixtures.fake_llm import (
    FakeLLMClient,
    error_response,
    text_response,
    thinking_response,
    tool_call_response,
)
from tests.fixtures.fake_tools import FakeApprovalCallback, FakeParams, FakeTool

SYSTEM_PROMPT = "You are a test agent."


@dataclass
class Harness:
    """Everything a test needs to drive one agent."""

    agent: Agent
    session: Session
    client: FakeLLMClient
    quota: QuotaTracker
    registry: ToolRegistry


def build_harness(
    tmp_path: Path,
    responses: Sequence[Sequence[StreamEvent] | BaseException],
    *,
    tools: Sequence[FakeTool] = (),
    approval: ApprovalManager | None = None,
    max_iterations: int = 25,
    max_input_chars: int = 32_000,
) -> Harness:
    """Build an agent wired to deterministic collaborators."""
    settings = Settings(
        _env_file=None,
        allowed_roots=[tmp_path],
        max_agent_iterations=max_iterations,
        max_input_chars=max_input_chars,
    )
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)

    quota = QuotaTracker(
        tmp_path / "quota.json",
        five_hour_limit=10_000,
        weekly_limit=100_000,
        now=lambda: datetime(2026, 1, 5, 12, 0, 0),
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
        registry=registry,
    )


async def collect(harness: Harness, message: str = "hello") -> list[AgentEvent]:
    return [event async for event in harness.agent.run(harness.session, message)]


def event_types(events: Sequence[AgentEvent]) -> list[AgentEventType]:
    return [event.event_type for event in events]


def make_call(
    call_id: str = "call_1",
    name: str = "fake_tool",
    value: str = "x",
) -> ToolCallRequest:
    return ToolCallRequest(id=call_id, name=name, arguments={"value": value})


async def test_text_only_turn(tmp_path: Path) -> None:
    harness = build_harness(tmp_path, [text_response("Hello there")])

    events = await collect(harness)

    assert event_types(events) == [AgentEventType.TEXT_CHUNK, AgentEventType.DONE]
    assert events[0].data == "Hello there"
    assert [message.role for message in harness.session.messages] == [
        "user",
        "assistant",
    ]
    assert harness.session.messages[1].content == "Hello there"
    assert harness.session.total_input_tokens == 10
    assert harness.session.total_output_tokens == 5
    assert harness.session.iteration_count == 1
    assert harness.quota.get_5h_status().used_tokens == 15


async def test_request_shape(tmp_path: Path) -> None:
    tool = FakeTool()
    harness = build_harness(tmp_path, [text_response("hi")], tools=[tool])

    await collect(harness, "what is this?")

    request = harness.client.calls[0]
    assert request["system_instruction"] == SYSTEM_PROMPT
    assert all(isinstance(message, Message) for message in request["messages"])
    assert request["messages"][0].role == "user"
    assert request["messages"][0].content == "what is this?"
    assert [definition.name for definition in request["tools"]] == ["fake_tool"]
    assert "value" in request["tools"][0].parameters_json_schema["properties"]


async def test_tool_call_round_trip(tmp_path: Path) -> None:
    tool = FakeTool(result=ToolResult(success=True, output="file contents"))
    call = make_call()
    harness = build_harness(
        tmp_path,
        [tool_call_response([call]), text_response("All done")],
        tools=[tool],
    )

    events = await collect(harness)

    assert event_types(events) == [
        AgentEventType.TOOL_CALL_START,
        AgentEventType.TOOL_CALL_END,
        AgentEventType.TEXT_CHUNK,
        AgentEventType.DONE,
    ]
    assert events[0].data is call
    assert events[0].metadata == {"tool": "fake_tool", "call_id": "call_1"}
    assert events[1].data == ToolResult(success=True, output="file contents")
    assert events[1].metadata == {
        "tool": "fake_tool",
        "call_id": "call_1",
        "status": "success",
    }

    assert tool.calls == [FakeParams(value="x")]
    assert [message.role for message in harness.session.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    observation = harness.session.messages[2]
    assert observation.name == "fake_tool"
    assert observation.tool_call_id == "call_1"
    assert observation.content == "file contents"

    assert len(harness.client.calls) == 2
    assert len(harness.client.calls[1]["messages"]) == 3
    assert harness.session.total_input_tokens == 20
    assert harness.session.total_output_tokens == 10
    assert harness.quota.get_5h_status().used_tokens == 30


async def test_multiple_tool_calls_in_one_turn(tmp_path: Path) -> None:
    tool = FakeTool()
    harness = build_harness(
        tmp_path,
        [
            tool_call_response([make_call(value="a"), make_call("call_2", value="b")]),
            text_response("done"),
        ],
        tools=[tool],
    )

    events = await collect(harness)

    # Both requests are announced while the stream is consumed, then each
    # execution completes in order.
    assert event_types(events) == [
        AgentEventType.TOOL_CALL_START,
        AgentEventType.TOOL_CALL_START,
        AgentEventType.TOOL_CALL_END,
        AgentEventType.TOOL_CALL_END,
        AgentEventType.TEXT_CHUNK,
        AgentEventType.DONE,
    ]
    assert [call.value for call in tool.calls] == ["a", "b"]
    observations = [
        message for message in harness.session.messages if message.role == "tool"
    ]
    assert len(observations) == 2


async def test_iteration_limit_stops_the_loop(tmp_path: Path) -> None:
    tool = FakeTool()
    harness = build_harness(
        tmp_path,
        [
            tool_call_response([make_call("call_1")]),
            tool_call_response([make_call("call_2")]),
            text_response("never requested"),
        ],
        tools=[tool],
        max_iterations=2,
    )

    events = await collect(harness)

    assert event_types(events) == [
        AgentEventType.TOOL_CALL_START,
        AgentEventType.TOOL_CALL_END,
        AgentEventType.TOOL_CALL_START,
        AgentEventType.TOOL_CALL_END,
        AgentEventType.ERROR,
        AgentEventType.DONE,
    ]
    assert "maximum" in events[-2].data
    assert "2" in events[-2].data
    assert harness.session.iteration_count == 2
    assert harness.client.remaining == 1


async def test_denied_tool_call_is_observed_and_loop_continues(
    tmp_path: Path,
) -> None:
    callback = FakeApprovalCallback(ApprovalDecision.DENIED)
    tool = FakeTool(risk_level=RiskLevel.WRITE)
    harness = build_harness(
        tmp_path,
        [tool_call_response([make_call()]), text_response("Understood")],
        tools=[tool],
        approval=ApprovalManager(callback),
    )

    events = await collect(harness)

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
    assert len(callback.calls) == 1


async def test_missing_approval_handler_denies_destructive_tool(
    tmp_path: Path,
) -> None:
    tool = FakeTool(risk_level=RiskLevel.DESTRUCTIVE)
    harness = build_harness(
        tmp_path,
        [tool_call_response([make_call()]), text_response("Understood")],
        tools=[tool],
    )

    events = await collect(harness)

    assert event_types(events) == [
        AgentEventType.TOOL_CALL_START,
        AgentEventType.TOOL_CALL_END,
        AgentEventType.TEXT_CHUNK,
        AgentEventType.DONE,
    ]
    assert events[1].metadata is not None
    assert events[1].metadata["status"] == "denied"
    assert tool.calls == []
    assert "requires approval" in harness.session.messages[2].content


async def test_unknown_tool_produces_an_observation(tmp_path: Path) -> None:
    harness = build_harness(
        tmp_path,
        [
            tool_call_response([make_call(name="missing_tool")]),
            text_response("recovered"),
        ],
    )

    events = await collect(harness)

    assert event_types(events) == [
        AgentEventType.TOOL_CALL_START,
        AgentEventType.TOOL_CALL_END,
        AgentEventType.TEXT_CHUNK,
        AgentEventType.DONE,
    ]
    assert events[1].metadata is not None
    assert events[1].metadata["status"] == "error"
    assert "Unknown tool" in harness.session.messages[2].content
    assert harness.session.messages[-1].content == "recovered"


async def test_tool_failure_is_reported_without_breaking_the_loop(
    tmp_path: Path,
) -> None:
    tool = FakeTool(result=ToolResult(success=False, output="partial", error="boom"))
    harness = build_harness(
        tmp_path,
        [tool_call_response([make_call()]), text_response("noted")],
        tools=[tool],
    )

    events = await collect(harness)

    assert events[1].metadata is not None
    assert events[1].metadata["status"] == "error"
    assert harness.session.messages[2].content == "boom\n\npartial"
    assert harness.session.messages[-1].content == "noted"


async def test_empty_message_is_rejected(tmp_path: Path) -> None:
    harness = build_harness(tmp_path, [])

    events = await collect(harness, "   ")

    assert event_types(events) == [AgentEventType.ERROR, AgentEventType.DONE]
    assert "must not be empty" in events[0].data
    assert harness.client.calls == []
    assert harness.session.messages == []


async def test_oversized_message_is_rejected(tmp_path: Path) -> None:
    harness = build_harness(tmp_path, [], max_input_chars=100)

    events = await collect(harness, "x" * 101)

    assert event_types(events) == [AgentEventType.ERROR, AgentEventType.DONE]
    assert "100" in events[0].data
    assert harness.client.calls == []
    assert harness.session.messages == []


async def test_stream_error_event_ends_the_turn(tmp_path: Path) -> None:
    harness = build_harness(
        tmp_path,
        [error_response("rate limited", status="rate_limited")],
    )

    events = await collect(harness)

    assert event_types(events) == [AgentEventType.ERROR, AgentEventType.DONE]
    assert events[0].data == "rate limited"
    assert events[0].metadata == {"status": "rate_limited"}
    assert [message.role for message in harness.session.messages] == ["user"]


async def test_raised_llm_error_ends_the_turn(tmp_path: Path) -> None:
    harness = build_harness(tmp_path, [LLMError("model unavailable")])

    events = await collect(harness)

    assert event_types(events) == [AgentEventType.ERROR, AgentEventType.DONE]
    assert events[0].data == "model unavailable"


async def test_read_tool_is_auto_approved(tmp_path: Path) -> None:
    callback = FakeApprovalCallback()
    tool = FakeTool(risk_level=RiskLevel.READ)
    harness = build_harness(
        tmp_path,
        [tool_call_response([make_call()]), text_response("done")],
        tools=[tool],
        approval=ApprovalManager(callback),
    )

    await collect(harness)

    assert callback.calls == []
    assert len(tool.calls) == 1


async def test_write_tool_asks_for_approval(tmp_path: Path) -> None:
    callback = FakeApprovalCallback(ApprovalDecision.APPROVED)
    tool = FakeTool(risk_level=RiskLevel.WRITE)
    harness = build_harness(
        tmp_path,
        [tool_call_response([make_call()]), text_response("done")],
        tools=[tool],
        approval=ApprovalManager(callback),
    )

    await collect(harness)

    assert len(callback.calls) == 1
    assert callback.calls[0][1] == "write"
    assert len(tool.calls) == 1


async def test_thinking_is_shown_but_not_replayed(tmp_path: Path) -> None:
    harness = build_harness(
        tmp_path,
        [thinking_response("pondering", "answer")],
    )

    events = await collect(harness)

    assert event_types(events) == [
        AgentEventType.THINKING,
        AgentEventType.TEXT_CHUNK,
        AgentEventType.DONE,
    ]
    assert events[0].data == "pondering"
    assert harness.session.messages[1].content == "answer"


async def test_done_is_emitted_exactly_once(tmp_path: Path) -> None:
    harness = build_harness(tmp_path, [text_response("hi")])

    events = await collect(harness)

    assert event_types(events).count(AgentEventType.DONE) == 1
