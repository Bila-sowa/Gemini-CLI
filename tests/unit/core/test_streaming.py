"""Tests for the stable agent event stream."""

from __future__ import annotations

import pytest

from gemini_cli.core.streaming import (
    AgentEvent,
    AgentEventType,
    done_event,
    error_event,
    from_stream_event,
    tool_call_end,
)
from gemini_cli.llm import StreamEvent, ToolCallRequest
from gemini_cli.tools.base import ToolResult


def call(call_id: str = "call_1", name: str = "read_file") -> ToolCallRequest:
    return ToolCallRequest(id=call_id, name=name, arguments={"path": "a.txt"})


def test_event_types_have_stable_values() -> None:
    assert AgentEventType.TEXT_CHUNK == "text_chunk"
    assert AgentEventType.TOOL_CALL_START == "tool_call_start"
    assert AgentEventType.TOOL_CALL_END == "tool_call_end"
    assert AgentEventType.THINKING == "thinking"
    assert AgentEventType.ERROR == "error"
    assert AgentEventType.DONE == "done"


def test_agent_event_defaults() -> None:
    event = AgentEvent(AgentEventType.DONE)

    assert event.data is None
    assert event.metadata is None


def test_agent_event_is_frozen() -> None:
    event = AgentEvent(AgentEventType.DONE)

    with pytest.raises(AttributeError):
        event.data = "changed"  # type: ignore[misc]


def test_text_delta_becomes_text_chunk() -> None:
    event = StreamEvent("text_delta", data="hello")

    mapped = from_stream_event(event)

    assert mapped is not None
    assert mapped.event_type is AgentEventType.TEXT_CHUNK
    assert mapped.data == "hello"
    assert mapped.metadata is None


def test_thought_delta_becomes_thinking() -> None:
    event = StreamEvent("text_delta", data="hmm", metadata={"thought": True})

    mapped = from_stream_event(event)

    assert mapped is not None
    assert mapped.event_type is AgentEventType.THINKING
    assert mapped.data == "hmm"


def test_non_string_text_data_becomes_empty() -> None:
    mapped = from_stream_event(StreamEvent("text_delta", data=None))

    assert mapped is not None
    assert mapped.data == ""


def test_tool_call_delta_becomes_tool_call_start() -> None:
    requested = call()

    mapped = from_stream_event(StreamEvent("tool_call_delta", data=requested))

    assert mapped is not None
    assert mapped.event_type is AgentEventType.TOOL_CALL_START
    assert mapped.data is requested
    assert mapped.metadata == {"tool": "read_file", "call_id": "call_1"}


def test_tool_call_delta_with_unknown_payload_is_ignored() -> None:
    assert from_stream_event(StreamEvent("tool_call_delta", data="oops")) is None


def test_error_event_keeps_status_metadata() -> None:
    event = StreamEvent(
        "error",
        data=RuntimeError("rate limited"),
        metadata={"status": "rate_limited"},
    )

    mapped = from_stream_event(event)

    assert mapped is not None
    assert mapped.event_type is AgentEventType.ERROR
    assert mapped.data == "rate limited"
    assert mapped.metadata == {"status": "rate_limited"}


def test_error_without_metadata_has_none() -> None:
    mapped = from_stream_event(StreamEvent("error", data=RuntimeError("boom")))

    assert mapped is not None
    assert mapped.metadata is None


@pytest.mark.parametrize("event_type", ["usage", "done"])
def test_bookkeeping_events_are_not_converted(event_type: str) -> None:
    assert from_stream_event(StreamEvent(event_type, data=object())) is None


def test_tool_call_end_carries_result_and_status() -> None:
    requested = call()
    result = ToolResult(success=True, output="ok")

    event = tool_call_end(requested, result, status="success")

    assert event.event_type is AgentEventType.TOOL_CALL_END
    assert event.data is result
    assert event.metadata == {
        "tool": "read_file",
        "call_id": "call_1",
        "status": "success",
    }


def test_done_event_helper() -> None:
    event = done_event()

    assert event.event_type is AgentEventType.DONE
    assert event.data is None


def test_error_event_helper_is_recoverable_by_default() -> None:
    event = error_event("iteration limit reached")

    assert event.event_type is AgentEventType.ERROR
    assert event.data == "iteration limit reached"
    assert event.metadata == {"recoverable": True}


def test_error_event_helper_can_be_fatal() -> None:
    event = error_event("configuration error", recoverable=False)

    assert event.metadata == {"recoverable": False}
