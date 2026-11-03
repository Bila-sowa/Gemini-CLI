"""Tests for conversation session state."""

from __future__ import annotations

from datetime import datetime

import pytest

from gemini_cli.core.session import MAX_OBSERVATION_CHARS, Session
from gemini_cli.llm import ToolCallRequest, UsageInfo
from gemini_cli.tools.base import ToolResult


def call(call_id: str = "call_1", name: str = "read_file") -> ToolCallRequest:
    """Build a tool call for the tests."""
    return ToolCallRequest(id=call_id, name=name, arguments={"path": "a.txt"})


def test_new_session_defaults() -> None:
    session = Session()

    assert session.id
    assert len(session.id) == 32
    assert isinstance(session.created_at, datetime)
    assert session.messages == []
    assert session.model == ""
    assert session.role == "basic"
    assert session.total_input_tokens == 0
    assert session.total_output_tokens == 0
    assert session.total_tokens == 0
    assert session.iteration_count == 0
    assert session.last_message() is None


def test_session_ids_are_unique() -> None:
    assert Session().id != Session().id


def test_add_user_message() -> None:
    session = Session()

    message = session.add_user_message("hello")

    assert message.role == "user"
    assert message.content == "hello"
    assert session.messages == [message]
    assert session.last_message() is message


def test_add_assistant_message() -> None:
    session = Session()

    message = session.add_assistant_message("hi there")

    assert message.role == "assistant"
    assert message.content == "hi there"
    assert message.tool_calls == ()


def test_assistant_message_records_tool_calls() -> None:
    session = Session()

    message = session.add_assistant_message(tool_calls=(call(),))

    assert message.tool_calls == (call(),)
    # The name is remembered for the matching observation.
    observation = session.add_tool_result("call_1", ToolResult(True, "ok"))
    assert observation.name == "read_file"


def test_add_tool_call_appends_to_trailing_assistant_turn() -> None:
    session = Session()
    session.add_user_message("do it")
    session.add_assistant_message("working")

    message = session.add_tool_call(call())

    assert message.role == "assistant"
    assert message.content == "working"
    assert message.tool_calls == (call(),)
    assert session.messages[-1] is message
    assert len(session.messages) == 2


def test_add_tool_call_starts_a_turn_when_needed() -> None:
    session = Session()
    session.add_user_message("do it")

    session.add_tool_call(call())

    assert session.messages[-1].role == "assistant"
    assert session.messages[-1].tool_calls == (call(),)


def test_tool_result_uses_recorded_name() -> None:
    session = Session()
    session.add_assistant_message(tool_calls=(call(),))

    message = session.add_tool_result("call_1", ToolResult(True, "line 1\n"))

    assert message.role == "tool"
    assert message.tool_call_id == "call_1"
    assert message.name == "read_file"
    assert message.content == "line 1\n"


def test_tool_result_for_unknown_call_raises() -> None:
    session = Session()

    with pytest.raises(ValueError, match="No tool call with id"):
        session.add_tool_result("call_missing", ToolResult(True, "ok"))


def test_failed_result_uses_error_text() -> None:
    session = Session()
    session.add_assistant_message(tool_calls=(call(),))

    message = session.add_tool_result(
        "call_1",
        ToolResult(success=False, output="", error="file not found"),
    )

    assert message.content == "file not found"


def test_failed_result_keeps_partial_output() -> None:
    session = Session()
    session.add_assistant_message(tool_calls=(call(),))

    message = session.add_tool_result(
        "call_1",
        ToolResult(success=False, output="partial", error="exit code 2"),
    )

    assert message.content == "exit code 2\n\npartial"


def test_failed_result_without_error_has_fallback_text() -> None:
    session = Session()
    session.add_assistant_message(tool_calls=(call(),))

    message = session.add_tool_result(
        "call_1",
        ToolResult(success=False, output=""),
    )

    assert message.content == "Tool failed without an error message."


def test_long_observation_is_truncated() -> None:
    session = Session()
    session.add_assistant_message(tool_calls=(call(),))
    output = "x" * (MAX_OBSERVATION_CHARS + 50)

    message = session.add_tool_result("call_1", ToolResult(True, output))

    assert len(message.content) == MAX_OBSERVATION_CHARS
    assert message.content.endswith("...")


def test_get_history_returns_a_copy() -> None:
    session = Session()
    session.add_user_message("hello")

    history = session.get_history()
    history.clear()

    assert len(session.messages) == 1
    assert len(history) == 0


def test_update_usage_accumulates() -> None:
    session = Session()

    session.update_usage(UsageInfo(input_tokens=10, output_tokens=5))
    session.update_usage(UsageInfo(input_tokens=3, output_tokens=2))

    assert session.total_input_tokens == 13
    assert session.total_output_tokens == 7
    assert session.total_tokens == 20


def test_record_iteration_counts_cycles() -> None:
    session = Session()

    assert session.record_iteration() == 1
    assert session.record_iteration() == 2
    assert session.iteration_count == 2
