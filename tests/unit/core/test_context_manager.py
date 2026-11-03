"""Tests for the context window manager."""

from __future__ import annotations

import pytest

from gemini_cli.core.context_manager import (
    MESSAGE_OVERHEAD_TOKENS,
    ContextManager,
    estimate_message_tokens,
    estimate_messages,
    estimate_tokens,
)
from gemini_cli.llm import Message, ToolCallRequest


def user(content: str) -> Message:
    return Message(role="user", content=content)


def assistant(
    content: str = "",
    tool_calls: tuple[ToolCallRequest, ...] = (),
) -> Message:
    return Message(role="assistant", content=content, tool_calls=tool_calls)


def tool(call_id: str = "call_1", text: str = "ok") -> Message:
    return Message(role="tool", content=text, tool_call_id=call_id, name="read_file")


def call(call_id: str = "call_1") -> ToolCallRequest:
    return ToolCallRequest(id=call_id, name="read_file", arguments={})


def test_estimate_tokens_is_zero_for_empty_text() -> None:
    assert estimate_tokens("") == 0


def test_estimate_tokens_rounds_up() -> None:
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2
    assert estimate_tokens("a" * 400) == 100


def test_estimate_message_tokens_includes_overhead() -> None:
    assert estimate_message_tokens(user("abcd")) == MESSAGE_OVERHEAD_TOKENS + 1


def test_estimate_message_tokens_counts_tool_calls() -> None:
    message = assistant(tool_calls=(call(),))

    # overhead + name "read_file" + "{}" arguments
    assert estimate_message_tokens(message) == MESSAGE_OVERHEAD_TOKENS + 3 + 1


def test_estimate_messages_sums_messages() -> None:
    messages = [user("abcd"), user("efgh")]

    assert estimate_messages(messages) == 2 * (MESSAGE_OVERHEAD_TOKENS + 1)


def test_empty_history_returns_empty() -> None:
    assert ContextManager(100).fit_to_window([]) == []


def test_history_that_fits_is_unchanged() -> None:
    messages = [user("aaaa"), assistant("bbbb"), user("cccc")]
    manager = ContextManager(1_000)

    assert manager.fit_to_window(messages) == messages


def test_oldest_messages_are_dropped_first() -> None:
    first = user("aaaa")
    second = assistant("bbbb")
    third = user("cccc")
    manager = ContextManager(12, preserve_recent=1)

    assert manager.fit_to_window([first, second, third]) == [third]


def test_recent_messages_are_preserved_even_over_budget() -> None:
    first = user("aaaa")
    second = assistant("bbbb")
    third = user("cccc")
    manager = ContextManager(1, preserve_recent=2)

    result = manager.fit_to_window([first, second, third])

    assert second in result
    assert third in result
    assert result[-1] is third
    assert result[0].role == "user"


def test_tool_group_is_never_split() -> None:
    start = user("start")
    turn = assistant("x" * 400, tool_calls=(call(),))
    observation = tool()
    end = user("end")
    manager = ContextManager(15, preserve_recent=1)

    result = manager.fit_to_window([start, turn, observation, end])

    assert result == [end]
    assert observation not in result
    assert turn not in result


def test_tool_group_is_kept_with_its_observations() -> None:
    start = user("start")
    turn = assistant("x" * 400, tool_calls=(call(),))
    observation = tool()
    end = user("end")
    manager = ContextManager(130, preserve_recent=1)

    result = manager.fit_to_window([start, turn, observation, end])

    assert result == [start, turn, observation, end]
    assert result.index(turn) + 1 == result.index(observation)


def test_preserved_window_protects_a_spanning_tool_group() -> None:
    start = user("start")
    turn = assistant(tool_calls=(call(),))
    observation = tool()
    manager = ContextManager(12, preserve_recent=2)

    result = manager.fit_to_window([start, turn, observation])

    assert turn in result
    assert observation in result


def test_result_starts_with_user_when_one_exists() -> None:
    messages = [
        user("first"),
        assistant("answer"),
        user("second"),
        assistant("answer two"),
        user("third"),
    ]
    manager = ContextManager(30, preserve_recent=1)

    result = manager.fit_to_window(messages)

    assert result[0].role == "user"
    assert result[-1] is messages[-1]


def test_available_tokens_subtracts_fixed_parts() -> None:
    manager = ContextManager(100)

    assert manager.available_tokens(30, 20) == 50
    assert manager.available_tokens(80, 40) == 0


def test_fixed_parts_reduce_the_history_budget() -> None:
    first = user("aaaa")
    second = user("bbbb")
    third = user("cccc")
    manager = ContextManager(18, preserve_recent=1)

    result = manager.fit_to_window(
        [first, second, third],
        system_prompt_tokens=7,
        tool_definitions_tokens=5,
    )

    assert result == [third]


def test_large_protected_message_is_still_returned() -> None:
    message = user("y" * 1_000)
    manager = ContextManager(1, preserve_recent=1)

    assert manager.fit_to_window([message]) == [message]


def test_input_history_is_not_mutated() -> None:
    messages = [user("aaaa"), assistant("bbbb")]
    original = list(messages)
    manager = ContextManager(1)

    manager.fit_to_window(messages)

    assert messages == original


def test_properties_expose_configuration() -> None:
    manager = ContextManager(64, preserve_recent=2)

    assert manager.max_tokens == 64
    assert manager.preserve_recent == 2


@pytest.mark.parametrize("max_tokens", [0, -1])
def test_rejects_non_positive_budget(max_tokens: int) -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        ContextManager(max_tokens)


def test_rejects_negative_preserve_recent() -> None:
    with pytest.raises(ValueError, match="preserve_recent"):
        ContextManager(100, preserve_recent=-1)
