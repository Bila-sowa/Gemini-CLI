"""Tests for the context-window tracker."""

from __future__ import annotations

import pytest

from gemini_cli.core.context_manager import ContextManager, estimate_messages
from gemini_cli.core.context_tracker import ContextWindowTracker
from gemini_cli.llm import Message


def test_new_tracker_starts_empty() -> None:
    tracker = ContextWindowTracker(1_000, reserved_output_tokens=100)

    status = tracker.status()

    assert status.used_tokens == 0
    assert status.window_tokens == 1_000
    assert status.budget_tokens == 900
    assert status.reserved_output_tokens == 100
    assert status.percentage == 0.0
    assert status.remaining_tokens == 900
    assert status.trimmed is False


def test_observe_updates_percentage_and_remaining() -> None:
    tracker = ContextWindowTracker(1_000, reserved_output_tokens=100)

    tracker.observe(450, trimmed=True)
    status = tracker.status()

    assert status.used_tokens == 450
    assert status.percentage == 50.0
    assert status.remaining_tokens == 450
    assert status.trimmed is True


def test_percentage_is_rounded() -> None:
    tracker = ContextWindowTracker(300)

    tracker.observe(100)

    assert tracker.status().percentage == 33.33


def test_percentage_can_exceed_one_hundred() -> None:
    tracker = ContextWindowTracker(10)

    tracker.observe(20)

    assert tracker.status().percentage == 200.0
    assert tracker.status().remaining_tokens == 0


def test_reset_clears_the_previous_conversation() -> None:
    tracker = ContextWindowTracker(1_000)
    tracker.observe(500, trimmed=True)

    tracker.reset()

    assert tracker.status().used_tokens == 0
    assert tracker.status().trimmed is False


def test_budget_is_clamped_when_output_reserve_covers_the_window() -> None:
    tracker = ContextWindowTracker(1_000, reserved_output_tokens=2_000)

    assert tracker.budget_tokens == 1


def test_refresh_recomputes_from_a_message_list() -> None:
    tracker = ContextWindowTracker(1_000, reserved_output_tokens=100)
    messages = [
        Message(role="user", content="hello"),
        Message(role="assistant", content="world"),
    ]
    manager = ContextManager(900)

    status = tracker.refresh(
        messages,
        context_manager=manager,
        system_prompt_tokens=10,
        tool_definitions_tokens=5,
    )

    assert status.used_tokens == 10 + 5 + estimate_messages(messages)
    assert status.trimmed is False


def test_refresh_reports_trimmed_history() -> None:
    tracker = ContextWindowTracker(60)
    messages = [
        Message(role="user", content="first " * 20),
        Message(role="assistant", content="second " * 20),
        Message(role="user", content="latest"),
    ]
    manager = ContextManager(30, preserve_recent=1)

    status = tracker.refresh(
        messages,
        context_manager=manager,
        system_prompt_tokens=0,
        tool_definitions_tokens=0,
    )

    assert status.trimmed is True
    assert status.used_tokens <= status.budget_tokens


def test_refresh_is_independent_of_previous_observations() -> None:
    tracker = ContextWindowTracker(1_000)
    tracker.observe(999)
    messages = [Message(role="user", content="short")]

    status = tracker.refresh(
        messages,
        context_manager=ContextManager(1_000),
        system_prompt_tokens=0,
        tool_definitions_tokens=0,
    )

    assert status.used_tokens == estimate_messages(messages)


@pytest.mark.parametrize("window", [0, -1])
def test_rejects_non_positive_window(window: int) -> None:
    with pytest.raises(ValueError, match="window_tokens"):
        ContextWindowTracker(window)


def test_rejects_negative_reserved_output() -> None:
    with pytest.raises(ValueError, match="reserved_output_tokens"):
        ContextWindowTracker(1_000, reserved_output_tokens=-1)


def test_rejects_negative_usage() -> None:
    tracker = ContextWindowTracker(1_000)

    with pytest.raises(ValueError, match="used_tokens"):
        tracker.observe(-1)
