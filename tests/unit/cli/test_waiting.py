"""Tests for the model-wait state machine and the Rich CLI spinner."""

from __future__ import annotations

import asyncio

import pytest

from gemini_cli.cli.waiting import ModelWaitTracker, RichWaitIndicator
from gemini_cli.core.streaming import AgentEventType


class FakeStatus:
    """Records the lifecycle calls made by RichWaitIndicator."""

    def __init__(self) -> None:
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


def test_tracker_starts_waiting() -> None:
    tracker = ModelWaitTracker()

    assert tracker.waiting is True


@pytest.mark.parametrize(
    "event_type",
    [
        AgentEventType.TEXT_CHUNK,
        AgentEventType.THINKING,
        AgentEventType.ERROR,
        AgentEventType.DONE,
    ],
)
def test_tracker_stops_on_response_or_terminal_events(
    event_type: AgentEventType,
) -> None:
    tracker = ModelWaitTracker()

    assert tracker.on_event(event_type) is False
    assert tracker.waiting is False


def test_tracker_resumes_after_the_last_parallel_tool() -> None:
    tracker = ModelWaitTracker()

    assert tracker.on_event(AgentEventType.TOOL_CALL_START) is False
    assert tracker.on_event(AgentEventType.TOOL_CALL_START) is False
    assert tracker.on_event(AgentEventType.TOOL_CALL_END) is False
    assert tracker.on_event(AgentEventType.TOOL_CALL_END) is True
    assert tracker.waiting is True


def test_tracker_ignores_an_unmatched_tool_end() -> None:
    tracker = ModelWaitTracker()

    assert tracker.on_event(AgentEventType.TOOL_CALL_END) is False
    assert tracker.waiting is False


def test_tracker_does_not_resume_after_it_finished() -> None:
    tracker = ModelWaitTracker()
    tracker.on_event(AgentEventType.TOOL_CALL_START)

    assert tracker.on_event(AgentEventType.ERROR) is False
    assert tracker.on_event(AgentEventType.TOOL_CALL_END) is False
    assert tracker.waiting is False


def test_rich_indicator_disabled_never_creates_a_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    indicator = RichWaitIndicator(enabled=False, delay=0)

    def unexpected_status() -> FakeStatus:
        raise AssertionError("disabled indicator must not start a spinner")

    monkeypatch.setattr(indicator, "_create_status", unexpected_status)

    indicator.set_waiting(True)
    indicator.set_waiting(False)
    indicator.close()


def test_rich_indicator_starts_and_stops_a_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = FakeStatus()
    indicator = RichWaitIndicator(enabled=True, delay=0)
    monkeypatch.setattr(indicator, "_create_status", lambda: status)

    indicator.set_waiting(True)
    indicator.set_waiting(True)

    assert status.started is True

    indicator.set_waiting(False)

    assert status.stopped is True

    indicator.close()


async def test_rich_indicator_cancels_a_delayed_show(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[FakeStatus] = []

    def make_status() -> FakeStatus:
        status = FakeStatus()
        created.append(status)
        return status

    indicator = RichWaitIndicator(enabled=True, delay=10)
    monkeypatch.setattr(indicator, "_create_status", make_status)

    indicator.set_waiting(True)
    indicator.set_waiting(False)

    await asyncio.sleep(0)

    assert created == []

    indicator.close()


async def test_rich_indicator_shows_after_the_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = FakeStatus()
    indicator = RichWaitIndicator(enabled=True, delay=0.001)
    monkeypatch.setattr(indicator, "_create_status", lambda: status)

    indicator.set_waiting(True)
    await asyncio.sleep(0.01)

    assert status.started is True

    indicator.close()

    assert status.stopped is True
