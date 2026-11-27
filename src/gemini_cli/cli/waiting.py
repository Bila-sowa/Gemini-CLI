"""Waiting indicators shared by the CLI presentation layers.

The loader animates only while a model request is in flight. It must never
overlap streamed text or run while a tool is being executed, so its state is
derived from the agent's tool-call events instead of from idle time.
"""

from __future__ import annotations

import asyncio

from rich.console import Console
from rich.status import Status

from gemini_cli.core.streaming import AgentEventType

WAITING_TEXT = "Waiting for Gemini..."
WAITING_SHOW_DELAY = 0.15


class ModelWaitTracker:
    """Derive whether the agent is waiting for the next model response."""

    def __init__(self) -> None:
        self.waiting = True
        self._pending_tools = 0
        self._finished = False

    def on_event(self, event_type: AgentEventType) -> bool:
        """Update the state for one agent event and return the new state."""
        if event_type is AgentEventType.ERROR or event_type is AgentEventType.DONE:
            self._finished = True
            self.waiting = False
            return False
        if self._finished:
            self.waiting = False
            return False

        if event_type is AgentEventType.TOOL_CALL_START:
            self._pending_tools += 1
            self.waiting = False
        elif event_type is AgentEventType.TOOL_CALL_END:
            if self._pending_tools == 0:
                # An unmatched completion must not start a spinner.
                self.waiting = False
            else:
                self._pending_tools -= 1
                self.waiting = self._pending_tools == 0
        else:
            # Text and thinking chunks mean the server has started answering.
            self.waiting = False
        return self.waiting


class RichWaitIndicator:
    """Animated Rich spinner for the non-TUI command and line fallback.

    The spinner writes to stderr so piped stdout stays clean. It is disabled
    automatically when stderr is not an interactive terminal, which also keeps
    test runners and CI logs free of control sequences.
    """

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        delay: float = WAITING_SHOW_DELAY,
        console: Console | None = None,
    ) -> None:
        self._console = console if console is not None else Console(stderr=True)
        self._enabled = self._console.is_terminal if enabled is None else enabled
        self._delay = max(0.0, delay)
        self._waiting = False
        self._closed = False
        self._status: Status | None = None
        self._show_handle: asyncio.TimerHandle | None = None

    def set_waiting(self, waiting: bool) -> None:
        """Show or hide the spinner according to *waiting*."""
        if self._closed:
            return

        if waiting:
            if self._waiting:
                return
            self._waiting = True
            if not self._enabled:
                return
            if self._delay <= 0:
                self._start()
                return
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                self._start()
                return
            self._show_handle = loop.call_later(self._delay, self._start)
            return

        self._waiting = False
        self._cancel_pending_show()
        self._stop()

    def close(self) -> None:
        """Stop the spinner; safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        self._waiting = False
        self._cancel_pending_show()
        self._stop()

    def _create_status(self) -> Status:
        return self._console.status(
            WAITING_TEXT,
            spinner="dots",
            spinner_style="cyan",
        )

    def _start(self) -> None:
        self._show_handle = None
        if self._closed or not self._waiting or not self._enabled:
            return
        if self._status is not None:
            return
        self._status = self._create_status()
        self._status.start()

    def _stop(self) -> None:
        if self._status is None:
            return
        self._status.stop()
        self._status = None

    def _cancel_pending_show(self) -> None:
        if self._show_handle is None:
            return
        self._show_handle.cancel()
        self._show_handle = None


__all__ = [
    "WAITING_SHOW_DELAY",
    "WAITING_TEXT",
    "ModelWaitTracker",
    "RichWaitIndicator",
]
