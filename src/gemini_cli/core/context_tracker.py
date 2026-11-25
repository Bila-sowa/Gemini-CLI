"""Track how much of the model's context window a conversation uses.

Context occupancy is a derived view of one session, not persisted state: it is
recomputed from the conversation history, the system prompt, the exported tool
definitions, and the model's context window. Saving and restoring a session
therefore only needs to restore the conversation; the status is rebuilt on the
next refresh.

The estimate intentionally matches :mod:`gemini_cli.core.context_manager`, so
the displayed percentage reflects the same budget that trims the request.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from gemini_cli.core.context_manager import ContextManager, estimate_messages
from gemini_cli.llm import Message


@dataclass(frozen=True)
class ContextWindowStatus:
    """How much of the model's input budget the conversation occupies."""

    used_tokens: int
    window_tokens: int
    budget_tokens: int
    reserved_output_tokens: int
    percentage: float
    remaining_tokens: int
    trimmed: bool = False


class ContextWindowTracker:
    """Derive context-window occupancy for the active conversation."""

    def __init__(
        self,
        window_tokens: int,
        *,
        reserved_output_tokens: int = 0,
    ) -> None:
        if window_tokens <= 0:
            msg = "window_tokens must be positive."
            raise ValueError(msg)
        if reserved_output_tokens < 0:
            msg = "reserved_output_tokens must not be negative."
            raise ValueError(msg)
        self._window_tokens = window_tokens
        self._reserved_output_tokens = reserved_output_tokens
        # A misconfigured max_output_tokens must not produce a zero budget.
        self._budget_tokens = max(1, window_tokens - reserved_output_tokens)
        self._used_tokens = 0
        self._trimmed = False

    @property
    def window_tokens(self) -> int:
        """The model's full context window."""
        return self._window_tokens

    @property
    def budget_tokens(self) -> int:
        """Tokens available for the request after reserving the response."""
        return self._budget_tokens

    @property
    def reserved_output_tokens(self) -> int:
        """Tokens reserved for the model's response."""
        return self._reserved_output_tokens

    def reset(self) -> None:
        """Forget the previous conversation, for example after ``/clear``."""
        self._used_tokens = 0
        self._trimmed = False

    def observe(self, used_tokens: int, *, trimmed: bool = False) -> None:
        """Record the size of an already fitted request.

        The agent uses this in its hot path after ``fit_to_window()`` so the
        history is not fitted twice. Use :meth:`refresh` when recomputing from
        an arbitrary conversation, such as a restored session.
        """
        if used_tokens < 0:
            msg = "used_tokens must not be negative."
            raise ValueError(msg)
        self._used_tokens = used_tokens
        self._trimmed = trimmed

    def refresh(
        self,
        messages: Sequence[Message],
        *,
        context_manager: ContextManager,
        system_prompt_tokens: int,
        tool_definitions_tokens: int,
    ) -> ContextWindowStatus:
        """Recompute the status from scratch for *messages*.

        This is the entry point for startup, model changes, and restored
        sessions: the tracker never keeps authoritative state of its own.
        """
        fitted = context_manager.fit_to_window(
            messages,
            system_prompt_tokens=system_prompt_tokens,
            tool_definitions_tokens=tool_definitions_tokens,
        )
        used = (
            system_prompt_tokens + tool_definitions_tokens + estimate_messages(fitted)
        )
        self.observe(used, trimmed=len(fitted) < len(messages))
        return self.status()

    def status(self) -> ContextWindowStatus:
        """Return the current occupancy snapshot."""
        return ContextWindowStatus(
            used_tokens=self._used_tokens,
            window_tokens=self._window_tokens,
            budget_tokens=self._budget_tokens,
            reserved_output_tokens=self._reserved_output_tokens,
            percentage=round(self._used_tokens / self._budget_tokens * 100, 2),
            remaining_tokens=max(0, self._budget_tokens - self._used_tokens),
            trimmed=self._trimmed,
        )


__all__ = ["ContextWindowStatus", "ContextWindowTracker"]
