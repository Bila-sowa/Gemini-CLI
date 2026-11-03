"""Keep requests within the model's context window.

The manager works on message groups rather than individual messages: an
assistant turn that requested tools always travels together with the tool
observations that answer those calls. Dropping half of such a group would
produce a history the provider rejects, so the whole group is kept or dropped.

Token counts are estimates. The package has no provider tokenizer dependency,
so a conservative characters-per-token heuristic is used; it only needs to be
good enough to keep a request below the window, not exact for billing.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence

from gemini_cli.llm import Message

logger = logging.getLogger(__name__)

# English prose averages roughly four characters per token. Code and JSON are
# denser, so the estimate intentionally errs on the side of larger counts.
CHARS_PER_TOKEN = 4
MESSAGE_OVERHEAD_TOKENS = 4


def estimate_tokens(text: str) -> int:
    """Approximate the token count of *text*."""
    if not text:
        return 0
    return max(1, (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN)


def estimate_message_tokens(message: Message) -> int:
    """Approximate the token cost of one message, including its tool calls."""
    total = MESSAGE_OVERHEAD_TOKENS + estimate_tokens(message.content)
    for call in message.tool_calls:
        try:
            arguments = json.dumps(call.arguments, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            arguments = str(call.arguments)
        total += estimate_tokens(call.name) + estimate_tokens(arguments)
    return total


def estimate_messages(messages: Sequence[Message]) -> int:
    """Approximate the token cost of a message list."""
    return sum(estimate_message_tokens(message) for message in messages)


def _group_spans(
    messages: Sequence[Message],
) -> list[tuple[list[Message], int, int]]:
    """Split history into indivisible groups with half-open index spans."""
    spans: list[tuple[list[Message], int, int]] = []
    index = 0
    while index < len(messages):
        start = index
        message = messages[index]
        group = [message]
        index += 1

        if message.role == "assistant" and message.tool_calls:
            call_ids = {call.id for call in message.tool_calls}
            while (
                index < len(messages)
                and messages[index].role == "tool"
                and messages[index].tool_call_id in call_ids
            ):
                group.append(messages[index])
                index += 1

        spans.append((group, start, index))
    return spans


class ContextManager:
    """Trim conversation history to fit a token budget."""

    def __init__(self, max_tokens: int, *, preserve_recent: int = 4) -> None:
        if max_tokens <= 0:
            msg = "max_tokens must be positive."
            raise ValueError(msg)
        if preserve_recent < 0:
            msg = "preserve_recent must not be negative."
            raise ValueError(msg)
        self._max_tokens = max_tokens
        self._preserve_recent = preserve_recent

    @property
    def max_tokens(self) -> int:
        """Total budget this manager was created with."""
        return self._max_tokens

    @property
    def preserve_recent(self) -> int:
        """Number of trailing messages that are never dropped."""
        return self._preserve_recent

    def available_tokens(
        self,
        system_prompt_tokens: int = 0,
        tool_definitions_tokens: int = 0,
    ) -> int:
        """Tokens left for conversation history after fixed request parts."""
        return max(
            0,
            self._max_tokens - system_prompt_tokens - tool_definitions_tokens,
        )

    def fit_to_window(
        self,
        messages: Sequence[Message],
        system_prompt_tokens: int = 0,
        tool_definitions_tokens: int = 0,
    ) -> list[Message]:
        """Return a history that fits within the context window.

        Guarantees:

        - the most recent ``preserve_recent`` messages are never removed;
        - an assistant tool call is never separated from its observations;
        - older messages are dropped from the beginning;
        - the returned list starts with a user message when one exists.

        The input is never mutated.
        """
        history = list(messages)
        if not history:
            return []

        budget = self.available_tokens(
            system_prompt_tokens,
            tool_definitions_tokens,
        )
        spans = _group_spans(history)
        preserved_start = max(0, len(history) - self._preserve_recent)

        selected: list[tuple[int, list[Message], bool]] = []
        used = 0
        for index in range(len(spans) - 1, -1, -1):
            group, _start, end = spans[index]
            cost = estimate_messages(group)
            protected = end > preserved_start
            if not protected and used + cost > budget:
                # Older groups cannot fit either once a group is dropped.
                break
            selected.append((index, group, protected))
            used += cost

        selected.reverse()

        # A provider requires a conversation to start with a user turn. Drop
        # leading assistant/tool groups that were kept only for their content;
        # never drop a protected group to satisfy the shape rule.
        while selected and selected[0][1][0].role != "user" and not selected[0][2]:
            selected.pop(0)

        if selected and selected[0][1][0].role != "user":
            selected = self._with_leading_user(selected, spans)

        result = [
            message for _index, group, _protected in selected for message in group
        ]
        total = estimate_messages(result)

        if total > budget:
            logger.warning(
                "Context history exceeds its budget (%d > %d estimated tokens); "
                "preserved %d mandatory messages.",
                total,
                budget,
                len(result),
            )

        return result

    @staticmethod
    def _with_leading_user(
        selected: list[tuple[int, list[Message], bool]],
        spans: Sequence[tuple[list[Message], int, int]],
    ) -> list[tuple[int, list[Message], bool]]:
        """Include the nearest earlier user turn so the history starts valid."""
        first_index = selected[0][0]
        for index in range(first_index - 1, -1, -1):
            group = spans[index][0]
            if group[0].role == "user":
                selected.insert(0, (index, group, False))
                break
        return selected


__all__ = [
    "CHARS_PER_TOKEN",
    "MESSAGE_OVERHEAD_TOKENS",
    "ContextManager",
    "estimate_message_tokens",
    "estimate_messages",
    "estimate_tokens",
]
