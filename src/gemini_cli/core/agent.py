"""Think-act-observe agent loop.

The agent is the only component that turns a user message into a sequence of
model turns, tool executions, and observations. It depends on small contracts
— a streaming chat client, the tool registry, the approval manager, the context
manager, and the quota tracker — so every collaborator can be replaced in
tests.

Safety rules enforced here:

- every tool call the model requests receives exactly one observation, even
  when the tool is unknown, denied, or fails, so the next provider request is
  valid;
- approval is checked before dispatch and the policy is fail-closed;
- the loop is bounded by ``max_agent_iterations``;
- tool arguments are never logged.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Sequence
from typing import Protocol

from gemini_cli.config import Settings
from gemini_cli.core.context_manager import ContextManager, estimate_tokens
from gemini_cli.core.quota_tracker import QuotaTracker
from gemini_cli.core.session import Session
from gemini_cli.core.streaming import (
    AgentEvent,
    AgentEventType,
    done_event,
    error_event,
    from_stream_event,
    tool_call_end,
)
from gemini_cli.llm import (
    LLMError,
    Message,
    StreamEvent,
    ToolCallRequest,
    ToolDefinition,
    UnknownModelError,
    UsageInfo,
)
from gemini_cli.sandbox.approval import (
    ApprovalDecision,
    ApprovalManager,
    ApprovalUnavailableError,
)
from gemini_cli.tools import ToolNotFoundError, ToolRegistry
from gemini_cli.tools.base import ToolResult

logger = logging.getLogger(__name__)

MAX_ERROR_CHARS = 2_000

_DENIED_MESSAGE = "The user denied this tool call."
_APPROVAL_UNAVAILABLE_MESSAGE = (
    "This tool call requires approval, but no approval handler is configured."
)


class ChatClient(Protocol):
    """Streaming LLM surface the agent depends on.

    ``GeminiClient`` satisfies this protocol structurally; tests provide a
    deterministic fake.
    """

    def generate_stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition] | None = None,
        system_instruction: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream one model response as domain events."""
        ...


def _safe_error(exc: BaseException) -> str:
    message = str(exc).strip() or type(exc).__name__
    if len(message) <= MAX_ERROR_CHARS:
        return message
    return f"{message[: MAX_ERROR_CHARS - 3]}..."


def _tool_definition_tokens(tools: Sequence[ToolDefinition]) -> int:
    """Estimate the token cost of the exported tool declarations."""
    total = 0
    for tool in tools:
        total += estimate_tokens(tool.name)
        total += estimate_tokens(tool.description)
        try:
            schema = json.dumps(
                tool.parameters_json_schema,
                ensure_ascii=False,
                sort_keys=True,
            )
        except (TypeError, ValueError):
            schema = str(tool.parameters_json_schema)
        total += estimate_tokens(schema)
    return total


class Agent:
    """Model-independent agent that drives conversation and tool use."""

    def __init__(
        self,
        llm_client: ChatClient,
        tool_registry: ToolRegistry,
        approval_manager: ApprovalManager,
        context_manager: ContextManager,
        quota_tracker: QuotaTracker,
        settings: Settings,
        system_prompt: str,
    ) -> None:
        self._llm_client = llm_client
        self._tool_registry = tool_registry
        self._approval_manager = approval_manager
        self._context_manager = context_manager
        self._quota_tracker = quota_tracker
        self._settings = settings
        self._system_prompt = system_prompt

    async def run(
        self,
        session: Session,
        user_message: str,
    ) -> AsyncIterator[AgentEvent]:
        """Process one user message and yield events until the turn is done."""
        if not user_message.strip():
            yield error_event("The user message must not be empty.")
            yield done_event()
            return

        if len(user_message) > self._settings.max_input_chars:
            yield error_event(
                "The user message exceeds the configured limit of "
                f"{self._settings.max_input_chars} characters."
            )
            yield done_event()
            return

        session.add_user_message(user_message)

        try:
            async for event in self._run_turns(session):
                yield event
        except (LLMError, UnknownModelError) as exc:
            yield error_event(_safe_error(exc))
            yield done_event()

    async def _run_turns(self, session: Session) -> AsyncIterator[AgentEvent]:
        tools = [
            ToolDefinition.from_export(entry)
            for entry in self._tool_registry.export_schemas()
        ]
        tool_tokens = _tool_definition_tokens(tools)
        system_tokens = estimate_tokens(self._system_prompt)

        while True:
            if session.iteration_count >= self._settings.max_agent_iterations:
                yield error_event(
                    "Reached the maximum number of agent iterations "
                    f"({self._settings.max_agent_iterations})."
                )
                yield done_event()
                return

            history = self._context_manager.fit_to_window(
                session.get_history(),
                system_prompt_tokens=system_tokens,
                tool_definitions_tokens=tool_tokens,
            )

            text_parts: list[str] = []
            calls: list[ToolCallRequest] = []
            usage = UsageInfo()
            failed = False

            async for stream_event in self._llm_client.generate_stream(
                history,
                tools or None,
                self._system_prompt,
            ):
                if stream_event.event_type == "usage":
                    if isinstance(stream_event.data, UsageInfo):
                        usage = stream_event.data
                    continue
                if stream_event.event_type == "done":
                    continue

                mapped = from_stream_event(stream_event)
                if mapped is None:
                    continue

                if mapped.event_type is AgentEventType.ERROR:
                    failed = True
                    yield mapped
                    continue

                if mapped.event_type is AgentEventType.TOOL_CALL_START:
                    if isinstance(mapped.data, ToolCallRequest):
                        calls.append(mapped.data)
                elif mapped.event_type is AgentEventType.TEXT_CHUNK:
                    text_parts.append(str(mapped.data))

                yield mapped

            if failed:
                yield done_event()
                return

            session.add_assistant_message("".join(text_parts), tuple(calls))
            session.update_usage(usage)
            if usage.input_tokens or usage.output_tokens:
                self._quota_tracker.record_usage(
                    self._settings.model,
                    usage.input_tokens,
                    usage.output_tokens,
                )
            session.record_iteration()

            if not calls:
                yield done_event()
                return

            for call in calls:
                async for event in self._handle_tool_call(session, call):
                    yield event

    async def _handle_tool_call(
        self,
        session: Session,
        call: ToolCallRequest,
    ) -> AsyncIterator[AgentEvent]:
        try:
            tool = self._tool_registry.get(call.name)
        except ToolNotFoundError as exc:
            result = ToolResult(success=False, output="", error=_safe_error(exc))
            session.add_tool_result(call.id, result)
            yield tool_call_end(call, result, status="error")
            return

        try:
            decision = await self._approval_manager.check(
                call.name,
                tool.risk_level,
                call.arguments,
            )
        except ApprovalUnavailableError:
            result = ToolResult(
                success=False,
                output="",
                error=_APPROVAL_UNAVAILABLE_MESSAGE,
            )
            session.add_tool_result(call.id, result)
            yield tool_call_end(call, result, status="denied")
            return

        if decision is ApprovalDecision.DENIED:
            logger.info("Tool call %s was denied by the user.", call.name)
            result = ToolResult(success=False, output="", error=_DENIED_MESSAGE)
            session.add_tool_result(call.id, result)
            yield tool_call_end(call, result, status="denied")
            return

        logger.debug("Executing tool %s", call.name)
        result = await self._tool_registry.dispatch(call.name, call.arguments)
        session.add_tool_result(call.id, result)
        yield tool_call_end(
            call,
            result,
            status="success" if result.success else "error",
        )


__all__ = ["Agent", "ChatClient"]
