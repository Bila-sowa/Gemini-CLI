"""Action classification and user-confirmation flow.

The approval manager is deliberately independent of the terminal UI: the UI
implements :class:`ApprovalCallback`, while the agent calls
:meth:`ApprovalManager.check` before dispatching a tool. The default policy is
fail-closed — destructive actions always ask, write actions ask unless
explicitly auto-approved, and an unknown risk level is treated as destructive.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Protocol

from gemini_cli.tools.base import RiskLevel

logger = logging.getLogger(__name__)

_SAFE_ARGUMENT_KEYS = frozenset(
    {
        "command",
        "create_directories",
        "end_line",
        "file_extensions",
        "include_pattern",
        "max_bytes",
        "max_results",
        "path",
        "pattern",
        "query",
        "root",
        "start_line",
        "timeout_seconds",
        "url",
        "working_directory",
    }
)
_CONTENT_ARGUMENT_KEYS = frozenset(
    {
        "content",
        "replacement_content",
        "target_content",
    }
)
_MAX_DETAIL_CHARS = 200


class ApprovalError(RuntimeError):
    """Base error for approval-flow failures."""


class ApprovalUnavailableError(ApprovalError):
    """Raised when approval is required but no callback is configured."""


class ApprovalDecision(StrEnum):
    """Result of an approval check."""

    APPROVED = "approved"
    DENIED = "denied"
    AUTO_APPROVED = "auto_approved"


class ApprovalCallback(Protocol):
    """UI-side contract for asking the user to confirm an action."""

    async def request_approval(
        self,
        action_description: str,
        risk_level: str,
        details: str,
    ) -> ApprovalDecision:
        """Return the user's decision for one action."""


class ApprovalManager:
    """Decide whether an action needs approval and coordinate the prompt."""

    def __init__(
        self,
        callback: ApprovalCallback | None = None,
        *,
        auto_approve_writes: bool = False,
    ) -> None:
        self._callback = callback
        self._auto_approve_writes = auto_approve_writes

    async def check(
        self,
        tool_name: str,
        risk_level: str | RiskLevel,
        arguments: Mapping[str, Any] | None = None,
    ) -> ApprovalDecision:
        """Classify the action and request approval when needed.

        Read-only tools are auto-approved. Write tools require confirmation by
        default. Destructive tools always require confirmation. An unknown risk
        value is treated as destructive so a malformed classification cannot
        silently bypass the prompt.
        """
        level = _normalize_risk(risk_level)

        if level is RiskLevel.READ:
            logger.debug("Auto-approved read-only tool %s", tool_name)
            return ApprovalDecision.AUTO_APPROVED

        if level is RiskLevel.WRITE and self._auto_approve_writes:
            logger.debug("Auto-approved write tool %s by policy", tool_name)
            return ApprovalDecision.AUTO_APPROVED

        if self._callback is None:
            msg = (
                f"Approval is required for tool '{tool_name}' "
                f"({level.value}), but no approval callback is configured."
            )
            raise ApprovalUnavailableError(msg)

        action = _action_description(tool_name, level)
        details = _describe_arguments(arguments or {})

        try:
            decision = await self._callback.request_approval(
                action,
                level.value,
                details,
            )
        except Exception as exc:
            # Do not include the exception text: a UI implementation may put
            # command arguments or other sensitive data into it.
            logger.warning(
                "Approval callback failed for tool %s (%s); denying the action.",
                tool_name,
                type(exc).__name__,
            )
            return ApprovalDecision.DENIED

        if not isinstance(decision, ApprovalDecision):
            logger.warning(
                "Approval callback returned an invalid decision for tool %s; "
                "denying the action.",
                tool_name,
            )
            return ApprovalDecision.DENIED

        logger.info(
            "Approval decision for tool %s (%s): %s",
            tool_name,
            level.value,
            decision.value,
        )
        return decision


def _normalize_risk(value: str | RiskLevel) -> RiskLevel:
    try:
        return RiskLevel(str(value).strip().lower())
    except ValueError:
        logger.warning(
            "Unknown risk level %r; treating it as destructive.",
            value,
        )
        return RiskLevel.DESTRUCTIVE


def _action_description(tool_name: str, level: RiskLevel) -> str:
    if level is RiskLevel.READ:
        return f"Read data with tool '{tool_name}'."
    if level is RiskLevel.WRITE:
        return f"Modify data with tool '{tool_name}'."
    return f"Run a destructive action with tool '{tool_name}'."


def _describe_arguments(arguments: Mapping[str, Any]) -> str:
    """Build a short, safe summary for the confirmation prompt.

    File contents are replaced by their length. Commands are shown because the
    user must know what they are approving; they are never written to logs.
    """
    if not arguments:
        return "No arguments."

    parts: list[str] = []
    for key in sorted(arguments):
        value = arguments[key]
        if key in _CONTENT_ARGUMENT_KEYS:
            parts.append(f"{key}: <{len(str(value))} characters>")
        elif key in _SAFE_ARGUMENT_KEYS:
            parts.append(f"{key}: {_short(value)}")
        else:
            parts.append(f"{key}: <provided>")
    return "; ".join(parts)


def _short(value: Any, limit: int = _MAX_DETAIL_CHARS) -> str:
    text = str(value).replace("\r", "\\r").replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


__all__ = [
    "ApprovalCallback",
    "ApprovalDecision",
    "ApprovalError",
    "ApprovalManager",
    "ApprovalUnavailableError",
]
