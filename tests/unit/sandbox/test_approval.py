"""Tests for the approval manager."""

from __future__ import annotations

import logging

import pytest

from gemini_cli.sandbox.approval import (
    ApprovalDecision,
    ApprovalManager,
    ApprovalUnavailableError,
)
from gemini_cli.tools.base import RiskLevel


class RecordingCallback:
    """Approval callback that records calls and returns a fixed decision."""

    def __init__(self, decision: ApprovalDecision) -> None:
        self.decision = decision
        self.calls: list[tuple[str, str, str]] = []

    async def request_approval(
        self,
        action_description: str,
        risk_level: str,
        details: str,
    ) -> ApprovalDecision:
        self.calls.append((action_description, risk_level, details))
        return self.decision


class ExplodingCallback:
    async def request_approval(
        self,
        action_description: str,
        risk_level: str,
        details: str,
    ) -> ApprovalDecision:
        del action_description, risk_level, details
        raise RuntimeError("secret command text")


class InvalidCallback:
    async def request_approval(
        self,
        action_description: str,
        risk_level: str,
        details: str,
    ) -> ApprovalDecision:
        del action_description, risk_level, details
        return "approved"  # type: ignore[return-value]


async def test_read_is_auto_approved_without_callback() -> None:
    manager = ApprovalManager()

    decision = await manager.check("read_file", RiskLevel.READ, {"path": "a.txt"})

    assert decision is ApprovalDecision.AUTO_APPROVED


async def test_write_requires_approval_by_default() -> None:
    manager = ApprovalManager()

    with pytest.raises(ApprovalUnavailableError, match="write_file"):
        await manager.check("write_file", RiskLevel.WRITE, {"path": "a.txt"})


async def test_write_can_be_auto_approved_by_policy() -> None:
    manager = ApprovalManager(auto_approve_writes=True)

    decision = await manager.check("write_file", RiskLevel.WRITE, {"path": "a.txt"})

    assert decision is ApprovalDecision.AUTO_APPROVED


async def test_destructive_action_requests_approval() -> None:
    callback = RecordingCallback(ApprovalDecision.APPROVED)
    manager = ApprovalManager(callback)

    decision = await manager.check(
        "run_command",
        RiskLevel.DESTRUCTIVE,
        {"command": "echo hello", "working_directory": "/tmp"},
    )

    assert decision is ApprovalDecision.APPROVED
    assert len(callback.calls) == 1
    action, risk, details = callback.calls[0]
    assert "run_command" in action
    assert risk == "destructive"
    assert "echo hello" in details
    assert "working_directory: /tmp" in details


async def test_denied_callback_returns_denied() -> None:
    callback = RecordingCallback(ApprovalDecision.DENIED)
    manager = ApprovalManager(callback)

    decision = await manager.check("run_command", RiskLevel.DESTRUCTIVE, {})

    assert decision is ApprovalDecision.DENIED


async def test_destructive_action_without_callback_raises() -> None:
    manager = ApprovalManager()

    with pytest.raises(ApprovalUnavailableError, match="run_command"):
        await manager.check("run_command", RiskLevel.DESTRUCTIVE, {})


async def test_unknown_risk_is_treated_as_destructive() -> None:
    callback = RecordingCallback(ApprovalDecision.DENIED)
    manager = ApprovalManager(callback)

    decision = await manager.check("mystery_tool", "unknown", {})

    assert decision is ApprovalDecision.DENIED
    assert callback.calls[0][1] == "destructive"


async def test_callback_exception_is_denied(
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = ApprovalManager(ExplodingCallback())

    with caplog.at_level(logging.WARNING):
        decision = await manager.check("run_command", RiskLevel.DESTRUCTIVE, {})

    assert decision is ApprovalDecision.DENIED
    assert "secret command text" not in caplog.text


async def test_invalid_callback_decision_is_denied() -> None:
    manager = ApprovalManager(InvalidCallback())

    decision = await manager.check("run_command", RiskLevel.DESTRUCTIVE, {})

    assert decision is ApprovalDecision.DENIED


async def test_file_contents_are_not_shown_in_details() -> None:
    callback = RecordingCallback(ApprovalDecision.APPROVED)
    manager = ApprovalManager(callback)

    await manager.check(
        "write_file",
        RiskLevel.WRITE,
        {"path": "a.txt", "content": "super-secret-content"},
    )

    details = callback.calls[0][2]
    assert "super-secret-content" not in details
    assert "characters" in details


async def test_arguments_are_not_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    callback = RecordingCallback(ApprovalDecision.DENIED)
    manager = ApprovalManager(callback)

    with caplog.at_level(logging.DEBUG):
        await manager.check(
            "run_command",
            RiskLevel.DESTRUCTIVE,
            {"command": "echo SUPER_SECRET_VALUE"},
        )

    assert "SUPER_SECRET_VALUE" not in caplog.text


async def test_unknown_argument_keys_are_not_echoed() -> None:
    callback = RecordingCallback(ApprovalDecision.APPROVED)
    manager = ApprovalManager(callback)

    await manager.check(
        "mystery_tool",
        RiskLevel.WRITE,
        {"unexpected": "do-not-echo"},
    )

    details = callback.calls[0][2]
    assert "do-not-echo" not in details
    assert "unexpected: <provided>" in details
