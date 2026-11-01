"""Tests for the fetch_url web tool."""

from __future__ import annotations

import ipaddress
from collections.abc import Callable

import httpx
import pytest
from pydantic import BaseModel, ValidationError

from gemini_cli.tools.base import RiskLevel, ToolCategory
from gemini_cli.tools.web import (
    MAX_RESPONSE_BYTES,
    FetchUrlParams,
    FetchUrlTool,
)
from gemini_cli.tools.web import fetch_url as fetch_url_module

PUBLIC_IP = "93.184.216.34"  # https://example.com

Handler = Callable[[httpx.Request], httpx.Response]


class UnexpectedParams(BaseModel):
    """Parameter model of a different tool."""

    other: str = ""


def _respond(status: int = 200, **kwargs: object) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, request=request, **kwargs)  # type: ignore[arg-type]

    return handler


def _tool(handler: Handler | None = None) -> FetchUrlTool:
    def no_network(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected network request to {request.url}")

    return FetchUrlTool(transport=httpx.MockTransport(handler or no_network))


@pytest.fixture
def public_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve every hostname to a public address."""
    monkeypatch.setattr(
        fetch_url_module,
        "_resolve_addresses",
        lambda host, port: [ipaddress.ip_address(PUBLIC_IP)],
    )


def _resolver(monkeypatch: pytest.MonkeyPatch, mapping: dict[str, str]) -> None:
    def resolve(
        host: str, port: int
    ) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
        return [ipaddress.ip_address(mapping.get(host, PUBLIC_IP))]

    monkeypatch.setattr(fetch_url_module, "_resolve_addresses", resolve)


async def test_fetches_text(public_resolver: None) -> None:
    tool = _tool(_respond(text="hello", headers={"content-type": "text/plain"}))

    result = await tool.execute(FetchUrlParams(url="https://example.com/page"))

    assert result.success is True
    assert result.output == "hello"
    assert result.metadata["status_code"] == 200
    assert result.metadata["content_type"] == "text/plain"
    assert result.metadata["truncated"] is False
    assert result.metadata["bytes_read"] == 5


async def test_accepts_json_content_type(public_resolver: None) -> None:
    tool = _tool(
        _respond(
            content=b'{"ok": true}',
            headers={"content-type": "application/json; charset=utf-8"},
        )
    )

    result = await tool.execute(FetchUrlParams(url="https://example.com/api"))

    assert result.success is True
    assert result.metadata["content_type"] == "application/json"


async def test_allows_missing_content_type(public_resolver: None) -> None:
    tool = _tool(_respond(content=b"plain"))

    result = await tool.execute(FetchUrlParams(url="https://example.com/file"))

    assert result.success is True
    assert result.output == "plain"


@pytest.mark.parametrize(
    "url",
    ["file:///etc/passwd", "ftp://example.com/file", "data:text/plain,hello"],
)
async def test_rejects_unsupported_schemes(url: str) -> None:
    result = await _tool().execute(FetchUrlParams(url=url))

    assert result.success is False
    assert "scheme" in (result.error or "")


async def test_rejects_credentials_in_url() -> None:
    result = await _tool().execute(
        FetchUrlParams(url="https://user:secret@example.com/")
    )

    assert result.success is False
    assert "credentials" in (result.error or "")
    assert "secret" not in (result.error or "")


async def test_rejects_url_without_host() -> None:
    result = await _tool().execute(FetchUrlParams(url="http://"))

    assert result.success is False


async def test_rejects_malformed_url() -> None:
    result = await _tool().execute(FetchUrlParams(url="not a url"))

    assert result.success is False


@pytest.mark.parametrize("url", ["http://localhost/", "http://app.localhost/"])
async def test_rejects_localhost_names(url: str) -> None:
    result = await _tool().execute(FetchUrlParams(url=url))

    assert result.success is False
    assert "not allowed" in (result.error or "")


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://10.0.0.1/",
        "http://192.168.1.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://100.64.0.1/",
        "http://[::1]/",
    ],
)
async def test_rejects_private_literal_addresses(url: str) -> None:
    result = await _tool().execute(FetchUrlParams(url=url))

    assert result.success is False
    assert "non-public address" in (result.error or "")


async def test_rejects_hostname_resolving_to_private_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _resolver(monkeypatch, {"internal.example": "10.1.2.3"})

    result = await _tool().execute(FetchUrlParams(url="https://internal.example/"))

    assert result.success is False
    assert "10.1.2.3" in (result.error or "")


async def test_rejects_when_any_resolved_address_is_private(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        fetch_url_module,
        "_resolve_addresses",
        lambda host, port: [
            ipaddress.ip_address(PUBLIC_IP),
            ipaddress.ip_address("127.0.0.1"),
        ],
    )

    result = await _tool().execute(FetchUrlParams(url="https://example.com/"))

    assert result.success is False
    assert "non-public address" in (result.error or "")


async def test_does_not_resolve_ip_literals(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(host: str, port: int) -> list[object]:
        raise AssertionError("resolver must not be called for IP literals")

    monkeypatch.setattr(fetch_url_module, "_resolve_addresses", explode)
    tool = _tool(_respond(text="ok", headers={"content-type": "text/plain"}))

    result = await tool.execute(FetchUrlParams(url="http://93.184.216.34/"))

    assert result.success is True


async def test_truncates_oversized_response(public_resolver: None) -> None:
    tool = _tool(_respond(text="x" * 100, headers={"content-type": "text/plain"}))

    result = await tool.execute(
        FetchUrlParams(url="https://example.com/big", max_bytes=10)
    )

    assert result.success is True
    assert result.metadata["truncated"] is True
    assert result.metadata["bytes_read"] == 10
    assert result.output.startswith("x" * 10)
    assert "truncated at 10 bytes" in result.output


async def test_rejects_binary_content_type(public_resolver: None) -> None:
    tool = _tool(_respond(content=b"\x89PNG", headers={"content-type": "image/png"}))

    result = await tool.execute(FetchUrlParams(url="https://example.com/image"))

    assert result.success is False
    assert "Unsupported content type" in (result.error or "")


async def test_reports_http_error(public_resolver: None) -> None:
    tool = _tool(_respond(status=404))

    result = await tool.execute(FetchUrlParams(url="https://example.com/missing"))

    assert result.success is False
    assert "HTTP 404" in (result.error or "")


async def test_follows_public_redirect(public_resolver: None) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(
                302,
                headers={"location": "https://example.com/final"},
                request=request,
            )
        return httpx.Response(
            200,
            text="landed",
            headers={"content-type": "text/plain"},
            request=request,
        )

    result = await _tool(handler).execute(
        FetchUrlParams(url="https://example.com/start")
    )

    assert result.success is True
    assert result.output == "landed"
    assert result.metadata["url"] == "https://example.com/final"


async def test_blocks_redirect_to_private_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _resolver(monkeypatch, {"internal.example": "127.0.0.1"})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302,
            headers={"location": "http://internal.example/secret"},
            request=request,
        )

    result = await _tool(handler).execute(
        FetchUrlParams(url="https://example.com/start")
    )

    assert result.success is False
    assert "non-public address" in (result.error or "")


async def test_blocks_redirect_without_location(public_resolver: None) -> None:
    tool = _tool(_respond(status=302))

    result = await tool.execute(FetchUrlParams(url="https://example.com/start"))

    assert result.success is False
    assert "Location" in (result.error or "")


async def test_rejects_too_many_redirects(public_resolver: None) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302,
            headers={"location": "https://example.com/loop"},
            request=request,
        )

    result = await _tool(handler).execute(
        FetchUrlParams(url="https://example.com/loop")
    )

    assert result.success is False
    assert "Too many redirects" in (result.error or "")


async def test_reports_timeout(public_resolver: None) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    result = await _tool(handler).execute(FetchUrlParams(url="https://example.com/"))

    assert result.success is False
    assert "timed out" in (result.error or "")


async def test_reports_transport_error(public_resolver: None) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    result = await _tool(handler).execute(FetchUrlParams(url="https://example.com/"))

    assert result.success is False
    assert "failed" in (result.error or "")


async def test_rejects_unexpected_parameter_type() -> None:
    result = await _tool().execute(UnexpectedParams())

    assert result.success is False
    assert "Unexpected parameter type" in (result.error or "")


@pytest.mark.parametrize("max_bytes", [0, -1, MAX_RESPONSE_BYTES + 1])
def test_rejects_out_of_range_max_bytes(max_bytes: int) -> None:
    with pytest.raises(ValidationError):
        FetchUrlParams(url="https://example.com/", max_bytes=max_bytes)


def test_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        FetchUrlParams(url="https://example.com/", timeout=5)  # type: ignore[call-arg]


def test_tool_metadata() -> None:
    tool = FetchUrlTool()

    assert tool.name == "fetch_url"
    assert tool.description
    assert tool.parameters_schema is FetchUrlParams
    assert tool.category is ToolCategory.WEB
    assert tool.risk_level is RiskLevel.READ
