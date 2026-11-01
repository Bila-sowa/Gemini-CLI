"""Tool that fetches remote text resources over HTTP(S).

Requests come from an untrusted model, so the tool treats URLs as hostile
input: only ``http``/``https`` is allowed, credentials in the URL are rejected,
and every host must resolve exclusively to public addresses. Redirects are
followed manually so that a public site cannot bounce the agent into a private
network.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from gemini_cli import __version__
from gemini_cli.tools.base import BaseTool, RiskLevel, ToolCategory, ToolResult

DEFAULT_MAX_BYTES = 524_288
MAX_RESPONSE_BYTES = 5_242_880
DEFAULT_FETCH_TIMEOUT_SECONDS = 15.0
MAX_REDIRECTS = 3

ALLOWED_SCHEMES = frozenset({"http", "https"})
BLOCKED_HOSTNAMES = frozenset({"localhost"})
# Shared address space (RFC 6598) is not public even where ``is_global``
# classification differs between Python versions.
BLOCKED_NETWORKS = (ipaddress.ip_network("100.64.0.0/10"),)
TEXT_MEDIA_TYPES = frozenset(
    {
        "application/json",
        "application/xml",
        "application/xhtml+xml",
        "application/javascript",
    }
)
USER_AGENT = f"gemini-cli-py/{__version__}"

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address


def _failure(message: str) -> ToolResult:
    return ToolResult(success=False, output="", error=message)


def _is_blocked_address(address: IPAddress) -> bool:
    """Return whether *address* must not be contacted."""
    mapped = address.ipv4_mapped if isinstance(address, ipaddress.IPv6Address) else None
    if mapped is not None:
        address = mapped
    if not address.is_global:
        return True
    return any(address in network for network in BLOCKED_NETWORKS)


def _resolve_addresses(host: str, port: int) -> list[IPAddress]:
    """Resolve *host* to every address the platform reports."""
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    addresses: list[IPAddress] = []
    for info in infos:
        raw = info[4][0]
        try:
            addresses.append(ipaddress.ip_address(raw))
        except ValueError:
            continue
    return addresses


def _normalized_host(host: str) -> str:
    return host.strip().lower().rstrip(".")


def _literal_address(host: str) -> IPAddress | None:
    """Return *host* as an address when it is an IP literal."""
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return None


def _validate_url(url: httpx.URL) -> str | None:
    """Return an error message when the URL must not be requested."""
    if url.scheme not in ALLOWED_SCHEMES:
        return (
            f"Unsupported URL scheme '{url.scheme}'. Only http and https are allowed."
        )
    if not url.host:
        return "URL must include a host."
    if url.userinfo:
        return "URLs with embedded credentials are not allowed."

    host = _normalized_host(url.host)
    if host in BLOCKED_HOSTNAMES or host.endswith(".localhost"):
        return f"Host '{host}' is not allowed."
    return None


async def _validate_host(host: str, port: int) -> str | None:
    """Return an error message when *host* resolves to a blocked address."""
    literal = _literal_address(host)
    if literal is not None:
        if _is_blocked_address(literal):
            return (
                f"Host '{host}' is the non-public address {literal}, "
                "which is not allowed."
            )
        return None

    try:
        addresses = await asyncio.to_thread(_resolve_addresses, host, port)
    except OSError as exc:
        return f"Cannot resolve host '{host}': {exc}"

    if not addresses:
        return f"Cannot resolve host '{host}'."

    for address in addresses:
        if _is_blocked_address(address):
            return (
                f"Host '{host}' resolves to the non-public address {address}, "
                "which is not allowed."
            )
    return None


def _is_text_media_type(media_type: str) -> bool:
    return (
        media_type.startswith("text/")
        or media_type in TEXT_MEDIA_TYPES
        or media_type.endswith("+json")
        or media_type.endswith("+xml")
    )


def _default_port(url: httpx.URL) -> int:
    if url.port is not None:
        return url.port
    return 443 if url.scheme == "https" else 80


class FetchUrlParams(BaseModel):
    """Parameters for :class:`FetchUrlTool`."""

    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=1, description="Absolute http or https URL.")
    max_bytes: int = Field(
        default=DEFAULT_MAX_BYTES,
        ge=1,
        le=MAX_RESPONSE_BYTES,
        description="Maximum number of response bytes to return.",
    )


class FetchUrlTool(BaseTool):
    """Fetch a public text resource over HTTP(S)."""

    def __init__(self, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._transport = transport

    @property
    def name(self) -> str:
        return "fetch_url"

    @property
    def description(self) -> str:
        return (
            "Fetch a public http or https URL and return its text content. "
            "Private, loopback, link-local, and internal hosts are rejected, "
            "responses are capped at max_bytes, and only text responses are "
            "returned."
        )

    @property
    def parameters_schema(self) -> type[BaseModel]:
        return FetchUrlParams

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.WEB

    @property
    def risk_level(self) -> RiskLevel:
        return RiskLevel.READ

    async def execute(self, params: BaseModel) -> ToolResult:
        if not isinstance(params, FetchUrlParams):
            return _failure("Unexpected parameter type for fetch_url.")

        try:
            request_url = httpx.URL(params.url)
        except (httpx.InvalidURL, ValueError) as exc:
            return _failure(f"Invalid URL: {exc}")

        async with httpx.AsyncClient(
            timeout=DEFAULT_FETCH_TIMEOUT_SECONDS,
            follow_redirects=False,
            trust_env=False,
            transport=self._transport,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            for _ in range(MAX_REDIRECTS + 1):
                problem = _validate_url(request_url)
                if problem is not None:
                    return _failure(problem)

                host = _normalized_host(request_url.host)
                host_problem = await _validate_host(host, _default_port(request_url))
                if host_problem is not None:
                    return _failure(host_problem)

                try:
                    async with client.stream("GET", request_url) as response:
                        if response.is_redirect:
                            location = response.headers.get("location")
                            if not location:
                                return _failure(
                                    "Redirect response is missing a Location header."
                                )
                            request_url = request_url.join(location)
                            continue
                        return await self._read_response(response, params.max_bytes)
                except httpx.TimeoutException:
                    return _failure(f"Request to {request_url} timed out.")
                except httpx.HTTPError as exc:
                    return _failure(f"Request to {request_url} failed: {exc}")

        return _failure(f"Too many redirects (limit {MAX_REDIRECTS}).")

    @staticmethod
    async def _read_response(response: httpx.Response, max_bytes: int) -> ToolResult:
        if response.status_code >= 400:
            return _failure(
                f"Request failed with HTTP {response.status_code} "
                f"{response.reason_phrase}."
            )

        content_type = response.headers.get("content-type", "")
        media_type = content_type.split(";", 1)[0].strip().lower()
        if media_type and not _is_text_media_type(media_type):
            return _failure(
                f"Unsupported content type '{media_type}'; only text responses "
                "are fetched."
            )

        buffer = bytearray()
        truncated = False
        async for chunk in response.aiter_bytes():
            remaining = max_bytes - len(buffer)
            if remaining <= 0:
                truncated = True
                break
            if len(chunk) > remaining:
                buffer.extend(chunk[:remaining])
                truncated = True
                break
            buffer.extend(chunk)

        text = bytes(buffer).decode("utf-8", errors="replace")
        if truncated:
            text = f"{text}\n\n[response truncated at {max_bytes} bytes]"

        metadata: dict[str, Any] = {
            "url": str(response.url),
            "status_code": response.status_code,
            "content_type": media_type or "text/plain",
            "truncated": truncated,
            "bytes_read": len(buffer),
        }
        return ToolResult(success=True, output=text, metadata=metadata)


__all__ = [
    "DEFAULT_FETCH_TIMEOUT_SECONDS",
    "DEFAULT_MAX_BYTES",
    "MAX_RESPONSE_BYTES",
    "FetchUrlParams",
    "FetchUrlTool",
]
