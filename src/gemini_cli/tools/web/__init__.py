"""Web tools — public surface."""

from gemini_cli.tools.web.fetch_url import (
    DEFAULT_FETCH_TIMEOUT_SECONDS,
    DEFAULT_MAX_BYTES,
    MAX_RESPONSE_BYTES,
    FetchUrlParams,
    FetchUrlTool,
)

__all__ = [
    "DEFAULT_FETCH_TIMEOUT_SECONDS",
    "DEFAULT_MAX_BYTES",
    "MAX_RESPONSE_BYTES",
    "FetchUrlParams",
    "FetchUrlTool",
]
