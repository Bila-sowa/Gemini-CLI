"""Code tools — public surface."""

from gemini_cli.tools.code.code_search import (
    CONTEXT_LINES,
    DEFAULT_CODE_EXTENSIONS,
    CodeBlock,
    CodeSearchParams,
    CodeSearchTool,
    render_blocks,
    scan_code,
)

__all__ = [
    "CONTEXT_LINES",
    "DEFAULT_CODE_EXTENSIONS",
    "CodeBlock",
    "CodeSearchParams",
    "CodeSearchTool",
    "render_blocks",
    "scan_code",
]
