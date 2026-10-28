"""Safe built-in default values for all configuration options."""

from __future__ import annotations

# Model defaults
DEFAULT_MODEL: str = "gemini-2.5-flash"
DEFAULT_REQUEST_TIMEOUT_MS: int = 60_000
DEFAULT_RETRY_ATTEMPTS: int = 3

# Conversation limits
DEFAULT_MAX_HISTORY_TURNS: int = 20
DEFAULT_MAX_INPUT_CHARS: int = 32_000
DEFAULT_MAX_AGENT_ITERATIONS: int = 25
DEFAULT_MAX_OUTPUT_TOKENS: int = 8_192

# Filesystem safety
DEFAULT_ALLOWED_EXTENSIONS: tuple[str, ...] = (
    ".py",
    ".md",
    ".txt",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
    ".cfg",
    ".ini",
    ".sh",
    ".bash",
    ".html",
    ".css",
    ".js",
    ".ts",
    ".sql",
    ".csv",
    ".xml",
    ".env.example",
)
DEFAULT_MAX_FILE_SIZE_BYTES: int = 1_048_576  # 1 MB

# Shell execution
DEFAULT_SHELL_TIMEOUT_SECONDS: int = 30
DEFAULT_SHELL_MAX_OUTPUT_BYTES: int = 524_288  # 512 KB

# Project context
PROJECT_INSTRUCTION_FILES: tuple[str, ...] = ("GEMINI.md", "AGENTS.md", "AGENT.md")

# Telemetry
DEFAULT_TELEMETRY_ENABLED: bool = False

# Roles
DEFAULT_ROLE: str = "basic"
AVAILABLE_ROLES: tuple[str, ...] = (
    "basic",
    "code",
    "creator",
    "analyser",
    "poet",
)

# Config file paths
GLOBAL_CONFIG_DIR_NAME: str = ".gemini-cli"
GLOBAL_CONFIG_FILE_NAME: str = "config.toml"
PROJECT_CONFIG_FILE_NAME: str = ".gemini-cli.toml"
