# Development

This guide records the local development workflow for the `gemini-cli-py`
package: environment setup, running the CLI, and the checks that must pass
before a change is committed.

## Prerequisites

- Python 3.11–3.13 (3.12 is the recommended development version)
- [uv](https://docs.astral.sh/uv/) for dependency management and command
  execution

The project uses a `src` layout. Run Python commands through `uv run` so the
package is resolved from the local environment instead of the repository root.

## Setup

Sync the environment and install the development dependencies:

```bash
uv sync --all-extras
```

Optionally, copy the environment template and fill in a local API key:

```bash
cp .env.example .env
```

`.env` is ignored by git and stays on your machine; `.env.example` is the
committed template. Never put a real key into `.env.example`.

## Project instructions

Optionally, copy the project-instruction template to add repository-specific
guidance:

```bash
cp GEMINI.md.example GEMINI.md
```

`GEMINI.md` is committed with the repository and loaded automatically on
startup, whereas the `.example` file is never loaded on its own. Project
instructions are treated as untrusted input: they can guide behavior but
cannot override security controls, credential handling, or approval
requirements.

## Run the CLI

```bash
uv run gemini-cli-py --version
uv run python -m gemini_cli --version
uv run python -m gemini_cli --help
```

Both entry points should print the current package version prefixed with `v`
(see `__version__` in `src/gemini_cli/__init__.py`).

## Quality gates

Run these checks before committing:

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy
uv run pytest
```

## Continuous integration

GitHub Actions runs the same checks in `.github/workflows/ci.yml` for every
branch push (tag pushes are excluded) and for pull requests targeting `main`.
The `quality` job runs once per supported Python version (3.11, 3.12, 3.13):
it installs the locked dependency set, then runs Ruff lint, the formatting
check, mypy, and pytest with coverage.

Reproduce a CI run locally with the commands from
[Quality gates](#quality-gates). A failed lint, format, type, or test step
blocks the workflow; coverage is reported but has no threshold yet.

## Package import check

Every module in the package must be importable without credentials or
configuration. This is enforced by `tests/unit/test_imports.py`:

```bash
uv run pytest tests/unit/test_imports.py
```

The test parametrizes over every file under `src/gemini_cli/`, so a broken
module is reported by name.

## Logging and telemetry

`gemini_cli` uses structured logging and optional client-side telemetry:

- **Console logs:** Human-readable Rich output sent to `stderr`.
- **Application logs:** Saved as JSON Lines to `~/.gemini-cli/logs/app.jsonl`.
- **Telemetry logs:** Saved as JSON Lines to `~/.gemini-cli/logs/telemetry.jsonl` only when enabled (`GEMINI_TELEMETRY_ENABLED=true`). When telemetry is disabled (the default), no telemetry handler is attached, the telemetry logger is disabled, and events are discarded.
- **Secret redaction:** The `RedactingFilter` automatically redacts API keys, passwords, tokens, and `AIza...` strings from log records before they are written to console or file outputs. Both `key=value` and `key:value` forms are covered, including prefixed names such as `GEMINI_API_KEY`.
- **Telemetry privacy:** Telemetry events (`ModelCallEvent`, `ToolCallEvent`) record non-sensitive operational metrics (durations, token counts, statuses) and contain no prompt text or tool arguments.

## LLM client

`gemini_cli.llm` wraps `google-genai` behind domain types; SDK types never
leave the package. Runtime behavior is driven by `Settings`:

- `GEMINI_MODEL` selects a registered model (`gemini-2.5-flash`,
  `gemini-2.5-pro`); unknown names raise `UnknownModelError`.
- `GEMINI_RETRY_ATTEMPTS` sets the total number of attempts (minimum 1) for
  transient statuses `429`, `500`, and `503`. Client errors are never retried.
- `GEMINI_REQUEST_TIMEOUT_MS` bounds each attempt.
- `GEMINI_MAX_OUTPUT_TOKENS` is passed to the model.

`GeminiClient.generate()` returns a `ModelResponse` and, after retries are
exhausted on `404`, `429`, `500`, or `503`, retries once with the model's
registry fallback (`gemini-2.5-pro` → `gemini-2.5-flash`).
`GeminiClient.generate_stream()` yields `text_delta`, `tool_call_delta`,
`usage`, `done`, or `error` events; it is not retried, and tool calls are
emitted only after their arguments are complete. System prompts come from
`llm/prompts/templates/`; project instructions are appended inside an
untrusted, non-spoofable block.

## Tools

`gemini_cli.tools` defines the shared tool contract, the registry, and the
filesystem tools:

- `BaseTool` requires a unique `name`, a model-facing `description`, a Pydantic
  `parameters_schema`, a `ToolCategory`, a `RiskLevel`, and an async
  `execute(params) -> ToolResult`.
- `ToolResult` carries `success`, `output`, optional `error`, and safe
  structured `metadata`. Expected failures are returned as a failed result
  instead of being raised.
- `RiskLevel` (`read`, `write`, `destructive`) is the classification the
  approval layer will consume; `ToolCategory` groups capabilities for
  discovery and the UI.
- `ToolRegistry` rejects duplicate or malformed tool names, exports
  model-facing JSON Schema declarations via `export_schemas()`, and runs calls
  through `dispatch()`. Dispatch validates arguments against the tool's schema
  and converts unknown tools, invalid arguments, timeouts, and unexpected
  exceptions into structured failures, so a model-generated call cannot crash
  the agent loop. Validation errors never echo the rejected input values.
- Dispatch records a `ToolCallEvent` (tool name, duration, status) when
  telemetry is enabled; arguments and outputs are never recorded. Approval
  checks are not part of the registry and will be added by the agent runtime.

`ToolResult.metadata` must not contain secrets, prompts, or raw tool arguments.

### Filesystem tools

The `filesystem/` package provides `read_file`, `write_file`, `edit_file`,
`find_files`, and `grep`. Each tool receives the shared `Settings` instance and
uses `validate_path()` from `filesystem/path_validator.py`:

- Paths are resolved before the containment check, so `..` traversal and
  symlinks pointing outside the roots are rejected.
- `Settings.allowed_roots` defaults to an empty list, which disables filesystem
  access entirely. The CLI layer is responsible for adding the working
  directory.
- Files larger than `Settings.max_file_size_bytes` and binaries (detected by a
  NUL byte in the first 8 KB) are rejected. Content must be valid UTF-8.
- `write_file` and `edit_file` write atomically through a temporary file and
  `os.replace`, so a failure never leaves a partial file.
- `read_file` returns lines as `"     1 | text"`. `edit_file` requires
  `target_content` to occur exactly once in the file and inside the requested
  `start_line`..`end_line` range.
- `find_files` and `grep` skip common generated and version-control directories
  (`.git`, `.venv`, `node_modules`, `__pycache__`, and similar), reject glob
  patterns that escape the root, and stop after bounded work. `grep` also caps
  line length and total search time to limit catastrophic regular expressions.

### Shell tool

`run_command` is registered with `RiskLevel.DESTRUCTIVE` and
`ToolCategory.SHELL`, but does not execute commands yet: the sandbox backend
arrives with Phase 6. Until then it returns a structured failure
(`Shell execution is unavailable: no sandbox backend is configured.`) so the
agent loop keeps running and the model receives a clear reason.

The tool still validates its inputs:

- `working_directory` must resolve inside `Settings.allowed_roots` and be an
  existing directory; an empty `allowed_roots` rejects any explicit directory.
- `command` must not be blank and `timeout_seconds` must be positive.
- The command string is never repeated in the result, metadata, or logs,
  because it can contain secrets.

When the sandbox backend is added, the tool will delegate execution to it and
return a real `ExecutionResult` instead of the unavailable failure.

### Web tool

`fetch_url` (risk `READ`, category `WEB`) fetches public text resources over
HTTP(S) with `httpx`. It has no settings dependency and accepts an optional
`httpx.AsyncBaseTransport` for tests.

- Only `http` and `https` are allowed; `file://`, `data:`, and other schemes
  are rejected before any request. URLs containing credentials are rejected.
- The host must resolve exclusively to public addresses. Loopback, private,
  link-local, reserved, multicast, and shared (`100.64.0.0/10`) addresses are
  rejected, including `169.254.169.254`. IP literals are checked directly
  without a DNS lookup. `localhost` and `*.localhost` are rejected by name.
- Redirects are followed manually (maximum 3), and every hop is re-validated,
  so a public site cannot redirect the agent into an internal network.
- Responses larger than `max_bytes` (default 512 KB, maximum 5 MB) are
  truncated, flagged in `metadata.truncated`, and marked in the returned text.
- Only text, JSON, XML, and XHTML responses are returned; other content types
  are rejected. HTTP errors become structured failures.
- System proxy configuration is ignored (`trust_env=False`) so the address
  checks cannot be bypassed by an environment variable.

DNS rebinding remains a residual risk: the hostname is resolved for
validation and again by the HTTP client. Closing that gap fully would require
pinning the validated address in the TLS connection.

### Code tool

`code_search` (risk `READ`, category `CODE`) searches source files with a
Python regular expression and returns each match with its surrounding lines.

- Only the first configured allowed root is searched. Without allowed roots the
  tool refuses to run.
- By default it looks at common source extensions (`.py`, `.js`, `.ts`, `.go`,
  `.rs`, `.java`, `.c`, `.cpp`, `.cs`, and similar). `file_extensions` replaces
  that set; values are normalized, so `py`, `.PY`, and `.py` all mean `.py`.
- Each result is rendered as `path:line` followed by the match line marked with
  `>` and up to two context lines above and below.
- Matches whose context would overlap share a single block instead of repeating
  lines.
- The same safety bounds as `grep` apply: skipped generated directories, ignored
  binary and oversized files, a cap on scanned files, a scan deadline, and a
  per-line length limit. `max_results` counts blocks, not raw matches.

### Acceptance suite

`tests/unit/tools/test_builtin_tools.py` registers all eight built-in tools
together and exercises the registry as the agent will: unique Gemini-compatible
names, flat JSON Schema declarations for every tool, and the security guards
(`/etc/passwd`, `../../etc/passwd`, `file:///`, and the unavailable shell tool)
checked through `dispatch()`.

## Current status

| Area | Status |
|---|---|
| Package install / `uv sync` | Available |
| CLI `--version` and `--help` | Available |
| Ruff / mypy | Available |
| Python version (`.python-version`) | Available (3.12) |
| Test suite | Available (446 tests: smoke, import, entry-point, configuration, authentication, logging/telemetry, LLM client, tool system, all built-in tools, and the Phase 5 acceptance suite) |
| Configuration (`Settings`, project context) | Available |
| Authentication (API key resolution, keyring store, OAuth stub) | Available |
| Logging and Telemetry (`configure_logging`, secret redaction) | Available |
| LLM client (`generate`, `generate_stream`, retry, fallback, prompts) | Available |
| Tool base contract (`BaseTool`, `ToolResult`, risk levels) | Available |
| Tool registry (`register`, `export_schemas`, `dispatch`, telemetry) | Available |
| Filesystem tools (`read_file`, `write_file`, `edit_file`, `find_files`, `grep`) | Available |
| Shell tool (`run_command`) | Registered, execution unavailable until the sandbox backend |
| Web tool (`fetch_url`) | Available (public HTTP(S), SSRF guards, size limit) |
| Code tool (`code_search`) | Available (regex search with context) |
| Environment template (`.env.example`) | Available |
| Project instructions template (`GEMINI.md.example`) | Available |
| CI pipeline (`.github/workflows/ci.yml`) | Available |
| Phase 0 acceptance | Passed |
| Phase 1 acceptance | Passed |
| Phase 2 acceptance | Passed |
| Phase 3 acceptance | Passed |
| Phase 4 acceptance | Passed |
| Phase 5 acceptance | Passed |

Placeholder modules are marked with `TODO(Phase N)` and do not implement
behavior yet. Do not treat their presence as a functional or security
guarantee.

## Troubleshooting

- `ModuleNotFoundError: gemini_cli` — run the command through `uv run`, or
  re-run `uv sync --all-extras`.
- `uv run pytest` exits with code 5 — no tests were collected; check that
  `testpaths` in `pyproject.toml` still points at the `tests` directory.
- `gemini-cli-py: command not found` — re-run `uv sync --all-extras` to
  install the console script.

## Related documentation

- [Implementation plan](./plan.md)
- [Architecture](./architecture.md)
