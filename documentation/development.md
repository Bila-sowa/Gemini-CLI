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

- `GEMINI_MODEL` selects a registered model. The default is
  `gemini-3.8-flash`; the registry also lists Gemini 4 Argon, Carbon, Neon,
  and Hydrogen; Gemini 3.8/3.7/3.6/3.5 Flash; 3.5/3.1 Flash-Lite; 3.1 Pro
  Preview; 3 Flash Preview; Gemma 4 31B and 26B; and the legacy 2.5 models.
  A well-formed name that is not in the registry is accepted as a custom model
  with generic metadata and no fallback, so a manual `.env` entry does not
  stop startup. Empty or malformed names still raise `UnknownModelError`.
- `GEMINI_RETRY_ATTEMPTS` sets the total number of attempts (minimum 1) for
  transient statuses `429`, `500`, and `503`. Client errors are never retried.
- `GEMINI_REQUEST_TIMEOUT_MS` bounds each attempt.
- `GEMINI_MAX_OUTPUT_TOKENS` is passed to the model.

`GeminiClient.generate()` returns a `ModelResponse` and, after retries are
exhausted on `404`, `429`, `500`, or `503`, retries once with the model's
registry fallback (`gemini-4-argon` and `gemini-3.1-pro-preview` →
`gemini-3.8-flash`; `gemini-3.8-flash` → `gemini-3.6-flash`). Custom models
have no fallback and report the provider error.
`GeminiClient.generate_stream()` yields `text_delta`, `tool_call_delta`,
`usage`, `done`, or `error` events; it is not retried, and tool calls are
emitted only after their arguments are complete. A `ToolCallRequest` also keeps
Gemini's opaque `thought_signature` (excluded from `repr`) and replays it
unchanged with the matching function call; unsigned parallel sibling calls stay
unsigned, and parallel tool observations are grouped into one user turn. System
prompts come from `llm/prompts/templates/`; project instructions are appended
inside an untrusted, non-spoofable block.

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
`ToolCategory.SHELL`. It validates its inputs and delegates execution to a
`SandboxBackend` injected by the caller. The tool never creates a backend and
never approves itself: the agent runtime is responsible for checking approval
before dispatch. Without a backend it returns a structured failure
(`Shell execution is unavailable: no sandbox backend is configured.`) so the
agent loop keeps running and the model receives a clear reason.

The tool validates:

- `working_directory` must resolve inside `Settings.allowed_roots` and be an
  existing directory; an empty `allowed_roots` rejects any explicit directory.
- `command` must not be blank and `timeout_seconds` must be positive.
- The configured `shell_timeout_seconds` and `shell_max_output_bytes` are used
  when the model does not override the timeout.

A real `ExecutionResult` is mapped to `ToolResult`: non-zero exit codes and
timeouts become failed results, and `exit_code`, `timed_out`,
`duration_seconds`, and truncation flags are exposed in `metadata`. The command
string is never repeated in the result, metadata, or logs, because it can
contain secrets.

### Sandbox and approvals

The `sandbox` package is the execution boundary for shell-capable tools.

- `sandbox/base.py` defines `ExecutionResult`, the `SandboxBackend` protocol,
  and `SandboxError`. A backend validates the working directory, enforces the
  timeout, caps output, and avoids inheriting application secrets.
- `SubprocessBackend` uses `asyncio.create_subprocess_shell`. It requires a
  non-empty `allowed_roots` list and defaults to the first root. A timeout kills
  the whole process group on POSIX, and both streams are drained after the
  output limit is reached so a full pipe cannot block the child. Only a small
  environment allowlist (`PATH`, locale, terminal, temporary directory) is
  inherited; explicitly passed variables are merged on top. This backend is a
  local execution boundary, not a kernel sandbox: a command can still reach the
  network or read host files outside the root. Use Docker when stronger
  isolation is required.
- `DockerBackend` is optional and uses the Docker CLI, so it adds no Python
  dependency. `is_available()` checks both the CLI and the daemon. Containers
  run ephemerally with no network by default, dropped capabilities,
  `no-new-privileges`, CPU/memory/PID limits, a read-only container root, and a
  read-only project mount. `--pull=never` avoids an unexpected image download;
  pull the configured image explicitly before first use. A timeout kills the
  container and removes it. `select_backend()` falls back to the subprocess
  backend when Docker is unavailable.
- `ApprovalManager` is fail-closed. Read-only tools are auto-approved, write
  tools ask by default, and destructive tools always ask. An unknown risk level
  is treated as destructive. If approval is required but no callback is
  configured, `ApprovalUnavailableError` is raised. Callback errors and invalid
  decisions become `DENIED`. File contents are replaced by their length in the
  confirmation details, and arguments are never written to logs.

The agent runtime (Phase 7) will wire the approval check and the selected
backend into the tool dispatch path.

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
(`/etc/passwd`, `../../etc/passwd`, and `file:///`) checked through
`dispatch()`. The shell tool is also checked without a backend, where it must
return the structured unavailable result rather than executing anything.

## Agent core runtime

The `core` package is the model-independent orchestrator. It has no dependency
on `cli` or `tui`; the acceptance suite enforces that with an import check.

- `Session` keeps the ordered history, assistant tool calls, observations,
  token totals, and iteration counter. Observations are keyed by call id, so
  the tool name is available when Gemini needs it.
- `ContextManager` estimates tokens locally (four characters per token) and
  trims the oldest complete groups first. It never separates an assistant tool
  call from its observations, never removes the most recent messages, and
  keeps the history starting with a user turn.
- `ContextWindowTracker` derives context occupancy from the system prompt, the
  exported tool declarations, and the fitted history, using the same token
  estimate as `ContextManager`. The percentage is measured against
  `context_window - max_output_tokens`. It keeps no state on disk, so a model
  change or a restored session recomputes the status from the conversation.
  `observe()` is the agent's hot-path update; `refresh()` rebuilds the status
  from an arbitrary message list and is the entry point for future session
  restore.
- `AgentEvent` and `AgentEventType` are the presentation contract. Text,
  thinking, tool, error, and done events stay stable across providers.
- `Agent.run()` streams model output, checks approval before every dispatch,
  records exactly one observation per tool call, and stops at
  `max_agent_iterations`. Denied, unknown, and approval-unavailable calls
  produce an observation instead of breaking the loop.

The agent depends on a small `ChatClient` protocol instead of importing the
Gemini client directly. Integration tests use the scripted `FakeLLMClient` and
`FakeTool` fixtures from `tests/fixtures/`.

## CLI and terminal UI

`gemini-cli-py` exposes `chat`, `run`, `login`, `logout`, `auth status`, and
`config show` / `config set`.

- `cli/runtime.py` builds the agent lazily. Only commands that need the agent
  resolve credentials; `--help`, `--version`, auth, and config work before a
  key exists. An empty `allowed_roots` defaults to the working directory.
- `run` streams assistant text to stdout and exits non-zero on an agent error.
  It passes no approval callback, so destructive tools remain denied.
- `chat` starts the full-screen **Textual** application in `tui/app.py`:
  the ASCII wordmark, bordered status bar, scrollable conversation, bordered
  input, and role footer match the mockups. Assistant text streams into the
  conversation and errors are red. While a model request is pending, an inline
  `WaitingIndicator` animates; it is hidden before each text chunk and while
  tools run.
- User and assistant messages are rendered as Markdown by `tui/markdown.py`.
  The speaker prefix is a separate widget, so a user turn is not bold as a
  whole. Assistant chunks go through `Markdown.get_stream()` and a final
  `update()` renders the completed document; user input is rendered once.
  Headings, emphasis, strikethrough, inline `<u>`, lists, block quotes, tables,
  inline code, and fenced code blocks are supported. Each code block has a
  `copy` button that copies the whole block to the clipboard. Releasing a mouse
  selection anywhere in the TUI copies the selected text automatically and
  confirms it with a short notification; plain clicks and empty selections are
  ignored. Both paths use `wl-copy` on Wayland, `xclip`/`xsel` on X11, or the
  platform equivalent, and fall back to OSC 52 only when the terminal supports
  it. VTE terminals (GNOME Terminal, Tilix, Terminator) ignore OSC 52; without
  `wl-clipboard` the code is saved to
  `~/.gemini-cli/clipboard/last-code-block.txt` and a selection is saved to
  `~/.gemini-cli/clipboard/last-selection.txt`, with a warning instead of a
  false success message. The toolbar stays fixed above the code area, which
  scrolls internally after 20 rows. Terminal control and ANSI sequences are
  stripped, raw HTML is not executed, and links are styled but never opened
  automatically.
- `tui/themes.py` defines the dark and light palettes shared by the widgets and
  derives the matching Textual themes (`gemini-dark`, `gemini-light`) for
  Markdown headings and links. H2 headings are bold without underline. Theme
  changes use Textual theme variables, so the main screen, modals, and Markdown
  blocks share one palette; existing message prefixes, info lines, and errors
  are re-styled with `MarkdownMessageView.apply_theme()` and
  `MessageView.apply_theme()` instead of keeping their creation-time colors.
- `run` and the line fallback show a Rich spinner on `stderr` while the model
  request is in flight. The spinner is disabled when `stderr` is not a
  terminal, so redirected stdout and CI logs stay clean.
- Sensitive tool calls open a modal `ApprovalScreen` with Approve/Deny buttons.
  `/role`, `/model`, `/theme`, and `/auth` open a modal `MenuScreen` with arrow
  navigation and Escape cancellation; model and role changes rebuild the
  runtime. `/model`, `/role`, and `/theme` also update `GEMINI_MODEL`,
  `GEMINI_ROLE`, or `GEMINI_THEME` in the project `.env` atomically, preserving
  comments and unrelated keys. If the same key is present in the process
  environment, `.env` is written but the environment value still takes
  precedence and the TUI warns about it.
- Slash commands are declared once in `cli/slash_commands.py` and shared by the
  TUI and the line fallback. Canonical names accept plural and short aliases:
  `/model` (`/models`, `/m`), `/role` (`/roles`), `/auth` (`/authentication`),
  `/context` (`/ctx`, `/window`), `/theme` (`/themes`, `/t`),
  `/clear` (`/cls`, `/new`), `/reset` (`/reset-key`, `/remove-key`, `/logout`),
  `/help` (`/h`, `/?`, `/commands`), and `/exit` (`/quit`, `/q`, `/bye`).
  Matching is case-insensitive. An exact alias wins over another command's
  prefix, and a unique prefix such as `/mod` also works. Ambiguous prefixes
  (`/c`) and misspellings (`/modl`) are reported with candidates or a
  suggestion instead of being executed. `/context` renders the context-window
  panel, and `/theme` opens the theme menu; the chosen palette is applied
  immediately and saved to `.env` for the next session.
- Commands accept validated arguments: `/model <name>`, `/role <role>`,
  `/theme <dark|light>`, and `/help <command>`. Only registered models are
  accepted; invalid models, roles, and themes produce a `Did you mean ...?`
  suggestion from `cli/command_arguments.py`. Commands that take no arguments
  reject extra text instead of ignoring it. In the TUI, omitting the argument
  still opens the corresponding menu.
- `tui/repl.py` is the plain line fallback used by `--no-tui`, pipes, and CI.
  It streams text and uses the same command registry, including `/help`,
  `/clear`, `/reset`, and `/exit`. `/model <name>` and `/role <role>` rebuild
  the runtime and save the choice to `.env`; `/help <command>` shows detailed
  help. Invoking `/model` or `/role` without a value, or `/theme` at all, points
  the user to the full-screen UI. Bare `exit` and `quit` remain accepted for
  compatibility.
- `login` verifies the key against the API unless `--no-verify` is passed.
  `config set` validates through `Settings`, refuses `api_key` and
  `allowed_roots`, and writes to the project config or `--global`.

## Current status

| Area | Status |
|---|---|
| Package install / `uv sync` | Available |
| CLI `--version` and `--help` | Available |
| Ruff / mypy | Available |
| Python version (`.python-version`) | Available (3.12) |
| Test suite | Available (873 tests: smoke, import, entry-point, configuration, dotenv updates, authentication, logging/telemetry, LLM client, tool system, all built-in tools, sandbox backends, approvals, agent core runtime, context-window tracking, CLI commands, slash-command registry, aliases, arguments and suggestions, Textual TUI, theme selection, persistence, and message restyling, Markdown rendering, clipboard integration, selection auto-copy, line fallback, and the Phase 5–8 acceptance checks) |
| Configuration (`Settings`, project context) | Available |
| Authentication (API key resolution, keyring store, OAuth stub) | Available |
| Logging and Telemetry (`configure_logging`, secret redaction) | Available |
| LLM client (`generate`, `generate_stream`, retry, fallback, custom models, prompts) | Available |
| Tool base contract (`BaseTool`, `ToolResult`, risk levels) | Available |
| Tool registry (`register`, `export_schemas`, `dispatch`, telemetry) | Available |
| Filesystem tools (`read_file`, `write_file`, `edit_file`, `find_files`, `grep`) | Available |
| Shell tool (`run_command`) | Available through an injected sandbox backend |
| Sandbox backends (`SubprocessBackend`, optional `DockerBackend`) | Available |
| Approval manager (`ApprovalManager`) | Available (fail-closed policy) |
| Conversation session (`Session`) | Available |
| Context manager (`ContextManager`) | Available (local token estimate, group trimming) |
| Context window tracker (`ContextWindowTracker`) | Available (local estimate against the model window, no persistence) |
| Agent events (`AgentEvent`, `AgentEventType`) | Available |
| Agent loop (`Agent`) | Available (streaming, approvals, iteration limit) |
| CLI runtime (`build_runtime`) | Available (lazy credentials, built-in tools, backend selection) |
| CLI commands (`chat`, `run`, `login`, `logout`, `auth status`, `config show/set`) | Available |
| Full-screen terminal UI (`GeminiApp`, streaming, modal approval) | Available |
| Markdown message rendering (`GeminiMarkdown`, streaming, inline `<u>`) | Available |
| TUI menus and slash commands (`MenuScreen`, `/role`, `/model`, `/auth`, `/theme`) | Available (aliases, prefixes, and validated arguments resolve through `cli/slash_commands.py` and `cli/command_arguments.py`; model, role, and theme choices persist to `.env`) |
| Context panel (`ContextPanel`, window and budget bar) | Available |
| Line fallback (`run_repl`, `--no-tui`, non-TTY) | Available |
| Terminal themes (`Theme`, dark and light) | Available |
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
| Phase 6 acceptance | Passed |
| Phase 7 acceptance | Passed |
| Phase 8 acceptance | Passed |

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
- Docker sandbox reports that it is unavailable — make sure the Docker CLI is
  installed, the daemon is running, and the configured image is already
  present. The backend intentionally uses `--pull=never`, so it never downloads
  an image on its own.

## Related documentation

- [Implementation plan](./plan.md)
- [Architecture](./architecture.md)
