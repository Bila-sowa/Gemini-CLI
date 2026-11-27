# Tools and Sandbox

This reference describes the tool contract, the built-in tools, and the
execution boundary they use. For the surrounding architecture, see
[Architecture](./architecture.md); for setup and quality gates, see
[Development](./development.md).

## Tool contract

`gemini_cli.tools` defines the shared tool contract, the registry, and the
filesystem tools:

- `BaseTool` requires a unique `name`, a model-facing `description`, a Pydantic
  `parameters_schema`, a `ToolCategory`, a `RiskLevel`, and an async
  `execute(params) -> ToolResult`.
- `ToolResult` carries `success`, `output`, optional `error`, and safe
  structured `metadata`. Expected failures are returned as a failed result
  instead of being raised.
- `RiskLevel` (`read`, `write`, `destructive`) is the classification the
  approval layer consumes; `ToolCategory` groups capabilities for discovery and
  the UI.
- `ToolRegistry` rejects duplicate or malformed tool names, exports model-facing
  JSON Schema declarations via `export_schemas()`, and runs calls through
  `dispatch()`. Dispatch validates arguments against the tool's schema and
  converts unknown tools, invalid arguments, timeouts, and unexpected
  exceptions into structured failures, so a model-generated call cannot crash
  the agent loop. Validation errors never echo the rejected input values.
- Dispatch records a `ToolCallEvent` (tool name, duration, status) when
  telemetry is enabled; arguments and outputs are never recorded. Approval
  checks are not part of the registry; the agent runtime applies them before
  dispatch.

`ToolResult.metadata` must not contain secrets, prompts, or raw tool arguments.

## Filesystem tools

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

## Shell tool

`run_command` is registered with `RiskLevel.DESTRUCTIVE` and
`ToolCategory.SHELL`. It validates its inputs and delegates execution to a
`SandboxBackend` injected by the caller. The tool never creates a backend and
never approves itself: the agent runtime checks approval before dispatch.
Without a backend it returns a structured failure
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

## Sandbox and approvals

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

## Web tool

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

DNS rebinding remains a residual risk: the hostname is resolved for validation
and again by the HTTP client. Closing that gap fully would require pinning the
validated address in the TLS connection.

## Code tool

`code_search` (risk `READ`, category `CODE`) searches source files with a Python
regular expression and returns each match with its surrounding lines.

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

## Acceptance suite

`tests/unit/tools/test_builtin_tools.py` registers all eight built-in tools
together and exercises the registry as the agent will: unique Gemini-compatible
names, flat JSON Schema declarations for every tool, and the security guards
(`/etc/passwd`, `../../etc/passwd`, and `file:///`) checked through
`dispatch()`. The shell tool is also checked without a backend, where it must
return the structured unavailable result rather than executing anything.
