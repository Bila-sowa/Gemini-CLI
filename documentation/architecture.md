# Gemini Agent Architecture

## Overview

`gemini_cli` is a command-line AI agent built around the Gemini API. The
project is organized as a layered Python package so that the user interface,
agent orchestration, model integration, tools, security boundaries, and
configuration can evolve independently.

The architecture has the following goals:

- provide both interactive and one-shot command-line workflows;
- keep the agent loop independent from a specific user interface;
- isolate Gemini SDK details behind a dedicated LLM client;
- expose built-in and MCP-provided capabilities through one tool abstraction;
- require explicit approval for potentially dangerous operations;
- support project-specific instructions and predictable configuration
  precedence;
- make core behavior testable without network access or real credentials.

This document describes the intended architecture represented by the proposed
project structure. It is a design target rather than a statement that every
module has already been implemented.

## High-Level Design

```text
User
  |
  v
CLI commands / interactive TUI
  |
  v
Agent loop <----> Session and context manager
  |
  +-----------> Gemini LLM client
  |
  +-----------> Tool registry
                   |
                   +-- Built-in tools
                   +-- MCP tools
                   +-- Sandbox and approval layer

Cross-cutting services: configuration, authentication, logging, and telemetry
```

The CLI accepts user input and delegates execution to the core agent. The
agent maintains conversation state, sends model requests, interprets tool
calls, executes approved tools, records observations, and repeats until it can
return a final response. The LLM and tool layers are accessed through narrow
interfaces so that they can be replaced with test doubles.

## Package Layout

```text
gemini_cli/
├── src/
│   └── gemini_cli/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli/
│       │   ├── __init__.py
│       │   ├── app.py
│       │   ├── commands/
│       │   │   ├── chat.py
│       │   │   ├── run.py
│       │   │   ├── auth.py
│       │   │   ├── mcp.py
│       │   │   └── config.py
│       │   └── tui/
│       │       ├── __init__.py
│       │       ├── repl.py
│       │       └── themes.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── agent.py
│       │   ├── session.py
│       │   ├── streaming.py
│       │   └── context_manager.py
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── client.py
│       │   ├── models.py
│       │   ├── retry.py
│       │   └── prompts/
│       │       ├── system.py
│       │       └── templates/
│       │           └── *.jinja2
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── registry.py
│       │   ├── filesystem/
│       │   ├── shell/
│       │   ├── web/
│       │   └── code/
│       ├── mcp/
│       │   ├── __init__.py
│       │   ├── client.py
│       │   ├── discovery.py
│       │   └── server_config.py
│       ├── sandbox/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── docker_backend.py
│       │   ├── subprocess_backend.py
│       │   └── approval.py
│       ├── auth/
│       │   ├── __init__.py
│       │   ├── oauth.py
│       │   ├── api_key.py
│       │   └── credentials_store.py
│       ├── config/
│       │   ├── __init__.py
│       │   ├── settings.py
│       │   ├── project_context.py
│       │   └── defaults.py
│       └── telemetry/
│           ├── __init__.py
│           └── logger.py
├── tests/
│   ├── conftest.py
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── documentation/
│   ├── architecture.md
│   └── tools.md
├── .env.example
├── .github/workflows/ci.yml
├── .python-version
├── pyproject.toml
├── README.md
├── GEMINI.md.example
└── uv.lock
```

The `src` layout prevents accidental imports from the repository root and
ensures that local development exercises the installed package in the same way
as production usage.

## Component Responsibilities

### Application entry points and CLI

`gemini_cli.__main__` is the module entry point for `python -m
gemini_cli`. It contains minimal bootstrapping and delegates command
registration and argument parsing to `cli/app.py`.

The `cli` package is an adapter between users and the application core:

- `app.py` registers commands and turns expected failures into exit codes;
- `runtime.py` builds the agent runtime lazily — it resolves the API key,
  loads project context, registers the built-in tools, selects the sandbox
  backend, and assembles the agent;
- `commands/chat.py` starts an interactive conversation;
- `commands/run.py` performs a single non-interactive request and is suitable
  for scripts and CI jobs;
- `commands/auth.py` manages login, logout, and status;
- `commands/config.py` reads and updates user-facing configuration;
- `commands/mcp.py` will manage configured MCP servers in Phase 9;
- `tui/app.py` is the full-screen Textual application: ASCII wordmark, status
  bar, scrollable conversation, prompt, quota panel, and role footer;
- `tui/widgets.py` contains the logo, status bar, message view, quota panel,
  approval modal, and selection menu;
- `tui/repl.py` is the plain line fallback for pipes, CI, and `--no-tui`;
- `tui/themes.py` keeps the dark and light palettes separate from behavior.

The TUI streams agent events into the conversation, opens a modal confirmation
for sensitive tool calls, and uses modal menus for role, model, and
authentication selection. A non-TTY environment falls back to line mode.

Commands that do not need the agent — help, version, auth, and config — never
resolve credentials. The one-shot `run` command uses no approval callback, so
destructive tools stay denied outside the interactive UI.

CLI modules should validate and normalize input, call application services,
and format output. They should not implement agent reasoning, credential
storage, or tool execution directly.

### Core agent runtime

The `core` package owns model-independent orchestration and never imports the
`cli` or `tui` packages.

`core/agent.py` implements the think-act-observe loop:

1. Validate the user message against `Settings.max_input_chars` and append it
   to the session.
2. Fit the history to the model window, export tool declarations, and stream a
   response.
3. Emit text and thinking events as they arrive and buffer complete tool calls.
4. Record the assistant turn, usage, and quota consumption, then check the
   iteration limit.
5. For each tool call, check approval, dispatch through the registry, and add
   exactly one observation — including denied, unknown, and failed calls.
6. Repeat until the model answers without tool calls or the limit is reached.

The agent depends on a small `ChatClient` protocol rather than the concrete
Gemini client, so the loop can be tested with scripted fakes.

`core/session.py` keeps the ordered history, tool calls, observations, token
totals, and iteration counter. Observations are keyed by call id so the tool
name is available when Gemini needs it.

`core/context_manager.py` estimates tokens locally and trims the oldest
indivisible groups first. It never separates an assistant tool call from its
observations, never removes the most recent messages, and keeps the history
starting with a user turn.

`core/quota_tracker.py` records reported usage in `~/.gemini-cli/quota.json`
using an atomic, private write and reports rolling five-hour and seven-day
windows. The limits are local policy derived from published free-tier
averages, not a provider guarantee.

`core/streaming.py` defines the stable `AgentEvent` contract. Both the
interactive TUI and non-interactive command can consume these events without
importing the Gemini SDK.

### LLM integration

The `llm` package isolates all Gemini-specific behavior:

- `client.py` wraps the `google-genai` SDK and maps SDK request and response
  types to internal application models;
- `models.py` defines supported models and controlled fallback behavior, such
  as switching from a Pro model to Flash when policy allows it;
- `retry.py` applies bounded exponential backoff to transient failures and
  rate limits;
- `prompts/system.py` composes trusted system instructions;
- `prompts/templates/` stores version-controlled Jinja templates for reusable
  prompts.

Untrusted user or tool content must remain separate from system instructions.
Fallbacks and retries must be bounded, observable, and restricted to errors
that are safe to retry.

Streaming yields text deltas as they arrive, but tool calls are buffered until
their arguments are complete so partial JSON never reaches tool dispatch.
`generate()` retries transient failures and then tries the registry fallback
once; a stream is never replayed, because that would duplicate emitted text.

### Tool system

The `tools` package provides a common interface for capabilities available to
the agent.

`tools/base.py` defines the `BaseTool` contract and Pydantic schemas for
validated parameters and structured results. `tools/registry.py` registers
tools, resolves unique names, exports model-facing schemas, and dispatches
validated calls.

Built-in tools are grouped by capability:

- `filesystem/` reads, writes, edits, and searches files;
- `shell/` executes commands through an approved sandbox backend;
- `web/` fetches resources and performs grounded Google Search;
- `code/` provides source-oriented search operations.

All tools should return structured errors rather than leaking backend-specific
exceptions into the agent loop. File paths, command arguments, URLs, and model
generated parameters must be validated before execution.

### Model Context Protocol integration

The `mcp` package extends the tool registry with tools exposed by external MCP
servers:

- `client.py` manages MCP sessions over supported transports such as stdio and
  SSE;
- `discovery.py` connects to configured servers, discovers their tools, and
  registers compatible definitions;
- `server_config.py` validates server declarations and transport settings.

MCP tools follow the same validation, approval, timeout, and result-handling
rules as built-in tools. Server configuration is data; it must not bypass
sandbox or approval policies.

### Sandbox and approvals

The `sandbox` package is the security boundary for operations that affect the
host system.

`sandbox/base.py` defines the backend protocol and `ExecutionResult`.
`subprocess_backend.py` provides a local execution boundary constrained by
working-directory validation, a timeout, bounded output, and a minimal
environment. It is not a kernel sandbox: commands can still reach the host
network and files outside the allowed root. `docker_backend.py` provides
stronger isolation through an optional ephemeral container with no network by
default, dropped capabilities, resource limits, and a read-only project mount.
`approval.py` classifies sensitive actions and coordinates explicit user
confirmation before execution; the policy is fail-closed.

The agent must use the sandbox abstraction rather than invoking shell commands
directly. Denied, timed-out, or resource-limited operations should be reported
as normal tool results so the agent can recover safely.

### Authentication and credentials

The `auth` package supports Google OAuth and API-key authentication.
`credentials_store.py` is the only component responsible for persistent secret
storage and should use the operating system keyring where available. Tokens and
API keys must never be written to source-controlled configuration, prompts, or
telemetry.

The CLI initiates authentication, but the LLM client consumes credentials
through the authentication abstraction. This separation prevents UI concerns
from leaking into model requests.

### Configuration and project context

The `config` package owns typed configuration and instruction discovery:

- `defaults.py` contains safe built-in defaults;
- `settings.py` loads and validates settings with the precedence `environment
  variables > project configuration > global configuration > defaults`;
- `project_context.py` discovers and reads project instructions such as
  `GEMINI.md` and `AGENTS.md`.

Configuration loading should produce one validated settings object during
startup. Lower layers receive explicit settings or focused configuration
objects instead of reading environment variables independently.

Project context is untrusted input. It can guide agent behavior, but it cannot
override security controls, credential handling, or approval requirements.

### Telemetry and logging

`telemetry/logger.py` configures structured application logging and optional
telemetry. Telemetry is disabled when the user opts out and must exclude prompt
content, secrets, personal data, and raw tool output by default. Useful events
include model selection, latency, token usage, retries, tool duration, and
sanitized error categories.

## Runtime Flows

### Interactive chat

1. `python -m gemini_cli chat` enters through `__main__.py` and `cli/app.py`.
2. Configuration, project context, credentials, and tool definitions are
   loaded.
3. The TUI creates a session and forwards user messages to the agent.
4. The agent streams model events through `core/streaming.py`.
5. Tool calls are validated, approved when required, and executed through the
   registry.
6. Observations are appended to the session and the loop continues.
7. The final response is rendered by the TUI and the session remains available
   for the next message.

### One-shot execution

The `run` command uses the same configuration, session, agent, LLM, and tool
components as interactive chat. It creates a short-lived session, processes one
request, writes the final response in a script-friendly form, and exits with a
meaningful status code. Presentation is the only intentional difference from
interactive mode.

### Tool execution

```text
Model tool request
      |
      v
Schema validation --> reject invalid request
      |
      v
Registry lookup ----> reject unknown or ambiguous tool
      |
      v
Policy and approval -> deny or ask the user when required
      |
      v
Sandbox / MCP client
      |
      v
Structured result --> session --> next model request
```

No model-produced command or argument is trusted implicitly. Validation and
authorization occur before side effects.

## Dependency Direction

Dependencies should point inward toward stable contracts:

- `cli` depends on `core`, `config`, and user-facing authentication services;
- `core` depends on LLM and tool interfaces, not on CLI implementations;
- `llm` depends on configuration and the Gemini SDK, not on the TUI;
- built-in tools depend on the sandbox where side effects require isolation;
- `mcp` adapts external tool definitions to the local tool contract;
- telemetry may be called by every layer but must not contain business logic.

Circular imports between these packages should be avoided. Shared data models
should live with the contract that owns them rather than in a generic utility
module.

## Testing Strategy

The test suite mirrors architectural boundaries:

- `tests/unit/tools/` validates schemas, registry behavior, path controls, and
  tool errors;
- `tests/unit/llm/` validates request mapping, streaming event conversion,
  fallback decisions, and retry limits with mocked SDK responses;
- `tests/unit/mcp/` validates configuration, discovery, and protocol adapters;
- `tests/integration/test_agent_loop.py` exercises complete think-act-observe
  scenarios with fake LLM and tool implementations;
- `tests/fixtures/mock_responses/` stores deterministic model and protocol
  responses.

Unit tests should not require network access, real credentials, Docker, or paid
API calls. A smaller explicitly marked integration or end-to-end suite may
exercise those dependencies when the environment provides them.

## Operational and Security Constraints

- Validate every tool argument before execution.
- Prevent path traversal and restrict filesystem tools to approved roots.
- Treat shell commands, remote content, project instructions, and MCP responses
  as untrusted input.
- Apply timeouts, output limits, and resource limits to external operations.
- Require explicit approval for destructive or high-impact actions.
- Redact credentials and personal data from logs and errors.
- Limit agent iterations, retries, and token consumption.
- Preserve a clear audit trail of approvals and tool outcomes without storing
  sensitive payloads.
- Keep optional telemetry transparent and easy to disable.

These constraints are enforced in code at the tool, sandbox, authentication,
and configuration boundaries; prompt instructions alone are not a sufficient
security mechanism.
