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

Both entry points print the current package version prefixed with `v` (see
`__version__` in `src/gemini_cli/__init__.py`).

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

## Project layout

The package is split into `cli`, `core`, `llm`, `tools`, `sandbox`, `auth`,
`config`, and `telemetry`. `core` has no dependency on `cli` or `tui`; the
import test enforces that boundary. See
[Architecture](./architecture.md) for the full layout and runtime flows.

## Logging

Application logs are written as JSON Lines to `~/.gemini-cli/logs/app.jsonl`.
Telemetry is optional and disabled by default; when enabled, events go to
`~/.gemini-cli/logs/telemetry.jsonl`. API keys, passwords, and tokens are
redacted before they reach console or file output. Details are in
[Architecture](./architecture.md).

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

- [Architecture](./architecture.md) — package layout, runtime flows, and design
  decisions.
- [Tools and sandbox](./tools.md) — tool contract and built-in tool behavior.
- [Implementation plan](./plan.md) — historical phase plan.
