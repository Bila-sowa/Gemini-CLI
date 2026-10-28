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

## Current status

| Area | Status |
|---|---|
| Package install / `uv sync` | Available |
| CLI `--version` and `--help` | Available |
| Ruff / mypy | Available |
| Python version (`.python-version`) | Available (3.12) |
| Test suite | Available (smoke, import, and entry-point tests) |
| Environment template (`.env.example`) | Available |
| Project instructions template (`GEMINI.md.example`) | Available |
| CI pipeline (`.github/workflows/ci.yml`) | Available |
| Phase 0 acceptance | Passed |

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
