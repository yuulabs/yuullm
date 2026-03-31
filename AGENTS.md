# Repository Guidelines

## Project Structure & Module Organization

`yuullm` is a standalone Python package in the workspace. Source code lives in
`src/yuullm/`, tests live in `tests/`, and repo-specific helpers live in
`scripts/`. Keep public exports in `src/yuullm/__init__.py`, provider-specific
code in `src/yuullm/providers/`, and design notes in `design/`.

## Build, Test, and Development Commands

Use `uv` for local work:

```bash
uv sync
uv run pytest
uv run pytest tests/test_client.py -v
uv run pytest tests/test_types.py -v
uv build
```

`scripts/setup-dev.sh` installs the repo’s Git hooks. The `pre-push` hook checks
that tag pushes match the version in `pyproject.toml`.

## Coding Style & Naming Conventions

Target Python 3.12+. Follow the codebase’s existing style: 4-space indentation,
type hints on public functions, `from __future__ import annotations` where
needed, and `snake_case` for modules/functions with `PascalCase` for classes.
Use `msgspec.Struct` for serialized stream types and `TypedDict` for content
items. Prefer small, explicit helpers over deep abstraction.

## Testing Guidelines

Tests use `pytest` with `pytest-asyncio` in auto mode. Name files
`tests/test_*.py`, keep unit tests close to the behavior they cover, and mock
provider SDK calls rather than hitting live services. Add focused tests for any
new provider conversion, pricing, or cache behavior.

## Commit & Pull Request Guidelines

Commit history in this repo uses short imperative subjects, often with
Conventional Commit prefixes such as `feat:`, `fix:`, `refactor(yuullm):`, and
`chore:`. Keep changes scoped, mention any version or hook impact in the PR
description, and include the exact verification commands you ran.
