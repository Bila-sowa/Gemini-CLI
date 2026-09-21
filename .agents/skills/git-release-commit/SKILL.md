---
name: git-release-commit
description: Analyze repository changes, bump the project version, create a Conventional Commit, and push it.
compatibility: Requires Git and permission to modify and push the repository.
---

---

# Git Release & Commit

Analyze all current repository changes, determine the release type, update the version, commit, and push.

## Workflow

1. Inspect all changes:

```bash
git status --short
git diff
git diff --cached
```

Read all relevant changed files before deciding the update type.

2. Determine the Semantic Versioning bump:

- `PATCH` — fixes, refactors, docs, minor internal changes.
- `MINOR` — new backward-compatible functionality.
- `MAJOR` — breaking changes.

Use repository-specific versioning rules if they exist.

3. Determine the Conventional Commit type:

`feat`, `fix`, `refactor`, `perf`, `docs`, `test`, `build`, `ci`, `chore`, or `style`.

4. Find the current project version and update all files that are supposed to contain the same version.

Common files include:

`package.json`, `package-lock.json`, `pyproject.toml`, `Cargo.toml`, `VERSION`, `manifest.json`.

Do not create a version if the project does not already use one.

Version bump:

```text
PATCH: X.Y.Z -> X.Y.(Z+1)
MINOR: X.Y.Z -> X.(Y+1).0
MAJOR: X.Y.Z -> (X+1).0.0
```

5. Review the final changes:

```bash
git diff
git status
```

Do not commit secrets, unrelated files, or generated artifacts that should be ignored.

6. Generate a commit:

```text
<type>(<optional-scope>): <short description>
```

Optionally add a short body describing important changes and the version bump.

7. Stage and review:

```bash
git add -A
git diff --cached
```

8. Commit and push:

```bash
git commit -m "<title>" -m "<description>"
git push
```

If the branch has no upstream:

```bash
git push -u origin <current-branch>
```

## Rules

- Analyze the complete diff before choosing the version bump.
- Never invent changes in the commit message.
- Keep version references synchronized.
- Never discard user changes.
- Never use force push automatically.
- Never use destructive Git commands such as `git reset --hard`.
- If push is rejected because remote history differs, fetch and inspect instead of force-pushing.
- Follow repository instructions such as `AGENTS.md` or `CONTRIBUTING.md` when present.

## Final output

Report only:

```text
Type: MINOR / feat
Version: 1.2.3 -> 1.3.0
Commit: feat(api): add branch support
Push: successful
```
