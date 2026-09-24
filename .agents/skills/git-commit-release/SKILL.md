---
name: git-commit-release
description: Automates the full release-commit workflow for a git repository - inspects every changed file, writes a commit title and body (including a per-file changelog list), decides whether the change is a semver major/minor/patch bump, updates the version number in every project manifest it can find (package.json, pyproject.toml, Cargo.toml, composer.json, *.csproj, Chart.yaml, VERSION, pom.xml, build.gradle, etc.), then commits, tags the release as vX.Y.Z, and pushes both - always showing the user the full commit message and getting explicit confirmation before anything is pushed. Use this whenever the user wants to commit their changes, ship a release, bump the version, tag a new version, or asks for a commit message/changelog for their current working-tree changes - including casual phrasing like "commit this", "запуш зміни", "release a new version", "зроби реліз і тег", "bump the version and tag it", or "push my changes with a version bump".
---

# Git Commit & Release

This skill turns a dirty working tree into a properly versioned, tagged, pushed
release: survey what changed → decide the bump → draft the commit → get the
user's sign-off → update version files → commit → tag → push.

## Why the order matters

A version bump and a pushed tag are not easily undone once someone else has
pulled them, so this workflow is built around one principle: **nothing
gets written to disk, committed, or pushed until the user has seen the exact
commit message and version change and approved it.** Everything before that
point is read-only analysis. Everything after it happens in one confirmed
batch, so the user isn't asked to approve five separate times.

## Step 1 — Survey the changes

Get the full picture of the working tree relative to the last commit (this
naturally covers staged, unstaged, and untracked files):

```bash
git status --porcelain=v1
git diff HEAD --stat
```

For each changed file, look at enough of the actual diff to understand what
happened — not just the filename:

```bash
git diff HEAD -- <file>
```

Classify each path as added, modified, deleted, or renamed (git reports
renames with a `R` status and a similarity score — trust it above a
manual guess).

## Step 2 — Decide the version bump

Read the diffs, not just the file list — the same set of filenames can mean
a one-line typo fix or a breaking rewrite. Weigh the change against semver:

- **major** — something that breaks existing usage: a removed or renamed
  public function/export/endpoint/CLI flag, an incompatible change to a
  function's signature or a config key's meaning, dropped support for
  something.
- **minor** — new, backward-compatible functionality: new files implementing
  a feature, a new exported function/endpoint/option that defaults to the
  old behavior.
- **patch** — bug fixes, refactors with no behavior change, docs, tests,
  formatting, dependency bumps, chores.

If one category clearly dominates the diff, decide it yourself and move on —
don't make the user rubber-stamp an obvious call. But if the diff genuinely
mixes categories in a way that changes the outcome (for example, it contains
both a plausible breaking change and an unrelated new feature, or the
diff's intent just isn't clear from the code), stop and ask the user which
level applies. Show your reasoning for each candidate level rather than
silently picking one — use `ask_user_input_v0` if it's available, otherwise
just ask in plain text.

## Step 3 — Find the current version

Run the bundled scanner (read-only, safe to run any time):

```bash
python3 <skill_dir>/scripts/bump_version.py detect --repo <repo-root>
```

It scans for a fixed set of well-known manifest files (`package.json`,
`pyproject.toml`, `Cargo.toml`, `composer.json`, `*.csproj`, `pom.xml`,
`build.gradle`/`build.gradle.kts`, `gradle.properties`, `Chart.yaml`,
`CMakeLists.txt`, `VERSION`, `version.txt`, `setup.py`, `setup.cfg`,
`__init__.py`/`_version.py`/`version.py`), and for each one reports the
version string it found. It deliberately narrows its search to the section
of the file where a project declares *its own* version — e.g. it reads
`[package]` in `Cargo.toml` but ignores the versions pinned under
`[dependencies]`, and it strips `<parent>` out of `pom.xml` before looking
for `<version>` so it doesn't grab the parent POM's version by mistake.

- If every file agrees, that's the current version.
- If they disagree, don't silently pick one — tell the user which files
  disagree and ask which is authoritative (or whether they're intentionally
  versioned independently, e.g. a sub-package in a monorepo).
- If it finds nothing at all, ask the user where the project's version
  lives. If the repo genuinely doesn't track one, that's fine — skip the
  version bump and tag entirely and just do the commit, and say so.

## Step 4 — Compute the new version

Apply the bump from Step 2 to the confirmed version from Step 3 using
ordinary semver rules (bumping a more significant part resets everything
below it):

| Current | Bump  | New   |
|---------|-------|-------|
| 2.2.1   | patch | 2.2.2 |
| 2.2.1   | minor | 2.3.0 |
| 2.2.1   | major | 3.0.0 |

## Step 5 — Draft the commit

**Title:** short, imperative, in English. A conventional-commit-style
prefix that matches the bump (`feat:` for minor, `fix:` for a patch that's
a bug fix, `chore:`/`refactor:`/`docs:` for other patches) is a nice touch
because it makes the bump decision legible in the log later — use it when
it fits naturally, but don't force it if nothing fits.

**Body:** one to three sentences summarizing what changed and why, based on
what you actually read in the diffs (not a generic restatement of the
title), then a blank line, then the changelog block, then the version line.

The changelog uses one line per file, in this exact dash-prefixed format:

```
-added: src/js/auth.js
-changed: src/js/index.js
-changed: index.html
-removed: src/js/legacy-login.js
```

Use `-added:` / `-changed:` / `-removed:` / `-renamed: old -> new` matching
each file's actual status from Step 1.

**Full example:**

```
feat: add JWT-based login flow

Adds a token-based auth flow and wires it into the existing login page.
No existing routes change behavior, so this is backward compatible.

-added: src/js/auth.js
-changed: src/js/index.js
-changed: index.html
-removed: src/js/legacy-login.js

Version: 2.2.1 -> 2.3.0 (minor)
```

## Step 6 — Show the user and get confirmation

Before writing or running anything else, show the user:

- the full commit title + body (with the changelog)
- the bump you decided (or the question, if you had to ask) and why
- old version → new version
- exactly which files will have their version field updated
- the tag name (`v<new-version>`)
- the remote and branch this will push to (`git remote -v`, `git branch --show-current`)

Then explicitly ask them to confirm before anything is written, committed,
or pushed. This is a hard requirement, not a nicety — treat it the same way
regardless of how simple the change looks. If the user asks for changes
(different title, different bump, drop a file from the changelog), revise
and show the updated version before proceeding — don't just proceed on the
assumption they'll be fine with it.

## Step 7 — Apply the version bump

Once confirmed, write the new version into exactly the files confirmed in
Step 3:

```bash
python3 <skill_dir>/scripts/bump_version.py apply \
  --repo <repo-root> --old <old-version> --new <new-version> \
  --files package.json,pyproject.toml,...
```

It edits only the version field in each file — nothing else in the file
moves or reformats — and prints what it changed so you can sanity-check it
before committing. If any file's expected old version doesn't match (e.g.
someone edited it since Step 3), it refuses to touch that file and reports
why; re-run Step 3 rather than forcing it.

## Step 8 — Stage, commit, tag

```bash
git add -A
git commit -m "<title>" -m "<body>"
git tag -a v<new-version> -m "<title>"
```

Use an annotated tag (`-a`) so the tag carries its own message rather than
just being a bare pointer.

## Step 9 — Push

```bash
git push
git push origin v<new-version>
```

If the current branch has no upstream yet, use `git push -u origin <branch>`
for the first push instead.

## Step 10 — Report back

Tell the user the commit hash, the tag name, and confirm both pushed
successfully. If a push is rejected (e.g. the remote has commits this
branch doesn't), don't force-push or retry blindly — that's exactly the
kind of surprising, hard-to-undo action this workflow exists to avoid.
Explain what happened and let the user decide how to resolve it (usually
pulling/rebasing first).

## Edge cases

- **Nothing changed** — say so and stop; there's nothing to commit.
- **Not a git repo / no remote configured** — tell the user, don't invent one.
- **No version file anywhere the scanner recognizes** — commit normally,
  skip the bump and tag, and mention that to the user (they may want to
  add a VERSION file, or this may just be a project that doesn't version
  itself that way).
- **Monorepo with independently-versioned packages** — don't assume one
  global version bump; ask which package(s) the change actually touches
  and only bump those manifests.
- **Detached HEAD, mid-merge/rebase, or a protected branch** — flag it and
  don't push blind; ask the user how they want to proceed.
- **Version file mismatch discovered mid-flow** (someone else edited a
  manifest between Step 3 and Step 7) — the apply script will refuse and
  say why; re-detect rather than overriding it.
