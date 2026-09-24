#!/usr/bin/env python3
"""
bump_version.py - find and update the version string in common project
manifest files, without disturbing anything else in those files.

Two subcommands:

  detect  - scan a repo for files that look like they hold "the" project
            version, and print what it found (file, section/key, value).
            Read-only, safe to run any time.

  apply   - given an explicit list of files (normally the ones `detect`
            just reported, after a human/agent has confirmed they agree
            on the current version), replace the old version string with
            a new one in each file, in place, doing the smallest possible
            text edit. Prints a summary of every change it made.

This script is deliberately conservative: it only recognizes a fixed set
of well-known manifest filenames and narrows its search to the section of
the file where a project typically declares its OWN version (as opposed
to a dependency's version, which the same file format often also
contains). If it can't find a confident, unambiguous match in a file, it
skips that file and says so rather than guessing.

Usage:
    python3 bump_version.py detect --repo /path/to/repo
    python3 bump_version.py apply  --repo /path/to/repo \
        --old 2.2.1 --new 2.2.2 --files package.json,pyproject.toml
"""

import argparse
import json
import re
import sys
from pathlib import Path

EXCLUDE_DIRS = {
    ".git", "node_modules", "vendor", "dist", "build", "out", "target",
    "venv", ".venv", "__pycache__", ".tox", ".mypy_cache", "coverage",
}

SEMVER_CORE = r"(\d+)\.(\d+)\.(\d+)"
VERSION_RE = re.compile(r"v?" + SEMVER_CORE + r"(-[0-9A-Za-z.\-]+)?(\+[0-9A-Za-z.\-]+)?")


def iter_candidate_files(repo: Path):
    """Yield (path, kind) for every file in repo that matches a known
    manifest filename, skipping vendored/build directories."""
    targets = {
        "package.json": "json_root",
        "composer.json": "json_root",
        "Cargo.toml": "toml_section:package",
        "pyproject.toml": "toml_section:project,tool.poetry",
        "setup.cfg": "ini_section:metadata",
        "build.gradle": "gradle",
        "build.gradle.kts": "gradle",
        "gradle.properties": "props",
        "Chart.yaml": "yaml_key",
        "VERSION": "plain",
        "version.txt": "plain",
        "CMakeLists.txt": "cmake",
        "pom.xml": "pom",
    }
    for path in repo.rglob("*"):
        if not path.is_file():
            continue
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        name = path.name
        if name in targets:
            yield path, targets[name]
        elif name.endswith(".csproj"):
            yield path, "csproj"
        elif name in ("setup.py",):
            yield path, "setup_py"
        elif name in ("_version.py", "version.py", "__init__.py"):
            yield path, "python_dunder"


def find_section_span(text: str, section_names):
    """For a TOML-like file, return (start, end) char offsets of the
    first matching top-level [section] (or [section.sub]) body, for any
    of the given dotted section_names. Body ends at the next top-level
    '[' header or EOF."""
    for section in section_names:
        # Matches [project] or [tool.poetry] etc exactly.
        pattern = re.compile(
            r"^\[" + re.escape(section) + r"\]\s*$", re.MULTILINE
        )
        m = pattern.search(text)
        if not m:
            continue
        start = m.end()
        next_header = re.search(r"^\[", text[start:], re.MULTILINE)
        end = start + next_header.start() if next_header else len(text)
        return start, end
    return None


def scan_toml_section(text: str, section_names):
    span = find_section_span(text, section_names)
    if not span:
        return None
    start, end = span
    body = text[start:end]
    m = re.search(r'(?m)^\s*version\s*=\s*"([^"]+)"', body)
    if not m:
        return None
    abs_start = start + m.start(1)
    abs_end = start + m.end(1)
    return m.group(1), abs_start, abs_end


def scan_json_root(text: str):
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    version = data.get("version")
    if not isinstance(version, str):
        return None
    # Find the first "version": "..." occurrence — for normal
    # package.json/composer.json files the root key comes before any
    # nested object could plausibly repeat it.
    m = re.search(r'"version"\s*:\s*"([^"]*)"', text)
    if not m or m.group(1) != version:
        return None
    return version, m.start(1), m.end(1)


def scan_plain(text: str):
    stripped = text.strip()
    m = VERSION_RE.fullmatch(stripped)
    if not m:
        return None
    start = text.find(stripped)
    return stripped, start, start + len(stripped)


def scan_gradle(text: str):
    m = re.search(r"(?m)^\s*version\s*=?\s*['\"]([^'\"]+)['\"]", text)
    if not m:
        return None
    return m.group(1), m.start(1), m.end(1)


def scan_props(text: str):
    m = re.search(r"(?m)^version\s*=\s*(.+?)\s*$", text)
    if not m:
        return None
    return m.group(1), m.start(1), m.end(1)


def scan_yaml_key(text: str):
    m = re.search(r"(?m)^version:\s*([^\s#]+)", text)
    if not m:
        return None
    return m.group(1), m.start(1), m.end(1)


def scan_csproj(text: str):
    m = re.search(r"<Version>([^<]+)</Version>", text)
    if not m:
        return None
    return m.group(1), m.start(1), m.end(1)


def scan_cmake(text: str):
    m = re.search(r"project\([^)]*VERSION\s+" + SEMVER_CORE, text)
    if not m:
        return None
    val = m.group(0).split("VERSION", 1)[1].strip()
    start = m.start() + m.group(0).rfind(val)
    return val, start, start + len(val)


def scan_pom(text: str):
    # Strip the <parent>...</parent> block first so we don't grab the
    # parent POM's version instead of this module's own.
    stripped = re.sub(r"<parent>.*?</parent>", "", text, flags=re.DOTALL)
    m = re.search(r"<version>([^<]+)</version>", stripped)
    if not m:
        return None
    val = m.group(1)
    # Re-find in the original text so offsets are correct.
    m2 = re.search(re.escape(f"<version>{val}</version>"), text)
    if not m2:
        return None
    inner_start = m2.start() + len("<version>")
    return val, inner_start, inner_start + len(val)


def scan_setup_py(text: str):
    m = re.search(r"version\s*=\s*['\"]([^'\"]+)['\"]", text)
    if not m:
        return None
    return m.group(1), m.start(1), m.end(1)


def scan_python_dunder(text: str):
    m = re.search(r"(?m)^__version__\s*=\s*['\"]([^'\"]+)['\"]", text)
    if not m:
        return None
    return m.group(1), m.start(1), m.end(1)


def scan(path: Path, kind: str):
    text = path.read_text(encoding="utf-8", errors="ignore")
    if kind == "json_root":
        result = scan_json_root(text)
    elif kind.startswith("toml_section:"):
        sections = kind.split(":", 1)[1].split(",")
        result = scan_toml_section(text, sections)
    elif kind.startswith("ini_section:"):
        section = kind.split(":", 1)[1]
        result = scan_toml_section(text, [section])  # same bracket-header shape
    elif kind == "gradle":
        result = scan_gradle(text)
    elif kind == "props":
        result = scan_props(text)
    elif kind == "yaml_key":
        result = scan_yaml_key(text)
    elif kind == "plain":
        result = scan_plain(text)
    elif kind == "csproj":
        result = scan_csproj(text)
    elif kind == "cmake":
        result = scan_cmake(text)
    elif kind == "pom":
        result = scan_pom(text)
    elif kind == "setup_py":
        result = scan_setup_py(text)
    elif kind == "python_dunder":
        result = scan_python_dunder(text)
    else:
        result = None
    return result


def cmd_detect(args):
    repo = Path(args.repo).resolve()
    findings = []
    for path, kind in iter_candidate_files(repo):
        try:
            result = scan(path, kind)
        except Exception as exc:  # noqa: BLE001 - report and move on
            findings.append({
                "file": str(path.relative_to(repo)),
                "kind": kind,
                "error": str(exc),
            })
            continue
        if result:
            value, _, _ = result
            findings.append({
                "file": str(path.relative_to(repo)),
                "kind": kind,
                "version": value,
            })
    print(json.dumps({"repo": str(repo), "findings": findings}, indent=2))


def cmd_apply(args):
    repo = Path(args.repo).resolve()
    files = [f.strip() for f in args.files.split(",") if f.strip()]
    changes = []
    for rel in files:
        path = repo / rel
        if not path.is_file():
            changes.append({"file": rel, "status": "error", "detail": "file not found"})
            continue
        name = path.name
        # Re-derive kind the same way iter_candidate_files would.
        kind = None
        for p, k in iter_candidate_files(repo):
            if p == path:
                kind = k
                break
        if kind is None:
            changes.append({"file": rel, "status": "error", "detail": "not a recognized manifest filename"})
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        result = scan(path, kind)
        if not result:
            changes.append({"file": rel, "status": "error", "detail": "could not locate a version field"})
            continue
        found_value, start, end = result
        if found_value != args.old:
            changes.append({
                "file": rel,
                "status": "error",
                "detail": f"file has version '{found_value}', expected '{args.old}' — not touching it",
            })
            continue
        new_text = text[:start] + args.new + text[end:]
        path.write_text(new_text, encoding="utf-8")
        changes.append({
            "file": rel,
            "status": "updated",
            "old": found_value,
            "new": args.new,
        })
    print(json.dumps({"repo": str(repo), "changes": changes}, indent=2))
    if any(c["status"] == "error" for c in changes):
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_detect = sub.add_parser("detect", help="scan repo for version-bearing files")
    p_detect.add_argument("--repo", required=True)
    p_detect.set_defaults(func=cmd_detect)

    p_apply = sub.add_parser("apply", help="write a new version into confirmed files")
    p_apply.add_argument("--repo", required=True)
    p_apply.add_argument("--old", required=True, help="expected current version, e.g. 2.2.1")
    p_apply.add_argument("--new", required=True, help="new version to write, e.g. 2.2.2")
    p_apply.add_argument("--files", required=True, help="comma-separated relative paths, from `detect`")
    p_apply.set_defaults(func=cmd_apply)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
