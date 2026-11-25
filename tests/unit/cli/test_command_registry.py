"""Tests for the shared slash-command registry and resolver."""

from __future__ import annotations

import pytest

from gemini_cli.cli.slash_commands import (
    COMMANDS,
    CommandAction,
    CommandProblem,
    CommandSpec,
    ResolvedCommand,
    SlashCommand,
    _build_index,
    format_command_problem,
    format_unexpected_argument,
    parse_slash_command,
    render_command_help,
    render_help,
    resolve_command,
    resolve_command_name,
)


def resolve(text: str) -> ResolvedCommand | CommandProblem:
    command = parse_slash_command(text)
    assert command is not None
    return resolve_command(command)


def require_resolved(text: str) -> ResolvedCommand:
    result = resolve(text)
    assert isinstance(result, ResolvedCommand)
    return result


def require_problem(text: str) -> CommandProblem:
    result = resolve(text)
    assert isinstance(result, CommandProblem)
    return result


def test_every_canonical_name_resolves_to_its_command() -> None:
    for spec in COMMANDS:
        result = require_resolved(f"/{spec.name}")

        assert result.spec is spec
        assert result.argument == ""


@pytest.mark.parametrize(
    ("text", "action"),
    [
        ("/models", CommandAction.MODEL),
        ("/roles", CommandAction.ROLE),
        ("/themes", CommandAction.THEME),
        ("/ctx", CommandAction.CONTEXT),
        ("/window", CommandAction.CONTEXT),
        ("/h", CommandAction.HELP),
        ("/?", CommandAction.HELP),
        ("/m", CommandAction.MODEL),
        ("/t", CommandAction.THEME),
        ("/q", CommandAction.EXIT),
        ("/quit", CommandAction.EXIT),
        ("/bye", CommandAction.EXIT),
        ("/cls", CommandAction.CLEAR),
        ("/new", CommandAction.CLEAR),
        ("/logout", CommandAction.RESET),
    ],
)
def test_aliases_resolve_to_the_expected_command(
    text: str,
    action: CommandAction,
) -> None:
    assert require_resolved(text).spec.action is action


@pytest.mark.parametrize(
    ("text", "action"),
    [
        ("/mod", CommandAction.MODEL),
        ("/rol", CommandAction.ROLE),
        ("/the", CommandAction.THEME),
        ("/con", CommandAction.CONTEXT),
        ("/qui", CommandAction.EXIT),
        ("/win", CommandAction.CONTEXT),
    ],
)
def test_unique_prefixes_resolve(text: str, action: CommandAction) -> None:
    assert require_resolved(text).spec.action is action


def test_exact_alias_wins_over_another_command_prefix() -> None:
    assert require_resolved("/q").spec.action is CommandAction.EXIT


@pytest.mark.parametrize("text", ["/r", "/c"])
def test_ambiguous_prefix_is_reported(text: str) -> None:
    problem = require_problem(text)

    assert problem.candidates
    assert problem.suggestion is None


def test_ambiguous_prefix_lists_canonical_candidates() -> None:
    problem = require_problem("/r")

    assert {spec.action for spec in problem.candidates} == {
        CommandAction.ROLE,
        CommandAction.RESET,
    }


def test_unknown_command_suggests_a_close_match() -> None:
    problem = require_problem("/modl")

    assert problem.suggestion is not None
    assert problem.suggestion.action is CommandAction.MODEL


def test_unknown_command_without_close_match_has_no_suggestion() -> None:
    problem = require_problem("/zzzzzz")

    assert problem.suggestion is None


def test_resolution_is_case_insensitive() -> None:
    assert require_resolved("/MODELS").spec.action is CommandAction.MODEL


def test_arguments_are_preserved() -> None:
    result = require_resolved("/model gemini-3.8-flash")

    assert result.spec.action is CommandAction.MODEL
    assert result.argument == "gemini-3.8-flash"


def test_parser_accepts_tabs_and_extra_spaces() -> None:
    assert parse_slash_command("/MODEL\tgemini-3.8-flash") == SlashCommand(
        "/model",
        "gemini-3.8-flash",
    )
    assert parse_slash_command("  /help  ") == SlashCommand("/help")


def test_help_lists_canonical_names_and_aliases() -> None:
    text = render_help()

    assert "Commands:" in text
    assert "/model" in text
    assert "/models" in text
    assert "/quit" in text
    assert "/ctx" in text
    assert "/mod -> /model" in text


def test_help_shows_argument_usage() -> None:
    text = render_help()

    assert "/model [name]" in text
    assert "/role [role]" in text
    assert "/theme [dark|light]" in text
    assert "/help [command]" in text


def test_command_help_describes_arguments_and_aliases() -> None:
    resolved = resolve_command_name("model")
    assert isinstance(resolved, ResolvedCommand)

    text = render_command_help(resolved.spec)

    assert "/model [name]" in text
    assert "Change the model." in text
    assert "/models" in text
    assert "open the selection menu" in text


def test_help_resolves_command_aliases() -> None:
    resolved = resolve_command_name("/models")

    assert isinstance(resolved, ResolvedCommand)
    assert resolved.spec.action is CommandAction.MODEL


def test_help_reports_unknown_commands_with_a_suggestion() -> None:
    problem = resolve_command_name("modl")

    assert isinstance(problem, CommandProblem)
    assert problem.suggestion is not None
    assert problem.suggestion.action is CommandAction.MODEL


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("/model gemini-4-argon", None),
        ("/role code", None),
        ("/theme dark", None),
        ("/help model", None),
        ("/clear extra", "/clear does not take arguments."),
        ("/exit now", "/exit does not take arguments."),
    ],
)
def test_unexpected_arguments_are_reported(
    text: str,
    expected: str | None,
) -> None:
    result = require_resolved(text)

    assert format_unexpected_argument(result.spec, result.argument) == expected


def test_all_spellings_are_unique() -> None:
    spellings = [spelling for spec in COMMANDS for spelling in spec.spellings]

    assert len(spellings) == len(set(spellings))


def test_every_action_has_exactly_one_spec() -> None:
    actions = [spec.action for spec in COMMANDS]

    assert len(actions) == len(set(actions))
    assert set(actions) == set(CommandAction)


def test_build_index_rejects_duplicate_spellings() -> None:
    specs = (
        CommandSpec(CommandAction.HELP, "help"),
        CommandSpec(CommandAction.EXIT, "help"),
    )

    with pytest.raises(ValueError, match="Duplicate"):
        _build_index(specs)


def test_problem_messages_are_actionable() -> None:
    ambiguous = format_command_problem(require_problem("/c"))
    unknown = format_command_problem(require_problem("/modl"))

    assert "ambiguous" in ambiguous
    assert "/clear" in ambiguous
    assert "/context" in ambiguous
    assert "Did you mean /model?" in unknown
