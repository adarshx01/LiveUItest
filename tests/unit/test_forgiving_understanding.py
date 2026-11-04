"""Typos, synonyms, dropdown options, picker values and memory keys (no browser needed)."""

from __future__ import annotations

import datetime as dt

import pytest

from backend.agents.forms.option_matcher import OptionInfo, match_option, split_choices
from backend.agents.forms.value_normalizer import normalize_for_input
from backend.agents.grounding.fuzzy import match_score
from backend.agents.grounding.text_match import similarity
from backend.agents.planner.requirement_analyzer import RequirementAnalyzer
from backend.agents.planner.spelling import correct_instruction
from backend.agents.planner.test_compiler import TestCompiler
from backend.agents.planner.test_ingestion import ingest
from backend.memory.memory_manager import action_signature


# ---------------------------------------------------------------- forgiving matcher
@pytest.mark.parametrize("query,label", [
    ("Sing In", "Sign In"), ("Loign", "Login"), ("Indai", "India"), ("Untied States", "United States"),
    ("dashbaord", "Dashboard"), ("usrname", "Username"), ("pasword", "Password"), ("Submit", "Sumbit"),
    ("Sign In", "Log in"), ("pwd", "Password"), ("qty", "Quantity"), ("Delete", "Remove"), ("Phone", "Mobile number"),
    ("e-mail", "Email"), ("Sing up", "Register"), ("Add to cart", "Add to basket")])
def test_typos_synonyms_and_abbreviations_match(query, label):
    assert match_score(query, label) >= 0.9


@pytest.mark.parametrize("query,label", [
    ("Save", "Same"), ("Save", "Sale"), ("Edit", "Exit"), ("from", "form"), ("Sign In", "Sign Up"),
    ("Log In", "Logout"), ("Item 1", "Item 7"), ("Indai", "Indonesia"), ("Apply", "Apple"), ("Sign In", "Sign Out")])
def test_different_words_are_never_typos_of_each_other(query, label):
    assert match_score(query, label) < 0.7


def test_exact_always_outranks_repaired_and_strict_similarity_is_unchanged():
    assert match_score("Sign In", "Sign In") > match_score("Sing In", "Sign In")
    assert match_score("Login", "Login") > match_score("Login", "Sign in")
    # assertions keep using the strict comparison: a typo in an expected value must not pass
    assert similarity("Order confirmed", "Order confirmd") < 0.95


def test_ambiguous_typo_is_not_resolved_to_another_meaning():
    # "usrname" is one slip from both username and surname: the matcher must not prefer last name
    assert match_score("usrname", "Last name") < 0.5


# ---------------------------------------------------------------- spelling repair of instructions
@pytest.mark.parametrize("line,expected", [
    ("Clik on the Loign button", "Click on the Login button"),
    ("Entr usrname as abc", "Enter username as abc"),
    ("Clcik Sing In", "Click Sign In"),
    ("Verfy that the user is redirected to the dashbaord", "Verify that the user is redirected to the dashboard"),
    ("Selct Indai from the contry dropdown", "Select Indai from the country dropdown"),
    ("Doubel clik the row", "Double click the row"),
    ("Chek the Remeber me checkbox", "Check the Remember me checkbox"),
    ("Navigte to the setings page", "Navigate to the settings page"),
    ("Verify the eror mesage is shown", "Verify the error message is shown")])
def test_instruction_grammar_is_repaired(line, expected):
    assert correct_instruction(line).text == expected


@pytest.mark.parametrize("line", [
    "Enter usrname as Passwrd",          # value after "as"
    "Type pasword as {{test_password}}",  # variable
    "Verify that 'Welcom' is dispalyed",  # quoted expectation
    "Verify Welcom back is dispalyed",    # unquoted expectation
    "Search for Laptpo",                  # search term
    "Selct Indai from the contry dropdown",  # option (matched against the real list at run time)
])
def test_data_is_never_rewritten(line):
    fixed = correct_instruction(line).text
    for word in ("Passwrd", "{{test_password}}", "Welcom", "Laptpo", "Indai"):
        if word in line:
            assert word in fixed


async def test_misspelt_test_parses_like_the_clean_one():
    (typo,) = ingest("URL: https://x.test\n1. Clik on the Loign button\n2. Entr usrname as abc\n3. Clcik Sing In")
    (clean,) = ingest("URL: https://x.test\n1. Click on the Login button\n2. Enter username as abc\n3. Click Sign In")
    a = TestCompiler().compile(await RequirementAnalyzer().analyze(typo)).flat_steps()
    b = TestCompiler().compile(await RequirementAnalyzer().analyze(clean)).flat_steps()
    assert [(s.action, s.target.purpose if s.target else None) for s in a] == \
           [(s.action, s.target.purpose if s.target else None) for s in b]
    assert any(n.startswith("spelling:") for n in a[1].notes)
    assert a[1].original_text == "Clik on the Loign button"  # the report keeps the author's words


# ---------------------------------------------------------------- dropdown options
COUNTRIES = [{"label": "-- Select a country --", "value": ""}, {"label": "India", "value": "IN", "group": "Asia"},
             {"label": "United States", "value": "US"}, {"label": "United Kingdom", "value": "GB"},
             {"label": "Antarctica (unavailable)", "value": "AQ", "disabled": True}]


@pytest.mark.parametrize("wanted,label,how", [
    ("India", "India", "exact"), ("india", "India", "case_insensitive"), ("IN", "India", "value"),
    ("USA", "United States", "alias"), ("Indai", "India", "fuzzy"), ("Untied Kingdom", "United Kingdom", "fuzzy"),
    ("the second option", "United States", "ordinal")])
def test_option_matching_explains_itself(wanted, label, how):
    m = match_option(COUNTRIES, wanted)
    assert (m.label, m.how) == (label, how)


def test_disabled_and_placeholder_options():
    assert match_option(COUNTRIES, "Antarctica").option.disabled  # reported, not silently skipped
    assert match_option(COUNTRIES, "select a").label is None  # placeholders never win by fuzzy rules
    assert OptionInfo.coerce(COUNTRIES)[0].is_placeholder


def test_no_closest_guess_when_ambiguous():
    m = match_option(["United States", "United Kingdom"], "United")
    assert not m.found and m.how == "ambiguous"


def test_multi_choice_splitting_respects_real_options():
    assert split_choices("English and Malayalam", ["English", "Malayalam"]) == ["English", "Malayalam"]
    assert split_choices("Salt and pepper", ["Salt and pepper", "Salt"]) == ["Salt and pepper"]


# ---------------------------------------------------------------- browser-rendered pickers
TODAY = dt.date(2026, 11, 4)


@pytest.mark.parametrize("value,kind,kwargs,expected", [
    ("12/24/2026", "date", {}, "2026-12-24"), ("24/12/2026", "date", {}, "2026-12-24"),
    ("1 May 1990", "date", {}, "1990-05-01"), ("tomorrow", "date", {}, "2026-11-05"),
    ("2:30 PM", "time", {}, "14:30"), ("2024-05-01 14:30", "datetime-local", {}, "2024-05-01T14:30"),
    ("May 2024", "month", {}, "2024-05"), ("red", "color", {}, "#ff0000"), ("#F00", "color", {}, "#ff0000"),
    ("50%", "range", {"min_value": "0", "max_value": "200", "step": "5"}, "100"),
    ("2530", "range", {"min_value": "0", "max_value": "5000", "step": "100"}, "2500"),
    ("1,000", "number", {}, "1000")])
def test_picker_values_are_normalized(value, kind, kwargs, expected):
    out = normalize_for_input(value, kind, today=TODAY, **kwargs)
    assert out.ok and out.value == expected


def test_uninterpretable_picker_value_fails_with_a_reason():
    out = normalize_for_input("next blue moon", "date", today=TODAY)
    assert not out.ok and "could not interpret" in (out.note or "")


# ---------------------------------------------------------------- memory keys
async def test_steps_sharing_a_purpose_get_distinct_memory_keys():
    (case,) = ingest("URL: https://x.test\n1. Enter email as a@b.c\n2. Enter the newsletter email as a@b.c\n"
                     "3. Click on the Login button\n4. Click on the Sign In button")
    steps = TestCompiler().compile(await RequirementAnalyzer().analyze(case)).flat_steps()
    keys = [action_signature(s) for s in steps if s.target is not None and s.action.value != "navigate"]
    assert keys[0] != keys[1]  # the newsletter field must not reuse the shipping email's locator
    assert keys[2] == keys[3] == "click:submit_login"  # same meaning, same memory
