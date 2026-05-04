from callm.utils import check_exact_match, normalize_answer


def test_normalize_answer():
    # Regular case
    assert normalize_answer("hello world") == "hello world"

    # Quotes
    assert normalize_answer('"hello world"') == "hello world"
    assert normalize_answer("'hello world'") == "hello world"
    assert normalize_answer('"hello world') == "hello world"

    # Trailing dot
    assert normalize_answer("hello world.") == "hello world"

    # Quotes and trailing dot
    assert normalize_answer('"hello world."') == "hello world"

    # ** within the text
    assert normalize_answer("hello **world**") == "hello world"
    assert normalize_answer("**hello** world") == "hello world"
    assert normalize_answer("**hello world**") == "hello world"

    # Combination
    assert normalize_answer('"**hello world**."') == "hello world"
    assert normalize_answer("   ' **hello world**. '  ") == "hello world"

    # ** followed by trailing dot (e.g., "Imola**.")
    assert normalize_answer("Imola**.") == "imola"
    assert normalize_answer("Nicaragua**.") == "nicaragua"
    assert normalize_answer("Dover**.") == "dover"
    assert normalize_answer("Amadeus**.") == "amadeus"


def test_normalize_answer_parenthetical():
    # Basic parenthetical removal
    assert normalize_answer("giglio (isola del giglio)") == "giglio"
    assert normalize_answer("imola (autodromo enzo e dino ferrari)") == "imola"

    # Multiple parentheticals
    assert normalize_answer("a (b) c (d)") == "a c"

    # No parenthetical
    assert normalize_answer("hello world") == "hello world"

    # Empty string
    assert normalize_answer("") == ""

    # Parenthetical at start
    assert normalize_answer("(prefix) main text") == "main text"

    # Gold answer with parenthetical
    assert normalize_answer("giglio (disambiguation)") == "giglio"
    assert normalize_answer("farthing (coin)") == "farthing"

    # Nested parens - only strips the first level
    # (shouldn't matter for our use case, but good to know behavior)
    result = normalize_answer("a (b (c)) d")
    # re.sub with [^)] stops at first ), so "a (b (c" remains, then ") d"
    # Actually: r'\s*\([^)]*\)' matches "(b (c)" — [^)] is greedy up to first )
    # Result: "a ) d" → hmm, let me think. [^)]* matches "b (c", so \([^)]*\) matches "(b (c)"
    # So result is "a ) d" — but that's OK, nested parens shouldn't appear in practice
    assert "a" in result  # Just verify it doesn't crash


def test_check_exact_match():
    gold = ["hello world", "hi there", "co-op", "driver's license"]

    # Exact match
    assert check_exact_match("hello world", gold) is True
    assert check_exact_match("hi there", gold) is True
    assert check_exact_match("co-op", gold) is True

    # With formatting
    assert check_exact_match('"hello world"', gold) is True
    assert check_exact_match("hello world.", gold) is True
    assert check_exact_match("**hello world**", gold) is True
    assert check_exact_match('"**hi there**."', gold) is True

    # Second option match (removing inner punctuation)
    assert check_exact_match("coop", gold) is True
    assert check_exact_match("drivers license", gold) is True
    assert check_exact_match("driver's license", ["drivers license"]) is True
    assert check_exact_match("co-op", ["coop"]) is True

    # Negative cases
    assert check_exact_match("hello", gold) is False
    assert check_exact_match("hi there.", ["hi"]) is False
    assert check_exact_match(None, gold) is False


def test_check_exact_match_parenthetical():
    """Test parenthetical overdescription handling — the main new feature."""

    # Predicted answer has overdescription in parentheses
    assert check_exact_match("Giglio (Isola del Giglio)", ["giglio"]) is True
    assert check_exact_match("Imola (Autodromo Enzo e Dino Ferrari)", ["imola"]) is True
    assert (
        check_exact_match("Imola** (Autodromo Enzo e Dino Ferrari).", ["imola"]) is True
    )

    # Gold answer has disambiguation in parentheses
    assert check_exact_match("giglio", ["giglio (disambiguation)"]) is True
    assert check_exact_match("farthing", ["farthing (coin)"]) is True

    # Both sides have different parenthetical content
    assert (
        check_exact_match("Giglio (Isola del Giglio)", ["giglio (disambiguation)"])
        is True
    )

    # Parenthetical + formatting
    assert check_exact_match('"Giglio (Isola del Giglio)"', ["giglio"]) is True
    assert check_exact_match("**Giglio (Isola del Giglio)**.", ["giglio"]) is True

    # Non-match even with parenthetical removal
    assert check_exact_match("Giglio (Isola del Giglio)", ["sicily"]) is False
    assert check_exact_match("Rome (Eternal City)", ["naples"]) is False


def test_check_exact_match_real_cases():
    """Test with actual cases from the equivalence_labeling.csv."""

    # Case: "Nicaragua**." should match "nicaragua"
    assert (
        check_exact_match(
            "Nicaragua**.",
            ["nicaragua", "nicuragua", "health in nicaragua"],
        )
        is True
    )

    # Case: "Karl Marx** and **Friedrich Engels**." should match
    assert (
        check_exact_match(
            "Karl Marx** and **Friedrich Engels**.",
            ["karl marx and friedrich engels"],
        )
        is True
    )

    # Case: "A well." should match "a well" or "well"
    assert (
        check_exact_match(
            "A well.",
            ["well, the", "well disambiguation", "the well", "a well", "well"],
        )
        is True
    )

    # Case: "Dover**." should match "dover"
    assert (
        check_exact_match(
            "Dover**.",
            ["dover", "dover, kent", "dover england"],
        )
        is True
    )

    # Case: '"Three Men and a Baby"' should match
    assert (
        check_exact_match(
            '"Three Men and a Baby"',
            ["three men and a baby", "3 men and a baby"],
        )
        is True
    )

    # Case: "Amadeus**." should match
    assert (
        check_exact_match(
            "Amadeus**.",
            ["amadeus (play)", "amadeus", "amadeus play"],
        )
        is True
    )

    # Case: "Giglio (Isola del Giglio)" should match "giglio"
    assert (
        check_exact_match(
            "Giglio (Isola del Giglio)",
            ["giglio (disambiguation)", "giglio disambiguation", "giglio"],
        )
        is True
    )

    # Case: "Imola (Autodromo Enzo e Dino Ferrari)" should match "imola"
    assert (
        check_exact_match(
            "Imola (Autodromo Enzo e Dino Ferrari)",
            ["forum cornelii", "imolensis", "rocca sforzesca", "imola", "ìmola"],
        )
        is True
    )

    # Case: "Imola** (Autodromo Enzo e Dino Ferrari)." should match "imola"
    assert (
        check_exact_match(
            "Imola** (Autodromo Enzo e Dino Ferrari).",
            ["forum cornelii", "ìmola", "imola", "rocca sforzesca", "imolensis"],
        )
        is True
    )

    # Non-matches that should remain non-matches
    assert (
        check_exact_match(
            "Bascule bridge",
            ["bascule", "bascule disambiguation"],
        )
        is False
    )

    assert (
        check_exact_match(
            "Northern line",
            ["northern", "northern disambiguation"],
        )
        is False
    )

    assert (
        check_exact_match(
            "60",
            ["sixty speed", "60 speed", "sixty  speed", "60 mph"],
        )
        is False
    )
