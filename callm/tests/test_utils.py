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
