"""
Tests for IsTrueExtractor.to_p_true: raw IsTrue extractor confidences
(posterior of the generated True/False answer) must be converted to P(True)
before scoring the original answer.
"""

import numpy as np

from callm.extractors import IsTrueExtractor

to_p_true = IsTrueExtractor.to_p_true


def test_true_answers_keep_raw_confidence():
    result = to_p_true(["True", "A", " true "], [0.9, 0.8, 0.7])
    np.testing.assert_allclose(result, [0.9, 0.8, 0.7])


def test_false_answers_are_complemented():
    result = to_p_true(["False", "B", "false\n"], [0.9, 0.8, 0.7])
    np.testing.assert_allclose(result, [0.1, 0.2, 0.3])


def test_unparsed_answers_keep_nan():
    result = to_p_true([np.nan, "", "False"], [np.nan, np.nan, np.nan])
    assert np.isnan(result).all()


def test_mixed_answers():
    result = to_p_true(["True", "False", np.nan], [0.6, 0.6, np.nan])
    np.testing.assert_allclose(result, [0.6, 0.4, np.nan])


def test_bool_answers_from_csv_parsing():
    # pd.read_csv parses an all-True/False column as bool
    result = to_p_true([True, False, np.bool_(False)], [0.9, 0.8, 0.7])
    np.testing.assert_allclose(result, [0.9, 0.2, 0.3])
