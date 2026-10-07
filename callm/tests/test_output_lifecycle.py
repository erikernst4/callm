"""
Characterization tests for the output lifecycle of ``BaseLightningModule``
subclasses: what each module writes at epoch end (CSV content, logged metrics)
and how flushed/resumed temp files are named and merged.
"""

import csv
import math
from unittest.mock import Mock, patch

import pytest
import torch
from ecuas import (
    ConfidenceAUCScore,
    ConfidenceBrierScore,
    ConfidenceCrossEntropy,
    ConfidenceECUAS,
    ExpectedCalibrationError,
)

from callm.models.evaluator import EvaluatorModule
from callm.models.exact_answer_extractor import ExactAnswerExtractorModule
from callm.models.gcp_evaluator import GCPEvaluatorModule
from callm.models.gcp_llm import GCPLLM
from callm.models.llm import LLM

ANSWER_METRICS = {
    "val_ece": lambda: ExpectedCalibrationError(n_bins=10),
    "val_brier_score": ConfidenceBrierScore,
    "val_cross_entropy": ConfidenceCrossEntropy,
    "val_auc": ConfidenceAUCScore,
    "val_ccas": lambda: ConfidenceECUAS(n=0),
}


def _attach_trainer(module, log_dir, requires_semantic_equivalence=False):
    module._trainer = Mock(
        log_dir=str(log_dir),
        global_rank=0,
        datamodule=Mock(requires_semantic_equivalence=requires_semantic_equivalence),
    )
    module.log = Mock()


def _logged(module) -> dict:
    return {c.args[0]: float(c.args[1]) for c in module.log.call_args_list}


def _expected_metrics(confidences, correctness) -> dict:
    conf = torch.tensor(confidences, dtype=torch.float32)
    corr = torch.tensor(correctness, dtype=torch.float32)
    expected = {"val_accuracy": float(corr.mean())}
    for name, factory in ANSWER_METRICS.items():
        metric = factory()
        metric.update(conf, corr)
        expected[name] = float(metric.compute())
    return expected


def _assert_metrics(logged: dict, expected: dict):
    assert logged.keys() == expected.keys()
    for name, value in expected.items():
        assert logged[name] == pytest.approx(value, nan_ok=True), name


def _read_csv(path) -> list[list[str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


# ---------------------------------------------------------------------------
# Module factories
# ---------------------------------------------------------------------------


@pytest.fixture
def make_llm():
    with (
        patch("callm.models.llm.initialize_model") as init_model,
        patch("callm.models.llm.get_tokenizer_for_model") as get_tokenizer,
    ):

        def _make(is_seq2seq=False, config=None, tokenizer=None, **kwargs):
            model = Mock()
            model.parameters.return_value = iter([])
            model.config = config or Mock(
                spec=["pad_token_id", "eos_token_id"], pad_token_id=0, eos_token_id=1
            )
            init_model.return_value = (model, is_seq2seq)
            get_tokenizer.return_value = tokenizer or Mock()
            kwargs.setdefault("extractor", Mock())
            return LLM(model_name="m", **kwargs)

        yield _make


@pytest.fixture
def make_gcp_llm():
    with patch("callm.models.gcp_llm.genai"):
        yield lambda **kwargs: GCPLLM(**{"extractor": Mock(), **kwargs})


@pytest.fixture
def make_evaluator():
    with (
        patch("callm.models.evaluator.initialize_model") as init_model,
        patch("callm.models.evaluator.get_tokenizer_for_model") as get_tokenizer,
    ):
        model = Mock()
        model.parameters.return_value = iter([])
        model.config = Mock(pad_token_id=0, eos_token_id=1)
        init_model.return_value = (model, False)
        tokenizer = Mock()
        tokenizer.decode.side_effect = lambda ids, **_: {1: " Yes.", 2: "no"}[
            int(ids[0])
        ]
        get_tokenizer.return_value = tokenizer
        yield lambda **kwargs: EvaluatorModule(**kwargs)


@pytest.fixture
def make_gcp_evaluator():
    with patch("callm.models.gcp_evaluator.genai"):
        yield lambda **kwargs: GCPEvaluatorModule(**kwargs)


@pytest.fixture
def make_exact_extractor():
    with patch("callm.models.exact_answer_extractor.genai"):
        yield lambda **kwargs: ExactAnswerExtractorModule(**kwargs)


# ---------------------------------------------------------------------------
# Sample outputs (as stored by validation_step) and expected CSVs
# ---------------------------------------------------------------------------

LLM_HEADER = ["question", "gold_answers", "choices", "pred_answer", "confidence"]


def _llm_outputs():
    return [
        {
            "output_ids": torch.tensor([1]),
            "question": "q1",
            "gold_answers": ["A", "a"],
            "choices": "A) x",
        },
        {"output_ids": torch.tensor([2]), "question": "q2", "gold_answers": ["B"]},
        {"output_ids": torch.tensor([3]), "question": "q3", "gold_answers": []},
    ]


def _llm_extractor():
    extractor = Mock()
    extractor.side_effect = [("A", 0.9), (" b ", float("nan")), (None, None)]
    return extractor


def _llm_tokenizer():
    tokenizer = Mock()
    tokenizer.decode.side_effect = lambda ids, **_: {
        1: "line1\nline2 is long",
        2: "short",
        3: "",
    }[int(ids[0])]
    return tokenizer


def _evaluator_outputs():
    # Unsorted on purpose: the evaluators sort by index before writing
    return [
        {
            "index": 2,
            "question": "q3",
            "gold_answers": ["C"],
            "pred_answer": "z",
            "confidence": "nan",
            "raw_output": "r3",
            "exact_match": False,
            "output_ids": torch.tensor([2]),
        },
        {
            "index": 0,
            "question": "q1\nmore",
            "gold_answers": ["A"],
            "pred_answer": "A",
            "confidence": 0.8,
            "raw_output": "r" * 250,
            "exact_match": True,
            "output_ids": None,
        },
        {
            "index": 1,
            "question": "q2",
            "gold_answers": ["B"],
            "pred_answer": "b!",
            "confidence": 0.4,
            "raw_output": "r2",
            "exact_match": False,
            "output_ids": torch.tensor([1]),
        },
        {
            "index": 3,
            "question": "q4",
            "gold_answers": ["D"],
            "pred_answer": "d",
            "confidence": 0.1,
            "raw_output": "",
            "exact_match": False,
            "output_ids": None,
        },
    ]


def _gcp_evaluator_outputs():
    return [
        {
            "index": 1,
            "question": "q2",
            "gold_answers": ["B"],
            "pred_answer": "b!",
            "confidence": 0.4,
            "raw_output": "r2\nx",
            "exact_match": False,
            "cached_correct": None,
            "evaluator_response_raw": "  YES it is",
        },
        {
            "index": 0,
            "question": "q1",
            "gold_answers": ["A"],
            "pred_answer": "A",
            "confidence": 0.8,
            "raw_output": "r1",
            "exact_match": True,
            "cached_correct": None,
            "evaluator_response_raw": None,
        },
        {
            "index": 2,
            "question": "q3",
            "gold_answers": ["C"],
            "pred_answer": "z",
            "confidence": None,
            "raw_output": "r3",
            "exact_match": False,
            "cached_correct": 0,
            "evaluator_response_raw": "cached\nresp",
        },
        {
            "index": 3,
            "question": "q4",
            "gold_answers": ["D"],
            "pred_answer": "d",
            "confidence": 0.1,
            "raw_output": "r4",
            "exact_match": False,
            "cached_correct": None,
            "evaluator_response_raw": "No",
        },
    ]


def _exact_extractor_outputs():
    return [
        {
            "index": 1,
            "question": "q2\nx",
            "gold_answers": "B",
            "pred_answer": None,
            "correct": False,
            "raw_output": "r2\ny",
            "confidence": None,
            "gemini_output": "g\nout",
            "exact_answer": "NO ANSWER",
            "valid_exact_answer": False,
        },
        {
            "index": 0,
            "question": "q1",
            "gold_answers": ["A", "a"],
            "pred_answer": "A",
            "correct": True,
            "raw_output": "The answer is A",
            "confidence": 0.75,
            "gemini_output": "",
            "exact_answer": "A",
            "valid_exact_answer": True,
        },
    ]


EVALUATION_HEADER = [
    "Question",
    "Gold Answers",
    "Predicted Answer",
    "Confidence",
    "Correct",
    "Evaluator Response",
    "Raw Output",
]


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------


class TestLLMEpochEnd:
    def test_writes_outputs_and_exact_match_metrics(self, make_llm, tmp_path):
        llm = make_llm(
            extractor=_llm_extractor(),
            tokenizer=_llm_tokenizer(),
            raw_output_max_length=10,
        )
        _attach_trainer(llm, tmp_path)
        llm.outputs = _llm_outputs()

        llm.on_validation_epoch_end()

        assert _read_csv(tmp_path / "llm_outputs.csv") == [
            LLM_HEADER + ["raw_output"],
            ["q1", "A|a", "A) x", "A", "0.900000", "line1\\nline..."],
            ["q2", "B", "", " b ", "nan", "short"],
            ["q3", "", "", "", "nan", ""],
        ]
        # Extractor gets (raw_output, logits, output_ids)
        first_call = llm.extractor.call_args_list[0]
        assert first_call.args[0] == "line1\nline2 is long"
        assert first_call.args[1] is None
        assert torch.equal(first_call.args[2], torch.tensor([1]))

        # Letter match (strip + upper): q1 and q2 correct, q3 wrong
        expected = _expected_metrics([0.9, math.nan, math.nan], [1.0, 1.0, 0.0])
        _assert_metrics(_logged(llm), expected)
        rows = _read_csv(tmp_path / "metrics.csv")
        assert rows[0] == list(expected.keys())
        assert [float(v) for v in rows[1]] == pytest.approx(
            list(expected.values()), nan_ok=True
        )
        assert llm.outputs == []

    def test_no_choices_column_without_choices(self, make_llm, tmp_path):
        llm = make_llm(extractor=_llm_extractor(), tokenizer=_llm_tokenizer())
        _attach_trainer(llm, tmp_path)
        llm.outputs = [o for o in _llm_outputs() if "choices" not in o]
        llm.extractor.side_effect = [("B", 0.5), ("C", 0.5)]

        llm.on_validation_epoch_end()

        rows = _read_csv(tmp_path / "llm_outputs.csv")
        assert rows[0] == [
            "question",
            "gold_answers",
            "pred_answer",
            "confidence",
            "raw_output",
        ]
        # Without raw_output_max_length nothing is truncated
        assert rows[1] == ["q2", "B", "B", "0.500000", "short"]

    def test_semantic_equivalence_skips_metrics(self, make_llm, tmp_path):
        llm = make_llm(extractor=_llm_extractor(), tokenizer=_llm_tokenizer())
        _attach_trainer(llm, tmp_path, requires_semantic_equivalence=True)
        llm.outputs = _llm_outputs()

        llm.on_validation_epoch_end()

        assert (tmp_path / "llm_outputs.csv").exists()
        assert not (tmp_path / "metrics.csv").exists()
        llm.log.assert_not_called()

    def test_no_outputs_writes_nothing(self, make_llm, tmp_path):
        llm = make_llm()
        _attach_trainer(llm, tmp_path)
        llm.on_validation_epoch_end()
        assert list(tmp_path.iterdir()) == []


class TestLLMValidationStep:
    def test_stores_per_token_log_probs(self, make_llm):
        torch.manual_seed(0)
        llm = make_llm(return_logits=True)
        input_ids = torch.tensor([[5, 6], [7, 8]])
        sequences = torch.tensor([[5, 6, 1, 2, 3], [7, 8, 4, 0, 0]])
        scores = tuple(torch.randn(2, 10) for _ in range(3))
        llm.forward = Mock(return_value=Mock(sequences=sequences, scores=scores))

        llm.validation_step(
            {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "question": ["q1", "q2"],
                "gold_answers": [["a"], ["b"]],
                "choices": ["c1", "c2"],
            },
            0,
        )

        assert [o["choices"] for o in llm.outputs] == ["c1", "c2"]
        for i, out in enumerate(llm.outputs):
            assert torch.equal(out["output_ids"], sequences[i, 2:])
            expected = torch.stack(
                [
                    torch.log_softmax(scores[s][i], dim=-1)[sequences[i, 2 + s]]
                    for s in range(3)
                ]
            )
            assert torch.allclose(out["logits"], expected)

    def test_gold_answers_from_label_or_none(self, make_llm):
        llm = make_llm()
        llm.forward = Mock(return_value=torch.tensor([[1], [2]]))
        batch = {
            "input_ids": torch.zeros(2, 0, dtype=torch.long),
            "attention_mask": torch.zeros(2, 0, dtype=torch.long),
            "question": ["q1", "q2"],
        }
        llm.validation_step(batch, 0)
        assert [o["gold_answers"] for o in llm.outputs] == [None, None]
        assert all("logits" not in o and "choices" not in o for o in llm.outputs)

        llm.outputs = []
        llm.validation_step({**batch, "label": ["A", "B"]}, 0)
        assert [o["gold_answers"] for o in llm.outputs] == ["A", "B"]


class TestLLMForwardAndInit:
    def test_generate_false_returns_hidden_states(self, make_llm):
        llm = make_llm()
        llm.model.return_value = Mock(hidden_states=("h0", "h1"))
        ids, mask = torch.tensor([[1]]), torch.tensor([[1]])

        assert llm.forward(ids, mask, generate=False) == ("h0", "h1")
        llm.model.assert_called_once_with(
            input_ids=ids,
            attention_mask=mask,
            output_hidden_states=True,
            return_dict=True,
        )
        llm.model.generate.assert_not_called()

    def test_generate_with_hidden_states(self, make_llm):
        llm = make_llm(max_new_tokens=7)
        llm.model.generate.return_value = "generated"
        llm.model.return_value = Mock(hidden_states=("h0",))
        ids, mask = torch.tensor([[1]]), torch.tensor([[1]])

        assert llm.forward(ids, mask, return_hidden_states=True) == (
            "generated",
            ("h0",),
        )
        llm.model.generate.assert_called_once_with(
            input_ids=ids,
            attention_mask=mask,
            max_new_tokens=7,
            do_sample=False,
            output_scores=False,
            return_dict_in_generate=False,
            pad_token_id=0,
        )

    def test_pad_token_id_falls_back_to_tokenizer(self, make_llm):
        config = Mock(spec=["pad_token_id", "eos_token_id"])
        config.pad_token_id, config.eos_token_id = None, None
        llm = make_llm(config=config, tokenizer=Mock(pad_token_id=5))
        llm.forward(torch.tensor([[1]]), torch.tensor([[1]]))
        assert llm.model.generate.call_args.kwargs["pad_token_id"] == 5

    def test_seq2seq_does_not_pass_pad_token_id(self, make_llm):
        llm = make_llm(is_seq2seq=True)
        llm.forward(torch.tensor([[1]]), torch.tensor([[1]]))
        assert "pad_token_id" not in llm.model.generate.call_args.kwargs

    def test_pad_token_id_init(self, make_llm):
        config = Mock(spec=["pad_token_id", "eos_token_id"])
        config.pad_token_id, config.eos_token_id = None, 3
        assert make_llm(config=config).model.config.pad_token_id == 3

        config = Mock(pad_token_id=None, eos_token_id=3)
        config.text_config.pad_token_id = 9
        assert make_llm(config=config).model.config.pad_token_id == 9

    def test_train_flag_controls_freezing(self, make_llm):
        param = torch.nn.Parameter(torch.zeros(1))
        with (
            patch("callm.models.llm.initialize_model") as init_model,
            patch("callm.models.llm.get_tokenizer_for_model"),
        ):
            model = Mock()
            model.parameters.return_value = iter([param])
            model.config = Mock(spec=["pad_token_id", "eos_token_id"], pad_token_id=0)
            init_model.return_value = (model, False)
            LLM(extractor=Mock(), train=True)
            assert param.requires_grad
            model.eval.assert_not_called()

            model.parameters.return_value = iter([param])
            LLM(extractor=Mock(), train=False)
            assert not param.requires_grad
            model.eval.assert_called_once()

    def test_strip_prompt(self, make_llm):
        ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        prompt = torch.zeros(2, 2)
        assert [t.tolist() for t in make_llm().strip_prompt(ids, prompt)] == [
            [3],
            [6],
        ]
        assert [
            t.tolist() for t in make_llm(is_seq2seq=True).strip_prompt(ids, prompt)
        ] == [[1, 2, 3], [4, 5, 6]]


# ---------------------------------------------------------------------------
# GCPLLM
# ---------------------------------------------------------------------------


class TestGCPLLMEpochEnd:
    def test_writes_outputs_and_exact_match_metrics(self, make_gcp_llm, tmp_path):
        llm = make_gcp_llm(extractor=_llm_extractor())
        _attach_trainer(llm, tmp_path)
        llm.outputs = [
            {
                "question": "q1",
                "gold_answers": ["A", "a"],
                "choices": "A) x",
                "raw_output": "x" * 250,
                "logits": None,
            },
            {"question": "q2", "gold_answers": ["B"], "raw_output": "s\nt"},
            {"question": "q3", "gold_answers": [], "raw_output": ""},
        ]

        llm.on_validation_epoch_end()

        assert _read_csv(tmp_path / "llm_outputs.csv") == [
            LLM_HEADER + ["raw_output"],
            ["q1", "A|a", "A) x", "A", "0.900000", "x" * 200 + "..."],
            ["q2", "B", "", " b ", "nan", "s\\nt"],
            ["q3", "", "", "", "nan", ""],
        ]
        assert llm.extractor.call_args_list[0].args == ("x" * 250, None, None)

        expected = _expected_metrics([0.9, math.nan, math.nan], [1.0, 1.0, 0.0])
        _assert_metrics(_logged(llm), expected)
        assert _read_csv(tmp_path / "metrics.csv")[0] == list(expected.keys())
        assert llm.outputs == []

    def test_semantic_equivalence_skips_metrics(self, make_gcp_llm, tmp_path):
        llm = make_gcp_llm(extractor=_llm_extractor())
        _attach_trainer(llm, tmp_path, requires_semantic_equivalence=True)
        llm.outputs = [{"question": "q", "gold_answers": ["A"], "raw_output": "r"}]
        llm.on_validation_epoch_end()
        assert not (tmp_path / "metrics.csv").exists()


# ---------------------------------------------------------------------------
# Evaluators and exact answer extractor
# ---------------------------------------------------------------------------


class TestEvaluatorEpochEnd:
    def test_writes_sorted_results_and_metrics(self, make_evaluator, tmp_path):
        evaluator = make_evaluator()
        _attach_trainer(evaluator, tmp_path)
        evaluator.outputs = _evaluator_outputs()

        evaluator.on_validation_epoch_end()

        assert _read_csv(tmp_path / "evaluation_results.csv") == [
            EVALUATION_HEADER,
            ["q1\\nmore", "['A']", "A", "0.800000", "Yes", "", "r" * 200 + "..."],
            ["q2", "['B']", "b!", "0.400000", "Yes", " Yes.", "r2"],
            ["q3", "['C']", "z", "nan", "No", "no", "r3"],
            ["q4", "['D']", "d", "0.100000", "No", "", ""],
        ]
        _assert_metrics(
            _logged(evaluator),
            _expected_metrics([0.8, 0.4, math.nan, 0.1], [1.0, 1.0, 0.0, 0.0]),
        )
        assert evaluator.outputs == []


class TestGCPEvaluatorEpochEnd:
    def test_writes_sorted_results_and_metrics(self, make_gcp_evaluator, tmp_path):
        evaluator = make_gcp_evaluator()
        _attach_trainer(evaluator, tmp_path)
        evaluator.outputs = _gcp_evaluator_outputs()

        evaluator.on_validation_epoch_end()

        assert _read_csv(tmp_path / "evaluation_results.csv") == [
            EVALUATION_HEADER,
            ["q1", "['A']", "A", "0.800000", "Yes", "", "r1"],
            ["q2", "['B']", "b!", "0.400000", "Yes", "  YES it is", "r2\\nx"],
            ["q3", "['C']", "z", "nan", "No", "cached\\nresp", "r3"],
            ["q4", "['D']", "d", "0.100000", "No", "No", "r4"],
        ]
        _assert_metrics(
            _logged(evaluator),
            _expected_metrics([0.8, 0.4, math.nan, 0.1], [1.0, 1.0, 0.0, 0.0]),
        )
        assert evaluator.outputs == []


class TestExactAnswerExtractorEpochEnd:
    def test_writes_sorted_results(self, make_exact_extractor, tmp_path):
        extractor = make_exact_extractor()
        _attach_trainer(extractor, tmp_path)
        extractor.outputs = _exact_extractor_outputs()

        extractor.on_validation_epoch_end()

        assert _read_csv(tmp_path / "evaluation_results_exact.csv") == [
            [
                "Question",
                "Gold Answers",
                "Predicted Answer",
                "Confidence",
                "Correct",
                "Gemini Output",
                "Exact Answer",
                "Valid Exact Answer",
                "Raw Output",
            ],
            [
                "q1",
                "['A', 'a']",
                "A",
                "0.750000",
                "Yes",
                "",
                "A",
                "Yes",
                "The answer is A",
            ],
            ["q2\\nx", "B", "", "nan", "No", "g\\nout", "NO ANSWER", "No", "r2\\ny"],
        ]
        assert extractor.outputs == []


# ---------------------------------------------------------------------------
# Flushing and resuming
# ---------------------------------------------------------------------------

FLUSH_CASES = [
    ("make_llm", "temp_val_outputs", "llm_outputs.csv"),
    ("make_gcp_llm", "temp_val_outputs", "llm_outputs.csv"),
    ("make_evaluator", "temp_eval_results", "evaluation_results.csv"),
    ("make_gcp_evaluator", "temp_eval_results", "evaluation_results.csv"),
    (
        "make_exact_extractor",
        "temp_exact_answer_results",
        "evaluation_results_exact.csv",
    ),
]

SAMPLE_OUTPUTS = {
    "make_llm": _llm_outputs,
    "make_gcp_llm": lambda: [
        {"question": "q", "gold_answers": ["A"], "raw_output": "r"}
    ],
    "make_evaluator": _evaluator_outputs,
    "make_gcp_evaluator": _gcp_evaluator_outputs,
    "make_exact_extractor": _exact_extractor_outputs,
}


def _make_module(request, factory_name, **kwargs):
    factory = request.getfixturevalue(factory_name)
    if factory_name == "make_llm":
        kwargs.update(extractor=_llm_extractor(), tokenizer=_llm_tokenizer())
    elif factory_name == "make_gcp_llm":
        kwargs.update(extractor=_llm_extractor())
    return factory(**kwargs)


class TestFlushing:
    @pytest.mark.parametrize("factory_name,prefix,result_file", FLUSH_CASES)
    def test_remainder_flushed_and_kept_with_save_outputs(
        self, request, tmp_path, factory_name, prefix, result_file
    ):
        module = _make_module(
            request, factory_name, flush_outputs_every_n_steps=100, save_outputs=True
        )
        _attach_trainer(module, tmp_path)
        module.outputs = SAMPLE_OUTPUTS[factory_name]()

        module.on_validation_epoch_end()

        assert (tmp_path / f"{prefix}_rank0_0.pt").exists()
        assert (tmp_path / result_file).exists()
        assert module.flushed_output_files == []
        assert module.outputs == []

    @pytest.mark.parametrize("factory_name,prefix,result_file", FLUSH_CASES)
    def test_flushed_chunks_merged_and_removed(
        self, request, tmp_path, factory_name, prefix, result_file
    ):
        module = _make_module(request, factory_name, flush_outputs_every_n_steps=1)
        _attach_trainer(module, tmp_path)
        outputs = SAMPLE_OUTPUTS[factory_name]()
        module.outputs = outputs[:1]
        module._flush_outputs(prefix=prefix)
        module.outputs = outputs[1:]

        module.on_validation_epoch_end()

        assert not list(tmp_path.glob("*.pt"))
        # One header + one row per sample
        assert len(_read_csv(tmp_path / result_file)) == len(outputs) + 1

    def test_no_flush_by_default(self, make_llm, tmp_path):
        llm = make_llm(extractor=_llm_extractor(), tokenizer=_llm_tokenizer())
        _attach_trainer(llm, tmp_path)
        llm.outputs = _llm_outputs()
        llm.on_validation_epoch_end()
        assert not list(tmp_path.glob("*.pt"))

    def test_periodic_flush_in_validation_step(self, make_gcp_evaluator, tmp_path):
        evaluator = make_gcp_evaluator(flush_outputs_every_n_steps=2)
        _attach_trainer(evaluator, tmp_path)
        batch = {
            "exact_match": [True],
            "question": ["q"],
            "gold_answers": [["A"]],
            "pred_answer": ["A"],
            "confidence": [0.5],
            "raw_output": ["r"],
            "index": [torch.tensor(0)],
        }
        evaluator.validation_step(batch, 0)
        assert evaluator.flushed_output_files == []
        evaluator.validation_step(batch, 1)
        assert [p.split("/")[-1] for p in evaluator.flushed_output_files] == [
            "temp_eval_results_rank0_0.pt"
        ]
        assert evaluator.outputs == []


class TestResume:
    @pytest.mark.parametrize("factory_name", ["make_evaluator", "make_gcp_evaluator"])
    def test_resume_copies_chunks_in_numeric_order(
        self, request, tmp_path, factory_name
    ):
        resume_dir = tmp_path / "prev"
        log_dir = tmp_path / "new"
        resume_dir.mkdir()
        log_dir.mkdir()
        samples = (
            _evaluator_outputs()
            if factory_name == "make_evaluator"
            else _gcp_evaluator_outputs()
        )
        for chunk_idx, sample in zip([0, 1, 10], samples[:3]):
            torch.save([sample], resume_dir / f"temp_eval_results_rank0_{chunk_idx}.pt")
        torch.save([samples[3]], resume_dir / "unrelated_rank0_0.pt")

        module = request.getfixturevalue(factory_name)(resume_from=str(resume_dir))
        _attach_trainer(module, log_dir)
        module.on_validation_start()

        assert module.flushed_output_files == [
            str(log_dir / f"temp_eval_results_rank0_{i}.pt") for i in range(3)
        ]
        assert [
            torch.load(p, weights_only=False)[0]["index"]
            for p in module.flushed_output_files
        ] == [s["index"] for s in samples[:3]]

        module.on_validation_epoch_end()
        # Resumed samples are merged and sorted by index
        rows = _read_csv(log_dir / "evaluation_results.csv")
        assert len(rows) == 4
        # Originals are left untouched
        assert len(list(resume_dir.glob("temp_eval_results_*.pt"))) == 3

    @pytest.mark.parametrize("factory_name", ["make_evaluator", "make_gcp_evaluator"])
    def test_resume_from_missing_path_raises(self, request, tmp_path, factory_name):
        module = request.getfixturevalue(factory_name)(
            resume_from=str(tmp_path / "missing")
        )
        _attach_trainer(module, tmp_path)
        with pytest.raises(ValueError, match="does not exist"):
            module.on_validation_start()

    @pytest.mark.parametrize("factory_name", ["make_evaluator", "make_gcp_evaluator"])
    def test_no_resume_adopts_nothing(self, request, tmp_path, factory_name):
        module = request.getfixturevalue(factory_name)()
        _attach_trainer(module, tmp_path)
        module.on_validation_start()
        assert module.flushed_output_files == []


class TestActivationExtractorResume:
    def _extractor(self, tmp_path, resume_from):
        from callm.models.activation_extractor import ActivationExtractor

        llm = Mock()
        llm.parameters.return_value = []
        extractor = ActivationExtractor(
            llm=llm, output_dir=str(tmp_path / "out"), resume_from=str(resume_from)
        )
        _attach_trainer(extractor, tmp_path / "new")
        return extractor

    def _write_chunks(self, resume_dir):
        resume_dir.mkdir()
        for i in range(2):
            torch.save([{"index": i}], resume_dir / f"activations_chunk_rank0_{i}.pt")

    def test_hardlinks_and_replaces_existing_targets(self, tmp_path):
        import os

        self._write_chunks(tmp_path / "prev")
        (tmp_path / "new").mkdir()
        (tmp_path / "new" / "activations_chunk_rank0_0.pt").write_bytes(b"stale")
        extractor = self._extractor(tmp_path, tmp_path / "prev")

        extractor.on_validation_start()

        assert extractor.sample_counter == 2
        assert extractor.flushed_output_files == [
            str(tmp_path / "new" / f"activations_chunk_rank0_{i}.pt") for i in range(2)
        ]
        for i, path in enumerate(extractor.flushed_output_files):
            src = tmp_path / "prev" / f"activations_chunk_rank0_{i}.pt"
            assert os.path.samefile(path, src)

    def test_falls_back_to_copy_when_link_fails(self, tmp_path):
        import os

        self._write_chunks(tmp_path / "prev")
        extractor = self._extractor(tmp_path, tmp_path / "prev")

        with patch("os.link", side_effect=OSError):
            extractor.on_validation_start()

        for path in extractor.flushed_output_files:
            assert os.path.exists(path)
            assert not os.path.samefile(
                path, tmp_path / "prev" / os.path.basename(path)
            )
