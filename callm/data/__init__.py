"""
Data package — re-exports all data modules for backward compatibility.

TriviaQA data modules live in ``callm.data.triviaqa``;
MMLU data modules live in ``callm.data.mmlu``.
"""

# Base data module
from callm.data.answers_data import AnswersDataModule  # noqa: F401

# TriviaQA data modules
from callm.data.triviaqa import (  # noqa: F401
    TriviaQADataModule,
    UntokenizedTriviaQADataModule,
    IsTrueDataModule,
    UntokenizedIsTrueDataModule,
    EvaluatorDataModule,
    UntokenizedEvaluatorDataModule,
)

# MMLU data modules
from callm.data.mmlu import (  # noqa: F401
    MMLUDataModule,
    UntokenizedMMLUDataModule,
    MMLUIsTrueDataModule,
    UntokenizedMMLUIsTrueDataModule,
)

from .simulation import SimulationDataset, SimulationDataset1D
