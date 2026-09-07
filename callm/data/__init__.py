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
    CuratedTriviaQADataModule,
    UntokenizedTriviaQADataModule,
    IsTrueDataModule,
    UntokenizedIsTrueDataModule,
    EvaluatorDataModule,
    UntokenizedEvaluatorDataModule,
    ExactAnswerExtractorDataModule,
)


# MMLU data modules
from callm.data.mmlu import (  # noqa: F401
    MMLUDataModule,
    UntokenizedMMLUDataModule,
    MMLUIsTrueDataModule,
    UntokenizedMMLUIsTrueDataModule,
)

from .simulation import SimulationDataset, SimulationDataset1D  # noqa: F401
from callm.data.activations_data import (  # noqa: F401
    ActivationsDataModule,
    SelectedTokensActivationsDataModule,
)
from callm.data.on_the_fly_data import OnTheFlyActivationsDataModule  # noqa: F401
