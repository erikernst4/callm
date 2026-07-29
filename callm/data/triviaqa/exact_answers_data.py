"""
DataModule for exact answer extraction using Gemini or substring matching.
"""

import ast
import csv
from typing import Optional
from torch.utils.data import DataLoader
from datasets import Dataset

from callm.data.answers_data import AnswersDataModule
from callm.prompts import Prompt


class ExactAnswerExtractorDataModule(AnswersDataModule):
    """DataModule for batched exact answer extraction."""

    def __init__(self, prompt: Optional[Prompt] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt = prompt

    def load_evaluation_csv(self):
        self._resolve_llm_outputs_path()
        rows = []
        with open(self.llm_outputs_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            field_map = {col.lower(): col for col in reader.fieldnames or []}

            for i, row in enumerate(reader):
                q_key = field_map.get("question")
                g_key = field_map.get("gold_answers") or field_map.get("gold answers")
                p_key = field_map.get("pred_answer") or field_map.get(
                    "predicted answer"
                )
                c_key = field_map.get("correct")
                r_key = field_map.get("raw_output") or field_map.get("raw output")
                conf_key = field_map.get("confidence")

                question = row.get(q_key, "") if q_key else ""
                gold_str = row.get(g_key, "") if g_key else ""
                pred_answer = row.get(p_key, "") if p_key else ""
                correct_str = row.get(c_key, "no") if c_key else "no"
                raw_output = (row.get(r_key, "") if r_key else "").replace("\\n", "\n")
                confidence = row.get(conf_key, "nan") if conf_key else "nan"

                # Parse gold answers
                if gold_str.startswith("[") and gold_str.endswith("]"):
                    try:
                        gold_answers = ast.literal_eval(gold_str)
                    except Exception:
                        gold_answers = [
                            g.strip().lower() for g in gold_str.strip("[]").split(",")
                        ]
                else:
                    gold_answers = [g.strip().lower() for g in gold_str.split("|")]

                gold_answers = [str(g).lower().strip() for g in gold_answers]
                correct = correct_str.strip().lower() == "yes"

                rows.append(
                    {
                        "question": question,
                        "gold_answers": gold_answers,
                        "pred_answer": pred_answer,
                        "correct": correct,
                        "raw_output": raw_output,
                        "confidence": confidence,
                        "index": i,
                    }
                )
        return rows

    def setup(self, stage: str = None):
        rows = self.load_evaluation_csv()

        # Limit to max_samples if specified (e.g. 10 for pilot inspection)
        if self.max_samples is not None:
            rows = rows[: self.max_samples]

        prompts = []
        needs_extraction = []
        for row in rows:
            if row["correct"]:
                needs_extraction.append(False)
                prompts.append("")
            else:
                needs_extraction.append(True)
                prompt_text = f"""Extract the short, exact answer entity from the provided long answer (A:).

Rules:
1. The extracted short answer MUST be a verbatim, strict substring copied directly from the long answer (A:). Do NOT paraphrase, rephrase, or alter words.
2. Ensure the short answer is complete (extract full entity names, phrases, or titles).
3. Base your extraction ONLY on the text in the long answer (A:). Do NOT use external knowledge. Even if the long answer states a factually incorrect entity, extract the entity as stated in the long answer.
4. If the long answer (A:) does not contain an answer or states that it does not know, output NO ANSWER.

Q: Which musical featured the song The Street Where You Live?
A: The song "The Street Where You Live" is from the Lerner and Loewe musical **"My Fair Lady."** It is one of the most famous songs from the show, and it is sung by Professor Henry Higgins as he reflects on the transformation of Eliza Doolittle and the memories they have shared together.
Exact answer: My Fair Lady

Q: Which personality of the American West is buried next to her partner Frank Butler?
A: The American West personality buried next to her partner Frank Butler is **Dollie Madison** (often referred to as **Dollie**).
Exact answer: Dollie Madison

Q: What does 'Chipping' indicate in a place name such as Chipping Norton?
A: In place names like **Chipping Norton**, the term **"Chipping"** indicates the presence of a **market town** that historically held a **fair or market**.
Exact answer: market town

Q: What year was the internet invented?
A: I apologize, but I do not have access to that information and cannot answer this question.
Exact answer: NO ANSWER

Q: {row["question"]}
A: {row["raw_output"]}
Exact answer:"""

                prompts.append(prompt_text)

        self.dataset = Dataset.from_dict(
            {
                "input": prompts,
                "needs_extraction": needs_extraction,
                "question": [row["question"] for row in rows],
                "gold_answers": [row["gold_answers"] for row in rows],
                "pred_answer": [row["pred_answer"] for row in rows],
                "correct": [row["correct"] for row in rows],
                "raw_output": [row["raw_output"] for row in rows],
                "confidence": [row["confidence"] for row in rows],
                "index": [row["index"] for row in rows],
            }
        ).with_format("torch")

    def val_dataloader(self):
        def collate_fn(batch):
            return {
                "input": [item["input"] for item in batch],
                "needs_extraction": [item["needs_extraction"] for item in batch],
                "question": [item["question"] for item in batch],
                "gold_answers": [item["gold_answers"] for item in batch],
                "pred_answer": [item["pred_answer"] for item in batch],
                "correct": [item["correct"] for item in batch],
                "raw_output": [item["raw_output"] for item in batch],
                "confidence": [item["confidence"] for item in batch],
                "index": [item["index"] for item in batch],
            }

        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
        )
