"""
RAGAS Evaluation Service.

This service runs the RAGAS evaluation framework against a test dataset
and returns scores across 4 dimensions:
- Faithfulness: Does the answer stick to what the documents say?
- Answer Relevancy: Does the answer address the actual question?
- Context Precision: Are the retrieved chunks actually relevant?
- Context Recall: Are all needed chunks being retrieved?

Think of it as a quarterly performance review for your AI system.
Each metric tells you something different about where the system
excels or struggles.

The scores appear in the Streamlit dashboard so you can:
1. See overall quality at a glance
2. Identify which part of the pipeline needs improvement
3. Track quality changes as you add documents or tweak settings
"""
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset
from datetime import datetime
import json
import logging
import os

logger = logging.getLogger(__name__)


class EvaluationService:
    """
    Runs RAGAS evaluation on a test dataset.

    Usage:
        eval_service = EvaluationService(rag_service, retriever)
        results = eval_service.run_evaluation("tests/eval_data/test_questions.json")
    """

    def __init__(self, rag_ask_fn, retriever):
        """
        Args:
            rag_ask_fn: Function that takes a question string and returns an answer string
            retriever: LangChain retriever for getting context
        """
        self.rag_ask_fn = rag_ask_fn
        self.retriever = retriever

    def run_evaluation(self, test_file: str = "tests/eval_data/test_questions.json") -> dict:
        """
        Run RAGAS evaluation on the test dataset.

        For each test question:
        1. Ask the RAG system the question (gets the generated answer)
        2. Retrieve the context chunks (what the system used)
        3. Compare against the ground truth

        Returns:
            Dict with metric scores and metadata
        """
        # Load test questions
        if not os.path.exists(test_file):
            return {"error": f"Test file not found: {test_file}"}

        with open(test_file, "r") as f:
            test_data = json.load(f)

        questions = test_data["questions"]
        logger.info(f"Running RAGAS evaluation on {len(questions)} questions")

        # Collect system outputs for each question
        eval_questions = []
        eval_answers = []
        eval_contexts = []
        eval_ground_truths = []

        for q in questions:
            try:
                # Get the system's answer
                result = self.rag_ask_fn(q["question"])
                answer = result if isinstance(result, str) else result.get("answer", "")

                # Get the retrieved context
                retrieved_docs = self.retriever.invoke(q["question"])
                contexts = [doc.page_content for doc in retrieved_docs[:5]]

                eval_questions.append(q["question"])
                eval_answers.append(answer)
                eval_contexts.append(contexts)
                eval_ground_truths.append(q["ground_truth"])

            except Exception as e:
                logger.warning(f"Skipping question due to error: {e}")
                continue

        if not eval_questions:
            return {"error": "No questions could be evaluated"}

        # Build the RAGAS dataset
        dataset = Dataset.from_dict({
            "question": eval_questions,
            "answer": eval_answers,
            "contexts": eval_contexts,
            "ground_truth": eval_ground_truths,
        })

        # Run RAGAS evaluation
        try:
            results = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            )

            scores = {
                "faithfulness": round(results["faithfulness"], 4),
                "answer_relevancy": round(results["answer_relevancy"], 4),
                "context_precision": round(results["context_precision"], 4),
                "context_recall": round(results["context_recall"], 4),
            }

            # Calculate overall score (average of all metrics)
            scores["overall_score"] = round(
                sum(scores.values()) / len(scores), 4
            )
            scores["questions_evaluated"] = len(eval_questions)
            scores["timestamp"] = datetime.now().isoformat()

            logger.info(f"RAGAS evaluation complete: {scores}")
            return scores

        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return {"error": str(e)}
