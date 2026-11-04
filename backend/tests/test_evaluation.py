import math
import json
import tempfile
import unittest
from pathlib import Path

from evaluate_matching import load_jsonl, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank, validate_case


class RankingMetricTests(unittest.TestCase):
    def setUp(self):
        self.ranked_ids = ["irrelevant", "strong-match", "partial-match", "other"]
        self.relevance = {"strong-match": 2, "partial-match": 1}

    def test_precision_at_k_counts_relevant_items_over_k(self):
        self.assertAlmostEqual(precision_at_k(self.ranked_ids, self.relevance, 3), 2 / 3)

    def test_recall_at_k_counts_relevant_items_found(self):
        self.assertEqual(recall_at_k(self.ranked_ids, self.relevance, 2), 0.5)

    def test_reciprocal_rank_uses_first_relevant_position(self):
        self.assertEqual(reciprocal_rank(self.ranked_ids, self.relevance), 0.5)

    def test_ndcg_uses_graded_relevance_and_ideal_order(self):
        expected = (3 / math.log2(3) + 1 / math.log2(4)) / (3 + 1 / math.log2(3))
        self.assertAlmostEqual(ndcg_at_k(self.ranked_ids, self.relevance, 3), expected)

    def test_metrics_reject_nonpositive_cutoffs(self):
        with self.assertRaises(ValueError):
            precision_at_k(self.ranked_ids, self.relevance, 0)
        with self.assertRaises(ValueError):
            recall_at_k(self.ranked_ids, self.relevance, -1)

    def test_dataset_rejects_relevance_for_unknown_jobs(self):
        case = {
            "profile": {},
            "jobs": [{"id": "job-1"}],
            "relevance": {"job-2": 1},
        }

        with self.assertRaisesRegex(ValueError, "reference job IDs"):
            validate_case(case)

    def test_dataset_rejects_out_of_range_relevance_grades(self):
        case = {
            "profile": {},
            "jobs": [{"id": "job-1"}],
            "relevance": {"job-1": 4},
        }

        with self.assertRaisesRegex(ValueError, "integers from 0 to 3"):
            validate_case(case)

    def test_jsonl_loader_reads_external_cases(self):
        case = {"profile": {}, "jobs": [{"id": "job-1"}], "relevance": {"job-1": 2}}
        with tempfile.TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "cases.jsonl"
            dataset_path.write_text(json.dumps(case) + "\n", encoding="utf-8")

            self.assertEqual(load_jsonl(dataset_path), [case])


if __name__ == "__main__":
    unittest.main()