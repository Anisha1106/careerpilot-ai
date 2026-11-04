import unittest
from unittest.mock import patch

from app.ai import answer, cosine


def fake_client_response(content):
    message = type("Message", (), {"content": content})()
    completion = type("Completion", (), {"choices": [type("Choice", (), {"message": message})()]})()
    return type("Client", (), {"chat": type("Chat", (), {"completions": type("Completions", (), {"create": lambda self, **kwargs: completion})()})()})()


class RagTests(unittest.TestCase):
    def setUp(self):
        self.contexts = [{"source": "Resume", "content": "Python and FastAPI experience from a project."}]

    def test_local_fallback_returns_retrieved_context_and_source_names(self):
        with patch("app.ai.client", return_value=None):
            result = answer("Which skills are in my resume?", self.contexts)

        self.assertIn("Python", result["answer"])
        self.assertEqual(result["sources"], ["Resume"])

    def test_empty_context_does_not_invent_profile_details(self):
        with patch("app.ai.client", return_value=None):
            result = answer("What did I achieve?", [])

        self.assertIn("don't have enough", result["answer"])
        self.assertEqual(result["sources"], [])

    def test_llm_answer_keeps_source_names(self):
        with patch("app.ai.client", return_value=fake_client_response("Your resume mentions Python.")):
            result = answer("Which skills are in my resume?", self.contexts)

        self.assertEqual(result["answer"], "Your resume mentions Python.")
        self.assertEqual(result["sources"], ["Resume"])

    def test_torch_cosine_handles_equal_and_mismatched_vectors(self):
        self.assertAlmostEqual(cosine([1, 0], [1, 0]), 1.0)
        self.assertEqual(cosine([1], [1, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()