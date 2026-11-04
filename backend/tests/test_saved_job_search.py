import unittest

from app.main import search_saved_jobs


class FakeDatabase:
    def __init__(self, rows):
        self.rows = rows

    def execute(self, query, parameters):
        self.parameters = parameters
        return self

    def fetchall(self):
        return self.rows


class SavedJobSearchTests(unittest.TestCase):
    def test_search_filters_saved_jobs_by_keyword_and_location(self):
        job = {
            "id": 1,
            "owner_id": 7,
            "title": "Junior Python Developer",
            "company": "Example Labs",
            "location": "Remote",
            "description": "Build APIs with Python.",
            "url": "https://example.com/jobs/1",
            "source": "User supplied",
            "required_skills": ["Python"],
            "preferred_skills": [],
            "experience_min": 0,
        }
        profile = {
            "skills": ["Python"],
            "projects": [],
            "experience": [],
            "education": "",
            "resume_text": "",
            "preferences": {"roles": ["Python Developer"]},
        }

        results = search_saved_jobs(FakeDatabase([job]), 7, profile, "python", "remote")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], 1)
        self.assertEqual(results[0]["match"]["matching_required_skills"], ["python"])

    def test_search_returns_empty_for_nonmatching_terms(self):
        database = FakeDatabase([])

        results = search_saved_jobs(database, 7, {"skills": [], "projects": [], "experience": [], "preferences": {}}, "designer", "")

        self.assertEqual(results, [])
        self.assertEqual(database.parameters, (7,))


if __name__ == "__main__":
    unittest.main()