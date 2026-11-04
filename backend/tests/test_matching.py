import unittest

from app.matching import normalize, score_match


class MatchingTests(unittest.TestCase):
    def test_normalize_maps_common_skill_aliases(self):
        self.assertEqual(normalize("React.js, Fast API, JS"), "react fastapi javascript")

    def test_structured_requirements_report_matches_and_gaps(self):
        profile = {
            "skills": ["Python", "FastAPI"],
            "projects": [],
            "experience": [],
            "education": "",
            "resume_text": "",
            "preferences": {"roles": ["Backend Engineer"]},
        }
        job = {
            "title": "Backend Engineer",
            "description": "Build Python APIs with FastAPI and PostgreSQL.",
            "required_skills": ["Python", "FastAPI", "PostgreSQL"],
            "preferred_skills": [],
            "experience_min": None,
        }

        result = score_match(profile, job)

        self.assertEqual(result["matching_required_skills"], ["fastapi", "python"])
        self.assertEqual(result["missing_required_skills"], ["postgresql"])
        self.assertFalse(result["eligible"])
        self.assertEqual(result["skill_evidence_source"], "structured job requirements")

    def test_detected_skills_are_not_reported_as_verified_requirements(self):
        profile = {
            "skills": ["Python"],
            "projects": [],
            "experience": [],
            "education": "",
            "resume_text": "",
            "preferences": {"roles": ["API Developer"]},
        }
        job = {
            "title": "API Developer",
            "description": "Develop services with Python and FastAPI.",
            "required_skills": [],
            "preferred_skills": [],
            "experience_min": None,
        }

        result = score_match(profile, job)

        self.assertIsNone(result["eligible"])
        self.assertEqual(result["skill_evidence_source"], "skills detected in job title/description")
        self.assertIn("fastapi", result["missing_detected_skills"])

    def test_experience_below_listed_minimum_is_flagged(self):
        profile = {
            "skills": ["Python"],
            "projects": [],
            "experience": [],
            "education": "",
            "resume_text": "",
            "preferences": {"roles": [], "experience_years": 0},
        }
        job = {
            "title": "Python Engineer",
            "description": "Build Python services.",
            "required_skills": ["Python"],
            "preferred_skills": [],
            "experience_min": 2,
        }

        result = score_match(profile, job)

        self.assertFalse(result["experience_ok"])
        self.assertFalse(result["eligible"])

    def test_score_component_weights_sum_to_one_hundred(self):
        result = score_match(
            {"skills": [], "projects": [], "experience": [], "education": "", "resume_text": ""},
            {
                "title": "Role",
                "description": "",
                "required_skills": [],
                "preferred_skills": [],
                "experience_min": None,
            },
        )

        total_weight = sum(part["weight"] for part in result["score_breakdown"].values())
        self.assertEqual(total_weight, 100.0)
        self.assertEqual(result["score"], 0)


if __name__ == "__main__":
    unittest.main()