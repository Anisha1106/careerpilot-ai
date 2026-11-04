import unittest
from io import BytesIO

import pymupdf as fitz
from docx import Document

from app.resume import analyze, extract, extract_skills


class ResumeTests(unittest.TestCase):
    def test_extract_skills_normalizes_aliases_without_partial_java_match(self):
        skills = extract_skills("JavaScript, React.js, Postgres, and Fast API")

        self.assertEqual(skills, ["fastapi", "javascript", "postgresql", "react"])
        self.assertEqual(extract_skills("GitHub and Git"), ["git", "github"])

    def test_analyze_finds_sections_and_recommendations(self):
        result = analyze("Education\nSkills: Python\nProjects\nBuilt a small tool.")

        self.assertEqual(result["sections_detected"], ["education", "projects", "skills"])
        self.assertEqual(result["skills"], ["python"])
        self.assertEqual(result["word_count"], 8)
        self.assertTrue(any("experience" in item.lower() for item in result["recommendations"]))

    def test_extract_reads_docx_text(self):
        document = Document()
        document.add_paragraph("Python developer resume")
        buffer = BytesIO()
        document.save(buffer)

        self.assertEqual(extract("resume.docx", buffer.getvalue()), "Python developer resume")

    def test_extract_reads_docx_table_text(self):
        document = Document()
        table = document.add_table(rows=1, cols=2)
        table.cell(0, 0).text = "Skills"
        table.cell(0, 1).text = "Python, SQL"
        buffer = BytesIO()
        document.save(buffer)

        self.assertEqual(extract("resume.docx", buffer.getvalue()), "Skills\nPython, SQL")

    def test_extract_reads_pdf_text(self):
        document = fitz.open()
        page = document.new_page()
        page.insert_text((72, 72), "Python developer resume")
        content = document.tobytes()
        document.close()

        self.assertIn("Python developer resume", extract("resume.pdf", content))

    def test_extract_rejects_unsupported_file_types(self):
        with self.assertRaisesRegex(ValueError, "PDF or DOCX"):
            extract("resume.txt", b"plain text")


if __name__ == "__main__":
    unittest.main()